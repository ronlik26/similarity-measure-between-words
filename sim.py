import sys, numpy, re, os
from collections import Counter

import math
import scipy.sparse as ss
from nltk.util import ngrams

CLEAN_CORPUS = "clean_corpus"
DIGIT_REPR = "<!DIGIT!>"
BEGIN_S = re.compile("\s*<s>\s*")
END_S = re.compile("\s*</s>\s*")
START_D = re.compile("\s*<text id=\w+>\s*")
END_D = re.compile("\s*</text>\s*")
STRIP = re.compile("['.,:;()+\s\"]+")
DIGIT = re.compile("[,.\d+]")

#TODO make into class
def preprocess(path_to_corpus, relevance_treshold=None):
    """
    this function create new file to be the preprocessed corpus.
    the preprocessing lowering all words, ignoring digits and punctuation.

    :param path_to_corpus: the path to the file contains the corpus.
        the corpus format is as follows:
            Each line corresponds to a word, aside from these special symbols:
            <s>: beginning of sentence
            </s>: end of sentence
            <text id="">: beginning of document
            </text>: end of document

    :return words frequencies
    """
    words_count = Counter()
    with open(path_to_corpus) as raw_c:
        dir = os.path.abspath(
                os.path.join(path_to_corpus, os.pardir))  # TODO check that this is the parent directory of the file.
        print(dir)
        lineNum = 0
        printLine = 0
        with open(dir + os.sep + CLEAN_CORPUS, 'w+') as clean_c:
            sentence = ""
            line = raw_c.readline()
            while line != "":
                line = raw_c.readline()

                lineNum += 1
                printLine += 1
                if printLine == 1000000:
                    print(lineNum)
                    printLine = 0

                if BEGIN_S.match(line):
                    sentence = ""
                elif END_S.match(line):
                    clean_c.write(sentence + "\n")
                elif START_D.match(line):
                    continue
                elif END_D.match(line):
                    continue
                else:
                    clean_word = STRIP.sub("", line).lower()
                    if DIGIT.match(clean_word):
                        clean_word = DIGIT_REPR

                    add = " " if len(sentence) > 0 else ""
                    sentence += add + clean_word

                    # add word to frequency count
                    words_count[clean_word] += 1

    if relevance_treshold is not None:
        return Counter(dict(words_count.most_common(relevance_treshold)))

    return words_count


def get_simlex(path_to_simlex):
    """
    loads the words from the simlex file.
    :param path_to_simlex the file path.
    :return dictionary of words mapped into index representation.
    """
    index = 0
    simlex = {}
    with open(path_to_simlex) as simFile:
        line = simFile.readline()
        while line != "":
            simlex[line[:-1]] = index
            index += 1
            line = simFile.readline()

    return simlex


def create_freq_matrix(corpus_fn, row_dic, col_dic, k):
    # TODO Add Loop over the corpus

    M = ss.dok_matrix((row_dic.len(), col_dic.len()))
    corpus_f = open(corpus_fn)

    for sent in corpus_f:
        sent = k * ". " + sent + k * " ."
        add_to_matrix_gram(sent, row_dic, col_dic, M, k)

    return M


def add_to_matrix_gram(sent, row_dic, col_dic, M, k):
    n = 2 * k + 1
    middle = k
    grams = ngrams(sent.split(), n)
    for gr in grams:
        row_word = gr[middle]
        for word in gr:
            if (row_word in row_dic and word in col_dic):
                coordinates = (row_dic.get(row_word), col_dic.get(word))
                M[coordinates] += 1


def create_ppmi_matrix(row_dic, col_dic, M):
    # TODO Fix dictionary in every place they occuer
    N = 0
    for i in row_dic:
        for j in col_dic:
            coordinates = (row_dic.get(i), col_dic.get(j))
            N += M[coordinates]

    Pmatrix = ss.dok_matrix((row_dic.len(), col_dic.len()))
    for i in row_dic:
        add_to_ppmi_matrix(i, col_dic, M, N, Pmatrix)

def prob_row(cur_rword, row_dic, col_dic, M, N):
    p = 0
    for j in col_dic:
        coordinates = (row_dic.get(cur_rword), col_dic.get(j))
        p += M[coordinates] / N
    return p


def prob_col(cur_cword, row_dic, col_dic, M, N):
    p = 0
    for i in row_dic:
        coordinates = (row_dic.get(i), col_dic.get(cur_cword))
        p += M[coordinates] / N
    return p


def prob_words(row_dic, col_dic, word1, word2, N, M):
    coordinates = (row_dic.get(word1), col_dic.get(word2))
    return M[coordinates] / N


def add_to_ppmi_matrix(row_dic, cur_word, col_dic, M, N, Pmatrix):
    for i in row_dic:
        for j in col_dic:
            coordinates = (row_dic.get(i), col_dic.get(j))
            probr = prob_row(i, row_dic, col_dic, M, N)
            probc = prob_col(j, row_dic, col_dic, M, N)
            probw = prob_words(i, j, N, M)
            ppmi = max(0, math.log(probw / (probr * probc), 2))
            Pmatrix[coordinates] = ppmi


def get_sim(word1, word2, PMatrix):
    pass

# initialization
index = 0
col_words_index_map = {}
for word in preprocess(r".\toy_corpus", relevance_treshold=20000):
    col_words_index_map[word] = index
    index += 1

col_index_word_map = {col_words_index_map[w] : w for w in col_words_index_map}

row_words_index_map = get_simlex(r".\simlex_words")
row_index_word_map = {row_words_index_map[w] : w for w in row_words_index_map}

count_matrix = create_freq_matrix(lambda x: x+1, row_words_index_map, col_words_index_map, 2)
ppmi_matrix = create_ppmi_matrix(row_words_index_map, col_words_index_map, count_matrix)

# evaluation
GOLD_STD_PAIR_PATT = re.compile("(?P<word1>\w+)\s+(?P<word2>\w+)\s+(?P<POS>[A-Z])\s+(?P<score>\d+\.\d+)\s*")
def load_simlex_goldstandard(path):
    test_simlex = {}
    with open(path, 'r') as gold_std_f:
        line = gold_std_f.readline()
        while (line != ""):
            line = gold_std_f.readline()
            pair_match = GOLD_STD_PAIR_PATT.match(line)
            word1 = pair_match.group("word1")
            word2 = pair_match.group("word2")

            # any pair of simlex words is mapped into triplet such as (POS, gold_standard score, our score)
            test_simlex[(word1, word2)] = (pair_match.group("POS"), pair_match.group("score"), get_sim(word1, word2, ppmi_matrix))

    return test_simlex
