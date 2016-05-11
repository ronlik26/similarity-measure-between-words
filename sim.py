import json
import sys, numpy, re, os, math

import re
import scipy.sparse as ss
from scipy.io import mmwrite, mmread
from scipy.spatial.distance import cosine
from collections import Counter

CLEAN_CORPUS = "clean_corpus"
DIGIT_REPR = "<!DIGIT!>"
BEGIN_S = re.compile("\s*<s>\s*")
END_S = re.compile("\s*</s>\s*")
START_D = re.compile("\s*<text id=\w+>\s*")
END_D = re.compile("\s*</text>\s*")
STRIP = re.compile("['.,:;()+\s\"]+")
DIGIT = re.compile("[,.\d+]")


def init():
    index = 0
    col_dic = {}
    for word in preprocess(r".\wacky_wiki_corpus_en1.words", relevance_treshold=20000):
        col_dic[word] = index
        index += 1

    # col_index_word_map = {col_words_index_map[w]: w for w in col_words_index_map}

    row_dic = get_simlex(r".\simlex_words")
    # row_index_word_map = {row_words_index_map[w]: w for w in row_words_index_map}
    return row_dic, col_dic


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


def create_freq_matrix(row_dic, col_dic, k, corpus_fn):
    # TODO Add Loop over the corpus

    M = ss.dok_matrix((len(row_dic), len(col_dic)))
    corpus_f = open(corpus_fn)
    for sent in corpus_f:
        sent = k * ". " + sent + k * " ."
        add_to_matrix_gram(sent, row_dic, col_dic, M, k)

    return M


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def add_to_matrix_gram(sent, row_dic, col_dic, M, k):
    n = 2 * k + 1
    middle = k
    grams = ngrams(sent, n)
    # print grams
    # grams = []
    for gr in grams:
        row_word = gr[middle]
        for word in gr:
            if row_word in row_dic and word in col_dic:
                coordinates = (row_dic.get(row_word), col_dic.get(word))
                M[coordinates] += 1
                # print M[coordinates]


def calculate_probabilies(row_dic, col_dic, M):
    N = M.sum()
    result = ss.dok_matrix((len(row_dic), len(col_dic)))
    for sim_word in row_dic:
        for context_word in col_dic:
            w = row_dic[sim_word]
            c = col_dic[context_word]
            result[(w, c)] = M[(w, c)] / N

    return result


def calculate_smoothed_probabilies(row_dic, col_dic, M, factor):
    N = (factor * len(row_dic) * len(col_dic)) + M.sum()
    for sim_word in row_dic:
        for context_word in col_dic:
            w = row_dic[sim_word]
            c = col_dic[context_word]
            M[(w, c)] = (M[(w, c)] + factor) / N


def calculate_ppmi(row_dic, col_dic, M):
    col_probs = [-1 for _ in range(len(col_dic))]
    for sim_word in row_dic:
        w = row_dic[sim_word]
        prob_sim_word = M[w].sum()
        for context_word in col_dic:
            c = col_dic[context_word]
            if M[(w, c)] == 0:
                continue
            if col_probs[c] == -1:
                col_probs[c] = M[:, c].sum()

            prob_context_word = col_probs[c]
            M[(w, c)] = max(0, math.log2(M[(w, c)] / (prob_sim_word * prob_context_word)))

def get_similarity(w1, w2, row_dic, M):
    vec1 = M[row_dic[w1]]
    vec2 = M[row_dic[w2]]

    return cosine(vec1, vec2)

# evaluation		 +
GOLD_STD_PAIR_PATT = re.compile("(\w+)\s+(\w+)\s+([A-Z])\s+(\d+(?:\.\d+)?)\s*")
def load_simlex_goldstandard(path, words_dic, M):
    test_simlex = {}
    with open(path, 'r') as gold_std_f:
        line = gold_std_f.readline()
        while (line != ""):
            line = gold_std_f.readline()
            pair_match = GOLD_STD_PAIR_PATT.match(line)
            word1 = pair_match.group(1)
            word2 = pair_match.group(2)

            # any pair of simlex words is mapped into triplet such as (POS, gold_standard score, our score)
            result = tuple([pair_match.group(3), pair_match.group(4), get_similarity(word1, word2, words_dic, M)])
            test_simlex[(word1, word2)] = result

    return test_simlex

def main():
    # parse command line options

    # row_dic, col_dic = init()

    # preprocess(r".\wacky_wiki_corpus_en1.words", relevance_treshold=20000)
    with open(r".\rows_indices.json", 'r') as rows_file:
        row_dic = json.load(rows_file)
    with open(r".\cols_indices.json", 'r') as cols_file:
        col_dic = json.load(cols_file)

    # M = create_freq_matrix(row_dic, col_dic, 2, r".\clean_corpus")
    # mmwrite(r".\freq_matrix_2.mtx", M)
    # with open(r".\rows_indices.json", 'w+') as rows_file:
    #     json.dump(row_dic, rows_file)
    # with open(r".\cols_indices.json", 'w+') as cols_file:
    #     json.dump(col_dic, cols_file)

    # loading the matrix
    print("load matrix: context windows of length 5")
    M = mmread(r".\results\freq_matrix_5.mtx").todense()
    print("matrix loaded\n")

    # print("calculating smoothed probabilities")
    # calculate_smoothed_probabilies(row_dic, col_dic, M, 2)
    # print("calculate PPMI")
    # calculate_ppmi(row_dic, col_dic, M)
    # print("recording...")
    # mmwrite(r".\ppmi_smooth_5.mtx", M)

    # print("\n====================================")
    # print("loading matrix: context windows of length 2")
    # M = mmread(r".\freq_matrix_2.mtx").todense()
    # print("matrix loaded\n")
    #
    # print("calculating smoothed probabilities")
    # calculate_smoothed_probabilies(row_dic, col_dic, M, 2)
    # print("calculate PPMI")
    # calculate_ppmi(row_dic, col_dic, M)
    # print("recording...")
    # mmwrite(r".\ppmi_smooth_2.mtx", M)
    #
    # print("finish")

    print(load_simlex_goldstandard(r".\simlex_gold_standard.txt", row_dic, M))


def convert_to_sparse(path_in, path_out):
    with open(path_in, 'r') as inMatrix:
        inMatrix.readline()
        inMatrix.readline()
        shape = tuple([int(num) for num in inMatrix.readline()[:-1].split(" ")])
        with open(path_out, 'w+') as outMatrix:
            outMatrix.write("%%MatrixMarket matrix coordinate integer general\n")
            outMatrix.write("%\n")
            for i in range(shape[0]):
                print(i)
                for j in range(shape[1]):
                    val = inMatrix.readline()
                    if eval(val) == 0:
                        continue

                    outMatrix.write("%d %d %s\n" % (i, j, val))

def test():
    testM = ss.dok_matrix((4, 5))
    col_dic = {
        "computer": 0,
        "data": 1,
        "pinch": 2,
        "result": 3,
        "sugar": 4
    }

    row_dic = {
        "apricot": 0,
        "pineapple": 1,
        "digital": 2,
        "information": 3
    }

    testM[(0, 2)] = 1
    testM[(0, 4)] = 1
    testM[(1, 2)] = 1
    testM[(1, 4)] = 1
    testM[(2, 0)] = 2
    testM[(2, 1)] = 1
    testM[(2, 3)] = 1
    testM[(3, 0)] = 1
    testM[(3, 1)] = 6
    testM[(3, 3)] = 4

    calculate_smoothed_probabilies(row_dic, col_dic, testM, 2)
    # print(probM)
    calculate_ppmi(row_dic, col_dic, testM)
    print(testM)


# test()
# convert_to_sparse(r".\ppmi_smooth_2.mtx", r".\ppmi_smooth_2_sparse.mtx")

if __name__ == "__main__":
    main()
