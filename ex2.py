import json
import sys, numpy, re, os, math
import scipy.stats as st
import re
import scipy.sparse as ss
from scipy.io import mmwrite, mmread
from scipy.spatial.distance import cosine
from collections import Counter

CLEAN_CORPUS = "clean_corpus"
BIG_CORPUS_PATH = r".\wacky_wiki_corpus_en1.words"
SIMLEX_FILE_PATH = r".\simlex_words"
ROWS_INDICES_FILE = r".\rows_indices.json"
COLS_INDICES_FILE = r".\cols_indices.json"
FREQ_MATRIX_2 = r".\freq_matrix_2.mtx"
FREQ_MATRIX_5 = r".\freq_matrix_5.mtx"
PPMI_MATRIX_2 = r".\ppmi_matrix_2.mtx"
PPMI_MATRIX_5 = r".\ppmi_matrix_5.mtx"
CORRELATION_PATH = "correlation.txt"
GOLD_STANDARD_SIMLEX = r".\simlex_gold_standard.txt"
COMP_FREQ_5 = "comp_freq_5.txt"
COMP_FREQ_2 = "comp_freq_2.txt"
COMP_PPMI_5 = "comp_ppmi_5.txt"
COMP_PPMI_2 = "comp_ppmi_2.txt"
ADJ = "A"
NOUN = "N"
VERB = "V"

USAGE_MESSAGE = "Usage:"

DIGIT_REPR = "<!DIGIT!>"
BEGIN_S = re.compile("\s*<s>\s*")
END_S = re.compile("\s*</s>\s*")
START_D = re.compile("\s*<text id=\w+>\s*")
END_D = re.compile("\s*</text>\s*")
STRIP = re.compile("['.,:;()+\s\"]+")
DIGIT = re.compile("[,.\d+]")
GOLD_STD_PAIR_PATT = re.compile("(\w+)\s+(\w+)\s+([A-Z])\s+(\d+(?:\.\d+)?)\s*")

SMOOTH_FACTOR = 2


def init():
    index = 0
    # the columns are the 20,000 most common words
    col_dic = {}
    for word in preprocess(BIG_CORPUS_PATH, relevance_treshold=20000):
        col_dic[word] = index
        index += 1

    # the rows represents the words to measure the similarity between them.
    row_dic = get_simlex(SIMLEX_FILE_PATH)
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
    """
    takes a matrix where each cell is a count,
    and calculates the probability of each cell (co occurrence of words) to occurre
    :param row_dic: map from words to indices for the words at the rows
    :param col_dic: map from words to indices for the words at the columns
    :param M: the matrix to calculate its probabilities
    """
    N = M.sum()
    for sim_word in row_dic:
        for context_word in col_dic:
            w = row_dic[sim_word]
            c = col_dic[context_word]
            M[(w, c)] = M[(w, c)] / N


def calculate_smoothed_probabilies(row_dic, col_dic, M, factor):
    """
    takes a matrix where each cell is a count,
    and calculates the smoothed probability of each cell (co occurrence of words) to occurre,
    means no zero probabilities are allowed!
    :param row_dic: map from words to indices for the words at the rows
    :param col_dic: map from words to indices for the words at the columns
    :param M: the matrix to calculate its probabilities
    """
    N = (factor * len(row_dic) * len(col_dic)) + M.sum()
    for sim_word in row_dic:
        for context_word in col_dic:
            w = row_dic[sim_word]
            c = col_dic[context_word]
            M[(w, c)] = (M[(w, c)] + factor) / N


def calculate_ppmi(row_dic, col_dic, M):
    """
    takes a matrix with probabilities and calculates the PPMI of
    each cell (simlex word with common word).
    :param row_dic: map from words to indices for the words at the rows
    :param col_dic: map from words to indices for the words at the columns
    :param M: the matrix to calculate its PPMI values
    """
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
            M[(w, c)] = max(0, math.log(M[(w, c)] / (prob_sim_word * prob_context_word), 2))


def get_similarity(w1, w2, row_dic, M):
    """
    calculates the similarity between two words
    according to the PPMI values in the given matrix.
    :param row_dic: map of words to indices of the simlex words,
    which represented as rows.
    :param M: the matrix of PPMI values
    :return:
    """
    vec1 = M[row_dic[w1]]
    vec2 = M[row_dic[w2]]

    return cosine(vec1, vec2)


# evaluation		 +
def compare_simlex_words(sim_path, out_path, words_dic, M):
    """
    loading the relevant pairs from the gold standard file
    and calculates our own similarity values according to the values
    in the given matrix.
    :param sim_path: path to the gold standard file.
    :param out_path:patht o save our own file in the same format
    :param words_dic: map form words to indices.
    :param M: matrix with values to calculate similarity.
    """
    with open(sim_path, 'r') as gold_std_f:
        with open(out_path, 'w+') as sim_comp:
            line = gold_std_f.readline()
            while (line != ""):
                line = gold_std_f.readline()
                pair_match = GOLD_STD_PAIR_PATT.match(line)
                if pair_match == None:
                    break
                word1 = pair_match.group(1)
                word2 = pair_match.group(2)
                try:
                    sim_val = get_similarity(word1, word2, words_dic, M)
                except:
                    continue

                sim_comp.write("%s\t%s\t%s\t%s\n" % (word1, word2, pair_match.group(3), sim_val))


def get_simliarty_list(path, POS=None):
    simList = []
    file = open(path, "r")
    for line in file.readlines():
        sim = float(line.split()[3])
        posf = line.split()[2]
        if POS is not None:
            if POS == posf:
                simList.append(sim)
        else:
            simList.append(sim)
    file.close()
    return simList


def calc_correlation(simlex_path, myfile_path, output):
    POS = [None, ADJ, NOUN, VERB]
    for p in POS:
        simlex_sim = get_simliarty_list(simlex_path, p)
        my_sim = get_simliarty_list(myfile_path, p)
        # Full dataset
        if p is None:
            print ("Entire dataset")
            correlation = st.spearmanr(simlex_sim, my_sim)
            output.write("Full correlation " + str(correlation) + "\n")
        # Dataset by specific POS
        else:
            print "POS: " + p
            correlation = st.spearmanr(simlex_sim, my_sim)
            output.write(p + " correlation " + str(correlation) + "\n")


def main():
    # parse command line options
    checkpoint = 0
    if len(sys.argv) > 1:
        try:
            checkpoint = int(sys.argv[1])
        except:
            print(USAGE_MESSAGE)

    if checkpoint != 0:
        with open(ROWS_INDICES_FILE, 'r') as rows_file:
            row_dic = json.load(rows_file)
        with open(COLS_INDICES_FILE, 'r') as cols_file:
            col_dic = json.load(cols_file)

    else:
        row_dic, col_dic = init()
        with open(ROWS_INDICES_FILE, 'w+') as rows_file:
            json.dump(row_dic, rows_file)
        with open(COLS_INDICES_FILE, 'w+') as cols_file:
            json.dump(col_dic, cols_file)

    if checkpoint <= 1:
        print("creating frequency matrix with context windows of 2")
        M = create_freq_matrix(row_dic, col_dic, 2, CLEAN_CORPUS)
        mmwrite(FREQ_MATRIX_2, M)

        M.clear()

        print("creating frequency matrix with context windows of 2")
        M = create_freq_matrix(row_dic, col_dic, 5, CLEAN_CORPUS)
        mmwrite(FREQ_MATRIX_2, M)

        M.clear()

    if checkpoint <= 2:
        print("load matrix: context windows of length 5")
        M = mmread(FREQ_MATRIX_5).todense()
        print("matrix loaded\n")

        print("calculating smoothed probabilities")
        calculate_smoothed_probabilies(row_dic, col_dic, M, SMOOTH_FACTOR)
        print("calculate PPMI")
        calculate_ppmi(row_dic, col_dic, M)
        print("recording...")
        mmwrite(PPMI_MATRIX_5, M)

        M.clear()

        print("\nloading matrix: context windows of length 2")
        M = mmread(FREQ_MATRIX_2).todense()
        print("matrix loaded\n")

        print("calculating smoothed probabilities")
        calculate_smoothed_probabilies(row_dic, col_dic, M, SMOOTH_FACTOR)
        print("calculate PPMI")
        calculate_ppmi(row_dic, col_dic, M)
        print("recording...")
        mmwrite(PPMI_MATRIX_2, M)

        M.clear()

    if checkpoint <= 3:
        print("Compare Stage\n")
        print("load matrix: context windows of length 5")
        M = mmread(FREQ_MATRIX_5).todense()
        print("matrix loaded\n")
        print("compare...")
        compare_simlex_words(GOLD_STANDARD_SIMLEX, COMP_FREQ_5, row_dic, M)
        print("finished")
        M.clear()

        print("load matrix: context windows of length 2")
        M = mmread(FREQ_MATRIX_2).todense()
        print("matrix loaded\n")
        print("compare...")
        compare_simlex_words(GOLD_STANDARD_SIMLEX, COMP_FREQ_2, row_dic, M)
        print("finished")
        M.clear()

        print("load matrix: context windows of length 5")
        M = mmread(PPMI_MATRIX_5).todense()
        print("matrix loaded\n")
        print("compare...")
        compare_simlex_words(GOLD_STANDARD_SIMLEX, COMP_PPMI_5, row_dic, M)
        print("finished")
        M.clear()

        print("load matrix: context windows of length 2")
        M = mmread(PPMI_MATRIX_2).todense()
        print("matrix loaded\n")
        print("compare...")
        compare_simlex_words(GOLD_STANDARD_SIMLEX, COMP_PPMI_2, row_dic, M)
        print("finished")
        M.clear()


    output = open(CORRELATION_PATH, "w")
    print ("frequency 2 window")
    output.write("frequency 2 window\n")
    calc_correlation(GOLD_STANDARD_SIMLEX, COMP_FREQ_2,output)
    print ("frequency 5 window")
    output.write("frequency 5 window\n")
    calc_correlation(GOLD_STANDARD_SIMLEX, COMP_FREQ_5,output)
    print ("ppmi 2 window")
    output.write("ppmi 2 window\n")
    calc_correlation(GOLD_STANDARD_SIMLEX, COMP_PPMI_2,output)
    print ("ppmi 5 window")
    output.write("ppmi 5 window\n")
    calc_correlation(GOLD_STANDARD_SIMLEX, COMP_PPMI_5,output)
    print ("finished")
    output.close()


def test():
    """
    checking the claculation of the ppmi as
    presented in class
    """
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

if __name__ == "__main__":
    main()
