
import sys, numpy
import scipy.sparse as ss
from nltk.util import ngrams




def create_freq_matrix(row_dic, col_dic, k):

#TODO Add Loop over the corpus

	M = ss.dok_matrix((row_dic.len(),col_dic.len()))
	corpus_f = open(corpus_fn)
    for sent in corpus_f:
			sent = k*". " + sent + k*" ."	
			add_to_matrix(sent, row_dic, col_dic, M, k)
	
	return M
	
	
	
	
def add_to_matrix_gram(sent, row_dic, col_dic, M, k):

	n = 2*k + 1
	middle = k
	grams = ngrams(sent.split(), n)
	for gr in grams:
		row_word = gr[middle]
		for word in gr:
			if (row_word in row_dic and word in col_dic):
				coordinates = (row_dic.get(row_word),col_dic.get(word))
				M[coordinates] += 1
				
				
				
def create_ppmi_matrix(row_dic, col_dic, M):
#TODO Fix dictionary in every place they occuer
	N = 0
	for i in row_dic:
		for j in col_dic:
			coordinates = (row_dic.get(i),col_dic.get(j))
			N += M[coordinates]
			
	for i in row_dic:
		add_to_ppmi_matrix(i, col_dic, M, N, Pmatrix)
		
		
def prob_row(cur_rword, row_dic, col_dic, M, N):
	p = 0
	for j in col_dic:
		coordinates = (row_dic.get(cur_rword),col_dic.get(j))
		p += M[coordinates] / N
	return p
	
def prob_col(cur_cword, row_dic, col_dic, M, N):
	p = 0
	for i in row_dic:
		coordinates = (row_dic.get(i),col_dic.get(cur_cword))
		p += M[coordinates] / N
	return p
	
def prob_words(word1, word2, N, M):
	coordinates = (row_dic.get(word1),col_dic.get(word2))
	return M[coordinates] / N
		
		
		
def add_to_ppmi_matrix(cur_word, col_dic, M, N, Pmatrix):
	for i in row_dic:
		for j in col_dic:
			coordinates = (row_dic.get(i),col_dic.get(j))
			probr = prob_row(i, row_dic, col_dic, M, N)
			probc = prob_col(j, row_dic, col_dic, M, N)
			probw = prob_words(i, j, N, M)
			ppmi = max(0, math.log(probw / (probr * probc),2))
			Pmatrix[coordinates] = ppmi
	

	
			
		
	







