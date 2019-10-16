import re
import numpy as np
import nltk
from pp import *
import networkx as nx
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

def read_file(path):
    return open(path, 'r', encoding='utf-8').read()

def preprocess(text):
    # remove highlights
    article = rm_highlight(text.lower())
    # replace contractions
    article = replace_contractions(article)
    # join words
    article = join_words(article)
    sentences = article
    # remove numbers and symbols
    article = rm_num_symb(article)
    sentences = rm_symb(sentences)
    # remove useless spaces
    article = rm_useless_spaces(article)
    sentences = rm_useless_spaces(sentences)
    # split to sentences
    article = sent_tokenize(article)
    sentences = sent_tokenize(sentences)
    # remove '.' 
    article = [rm_dot(sen) for sen in article]
    # remove stopwords,len 1 words and lemmatize
    article = [rm_stop_lemma(sen.split()) for sen in article]
    return sentences, article

def get_embs(file_path):
    word_embeddings = {}
    f = open(file_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def get_vec(sents, word_embeddings):
    dim = len(word_embeddings[','])
    sentence_vectors = []
    for sen in sents:
        if len(sen) != 0:
            v = sum([word_embeddings.get(w, np.zeros((dim,))) for w in sen.split()])/(len(sen.split())+0.001)
        else:
            v = np.zeros((dim,))
        sentence_vectors.append(v)
    return sentence_vectors

def summary(k, sen, scores,pr=True):
    imp_lines = []
    ranked = sorted(((scores[i],s) for i,s in enumerate(sen)), reverse=True)
    if pr: print('\nExtractive Summary:\n')
    for i in range(k):
        if not pr:
            imp_lines.append(ranked[i][1])
        else: print(ranked[i][1]) 
    return imp_lines
        
if __name__ == "__main__":
	# get input path
	file_path = input('Enter the absolute path to document: ')
	article = read_file(file_path)
	# preprocess text
	sen, cl_sen = preprocess(article)

	emb_path = input('\nEnter the absolute path to pre-trained vectors: ')
	# load pre_trained word vectors
	word_embeddings = get_embs(emb_path)

	# generate sentence vectors
	sentence_vectors = get_vec(cl_sen, word_embeddings)

	# similarity matrix
	sim_mat = np.zeros([len(sen), len(sen)])
	nx_graph = nx.from_numpy_array(sim_mat)
	scores = nx.pagerank(nx_graph)

	# print summary
	k = int(input('Number of lines to Sumamrize text in: '))
	summary(k, sen, scores, pr=True)