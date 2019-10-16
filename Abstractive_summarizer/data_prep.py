import numpy as np
from pp import *
from ext_summ import *
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split


import numpy as np
from tqdm import tqdm
from pp import *
from ext_summ import *
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def ext_lines(text, w_embs, k=2):
    # extract only important sentences
    k = k
    sen, cl_sen = preprocess(text)
    sentence_vectors = get_vec(cl_sen, w_embs)
    sim_mat = np.zeros([len(sen), len(sen)])
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    imp_lines = summary(k, sen, scores,pr=False)
    return imp_lines

def load_vectors(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# load embeddings
word_embeddings = get_embs(input('Enter the absolute path to Embeddings:'))

# take only a smaple for training
np.random.seed(5)
idx = np.random.randint(low=0, high=90000, size=400)

train_folder_path = input('Enter the absolute path to train folder containing the articles:')
articles = []
for file in np.array(glob(train_folder_path+'/*.story'))[idx]:
    articles.append(open(file, 'r', encoding='utf-8').read())

train = pd.DataFrame([get_xy(art) for art in articles], columns=['source', 'target'])

# extarct the top most important sentences
train.source = train.source.apply(lambda x: ' '.join(ext_lines(x, word_embeddings, 3)))

# preprocess data 
train.source = train.source.apply(lambda x: rm_stop(pr_proc(x)))
train.target = train.target.apply(lambda x: rm_stop(' '.join([sen for sen in pr_proc(x).split('\n') if len(sen) > 1])))

# take only articles with short full text
src_lens = [len(src) for src in train.source]
src_wl = np.percentile(src_lens, 30)
msk = [len(str(src)) < src_wl for src in train.source.values]
train = train[msk]

# split to train and test 
train , valid = train_test_split(train, test_size=0.1, random_state=0)

# save splits 
train.to_csv('train.csv', index=None)
valid.to_csv('valid.csv', index=None)
print('Completed Preparation')