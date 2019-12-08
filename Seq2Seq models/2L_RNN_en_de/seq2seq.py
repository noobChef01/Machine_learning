# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# torchtext libraries
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

# utils
import numpy as np
import spacy

# std libraries
import math
import random
import time

# set seed for deterministic results
SEED = 89

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# load spacy models
en_model = spacy.load("en")
de_model = spacy.load("de")

# custom tokenizers: can be passed to torchtext which takes in 
# a sentence and tokenizes it+ 
def tokenize_de(text):
    # for the german tokens it was found that reversing the order captured 
    # short term dependencies 
    return [token.text for token in de_model.tokenizer(text)][::-1]

def tokenize_en(text):
    return [token.text for token in en_model.tokenizer(text)]

# pre-processing pipeline
SRC = Field(tokenize=tokenize_de, 
            init_token="<sos>", 
            eos_token="<eos>",
            lower=True)

TRG = Field(tokenize=tokenize_en, 
            init_token="<sos>", 
            eos_token="<eos>",
            lower=True)

# load dataset
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"),
               fields=(SRC, TRG))

# check if loaded properly
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

# print out a sentence
print(vars(train_data.examples[0]))

# build vocab: each unique token is given an id for one-hot encoding
# the vocabs for src and target here is different using two fields
# the vocab should be built using train data only to prevent info leakage 
# where we get artificially inflated scores in the valid/test set
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# no of unique tokens:
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

# define the iterators
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bs = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
                                                    (train_data, valid_data, test_data), 
                                                     batch_sizes=(bs, bs*2, bs*2),  
                                                     device=device)


# define encoder
class Encoder(nn.Module):

    def __init__(self, in_sz, emb_sz, hidden_sz, n_layers, dropout):
        self.enc_emb = nn.Embedding(vocab_sz, emb_sz)
        self.enc_rnn = nn.LSTM(emb_sz, hidden_sz, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden_sz = hidden_sz
        self.n_layers = n_layers

    def forward(self, src):
        # [src len, batch sz]
        x = self.dropout(self.enc_emb(src))
        # [src len, batch sz, emb_sz]
        output, (hidden, cell) = self.enc_rnn(x)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer

        return hidden, cell


# define decoder
class Decoder(nn.Module):

    def __init__(self, out_sz, emb_sz, hidden_sz, n_layers, dropout):
        self.hidden_sz = hidden_sz
        self.n_layers = n_layers

        self.dec_emb = nn.Embedding(out_sz, emb_sz)
        self.dec_rnn = nn.LSTM(emb_sz, hidden_sz, n_layers,dropout=dropout)
        self.fc_out = nn.Linear(hidden_sz, out_sz) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input [batch_sz]
        # add sequence length one
        input = input.unsqueeze(0)
        # input [1, batch_sz]
        x = self.dropout(self.dec_emb(input))
        # pass in the encoder hidden and cell
        output, (hidden, cell) = self.dec_rnn(x, (hidden, cell))

        # remove the sequence length
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell
                                            

                                            