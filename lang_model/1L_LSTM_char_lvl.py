#!/usr/bin/env conda run -n torch python
# std libraries
import math
import random
import time

# utils
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
# torchtext libraries
from torchtext.data import BPTTIterator, Field
from torchtext.datasets import LanguageModelingDataset
# tensorboard
from tensorboardX import SummaryWriter

# set seed for deterministic results
def set_seed(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

SEED = 789
set_seed(SEED)

# tensorboard writer
writer = SummaryWriter("runs/exp1")

# custom tokenizer
def tokenizer(text):
    return list(text)

# pre-processing pipeline
TEXT = Field(tokenize=tokenizer, 
            init_token="<sos>", 
            eos_token="<eos>",
            lower=True)

# split and write lyrics.txt to train.txt and valid.txt
lyrics_df = pd.read_csv("lyrics.txt", header=None, 
                        sep="\n", names=None, 
                        index_col=None)
train, valid = train_test_split(lyrics_df, test_size=0.3, 
                                random_state=SEED, shuffle=False)
train.to_csv("train.txt", index=None, 
             header=None, sep="\n")
valid.to_csv("valid.txt", index=None, 
             header=None, sep="\n")

# make datasets from train and test txt files
train_dataset = LanguageModelingDataset("train.txt", TEXT)
valid_dataset = LanguageModelingDataset("valid.txt", TEXT)

# build vocab from train set
TEXT.build_vocab(train_dataset)

# make iters from datasets
batch_sz = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter, valid_iter = BPTTIterator.splits((train_dataset, valid_dataset), 
                    batch_sizes=(batch_sz, batch_sz*2), 
                    bptt_len=30, device=device, repeat=False)

# check out a example batch
batch = next(iter(train_iter))
print(vars(batch).keys())

# check out an example source and target sequence
"".join([TEXT.vocab.itos[idx] for idx in batch.text[:, 0]])
"".join([TEXT.vocab.itos[idx] for idx in batch.target[:, 0]])


# define model
class LanguageModel(nn.Module):

    def __init__(self, vocab_size, emb_sz, hidden_sz, dropout):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_sz)
        self.encoder = nn.LSTM(emb_sz, hidden_sz, bidirectional=False, dropout=0)
        self.fc = nn.Linear(hidden_sz, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text [src_len, batch_sz]
        
        embeddings = self.dropout(self.embedding_layer(text))
        # embeddings [src_len, batch_sz, emb_sz]

        outputs, (hidden, cell) = self.encoder(embeddings)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # squeeze hidden state
        hidden = self.dropout(hidden.squeeze(0))

        return self.fc(hidden)


# Model training
# initialize hyperparams
vocab_size = len(TEXT.vocab) 
emb_sz = 256 
hidden_sz = 512 
# n_layers = 2
dropout = 0.3

# initailize model
model = LanguageModel(vocab_size, emb_sz, hidden_sz, dropout).to(device)

# initialize all named weights using model.apply func and nn.init
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
        
model.apply(init_weights)

# count total trainable paramns in model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_params(model):,} trainable parameters")

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ignore target pad token while calculating loss, as it is calculated per char basis
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

# define loss
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# define training loop
def train(model, iterator, optimizer, criterion, clip):
    # set for training
    model.train()
    running_loss = 0
    for batch in iterator:
        # get input and target sentence
        src = batch.text
        trg = batch.target
        # zero the grads
        optimizer.zero_grad()
        
        preds = model(src) 
        
        # compute loss
        loss = criterion(preds, trg)
        loss.backward()
        
        # clip grads to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(iterator)       

# define evalution loop
def evaluation(model, iterator, criterion):
    # set for evalutation
    model.eval()
    running_loss = 0
    
    # turn off auto-grad
    with torch.no_grad():
        for batch in iterator:
            # get input and target sentences
            src = batch.src
            trg = batch.trg
            
            # get preds and turn off teacher forcing
            output = model(src, trg, teacher_force_ratio=0)
            
            # cut-off first element from trg and output
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            trg = trg[1:].view(-1)
            output = output[1:].view(-1, output_dim)
            # trg = [(trg len-1)*batch size]
            # output = [(trg len-1)*batch size, out_dim]
            
            loss = criterion(output, trg)
            running_loss += loss.item()
        return running_loss / len(iterator)

# time each epoch
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

n_epochs = 10
clip = 1

best_val_loss = float("inf")
for epoch in range(n_epochs):
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, clip)
    valid_loss = evaluation(model, valid_iterator, criterion)   
    end_time = time.time()
    elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)

    # write to tensorboard
    writer.add_scalar("Train Loss", train_loss, global_step=epoch)
    writer.add_scalar("Train PPL", math.exp(train_loss), global_step=epoch)
    writer.add_scalar("Validation Loss", valid_loss, global_step=epoch)
    writer.add_scalar("Validation PPL", math.exp(valid_loss), global_step=epoch)
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), f"checkpoint.pth")
        
    # print(f"Epoch: {epoch+1:02} | Time: {elapsed_mins}min  {elapsed_secs}secs")
    # print(f"\tTrain Loss {train_loss:.3f}| Train PPL {math.exp(train_loss):7.3f}")
    # print(f"\t Valid Loss {valid_loss:.3f}| Valid PPL {math.exp(valid_loss):7.3f}")



