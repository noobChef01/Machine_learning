#!/usr/bin/env conda run -n torch
# std libraries
import math
import random
import time

# utils
import numpy as np
import spacy
# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator, Field
# torchtext libraries
from torchtext.datasets import Multi30k, TranslationDataset
from tensorboardX import SummaryWriter

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

# tensorboard writer
writer = SummaryWriter("runs/exp1")

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
        super().__init__()
        self.hidden_sz = hidden_sz
        self.n_layers = n_layers
        
        self.enc_emb = nn.Embedding(in_sz, emb_sz)
        self.enc_rnn = nn.LSTM(emb_sz, hidden_sz, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)        

    def forward(self, src):
        # [src len, batch sz]
        x = self.dropout(self.enc_emb(src))
        # [src len, batch sz, emb_sz]
        output, (hidden, cell) = self.enc_rnn(x)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer for each time step 
        # in sequence
        return hidden, cell

# define decoder
class Decoder(nn.Module):

    def __init__(self, out_sz, emb_sz, hidden_sz, n_layers, dropout):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.n_layers = n_layers
        self.out_sz = out_sz

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

# define seq2seq model
class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # for this case encoder and decoder hidden_sz should be same 
        # same for the case of encoder and decoder n_layers
        assert encoder.hidden_sz == decoder.hidden_sz
        assert encoder.n_layers == decoder.n_layers
        
    def forward(self, src, trg, teacher_force_ratio=0.5):
        # src[src len, batch_sz]
        # trg[trg len, batch_sz]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_sz = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_sz = self.decoder.out_sz
        
        # tensor to store predictions
        output = torch.zeros(trg_len, batch_sz, trg_vocab_sz).to(self.device)
        
        # encoder results 
        hidden, cell = self.encoder(src)
        
        # first token of trg
        input = trg[0, :]
        
        # run the decoding loop
        for t in range(1, trg_len):
            # decode the encoder inputs
            pred, hidden, cell = self.decoder(input, hidden, cell)
            
            # place preds in output tensor  
            output[t] = pred
            
            # top 1 prediction
            top1 = pred.argmax(1)
            
            # decide if we will use teacher forcing or not
            teacher_force = random.random() < teacher_force_ratio
            input = trg[t] if teacher_force else top1
        return output     

# Model training

# initialize hyperparams
in_sz = len(SRC.vocab)
out_sz = len(TRG.vocab)
# vocab is huge try compressing 
emb_sz = 256 
# final size of the context vectors try to capture more info
hidden_sz = 512 
n_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# initailize model parts
encoder = Encoder(in_sz, emb_sz, hidden_sz, n_layers, enc_dropout)
decoder = Decoder(out_sz, emb_sz, hidden_sz, n_layers, dec_dropout)

# initialize model and pass it to device
model = Seq2Seq(encoder, decoder, device).to(device)

# initialize all named weights using model.apply func and nn.init
# here we'll use a uniform distribution between -0.08 and +0.08 (as in orginal paper) 
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

# count total trainable paramns in model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_params(model):,} trainable parameters")

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ignore target pad token while calculating loss, as it is calculated per word basis
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

# define loss
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# define training loop
def train(model, iterator, optimizer, criterion, clip):
    # set for training
    model.train()
    running_loss = 0
    for batch in iterator:
        # get input and target sentence
        src = batch.src
        trg = batch.trg
        # zero the grads
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # cut-off first element from trg and output
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        trg = trg[1:].view(-1)
        output = output[1:].view(-1, output_dim)
        
        # trg [(trg len-1)*batch size]
        # output[(trg len-1)*batch size, output dim]
        
        loss = criterion(output, trg)
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

# Out-of sample metrics
model.load_state_dict(torch.load(f"checkpoint.pth"))

test_loss = evaluation(model, test_iterator, criterion)
print(f"\tTest Loss {test_loss:.3f}| Train PPL {math.exp(test_loss):7.3f}")

def save_checkpoint(h_params, model, time_stamp, model_path):
    model_meta = h_params
    torch.save(model_meta, f'{model_path}{model.__class__.__name__}_{time_stamp}.pth')
    return f'Model saved to {model_path}'

writer.export_scalars_to_json("./all_scalars.json")
writer.close()