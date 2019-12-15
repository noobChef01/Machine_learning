#!/usr/bin/env conda run -n torch python
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
def set_seed(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

SEED = 123
set_seed(SEED)

# load spacy models
en_model = spacy.load("en")
de_model = spacy.load("de")

# tensorboard writer
writer = SummaryWriter("runs/exp1")

# custom tokenizers: can be passed to torchtext which takes in 
# a sentence and tokenizes it
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

    def __init__(self, input_sz, emb_sz, hidden_sz, dropout):
        super().__init__()
        self.enc_emd = nn.Embedding(input_sz, emb_sz)
        self.enc_gru = nn.GRU(emb_sz, hidden_sz)
        self.dropout = nn.Dropout(dropout)
        self.hidden_sz = hidden_sz

    def forward(self, src):
        # src [src_len, batch_sz]
        embeddings = self.dropout(self.enc_emd(src))
        # embeddings [src_len, batch_sz, emb_dim]

        # no cell state
        outputs, hidden = self.enc_gru(embeddings)
        # output [src_len, batch_sz, n_dir*hidden_sz] 
        # hidden [n_layers*n_dir, batch_sz, hidden_sz]

        # return only context vector (last hidden from the layer)
        return hidden


# Define decoder
class Decoder(nn.Module):

    def __init__(self, out_sz, emb_sz, hidden_sz, dropout):
        super().__init__()
        self.dec_emb = nn.Embedding(out_sz, emb_sz)
        self.dec_gru = nn.GRU(emb_sz+hidden_sz, hidden_sz)
        self.fc = nn.Linear(emb_sz + hidden_sz*2, out_sz)
        self.dropout = nn.Dropout(dropout)
        self.hidden_sz = hidden_sz
        self.out_sz = out_sz

    def forward(self, input, hidden, context):
        # input [batch_sz]
        # hidden [1, batch_sz, hidden_sz]
        # context [1, batch_sz, hidden_sz]

        # unsqueeze inputs
        input = input.unsqueeze(0)
        # input [1, batch_sz]

        embeddings = self.dropout(self.dec_emb(input))
        # embeddings [1, batch_sz, emb_sz]

        emb_cont = torch.cat((embeddings, context), dim=2)
        # emb_cont [1, batch_sz, emb_sz + hidden_sz]

        outputs, hidden = self.dec_gru(emb_cont, hidden)
        # outputs [1, batch_sz, hidden_sz]
        # hidden [1, batch_sz, hidden_sz]

        fc_in = torch.cat((outputs.squeeze(0), 
                           embeddings.squeeze(0), 
                           context.squeeze(0)), 
                           dim=1)

        pred = self.fc(fc_in)

        return pred, hidden


# define seq2seq
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_sz == decoder.hidden_sz, \
            "Hidden dimension of encoder and decoder must be same!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [src_len, batch_sz]
        # trg [trg_len, batch_sz]

        # create outputs vector (of preds) same dim as trg 
        batch_sz = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_sz = self.decoder.out_sz

        outputs = torch.zeros(trg_len, batch_sz, trg_vocab_sz).to(self.device)

        # context vector
        context = self.encoder(src)

        # set initial hidden state of decoder to context
        hidden = context

        # slice first token to input to decoder
        input = trg[0, :]

        # decoder loop
        for t in range(1, trg_len):
            # pass inputs and states to decoder 
            pred, hidden = self.decoder(input, hidden, context)

            # place pred in outputs vector
            outputs[t] = pred

            # decide to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get top 1 pred
            top1 = pred.argmax(1)

            # based on teacher_force modify input or not
            input = trg[t] if teacher_force else top1

        return outputs


# Model training

# initialize hyperparams
in_sz = len(SRC.vocab)
out_sz = len(TRG.vocab)
# vocab is huge try compressing 
emb_sz = 256 
# final size of the context vectors try to capture more info
hidden_sz = 512 
enc_dropout = 0.5
dec_dropout = 0.5

# initailize model parts
encoder = Encoder(in_sz, emb_sz, hidden_sz, enc_dropout)
decoder = Decoder(out_sz, emb_sz, hidden_sz, dec_dropout)

# initialize model and pass it to device
model = Seq2Seq(encoder, decoder, device).to(device)

# do weight initialization using normal distribution
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
            output = model(src, trg, teacher_forcing_ratio=0)
            
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
        
    print(f"Epoch: {epoch+1:02} | Time: {elapsed_mins}min  {elapsed_secs}secs")
    print(f"\tTrain Loss {train_loss:.3f}| Train PPL {math.exp(train_loss):7.3f}")
    print(f"\t Valid Loss {valid_loss:.3f}| Valid PPL {math.exp(valid_loss):7.3f}")

# Out-of sample metrics
model.load_state_dict(torch.load(f"checkpoint.pth"))

test_loss = evaluation(model, test_iterator, criterion)
print(f"\tTest Loss {test_loss:.3f}| Train PPL {math.exp(test_loss):7.3f}")

def save_checkpoint(h_params, model, time_stamp, model_path):
    model_meta = h_params
    torch.save(model_meta, f"{model_path}{model.__class__.__name__}_{time_stamp}.pth")
    return f"Model saved to {model_path}"

# visualize trained embeddings
model = model.to("cpu")
embedding = model.encoder.enc_emd.weight
labels = [SRC.vocab.itos[i] for i in range(len(SRC.vocab.itos))]
writer.add_embedding(embedding)

writer.export_scalars_to_json("all_scalars.json")
writer.close()  




