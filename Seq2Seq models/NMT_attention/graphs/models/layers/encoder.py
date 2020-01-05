"""
Encoder Layer (1 layer bi-RNN based)
name: encoder.py
date: Jan 2020
author: Sajid Mashroor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()        
        self.emb_layer = nn.Embedding(config["in_sz"], config["emb_sz"])        
        self.rnn = nn.GRU(config["emb_sz"], config["enc_hid_dim"], bidirectional = True)
        self.fc = nn.Linear(config["enc_hid_dim"] * 2, config["dec_hid_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, src):        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.emb_layer(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs