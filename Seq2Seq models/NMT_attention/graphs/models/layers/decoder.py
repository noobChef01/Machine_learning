"""
Decoder Layer (1 layer RNN based)
name: encoder.py
date: Jan 2020
author: Sajid Mashroor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.models.layers.attention import Attention


class Decoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.out_sz = config["out_sz"]
        self.emb_layer = nn.Embedding(config["out_sz"], config["emb_sz"])
        self.rnn = nn.GRU(config["enc_hid_dim"]*2 + config["emb_sz"], config["dec_hid_dim"])
        self.attn = Attention(config["enc_hid_dim"], config["dec_hid_dim"]) 
        self.fc_out = nn.Linear(config["enc_hid_dim"]*2 + config["dec_hid_dim"] + config["emb_sz"], config["out_sz"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, input, hidden, enc_outs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)
        # input [1, batch_sz]

        embedded = self.emb_layer(input)
        # embedded [1, batch_sz, emb_sz]

        a = self.attn(hidden, enc_outs)
        # a [batch_sz, src_len]
        a = a.unsqueeze(1)
        # a [batch_sz, 1, src_len]

        enc_outs = enc_outs.permute(1, 0, 2)
        # enc_outs [batch_sz, src_len, enc_hid_dim*2]

        weighted = torch.bmm(a, enc_outs)
        # weighted [batch_sz, 1, enc_hid_dim*2]

        weighted = weighted.permute(1, 0, 2)
        # weighted [1, batch_sz, enc_hid_dim*2]

        rnn_in = torch.cat((embedded, weighted), dim=2)
        # unsqueeze hidden at 0 to satisfy rnn hidden size [nl*nd, batch_sz, dec_hid_dim]
        outputs, hidden = self.rnn(rnn_in, hidden.unsqueeze(0))
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        assert (outputs == hidden).all() # boolean list check

        embedded = embedded.squeeze(0)
        outputs = outputs.squeeze(0)
        weighted = weighted.squeeze(0)
        
        pred = self.fc_out(torch.cat((outputs, weighted, embedded), dim = 1))
        # pred [batch_sz, out]
        
        return pred, hidden.squeeze(0)