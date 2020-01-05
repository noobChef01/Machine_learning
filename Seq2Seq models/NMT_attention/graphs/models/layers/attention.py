"""
Bahdanau Attention layer
name: attention.py
date: Jan 2020
author: Sajid Mashroor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.fc = nn.Linear(enc_hid_dim*2 + dec_hid_dim, dec_hid_dim)
        # normalize the hidden values and get vector 
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, enc_outs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = enc_outs.shape[1]
        src_len = enc_outs.shape[0]
        
        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        enc_outs = enc_outs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.fc(torch.cat((hidden, enc_outs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]
        
        energy = energy.permute(0, 2, 1)
        
        #energy = [batch size, dec hid dim, src len]
        
        #v = [dec hid dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        #v = [batch size, 1, dec hid dim]
                
        attention = torch.bmm(v, energy).squeeze(1)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)