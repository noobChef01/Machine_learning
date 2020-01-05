"""
Seq2Seq Network
name: network.py
date: Jan 2020
author: Sajid Mashroor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.models.layers.encoder import Encoder
from graphs.models.layers.decoder import Decoder
import random


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [src_len, batch_sz]
        # trg [trg_len, batch_sz]

        trg_len = trg.shape[0]
        batch_sz = trg.shape[1]
        out_sz = self.decoder.out_sz

        outputs = torch.zeros(trg_len, batch_sz, out_sz).to(self.device)

        # pass src to encoder
        enc_outs, hidden = self.encoder(src)

        # first input to decoder is <sos>
        input = trg[0, :]

        for t in range(1, trg_len):
            pred, hidden = self.decoder(input, hidden, enc_outs)
            outputs[t] = pred
            teacher_force = random.random() < teacher_forcing_ratio
            # best prediction
            top1 = pred.argmax(1)
            # use teacher force
            input = trg[t] if teacher_force else top1
        return outputs     