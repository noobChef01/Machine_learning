class Seq2SeqRNN(nn.Module):
    def __init__(self, vecs, itos, em_sz, nh, out_sl, nl=2):
        super().__init__()
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.emb = create_emb(vecs, itos, em_sz)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(em_sz, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz, bias=False)
        
        #self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz, em_sz, num_layers=nl, dropout=0.1)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz, len(itos))
        self.out.weight.data = self.emb.weight.data
        
    def forward(self, inp):
        #inp = torch.stack(inp).to(dev)
        sl,bs = inp.size()
        h = self.initHidden(bs).to(dev)
        emb = self.emb_enc_drop(self.emb(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = self.out_enc(h)

        dec_inp = torch.zeros(bs).long().to(dev)
        res = []
        for i in range(self.out_sl):
            emb = self.emb(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = outp.data.max(1)[1]
            if (dec_inp==1).all(): break
        return torch.stack(res)
    
    def initHidden(self, bs): return torch.zeros(self.nl, bs, self.nh)


# bidirectional
class Seq2SeqRNN_Bidir(nn.Module):
    def __init__(self, vecs, itos, em_sz, nh, out_sl, nl=2):
        super().__init__()
        self.emb = create_emb(vecs, itos, em_sz)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.gru_enc = nn.GRU(em_sz, nh, num_layers=nl,
                              dropout=0.25, bidirectional=True)
        self.out_enc = nn.Linear(nh*2, em_sz, bias=False)
        self.drop_enc = nn.Dropout(0.05)
        self.gru_dec = nn.GRU(em_sz, em_sz, num_layers=nl,
                              dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz, len(itos))
        self.out.weight.data = self.emb.weight.data
        
    def forward(self, inp):
        sl,bs = inp.size()
        h = self.initHidden(bs).to(dev)
        emb = self.emb_enc_drop(self.emb(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = h.view(2,2,bs,-1).permute(0,2,1,3).contiguous().view(2,bs,-1)
        h = self.out_enc(self.drop_enc(h))
        dec_inp = torch.zeros(bs).long().to(dev)
        res = []
        for i in range(self.out_sl):
            emb = self.emb(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = outp.data.max(1)[1]
            if (dec_inp==1).all(): break
        return torch.stack(res)
    
    def initHidden(self, bs): 
        return torch.zeros(self.nl*2, bs, self.nh)