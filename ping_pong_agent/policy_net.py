import torch 
import torch.nn as nn
import torch.nn.functional as F


class PongAgent(nn.Module):

    def __init__(self, in_size, nh, out_sz):
        super().__init__()
        # after wards change the bias
        self.fc1 = nn.Linear(in_size, nh) # weights check the positon of paddles and ball 
        self.fc_out = nn.Linear(nh, out_sz) # decide to go up or down

    def forward(self, x):
        # x [batch_sz x in_sz] images after (pre-processing)
        
        h = F.relu(self.fc1(x))
        lop_p = self.fc_out(h)
        return torch.sigmoid(lop_p)
