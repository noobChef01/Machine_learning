import torch 
import torch.nn as nn
import torch.nn.functional as F


class PongAgent(nn.Module):

    def __init__(self, in_size, nh, out_sz):
        super().__init__()
        self.fc1 = nn.Linear(in_size, nh) # weights check the environment 
        self.fc_out = nn.Linear(nh, out_sz) # decide to go up or down

    def forward(self, x):
        # x [in_sz] image after (pre-processing)
        if len(x.size()) == 1:
            x = x.squeeze(0)
        # add batch sz dim
        # x [1 x in_sz]
        
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc_out(h))
