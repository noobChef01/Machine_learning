import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.nn.functional import one_hot, logsigmoid, log_softmax, softmax

import os
from glob import glob
import json
import numpy as np
from datetime import datetime

import gym


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    
    # crop image
    I = I[35:195]
    
    # downsample, take every second element and set to grey scale 
    I = I[::2,::2,0]
    
    I[I == 144] = 0 # erase background (type I) set to black 
    I[I == 109] = 0 # erase background (type II) set to black 
    
    # everything else erase (ball, paddle)
    I[I != 0] = 1 
    
    return I.astype(np.float32).ravel()


class Agent(nn.Module):

    def __init__(self, in_sz, nh, out_sz):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, nh) # weights check the positon of paddles and ball 
        self.fc_out = nn.Linear(nh, out_sz) # decide to go up or down

    def forward(self, x):
        # x [batch_sz x in_sz] images after (pre-processing)
        h = F.relu(self.fc1(x))
        lop_p = self.fc_out(h)
        return lop_p

env = gym.make("Pong-v0")
agent = Agent(6400,200, env.action_space.n)
agent.load_state_dict(torch.load("saved_weights/b_16/Agent_04.24.2020.00_52_00_5.pth"))



state = env.reset()
prev_x = None

while True:
    
    env.render()
    cur_x = prepro(state)    
    x = cur_x - prev_x if prev_x is not None else np.zeros(6400).astype(np.float32)
    prev_x = cur_x

    action_logp = agent(torch.tensor(x).float().unsqueeze(0))
    action_p = softmax(action_logp, dim=1).detach().numpy()
    action = np.argmax(action_p)

    state, reward, done, _ = env.step(action=action)
    print(reward)
    # time.sleep(2)

    if done:
        state = env.reset()
        prev_x = None