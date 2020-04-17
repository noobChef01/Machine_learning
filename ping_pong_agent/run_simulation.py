import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

import os
from glob import glob
import json
import numpy as np
from datetime import datetime

from agent import PongAgent
import gym

###### set seed for deterministic results #########

SEED = 89
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

######## helper functions ############

def prepro(img):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    # crop image
    img = img[35:195]
    
    # downsample, take every second element and set to grey scale 
    img = img[::2,::2,0]
    
    img[img == 144] = 0 # erase background (type I) set to black 
    img[img == 109] = 0 # erase background (type II) set to black 
    
    # everything else erase (ball, paddle)
    img[img != 0] = 1 
    
    return img.astype(np.float32).ravel()

def discount_rewards(r):
    """Take input 1D float array of rewards and return discounted rewards."""

    # placeholder for discounted rewards
    discounted_r = np.zeros_like(r)
    
    running_sum = 0
    for t in range(len(r)-1, 0, -1):
        if r[t] != 0.0: running_sum = 0
        running_sum = running_sum * gamma + r[t]
        discounted_r[t] = running_sum
    
    return discounted_r

def train(model, X, Y, dR, optimizer, criterion):
    # set for training
    model.train()
    
    # get input and target sentence
    # zero the grads
    optimizer.zero_grad()
    
    loss = criterion(model(X).squeeze(1), Y.squeeze(1))
    # loss.grad *= dR
    loss.backward()
    optimizer.step()
    print(f"Trained for batch of episodes #{rollout_sz}")  

def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data)

def save_checkpoint(m, ts, m_pth, ver):
    torch.save(m.state_dict(), os.path.join(m_pth, f"{m.__class__.__name__}_{ts}_{ver}.pth"))
    print(f"Model saved to {m_pth} as {m.__class__.__name__}_{ts}_{ver}.pth")

###### meta data and hyperparams #######
with open("config-v0.json") as json_f:
    config = json.load(json_f)

# agent hyperparams
nh = config["agent"]["nh"]
in_sz = config["agent"]["in_sz"]
out_sz = config["agent"]["out_sz"]

# env hyperparams
env_name = config["env"]["env_name"]
render = config["env"]["render"]

# train hyperparams 
rollout_sz = config["train"]["rollout_sz"]
save_every = config["train"]["save_every"]
gamma = config["train"]["gamma"] # discount factor
lr = config["train"]["lr"]
ver = config["train"]["ver"]

######## experiment variables ##############
model = PongAgent(in_sz, nh, out_sz)
model.apply(init_weights)

xs, rs, ys = [], [], [] # episodes data
b_xs, b_drs, b_ys = [], [], [] # batch of episodes 
w_running_reward = 0
reward_sum = 0

env = gym.make(env_name)
observation = env.reset()
prev_x = None

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# define criterion
# criterion = nn.BCELoss()


######### simulate, collect data and train agent ############
for i in range(16):
    ep_n = 0
    if os.listdir("./saved_weights/"):
        # get most recent file
        list_of_files = os.listdir("./saved_weights/")
        latest_file = max([f"./saved_weights/{f}" for f in list_of_files], key=os.path.getctime)
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), latest_file)))
        print(latest_file)

    while ep_n < save_every:    
        # visualize environment
        if render:
            env.render()

        # input as difference of frames to account for motion
        cur_x = prepro(observation)    
        x = cur_x - prev_x if prev_x is not None else np.zeros(in_sz).astype(np.float32)
        prev_x = cur_x

        # convert to tensor for model 
        x_in = torch.from_numpy(x)
        # forward the network ## model need to be in cpu mode 
        with torch.no_grad():
            action_p = model(x_in)

            # roll die and sample and action from returned probability
            action = 2 if np.random.uniform() < action_p.data.item() else 3

        y = 1 if action == 2 else 0 # fake label

        # record the observation and its fake label, i.e, training data
        xs.append(x)
        ys.append(y)

        # sample and execute a action to get new state
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rs.append(reward) 

        # if reward != 0:
        #     win = "win" if reward == 1 else "loss"
        #     print(f"Game finished with {reward}, ep_no: {ep_n}, status: {win}")

        if done: # end of episode
            ep_n += 1

            ep_xs = np.vstack(xs)
            ep_rs = np.vstack(rs)
            ep_ys = np.vstack(ys)
            xs, rs, ys = [], [], [] # re-set memory

            # compute discounted reward for each time-step in the episode
            ep_drs = discount_rewards(ep_rs)
            # scale the data for variance reduction 
            ep_drs -= np.mean(ep_drs)
            ep_drs /= np.std(ep_drs)

            # add to batch collection 
            b_xs.append(ep_xs)
            b_ys.append(ep_ys)
            b_drs.append(ep_drs)

            # accumulate batch data and train model and update its weights
            if ep_n % rollout_sz == 0:
                # gather data
                X, Y, dR = map(torch.from_numpy, (np.vstack(b_xs), np.vstack(b_ys), np.vstack(b_drs)))
                Y = Y.float()
                dR = dR.squeeze(1).float()
                b_xs, b_ys, b_drs = [], [], [] # re-set memory 

                # define criterion
                criterion = nn.BCELoss(weight=dR)
                # optimizer
                # optimizer = optim.Adam

                # train model
                train(model, X, Y, dR, optimizer, criterion)
                X, Y, dR = None, None, None

            w_running_reward = reward_sum if not w_running_reward else w_running_reward * 0.99 + 0.01 * reward_sum
            if ep_n % (rollout_sz*2) == 0:
                print(f"Re-setting env, total episode reward is {reward_sum}, running weighted reward sum is {w_running_reward}")

            # reset environment to default settings
            reward_sum = 0
            observation = env.reset()
            prev_x = None     

    ### save model to disk
    cdir = os.getcwd()
    ts = datetime.now().strftime("%m.%d.%Y.%H_%M_%S")
    save_checkpoint(model, ts, os.path.join(cdir, "saved_weights"), ver)

# close simulation env 
env.close()