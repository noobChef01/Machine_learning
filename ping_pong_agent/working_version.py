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

###### set seed for deterministic results #########

SEED = 89
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

## policy network/agent

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

# functions
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data)

def save_checkpoint(m, ts, m_pth, ver):
    torch.save(m.state_dict(), os.path.join(m_pth, f"{m.__class__.__name__}_{ts}_{ver}.pth"))
    print(f"Model saved to {m_pth} as {m.__class__.__name__}_{ts}_{ver}.pth")
    
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

def discount_rewards(r, gamma=0.99):
    """Take input 1D float array of rewards and return discounted rewards."""

    # placeholder for discounted rewards
    discounted_r = np.zeros_like(r)
    
    running_sum = 0
    for t in range(len(r)-1, 0, -1):
        if r[t] != 0.0: running_sum = 0
        running_sum = running_sum * gamma + r[t]
        discounted_r[t] = running_sum
        
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    
    return discounted_r


# read hyperparams file
with open("config-v1.json") as json_f:
    config = json.load(json_f)

# instantiate tensorboard
writer = SummaryWriter()

# instantiate env obj
env = gym.make(config["env_name"])
render_env = config["render"]

gamma = config["gamma"]
save_every = config["save_every"]
rollout_sz = config["rollout_sz"]
n_epochs = config["n_epochs"]
ent_beta = config["ent_beta"]

# device to use
dev = "cuda" if torch.cuda.is_available() else "cpu"
#dev = "cpu"
device = torch.device(dev)
print(f"Runnign on device {dev}")

# agent to drive our paddle 
policy_net = Agent(config["agent"]["in_sz"], config["agent"]["nh"],env.action_space.n) 
#policy_net.apply(init_weights)
policy_net.to(device)

# Adam for agent weight updates
optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])

def calculate_loss(batch_logits, batch_weighted_log_p, beta=0.1):
    policy_loss = -1 * torch.mean(batch_weighted_log_p)
    
    # add the entropy bonus
    p = softmax(batch_logits, dim=1)
    
    log_p = log_softmax(batch_logits, dim=1)
    
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    
    entropy_bonus = -1 * beta * entropy
    
    return policy_loss + entropy_bonus


## run experiments 
def train():
    epoch = 0
    while epoch <= n_epochs:
        ep_n = 0
        
        w_running_reward = None
        # buffers
        batch_weighted_log_p = torch.empty(size=(0,), dtype=torch.float, device=device)
        batch_logits = torch.empty(size=(0, env.action_space.n), device=device)

        while ep_n < save_every:
            # play episode and collect loss and logits
            episode_weighted_log_p, episode_logits, episode_rewards = play_ep()
            w_running_reward = np.sum(episode_rewards) if not w_running_reward else w_running_reward * 0.99 + 0.01 * np.sum(episode_rewards)
            print('resetting env. episode reward total was %f. running mean: %f' % (np.sum(episode_rewards), w_running_reward))
            batch_weighted_log_p = torch.cat((batch_weighted_log_p, episode_weighted_log_p), dim=0)
            batch_logits = torch.cat((batch_logits, episode_logits), dim=0)
            ep_n += 1
            # if ep_n % 10 == 0:
            print(f"Ep_no: {ep_n}")
            
            if ep_n % rollout_sz == 0:
                # compute loss 
                loss = calculate_loss(batch_logits, batch_weighted_log_p, ent_beta)
                # zero the grads
                optimizer.zero_grad()
                # backprop
                loss.backward()
                optimizer.step()
                print(f"batch loss {loss}")
                # reset the epoch arrays
                # used for entropy calculation
                batch_logits = torch.empty(size=(0, env.action_space.n), device=device)
                batch_weighted_log_p = torch.empty(size=(0,), dtype=torch.float, device=device)
                # print(f"Running weighted reward {w_running_reward}")

        cdir = os.getcwd()
        ts = datetime.now().strftime("%m.%d.%Y.%H_%M_%S")
        save_checkpoint(policy_net, ts, os.path.join(cdir, "saved_weights"), epoch)
        # writer.add_scalar(tag=f'Running batch Loss after {save_every} episodes', 
        #                   scalar_value=loss,
        #                   global_step=epoch)        
        # writer.add_scalar(tag=f'Weighted Return over {save_every} episodes', 
        #                   scalar_value=w_running_reward,
        #                   global_step=epoch)  
        epoch += 1
        print(f"epoch no {epoch}")

## play episode
def play_ep():
    # reset env state after every episode
    state = env.reset() 
    prev_x = None
    episode_actions = torch.empty(size=(0,), dtype=torch.long,device=device)
    episode_logits = torch.empty(size=(0, env.action_space.n),device=device)
    average_rewards = np.empty(shape=(0,), dtype=np.float)
    episode_rewards = np.empty(shape=(0,), dtype=np.float)
    
    while True:
        # render env for display 
        if render_env:
            env.render()

        # pre-preprocess current the state and subtract fromprevious state to add in motion information
        cur_x = prepro(state)    
        x = cur_x - prev_x if prev_x is not None else np.zeros(config["agent"]["in_sz"]).astype(np.float32)
        prev_x = cur_x

        # get choice from network
        action_log_p = policy_net(torch.tensor(x).float().unsqueeze(0).to(device))
        # add to buffer
        episode_logits = torch.cat((episode_logits, action_log_p),dim=0)
        # sample and action and execute the action 
        action = Categorical(logits=action_log_p).sample()
        # add to buffer
        episode_actions = torch.cat((episode_actions, action),dim=0)

        state, reward, done, _ = env.step(action=action.cpu().item())
        
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print(f"ep #: game finished, reward: {reward}" + ('' if reward == -1 else ' !!!!!!!!'))

        # add to buffer 
        episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

        # average rewards after every action and add to buffer for computing baseline
        # like averaging from 1 to nth time step (on-average return till that time step)
        average_rewards = np.concatenate((average_rewards, np.expand_dims(np.mean(episode_rewards), axis=0)), axis=0)
            
        if done: # end of episode
            # get discounted rewards and normalize the return
            discounted_rewards = discount_rewards(episode_rewards, gamma=gamma)
                
            # subtract baseline rewards 
            discounted_rewards -= average_rewards
                
            # set mask for the actions executed 
            mask = one_hot(episode_actions, num_classes=env.action_space.n)
                
            # similar to cross-entropy for classification but with fake labels and our action confidence
            episode_loss = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
                
            # weight the loss with the discounted rewards to get expected reward from distribution 
            episode_weighted_log_p = episode_loss * torch.tensor(discounted_rewards).float().to(device)
            
            return episode_weighted_log_p, episode_logits, episode_rewards

train()
env.close()