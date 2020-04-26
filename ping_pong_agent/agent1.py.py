'''Using Class and torch (the other guy implementation)'''


import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.nn.functional import one_hot, log_softmax, softmax

import os
from glob import glob
import json
import numpy as np
from datetime import datetime

import gym


def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

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

def calculate_loss(batch_logits, batch_weighted_loss, beta=0.1):
    policy_loss = -1 * torch.mean(batch_weighted_loss)
    
    # add the entropy bonus
    p = softmax(batch_logits, dim=1)
    
    log_p = log_softmax(batch_logits, dim=1)
    
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    
    entropy_bonus = -1 * beta * entropy
    
    return policy_loss + entropy_bonus



class PolicyNetwork(nn.Module):

    def __init__(self, in_sz, nh, out_sz):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, nh) # weights check the positon of paddles and ball 
        # self.fc2 = nn.Linear(nh, 200)
        self.fc_out = nn.Linear(nh, out_sz) # decide to go up or down

    def forward(self, x):
        # x [batch_sz x in_sz] images after (pre-processing)
        h = F.relu(self.fc1(x))
        logit = self.fc_out(h)
        return logit


class PolicyGradient:
    def __init__(self, config):
        self.config = config
        # instantiate tensorboard
        self.writer = SummaryWriter()

        # instantiate env obj
        self.env = gym.make(config["env_name"])
        self.render_env = config["render"]

        self.gamma = config["discount_factor"]
        self.save_every = config["save_every"]
        self.rollout_sz = config["rollout_sz"]
        self.n_epochs = config["n_epochs"]
        self.entropy_beta = config["ent_beta"]
        self.n_epochs = config["n_epochs"]
        # device to use
        use_cuda = config["cuda"]
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        if use_cuda:
            print("--------------------Running on GPU------------------------")
        else:
            print("--------------------Running on CPU------------------------") 
        ### policy network hyperparams ###
        self.in_sz = config["agent"]["in_sz"]
        nh = config["agent"]["nh"]
        
        self.agent = PolicyNetwork(self.in_sz, nh, 2) 
        #self.policy_net.apply(init_weights)
        self.agent.to(self.device)

        # Adam for agent weight updates
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["lr"])

    def play_ep(self):
        # reset env state after every episode
        state = self.env.reset() 
        prev_x = None
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.device)
        episode_logits = torch.empty(size=(0, 2),device=self.device)
        average_rewards = np.empty(shape=(0,), dtype=np.float)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)
    
        while True:
            # render env for display 
            if self.render_env:
                self.env.render()

            # pre-preprocess current the state and subtract from previous state to add-in motion information
            cur_x = prepro(state)    
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.in_sz).astype(np.float32)
            prev_x = cur_x

            # get choice from network
            action_logit = self.agent(torch.tensor(x).float().unsqueeze(0).to(self.device))
            # add to buffer
            episode_logits = torch.cat((episode_logits, action_logit), dim=0)
            # sample and action and execute the action 
            action = Categorical(logits=action_logit).sample()
            # add to buffer
            episode_actions = torch.cat((episode_actions, action),dim=0)

            state, reward, done, _ = self.env.step(action=action.cpu().item())

            # add to buffer 
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)
            
            # like averaging from 1 to nth time step (on-average return till that time step)
            average_rewards = np.concatenate((average_rewards, np.expand_dims(np.mean(episode_rewards), axis=0)), axis=0)

            if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                print(('ep #: game finished, reward: %f' % (reward)) + ('' if reward == -1 else ' !!!!!!!!'))
                
            if done: # end of episode
                # get discounted rewards and normalize the return
                discounted_rewards = discount_rewards(episode_rewards, gamma=self.gamma)
                    
                # subtract baseline rewards 
                discounted_rewards -= average_rewards
                    
                # set mask for the actions executed 
                mask = one_hot(episode_actions, num_classes=2)
                
                # similar to cross-entropy for classification but with fake labels and our action confidence
                weighted_ps = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
                    
                # weight the loss with the discounted rewards to get expected reward from distribution 
                episode_weighted_loss = weighted_ps * torch.tensor(discounted_rewards).float().to(self.device)
                
                return episode_weighted_loss, episode_logits, episode_rewards

     
    def solve_problem(self):
        epoch = 0
        while epoch <= self.n_epochs:
            ep_n = 0            
            w_running_reward = None
            # buffers
            batch_weighted_loss = torch.empty(size=(0,), dtype=torch.float, device=self.device)
            batch_logits = torch.empty(size=(0, 2), device=self.device)

            while ep_n < self.save_every:
                print(f"ep no: {ep_n}")

                # play episode and collect loss and logits
                episode_weighted_loss, episode_logits, episode_rewards = self.play_ep()
                w_running_reward = np.sum(episode_rewards) if not w_running_reward else w_running_reward * 0.99 + 0.01 * np.sum(episode_rewards)

                print(f"resetting env. episode reward total was {np.sum(episode_rewards)}, running mean: {w_running_reward}")

                batch_weighted_loss = torch.cat((batch_weighted_loss, episode_weighted_loss), dim=0)
                batch_logits = torch.cat((batch_logits, episode_logits), dim=0)
                ep_n += 1
                
                # if ep_n % 10 == 0:
                #     print(f"Ep_no: {ep_n}")

                if ep_n % self.rollout_sz == 0:
                    # zero the grads
                    self.optimizer.zero_grad()
                    # compute loss 
                    loss = calculate_loss(batch_logits, batch_weighted_loss, self.entropy_beta)
                    # backprop
                    loss.backward()
                    
                    self.optimizer.step()
                    
                    # print(f"batch loss {loss:.2f}")
                    # reset the epoch arrays
                    # used for entropy calculation
                    batch_logits = torch.empty(size=(0, 2), device=self.device)
                    batch_weighted_loss = torch.empty(size=(0,), dtype=torch.float, device=self.device)
                    
                    # print(f"Running weighted reward {w_running_reward:.2f}")


            cdir = os.getcwd()
            ts = datetime.now().strftime("%m.%d.%Y.%H_%M_%S")
            save_checkpoint(self.agent, ts, os.path.join(cdir, "saved_weights"), epoch)
            self.writer.add_scalar(tag=f'Running batch Loss after {self.save_every} episodes', 
                            scalar_value=loss,
                            global_step=epoch)        
            self.writer.add_scalar(tag=f'Weighted Return over {self.save_every} episodes', 
                            scalar_value=w_running_reward,
                            global_step=epoch)  
            epoch += 1
            print(f"epoch no {epoch}")


def main():
    # read hyperparams file
    with open("config-v1.json") as json_f:
        config = json.load(json_f)

    set_seed(config["SEED"])

    policy_gradient = PolicyGradient(config)
    policy_gradient.solve_problem()
    policy_gradient.env.close()

if __name__ == "__main__":
    main()