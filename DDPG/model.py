"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum (continuous action space) example.

"""
################################# DDPG #####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import set_init

class Actor(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim),
            nn.Tanh(), # use tanh to restrict aciton to [-1,1]
        )
    def forward(self, s, a_bound):
        #print(type(s))
        out = self.linear(s)
        out = torch.mul(out, a_bound)
        return out

class Critic(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(s_dim + a_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1), # no activation func
        )  
    def forward(self, s, a):
        out = torch.cat((s,a), 1)
        out = self.linear(out)
        return out


##################################### DQN ####################################

class Qnet(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim),
        )
    def forward(self, s):
        out = self.linear(s)
        return out


#################################### TD3 ######################################
# Use DDPG Net Frame
################################################################################

##################################### A3C (share net) ######################################

class A3CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal   

    def forward(self, s):
        a1 = F.relu6(self.a1(s))
        mu = 2*F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1))+0.001   #softplus is smooth version of relu
        c1 = F.relu6(self.c1(s))
        values = self.v(c1)
        return mu, sigma, values
    
    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s) # s should have a batch_size of 1
        m = self.distribution(mu.view(1,-1).data, sigma.view(1,-1).data)
        return m.sample().numpy()
     
    def loss_func(self, s, a, v_t):  # v_target
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2) # do not detach

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        # entropy of Normal Dist = 0.5*ln(2*pi)+ln(sigma)+0.5
        entropy = 0.5 + 0.5*math.log(2*math.pi) + torch.log(m.scale)
        # BETA=0.005 is exploration rate
        exp_v = log_prob*td.detach() + 0.005*entropy
        a_loss = -exp_v # maximize v, 
        total_loss = (a_loss + c_loss).mean()
        return total_loss

#############################################################################

################################ ACER (two heads) #############################
class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super().__init__()
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n  # discrete action space

        self.fc1 = nn.Linear(self.state_size, hidden_size)
        