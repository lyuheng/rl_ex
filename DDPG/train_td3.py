import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time

from model import Actor, Critic

MAX_EPISODES = 400
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
SIGMA = 0.1
D = 2

##################################################################################

RENDER = False
ENV_NAME = 'Pendulum-v0'

env = gym.make(ENV_NAME).unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0] # 3
a_dim = env.action_space.shape[0]      # 1
a_bound = env.action_space.high # low:[-2,] high:[2,]
#cuda = torch.cuda.is_available()

def to_var(x):
    return torch.Tensor(x)
    
a_bound = to_var(a_bound)

class TD3():
    def __init__(self):
        ##########  Actor ##########################
        self.AE   = Actor(s_dim, 30, a_dim)
        self.AT   = Actor(s_dim, 30, a_dim)
        ########## Critic ##########################
        self.CE_1 = Critic(s_dim, 30, a_dim)
        self.CE_2 = Critic(s_dim, 30, a_dim)
        self.CT_1 = Critic(s_dim, 30, a_dim)
        self.CT_2 = Critic(s_dim, 30, a_dim)
        self.Memory = np.zeros((MEMORY_CAPACITY, s_dim*2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.optimizer_c_1 = optim.Adam(self.CE_1.parameters(), lr=LR_C)
        self.optimizer_c_2 = optim.Adam(self.CE_2.parameters(), lr=LR_C)
        self.optimizer_a = optim.Adam(self.AE.parameters(), lr=LR_A)
        self.batch_size = BATCH_SIZE
        self.step_learn_counter = 0
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.Memory[index, :] = transition # store torch tensors!!
        self.pointer += 1
    
    def choose_action(self, s):
        a = self.AE(to_var(s[np.newaxis,:]), a_bound)[0] #####!!!!!
        return a

    def soft_replace(self):
        for t,e in zip(self.AT.parameters(),self.AE.parameters()):
            t.data = (1-TAU)*t.data + TAU*e.data
        for t,e in zip(self.CT_1.parameters(),self.CE_1.parameters()):
            t.data = (1-TAU)*t.data + TAU*e.data
        for t,e in zip(self.CT_2.parameters(),self.CE_2.parameters()):
            t.data = (1-TAU)*t.data + TAU*e.data
    
    def train(self):
        # sample data
        if self.pointer > MEMORY_CAPACITY:
            indices = np.random.choice(MEMORY_CAPACITY, size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)

        bt = self.Memory[indices, :]
        bs = to_var(bt[:, :s_dim])
        ba = to_var(bt[:, s_dim: s_dim + a_dim])
        br = to_var(bt[:, -s_dim - 1: -s_dim])
        bs_ = to_var(bt[:, -s_dim:])

        a_tilta = self.AT(bs_, a_bound).detach()
        # add randomness to action selection for exploration
        a_tilta = np.clip(np.random.normal(a_tilta.numpy(), SIGMA), -2, 2)
        qt1 = self.CT_1(bs_, to_var(a_tilta)).detach()
        qt2 = self.CT_2(bs_, to_var(a_tilta)).detach()
        # idx1 = qt1 > qt2
        # idx2 = qt1 <= qt2
        # q_t = qt1.clone() # q_t is min over qt1 and qt2
        # q_t[idx1] = qt2[idx1]
        # q_t[idx2] = qt1[idx2]
        q_t = torch.min(qt1, qt2)
        y = br+GAMMA*q_t

        qe1 = self.CE_1(bs, ba)
        qe2 = self.CE_2(bs, ba)

        theta_loss = ((y-qe1)**2/2).sum()/self.batch_size + ((y-qe2)**2/2).sum()/self.batch_size
        # update theta_1
        self.CE_1.zero_grad()
        theta_loss.backward(retain_graph=True)
        self.optimizer_c_1.step()
        
        # update theta_2
        self.CE_2.zero_grad()
        theta_loss.backward()
        self.optimizer_c_2.step()

        if self.step_learn_counter % D == 0:
            ############################ IMPROTANT !!! #####################################
            ac = self.AE(bs, a_bound)
            a_loss = -torch.mean(self.CE_1(bs,ac)) # maximize q
            ################################################################################
            self.AE.zero_grad()
            a_loss.backward()
            self.optimizer_a.step()
            self.soft_replace()

        self.step_learn_counter += 1


reward_his = []
td3 = TD3()
#var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = td3.choose_action(s).detach().numpy()
        a = np.clip(np.random.normal(a, SIGMA), -2, 2)    # add randomness to action selection for exploration
        #print(a)
        s_, r, done, info = env.step(a)

        td3.store_transition(s, a, r / 10, s_)
        # if memory has spaces, continue collecting data
        #if td3.pointer > MEMORY_CAPACITY:
            #var *= .9995    # decay the action randomness
        td3.train()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1: 
            print('Episode:', i, ' Reward: %i' % int(ep_reward))
            #if ep_reward > -300 :RENDER = True
            reward_his.append(int(ep_reward))
            break
print('Running time: ', time.time() - t1)

import matplotlib.pyplot as plt

plt.plot(reward_his)
plt.show()

np.save('./data/td3.npy', reward_his)