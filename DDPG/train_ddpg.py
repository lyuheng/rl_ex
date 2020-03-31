import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time

from model import Actor, Critic

############################### Hyperparameter ##################################

MAX_EPISODES = 400
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

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

class DDPG():
    def __init__(self):
        self.AE = Actor(s_dim, 30, a_dim)
        self.AT = Actor(s_dim, 30, a_dim)
        self.CE = Critic(s_dim, 30, a_dim)
        self.CT = Critic(s_dim, 30, a_dim)
        self.Memory = np.zeros((MEMORY_CAPACITY, s_dim*2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.optimizer_c = optim.Adam(self.CE.parameters(), lr=LR_C)
        self.optimizer_a = optim.Adam(self.AE.parameters(), lr=LR_A)


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
        for t,e in zip(self.CT.parameters(),self.CE.parameters()):
            t.data = (1-TAU)*t.data + TAU*e.data

    def train(self):
        #model.train()
        self.soft_replace()
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.Memory[indices, :]
        bs = to_var(bt[:, :s_dim])
        ba = to_var(bt[:, s_dim: s_dim + a_dim])
        br = to_var(bt[:, -s_dim - 1: -s_dim])
        bs_ = to_var(bt[:, -s_dim:])

    
        a  = self.AE(bs, a_bound)
        a_ = self.AT(bs_, a_bound)
        #######################  NOTICE !!! #####################################
        q1  = self.CE(bs,  a)  ###### here use action batch !!! for policy loss!!!
        q2  = self.CE(bs,  ba) ###### here use computed batch !!! for value loss!!!
        ########################################################################
        q_ = self.CT(bs_, a_).detach()

        q_target = br + GAMMA*q_
        #print(q.shape, q_target.shape)
        td_error = ((q2-q_target)**2/2).mean(0) # minimize td_error
        self.CE.zero_grad()
        td_error.backward(retain_graph=True)
        self.optimizer_c.step()

        a_loss = -torch.mean(q1) # maximize q
        self.AE.zero_grad()
        a_loss.backward(retain_graph=True)
        self.optimizer_a.step()


reward_his = []
ddpg = DDPG()
var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s).detach().numpy()
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        #print(a)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)
        # if memory has spaces, continue collecting data
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.train()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1: 
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            #if ep_reward > -300 :RENDER = True
            reward_his.append(int(ep_reward))
            break
print('Running time: ', time.time() - t1)

import matplotlib.pyplot as plt

plt.plot(reward_his)
plt.show()

np.save('./data/ddpg.npy', reward_his)