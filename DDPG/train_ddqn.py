"""
print(env.action_space) #动作空间，输出的内容看不懂
print(env.action_space.n) #当动作是离散的时，用该方法获取有多少个动作
# env.observation_space.shape[0] #当动作是连续的时，用该方法获取动作由几个数来表示
print(env.action_space.low) #动作的最小值
print(env.action_space.high) #动作的最大值
print(env.action_space.sample()) #从动作空间中随机选取一个动作
# 同理，还有 env.observation_space，也具有同样的属性和方法（.low和.high方法除外）

"""

"""
DDQN(double DQN) train on continous space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time

from model import Qnet

########################## HyperParameters #################################

RENDER = False
ENV_NAME = 'Pendulum-v0'

env = gym.make(ENV_NAME).unwrapped
env.seed(1)
np.random.seed(1)

s_dim = env.observation_space.shape[0]  # 3
#a_dim = env.action_space.shape[0]      # 1
a_bound = env.action_space.high # low:[-2,] high:[2,]

LR = 0.005
reward_decay = 0.9
epsilon_greedy = 0.9
replace_target_iter = 200
MEMORY_CAPACITY = 10000

MAX_EPISODES = 400
MAX_EP_STEPS = 200
#############################################################################

def to_var(x):
    return torch.Tensor(x)
    
a_bound = to_var(a_bound)


class DoubleDQN():
    def __init__(self):
        ##################### not make sense at all !!############################
        self.n_actions = 11     # for discretization
        #######################################################################
        self.batch_size = 32
        self.EN = Qnet(s_dim, 30, self.n_actions) # evaluation net
        self.TN = Qnet(s_dim, 30, self.n_actions)   # target net
        self.Memory = np.zeros((MEMORY_CAPACITY, s_dim*2+2))
        self.learn_step_counter = 0
        self.pointer = 0
        self.optimizer = optim.Adam(self.EN.parameters(), lr = LR)

    
    def store_transitions(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.Memory[index, :] = transition # store torch tensors!!
        self.pointer += 1
    
    def choose_actions(self, s): 
        s = to_var(s[np.newaxis,:]) # batch=1
        action = self.EN(s)
        action = torch.argmax(action).item()

        if np.random.uniform() > epsilon_greedy:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def hard_replace(self):
        for t, e in zip(self.TN.parameters(),self.EN.parameters()):
            t.data = e.data

    def train(self):
        if self.learn_step_counter % replace_target_iter == 0:
            self.hard_replace()

        if self.pointer > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.pointer, size=self.batch_size)
        bt = self.Memory[sample_index, :]

        bs = to_var(bt[:, :s_dim])
        ba = to_var(bt[:, s_dim: s_dim + 1]).long()
        br = to_var(bt[:, -s_dim - 1: -s_dim])
        bs_ = to_var(bt[:, -s_dim:])

        q_target4next = self.TN(bs_).detach()
        q_eval4next = self.EN(bs_)
        q_eval = self.EN(bs)

        q_target = q_eval.clone()
        
        batch_index = torch.arange(self.batch_size, dtype=torch.long)
        max_act = torch.argmax(q_eval4next, 1).long()
        selected_q_next = q_target4next[batch_index, max_act]
        
        # one_line not work !
        #q_eval4next = q_eval4next.type(torch.LongTensor)
        #selected_q_next = torch.gather(q_target4next, dim=1, index=q_eval4next).max(1)

        q_target[batch_index, ba.squeeze()] = br.squeeze() + reward_decay*selected_q_next
        loss = ((q_target-q_eval)**2/2).sum()/self.batch_size
        self.EN.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        
reward_his = []
ddqn = DoubleDQN()
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        action = ddqn.choose_actions(s)
        ###########################################################################################
        # f_action is to connected between env, while action is to update Q-learning !!!!         #
        ###########################################################################################
        f_action = (action-(ddqn.n_actions-1)/2)/((ddqn.n_actions-1)/4)   # convert to [-2 ~ 2] float actions
        
        s_, r, done, info = env.step(np.array([f_action]))
        
        # action is to update Q-learning !!!!
        ddqn.store_transitions(s, action, r / 10, s_)
        # if memory has spaces, continue collecting data
        if ddqn.pointer > MEMORY_CAPACITY:
            ddqn.train()

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

np.save('./data/ddqn.npy', reward_his)