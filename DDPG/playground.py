import numpy as np
import torch
import gym

RENDER = False
ENV_NAME = 'Pendulum-v0' # 49 Atari games in total

env = gym.make(ENV_NAME).unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0] # 3
a_dim = env.action_space.shape[0]    # 1
#a_bound = env.action_space.high # low:[-2,] high:[2,]
cuda = torch.cuda.is_available()
#print(cuda)
#print(s_dim, a_dim)

import Box2D
env = gym.make('CarRacing-v0')
print(env.observation_space) # Box(96, 96, 3)
print(env.action_space) # Box(3,)

env = gym.make("BipedalWalker-v2")
print(env.observation_space) # Box(24,)
print(env.action_space) # Box(4,)

env = gym.make('BipedalWalkerHardcore-v2')
print(env.observation_space) # Box(24,)
print(env.action_space) # Box(24,)

env = gym.make('LunarLander-v2')
print(env.observation_space) # Box(8,)
print(env.action_space) # Discrete(4)