import matplotlib.pyplot as plt
import numpy as np
reward_ddqn = np.load('./data/ddqn.npy')
reward_ddpg = np.load('./data/ddpg.npy')
reward_td3 = np.load('./data/td3.npy')

def smooth(s):
    for i in range(len(s)-1):
        s[i+1] = s[i+1]*0.1 + s[i]*0.9
    return s


plt.plot(smooth(reward_ddqn)[range(0,400,1)], color='b', label='Double-DQN')
plt.plot(smooth(reward_ddpg)[range(0,400,1)], color='r', label='DDPG')
plt.plot(smooth(reward_td3)[range(0,400,1)], color='green', label='TD3')

plt.title('Double-DQN, DDPG and TD3 on Pendulum-v0')
plt.legend(loc='best')
plt.show()