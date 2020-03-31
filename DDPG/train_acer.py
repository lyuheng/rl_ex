"""
discrete version of ACER on cartpole-v1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import time
import random

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0] # 4
N_A = env.action_space.n  #2

"""
some key points on implements:
1. Retrace method on estimationg Q value
2. TRPO update
3. Parallel training, A3C style
4. on-policy and off-policy updating
"""
######################### HyperParameters ###################################
DISCOUNT = 0.9
TRACE_MAX = 0.0
TRUST_REGION = True
TRUST_REGION_THRESHOLD = 0.0
MAX_GRADIENT_NORM = 
LR_DECAY = False
TRUST_REGION_DECAY = 0.99
SEED = 1
ENV_NAME = 
hidden_size = 
MEMORY_CAPACITY = 
NUM_PROCESS = 
MAX_EPISODE_LENGTH = 
T_MAX = 5e5        # number of training steps 
t_max =            # Max number of forward steps for A3C before update
REWARD_CLIP = False
REPLY_START = 
REPLY_RATIO =             # For Possion sampling
##############################################################################

def _possion(lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k+=1
        p*=random.uniform(0,1)
    return max(k-1,0)

def _adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _transfer_grads_to_shared_model(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return 
        share_model._grad = model.grad
 
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold, g, k):
    kl = -(ref_distribution * (distribution.log()- ref_distribution.log())).sum(1).mean(0)

    k_dot_g = (k*g).sum(1).mean(0)
    k_dot_k = (k**2).sum(1).mean(0)

    if k_dot_k.item() > 0:
        trust_factor = ((k_dot_g - threshold)/k_dot_k).clamp(min=0).detach()
    else:
        trust_factor = torch.zeros(1)
    trust_loss = loss + trust_factor*kl
    return trust_loss

def _update_networks(args, T, model, shared_model, shared_average_model, loss, optimizer):
    optimizer.zero_grads()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters, MAX_GRADIENT_NORM)

    _transfer_grads_to_shared_model(model, shared_model)
    optimizer.step()
    if LR_DECAY:
        _adjust_learning_rate(optimizer, max(LR*(T_MAX-T.value))/T_max, 1e-32)
    
    for share_param, shared_average_param in zip(shared_model.parameters(), shared_average_model.parameters()):
        shared_average_param.data = TRUST_REGION_DECAY*shared_average_param.data + (1-TRUST_REGION_DECAY)*share_param


def _train(args, T, model, shared_model, shared_average_model, optimizer, policies, Qs, Vs, actions,
            rewards, Qret, average_policies, old_policies=None):
    off_policy = old_policies is not None
    action_size = policies[0].size(1)
    policy_loss, value_loss = 0. , 0.

    t = len(rewards)
    for i in reverse(range(t)):
        if off_policy:
            # which means old_policies is not None # policies[i]:(batch,ac_dim)
            rho = policies[i].detach()/old_policies[i]
        else:
            rho = torch.ones(1, action_size) # batch=1, make sense!
        
        Qret = rewards[i] + DISCOUNT*Qret
        A = Qret - Vs[i] # (batch, ac_dim)-(batch,)
        
        log_prob = policies[i].gather(dim=1, index=actions[i]).log()
        # (9) first part also on-policy loss
        single_step_policy_loss = -(rho.gather(1,actions[i]).clamp(max=TRACE_MAX)*log_prob*A.detach()).mean(0)
        if off_policy:
            # (9) second part 
            bias_weight = (1-TRACE_MAX/rho).clamp(min=0)*policies[i]
            single_step_policy_loss -= (bias_weight*policies[i].log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(1).mean(0)
        if TRUST_REGION:
            # KL divergence
            k = - average_policies[i].gather(1, actions[i])/(policies[i].gather(1, actions[i])+1e-10)
            if off_policy: # why we divide policy ? 
                g = (rho.gather(1,actions[i]).clamp(max=TRACE_MAX)*A/(policies[i]+1e-10).gather(1,actions[i]) \
                    + (bias_weight*(Qs[i] - Vs[i].expand_as(Qs[i]))/(policies[i]+1e-10).gather(1,actions[i])).sum(1)).detach()
            else:
                g = (rho.gather(1,actions[i]).clamp(max=TRACE_MAX)*A/(policies[i]+1e-10).gather(1,actions[i])).detach()
            policy_loss += _trust_region_loss(model, policies[i].gather(1,actions[i]))+1e-10, \
                        average_policies[i].gather(1,actions[i])+1e-10, single_step_policy_loss, TRUST_REGION_THRESHOLD, g, k)
        else:
            policy_loss += single_step_policy_loss
        
        # entropy regularization
        policy_loss -= ENTROPY_WEIGHT * -(policies[i].log() * policies[i]).sum(1).mean(0)
        # value update
        Q = Qs[i].gather(1,actions[i])
        value_loss += ((Qret-Q)**2/2).mean(0) # MSELoss
        
        truncated_rho = rho.gather(1,actions[i]).clamp(max=1)
        Qret = truncated_rho*(Qret-Q.detach())+Vs[i].detach()
    
    _update_networks(args, T, model, shared_model, shared_average_model, policy_loss+value_loss, optimizer)


def train(rank, args, T, shared_model, shared_average_model, optimizer):
    torch.manual_seed(SEED + rank)
    env = gym.make(ENV_NAME)
    model = ActorCritic(observation_space, env.action_space, hidden_size)
    model.train()

    if not ON_POLICY:
        memory = EpisodeReplayMemory(MEMORY_CAPACITY) // NUM_PROCESS, MAX_EPISODE_LENGTH)
    t = 1
    done = True
    while T.value() <= T_MAX:
        while True:
            model.load_state_dict(share_model.state_dict())
            t_start = t
            if done:
                hx, avg_hx = torch.zeros(1, hidden_size), torch.zeros(1, hidden_size)
                cx, avg_cx = torch.zeros(1, hidden_size), torch.zeros(1, hidden_size)  # batch=1
                state = state_to_tensor(env.reset())
                done, episode_length = False, 0
            else:
                hx = hx.detach()
                cx = cx.detach()
            policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []
            while not done and t - t_start < t_max: # memory里存的最大长度是 t_max=100
                policies, Q, V (hx, cx) = model(state, (hx, cx))
                average_policies, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))
                # sample action
                action = torch.multinomial(policy, 1)[0,0] # one item
                next_state, reward, done, _ = env.step(action.item())
                reward = REWARD_CLIP and min(max(reward,-1),1) or reward
                done = done or episode_length >= MAX_EPISODE_LENGTH   # 不要再一条路上走太长 最长500steps
                episode_length += 1

                if not ON_POLICY:
                    memory.append(state, action, reward, policy.detach())
                [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies), \
                                                    (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]), average_policy))]
                t += 1
                T.increment()
                state = next_state               
            if done:
                Qret = torch.zeros(1,1)
                if not ON_POLICY:
                    memory.append(state, None, None, None)
            else:
                _, _, Qret, _ = model(state, (hx,cx))
                Qret = Qret.detach()
            _train(args, T, model, shared_model, shared_average_model, optimizer, policies, Qs, Vs, actions, rewards, Qret, average_policies)
            if done:
                break
        if not ON_POLICY and len(memory) >= REPLY_START:
            for _ in range(_possion(REPLY_RATIO)):
                trajectories = memory.sample_batch(BATCH_SIZE, t_max)
                hx, avg_hx = torch.zeros(BATCH_SIZE, hidden_size), torch.zeros(BATCH_SIZE, hidden_size)
                cx, avg_cx = torch.zeros(BATCH_SIZE, hidden_size), torch.zeros(BATCH_SIZE, hidden_size)  # batch=1
                policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []
                for i in range(len(trajectories)-1):  # (time_step, batch)
                    state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
                    action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
                    action = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
                    old_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)

                    policy, Q, V, (hx, cx) = model(state, (hx, cx)) # state:(batch, s_dim)
                    average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))
                    [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies), \
                                                    (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor(reward), average_policy))]
                    next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i+1]), 0)
                    done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i+1]]).unsqueeze(1)
                _, _,  Qret, _ = model(next_state, (hx, cx))
                Qret = ((1-done)*Qret).detach()

                _train(args, T, model, shared_model, shared_average_model, optimizer, policies, Qs, Vs, actions, rewards, Qret, average_policies)
        done=True
    env.close()

