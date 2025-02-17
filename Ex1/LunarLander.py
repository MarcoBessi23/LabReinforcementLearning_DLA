import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from NeuralNet import PolicyNet,LunarPolicy, BaselineNet, LunarBaseline, ReplayMemory
from torch.distributions import Categorical
import os

def select_action(env, obs, policy):
    
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

def compute_returns(rewards, gamma):
    trajectory_len = len(rewards)
    return_array = np.zeros((trajectory_len,))
    g_return = 0
    for i in range(trajectory_len-1,-1,-1):
        g_return = rewards[i] + gamma*g_return
        return_array[i] = g_return
        return_t = torch.FloatTensor(return_array)
    return return_t


# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=500):
    # Collect just about everything.
    observations = []
    actions      = []
    log_probs    = []
    rewards      = []

    # Reset the environment and start the episode.
    (obs, _) = env.reset()
    for _ in range(maxlen):
        # Get the current observation, run the policy and select an action.
        #env.render()
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        
        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, _) = env.step(action)
        rewards.append(reward)
        #Interrupt the training if you get a score that is to hard to recover from.
        if np.sum(rewards) < -250:
            term = True
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


def run_episode_epsilon_greedy(env, policy, eps,  maxlen=500):
    observations = []
    actions      = []
    log_probs    = []
    rewards      = []

    # Reset the environment and start the episode.
    (obs, _) = env.reset()
    for _ in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs)
        dist = Categorical(policy(obs))

        if random.random() < eps:
            #EPS
            m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            action = m.sample()
            log_prob = dist.log_prob(action)
            action = action.item()

        else :
            #GREEDY
            action = torch.argmax(dist.probs)
            log_prob = dist.log_prob(action)
            action = action.item()

        actions.append(action)
        log_probs.append(log_prob.reshape(1))
        observations.append(obs)

        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, _) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


def epsilon_schedule(episode, epsilon_initial, epsilon_final, decay_episodes):
    epsilon = epsilon_initial - (epsilon_initial - epsilon_final) * (episode / decay_episodes)
    return max(epsilon, epsilon_final)

# Lunar Lander is solved when we obtain a score of 200 over the last 100 iteration
SOLVED_SCORE = 200

def reinforce_epsilon_greedy(policy, env, gamma=0.9, num_episodes=500, baseline=None):

    epsilon_initial = 0.9
    epsilon_final   = 0.01
    decay_episodes  = 100

    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=0.01)

    if isinstance(baseline, nn.Module):
        opt_baseline = torch.optim.Adam(baseline.parameters(), lr=0.01)
        baseline.train()
        print('Training agent with baseline value network.')
    elif baseline == 'std':
        print('Training agent with standardization baseline.')
    else:
        print('Training agent with no baseline.')

    # Track episode rewards in a list.
    running_rewards = deque(maxlen=100)

    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        print(f' EPISODE NUMBER {episode}')
        eps = epsilon_schedule(episode, epsilon_initial, epsilon_final, decay_episodes)
        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode_epsilon_greedy(env, policy, eps)

        # Compute the discounted reward for every step of the episode.
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)

        running_rewards.append(np.sum(rewards))
        if len(running_rewards) == 100:
            average = sum(running_rewards)/len(running_rewards)
            
            if average >= SOLVED_SCORE:
                print('LANDED')
                break

        if isinstance(baseline, nn.Module):
            with torch.no_grad():
                target = returns - baseline(torch.stack(observations))
        elif baseline == 'std':
            target = (returns - returns.mean()) / returns.std()
        else:
            target = returns

        # Make an optimization step
        opt.zero_grad()

        # Update policy network
        loss = (-log_probs * target).mean()
        loss.backward()
        opt.step()

        # Update baseline network.
        if isinstance(baseline, nn.Module):
            opt_baseline.zero_grad()
            loss_baseline = F.mse_loss(baseline(torch.stack(observations)).squeeze(1), returns)
            loss_baseline.backward()
            opt_baseline.step()

    return running_rewards



def reinforce(policy, env, gamma=0.99, num_episodes=500, baseline=None):

    opt = torch.optim.Adam(policy.parameters(), lr=0.002)    
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=0.002)

    baseline.train()
    
    running_rewards = deque(maxlen=10)
    results = []

    policy.train()
    for episode in range(num_episodes):
        print(f' EPISODE NUMBER {episode}')

        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy)
        
        
        # Compute the discounted reward for every step of the episode.
        #returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        #here try to reduce variance in the rewards
        rew = torch.tensor(rewards, dtype=torch.float32) * 0.01
        returns = compute_returns(rew, gamma)
        running_rewards.append(np.sum(rewards))
        results.append(np.sum(rewards))

        if len(running_rewards) == 10:
            average = sum(running_rewards)/len(running_rewards)
            
            if average >= SOLVED_SCORE:
                print('LANDED')
                break

        print('ACTUAL REWARDS')
        print(results[-1])

        with torch.no_grad():
            target = returns - baseline(torch.stack(observations))
        # Make an optimization step
        opt.zero_grad()

        # Update policy network
        loss = (-log_probs * target).mean()
        loss.backward()
        opt.step()

        # Update baseline network.
        opt_baseline.zero_grad()
        loss_baseline = F.mse_loss(baseline(torch.stack(observations)).squeeze(1), returns)
        loss_baseline.backward()
        opt_baseline.step()
    
    return results


env = gym.make("LunarLander-v3")
#policy = PolicyNet(env, 128)
#baseline = BaselineNet(env, 128)
policy = LunarPolicy(env, 128)
baseline = LunarBaseline(env, 128)

# Train the agent.
plt.plot(reinforce(policy, env, num_episodes= 3000, gamma=0.99, 
                   baseline=baseline))

torch.save(policy.state_dict(), 'Ex1/Lunar_params.pth')
# Close up everything
env.close()

path_lunar = os.path.join(os.getcwd(), 'Ex1/Results/Lunar.png')
plt.savefig(path_lunar)

env_render = gym.make('LunarLander-v3', render_mode='human')
policy.eval()
scores = []
for _ in range(5):
    _,_,_, score = run_episode(env_render, policy)
    scores.append(np.sum(score))

print(np.mean(scores))
env_render.close()