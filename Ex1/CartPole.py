# Standard imports.
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from NeuralNet import PolicyNet, BaselineNet, ReplayMemory
from torch.distributions import Categorical

# Given an environment, observation, and policy, sample from pi(a | obs). Returns the
# selected action and the log probability of that action (needed for policy gradient).
def select_action(env, obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array. There's probably a better way to do this.
def compute_returns(rewards, gamma):
    
    return np.flip(np.cumsum([gamma**(i+1)*r for (i, r) in enumerate(rewards)][::-1]), 0).copy()

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
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


# Cartpole is solved when we obtain a score of 195 over the last 100 iteration
SOLVED_SCORE = 195

def reinforce(policy, env, gamma=0.99, num_episodes=500, baseline=None):

    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    if isinstance(baseline, nn.Module):
        opt_baseline = torch.optim.Adam(baseline.parameters(), lr=0.1)
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

        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy)

        # Compute the discounted reward for every step of the episode.
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)

        running_rewards.append(np.sum(rewards))
        if len(running_rewards) == 100:
            average = sum(running_rewards)/len(running_rewards)
            
            if average >= SOLVED_SCORE:
                print('CARTPOLE SOLVED')
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


env = gym.make('CartPole-v1')

# Make a policy network.
policy = PolicyNet(env, inner=128)
baseline = BaselineNet(env, inner=128)

# Train the agent.
plt.plot(reinforce(policy, env, num_episodes=400, gamma=0.99, 
                   baseline=baseline))

# Close up everything
env.close()

import os
path_cart = os.path.join(os.getcwd(), 'Ex1/Results/Cart.png')
plt.savefig(path_cart)


#torch.save(policy.state_dict(), 'Ex1/policy_params.pth')

env_render = gym.make('CartPole-v1', render_mode='human')
policy.eval()
scores = []
for _ in range(5):
    _,_,_, score = run_episode(env_render, policy)
    scores.append(np.sum(score))

print(np.mean(scores))
env_render.close()
