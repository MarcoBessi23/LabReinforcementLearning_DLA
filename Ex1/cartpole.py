import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from torch.distributions import Categorical
from NeuralNet import PolicyNet, BaselineNet
import os


class REINFORCEAgent_cartpole:
    def __init__(self, policy, env, gamma=0.99, num_episodes=500, lr=1e-2, baseline=None, solved_score=300):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.lr = lr
        self.solved_score = solved_score
        self.baseline = baseline

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=0.1)
        self.baseline.train()
        
        self.running_rewards = deque(maxlen=100)

    def select_action(self, obs):
        """ Seleziona un'azione dalla policy e restituisce l'azione e la log-probabilità. """
        dist = Categorical(self.policy(obs))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.reshape(1)

    def compute_returns(self, rewards):
        """ Calcola i ritorni scontati. """
        trajectory_len = len(rewards)
        return_array = np.zeros(trajectory_len)
        g_return = 0
        for i in range(trajectory_len - 1, -1, -1):
            g_return = rewards[i] + self.gamma * g_return
            return_array[i] = g_return
        return torch.tensor(return_array, dtype=torch.float32)

    def run_episode(self):
        """ Esegue un episodio e raccoglie osservazioni, azioni, log_prob e ricompense. """
        observations, actions, log_probs, rewards = [], [], [], []
        obs, _ = self.env.reset()

        while True:
            obs = torch.tensor(obs, dtype=torch.float32)
            action, log_prob = self.select_action(obs)
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)

            obs, reward, done, _, _ = self.env.step(action)
            rewards.append(reward)

            if done:
                break

        return observations, actions, torch.cat(log_probs), rewards

    def update_policy(self, observations, log_probs, returns):
        """ Aggiorna la policy network usando il policy gradient. """
        
        with torch.no_grad():
            target = returns - self.baseline(torch.stack(observations))
        
        loss = (-log_probs * target).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def update_baseline(self, observations, returns):
        """ Aggiorna la rete baseline, se esiste. """
        if isinstance(self.baseline, nn.Module):
            baseline_pred = self.baseline(torch.stack(observations)).squeeze(1)
            loss_baseline = F.mse_loss(baseline_pred, returns)

            self.baseline_optimizer.zero_grad()
            loss_baseline.backward()
            self.baseline_optimizer.step()

    def train(self):
        """ Addestra l'agente per un numero di episodi specificato. """
        self.policy.train()
        
        for episode in range(self.num_episodes):
            print(f' EPISODE NUMBER {episode}')

            observations, actions, log_probs, rewards = self.run_episode()
            returns = self.compute_returns(rewards)

            self.running_rewards.append(np.sum(rewards))
            if len(self.running_rewards) == 100:
                average = sum(self.running_rewards) / len(self.running_rewards)
                if average >= self.solved_score:
                    print('CARTPOLE SOLVED')
                    break

            self.update_policy(observations, log_probs, returns)
            self.update_baseline(observations, returns)

            print(f"Episode reward: {self.running_rewards[-1]}")

        return self.running_rewards


def main():
    env = gym.make('CartPole-v1')

    policy = PolicyNet(env, inner=128)
    baseline = BaselineNet(env, inner=128)

    agent = REINFORCEAgent_cartpole(policy, env, num_episodes=400, gamma=0.99, baseline=baseline)
    rewards = agent.train()

    plt.plot(rewards)
    path_cart = os.path.join(os.getcwd(), 'Ex1/Results/Cart.png')
    plt.savefig(path_cart)

    torch.save(policy.state_dict(), 'Ex1/policy_params.pth')

    env_render = gym.make('CartPole-v1', render_mode='human')
    policy.eval()
    scores = []
    agent.env = env_render

    for _ in range(5):
        _, _, _, score = agent.run_episode()
        scores.append(np.sum(score))

    print(f"Average score: {np.mean(scores)}")
    env_render.close()


if __name__ == "__main__":
    main()
