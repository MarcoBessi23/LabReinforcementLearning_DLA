import torch
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from torch.distributions import Categorical
import os
from NeuralNet import LunarBaseline, LunarPolicy


class REINFORCEAgent_Lunar:
    def __init__(self, policy, baseline, env, gamma=0.99, num_episodes=500, lr=0.002, solved_score=200):
        self.policy = policy
        self.baseline = baseline
        self.env = env
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.lr = lr
        self.solved_score = solved_score

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=self.lr)
        
        self.running_rewards = deque(maxlen=10)
        self.results = []

    def select_action(self, obs):
        # Sample an action from the policy and compute log_prob
        dist = Categorical(self.policy(obs))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.reshape(1)

    def compute_returns(self, rewards):
        # Compute the discounted cumulative rewards (returns)
        trajectory_len = len(rewards)
        return_array = np.zeros((trajectory_len,))
        g_return = 0
        for i in range(trajectory_len - 1, -1, -1):
            g_return = rewards[i] + self.gamma * g_return
            return_array[i] = g_return
        return torch.FloatTensor(return_array)

    def run_episode(self, maxlen=500):
        # Run an episode and collect observations, actions, log_probs, and rewards
        observations = []
        actions = []
        log_probs = []
        rewards = []

        # Reset the environment and start the episode
        obs, _ = self.env.reset()
        for _ in range(maxlen):
            obs = torch.tensor(obs, dtype=torch.float32)
            action, log_prob = self.select_action(obs)
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)

            # Advance the episode by executing the selected action
            obs, reward, done, truncated, _ = self.env.step(action)
            rewards.append(reward)

            # End the episode early if the reward is too low or it's done
            if np.sum(rewards) < -250 or done or truncated:
                break

        return observations, actions, torch.cat(log_probs), rewards

    def update_policy(self, observations, log_probs, returns):
        # Compute the advantage and update the policy
        target = returns - self.baseline(torch.stack(observations))
        loss = (-log_probs * target).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def update_baseline(self, observations, returns):
        # Update the baseline network
        baseline_predictions = self.baseline(torch.stack(observations)).squeeze(1)
        loss_baseline = F.mse_loss(baseline_predictions, returns)

        self.baseline_optimizer.zero_grad()
        loss_baseline.backward()
        self.baseline_optimizer.step()

    def train(self):
        self.policy.train()
        self.baseline.train()

        for episode in range(self.num_episodes):
            print(f"Episode {episode}/{self.num_episodes}")
            
            observations, actions, log_probs, rewards = self.run_episode()
            returns = self.compute_returns(rewards)
            self.running_rewards.append(np.sum(rewards))
            self.results.append(np.sum(rewards))

            if len(self.running_rewards) == 10:
                average = sum(self.running_rewards) / len(self.running_rewards)
                if average >= self.solved_score:
                    print("Environment solved!")
                    break

            # Update the policy and baseline
            self.update_policy(observations, log_probs, returns)
            self.update_baseline(observations, returns)

            print(f"Episode reward: {self.results[-1]}")

        return self.results


def main():
    env = gym.make("LunarLander-v3")
    policy = LunarPolicy(env, 128)
    baseline = LunarBaseline(env, 128)

    agent = REINFORCEAgent_Lunar(policy, baseline, env, num_episodes=3000, gamma=0.99)

    rewards = agent.train()

    # Save trained policy and plot results
    torch.save(policy.state_dict(), 'Ex1/Lunar_params.pth')

    plt.plot(rewards)
    path_lunar = os.path.join(os.getcwd(), 'Ex1/Results/Lunar_try.png')
    plt.savefig(path_lunar)

    env_render = gym.make("LunarLander-v3", render_mode='human')
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
