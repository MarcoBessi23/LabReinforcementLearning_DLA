import os
import random
import math
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque
import numpy as np
import argparse
from NeuralNet import ReplayMemory, Transition, DQN



class DQNAgent:
    def __init__(self, env, policy_net, target_net, batch_size=64, gamma=0.99, lr=0.0005, tau=0.005, memory_size=200000, eps_start=0.9, eps_end=0.05, eps_decay=1000):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.memory_size = memory_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.memory_size)
        self.n_actions = env.action_space.n
        self.state, _ = env.reset()
        self.n_observations = len(self.state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                  batch.next_state)), device=device, dtype=torch.bool)
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        #compute Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_last_100_iter(self, average):
        plt.plot(average)
        plt.xlabel('iterations')
        plt.ylabel('score')
        plt.title('score for the winning 100 iterations')
        plt.savefig(os.path.join(os.getcwd(), 'Ex2/Results', 'Lunar_try.png'))

    def train(self, num_episodes=500, update_freq=4):
    
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        episode_durations = []
        average_score = deque(maxlen=100)

        for i_episode in range(num_episodes):
            score = 0
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in range(10000):  # Arbitrary large number of time steps
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                score += reward

                done = terminated or truncated
                if terminated:
                    average_score.append(score)
                    next_state = None
                else :
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                #next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                #store the transition in memory
                self.memory.push(state, action, next_state, reward)
                #move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if i_episode % update_freq == 0:
                    self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                self.soft_update_target_network()

                if done:
                    average_score.append(score)
                    break

            print(np.average(average_score), f'episode: {i_episode}')
            if np.average(average_score) > 200:
                print('LANDED ON THE MOON')
                self.plot_last_100_iter(average_score)
                break

    def soft_update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def test(self, env, num_episodes=2000):
        self.policy_net.load_state_dict(torch.load('Ex2/Lunar_params.pth', weights_only=True))
        self.policy_net.eval()

        self.env = env
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        ##NEEDED TO FORCE THE AGENT TO USE ONLY THE POLICY NETWORK ACTIONS(EPSILON GREEDY)
        self.steps_done = 5000
        b = 0
        for _ in range(num_episodes):
            action = self.select_action(state)
            observation, reward, terminated, truncated, info = self.env.step(action.item())

            if terminated or truncated:
                if terminated:
                    b += 1
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        print(f"Number of successful landings: {b}")
        self.env.close()


#def get_args():
#    parser = argparse.ArgumentParser(description="Train or Test a DQN Agent on LunarLander")
#    parser.add_argument("mode", choices=["train", "test"], help="Choose mode: train or test")
#    return parser.parse_args()
#
#
#def main(mode):
#    env = gym.make("LunarLander-v3")
#    n_actions = env.action_space.n
#    state, _ = env.reset()
#    n_observations = len(state)
#
#    policy_net = DQN(n_observations, n_actions)
#    target_net = DQN(n_observations, n_actions)
#
#    agent = DQNAgent(env, policy_net, target_net)
#
#    if mode == "train":
#        agent.train()
#        torch.save(policy_net.state_dict(), "Ex2/Lunar_params.pth")
#    elif mode == "test": 
#       env = gym.make("LunarLander-v3", render_mode="human")
#       agent.test(env= env)
#    else:
#        print("Select one : 'train' or 'test'")
#
#if __name__ == "__main__":
#    args = get_args()
#    main(args.mode)



##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###USED ONLY TO REGISTER THE GIF FOR GITHUB README, DON'T UNCOMMENT
def test_and_record():
    env = gym.make("LunarLander-v3", render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True, name_prefix= "lunar_dqn")
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)
    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    agent = DQNAgent(env, policy_net, target_net)

    agent.test(env= env, num_episodes=1500)
    env.close()

if __name__ == "__main__":
    test_and_record()
