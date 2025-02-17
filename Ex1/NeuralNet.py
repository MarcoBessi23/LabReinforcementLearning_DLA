import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import random
import torch
from collections import deque, namedtuple


class PolicyNet(nn.Module):
    def __init__(self, env, inner=16):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], inner)
        self.fc2 = nn.Linear(inner, env.action_space.n)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s

class BaselineNet(nn.Module):
    def __init__(self, env, inner= 16):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], inner)
        self.fc2 = nn.Linear(inner, 1)
        
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        return s



class LunarPolicy(nn.Module):
    def __init__(self, env, inner):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], inner)
        self.fc2 = nn.Linear(inner, inner)
        self.fc3 = nn.Linear(inner, env.action_space.n)

    def forward(self, s):
        s = torch.clamp(s, -1, 1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.softmax(self.fc3(s), dim=-1)
        return s

class LunarBaseline(nn.Module):
    def __init__(self, env, inner):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], inner)
        self.fc2 = nn.Linear(inner, inner)
        self.fc3 = nn.Linear(inner, 1)
        
    def forward(self, s):
        s = torch.clamp(s, -1, 1)
        #s = F.normalize(s, dim=1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)

        return s

class ReplayMemory(object):

    def __init__(self, capacity = 500):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)