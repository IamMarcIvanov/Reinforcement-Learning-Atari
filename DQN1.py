import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Pong-v0")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        
        def flat(size, kernel_size, stride):
            return (size - kernel_size // stride) + 1
        
        def convBlock(in_channels, filters, *args, **kwargs):
            return nn.Sequential(
                nn.Conv2d(in_channels, filters, *args, **kwargs),
                nn.BatchNorm2d(filters),
                nn.LeakyReLU()
            )
        
        def linBlock(inDim, outDim):
            return nn.Sequential(
                nn.Linear(inDim, outDim),
                nn.BatchNorm2d(outDim)
                nn.LeakyReLU()
        )

        self.conv1 = convBlock(1, 32, kernel_size=8, stride=4)
        self.conv2 = convBlock(32, 64, kernel_size=4, stride=2)
        self.conv3 = convBlock(64, 64, kernel_size=3, stride=1)

        convw = flat(flat(flat(w, 8, 4), 4, 2), 3, 1)
        convh = flat(flat(flat(h, 8, 4), 4, 2), 3, 1)
        
        self.fc1 = linBlock(convw * convh * 64, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(out)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

resize = T.Compose([T.ToPILImage(), 
                    T.Grayscale(), 
                    T.Resize([84, 84], interpolation=InterpolationMode.NEAREST), 
                    T.ToTensor()])
screen = resize(env.render(mode='rgb_array'))

n_action = env.action_space.n
screen_height = screen.shape[1]
screen_width = screen.shape[2]
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

