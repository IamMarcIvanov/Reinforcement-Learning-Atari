import time
import torch
import cv2
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import collections
import gym.spaces
from pyvirtualdisplay import Display
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import math
import glob
import io
import base64
from IPython.display import HTML

from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40)  # error only

display = Display(visible=0, size=(1400, 900))
display.start()

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        # Convolution layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        out = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, inputs):
        conv_out = self.conv_layers(inputs).view(inputs.size()[0], -1)
        return self.fc_layers(conv_out)

    def __call__(self, inputs):
        return self.forward(inputs)


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 0.00025
SYNC_TARGET_FRAMES = 10000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 2 * pow(10, 5)
EPSILON_START = 1.0
EPSILON_FINAL = 0.1


TransitionTable = collections.namedtuple('TransitionTable', field_names=[
                                         'state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(
            len(self.buffer), (BATCH_SIZE), replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def step_env(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_actions = np.array([self.state], copy=False)
            state_vector = torch.tensor(state_actions).to(device)
            q_values_vector = net(state_vector)
            _, action_value = torch.max(q_values_vector, dim=1)
            action = int(action_value.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp_tuple = TransitionTable(
            self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp_tuple)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def compute_loss(batch, net, target_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_vector = torch.tensor(states).to(device)
    next_states_vector = torch.tensor(next_states).to(device)
    actions_vector = torch.tensor(actions).to(device)
    rewards_vector = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_vector).gather(
        1, actions_vector.unsqueeze(-1)).squeeze(-1)

    next_state_values = target_net(next_states_vector).max(1)[0]

    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_vector

    return nn.MSELoss()(state_action_values, expected_state_action_values)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_env(DEFAULT_ENV_NAME)

local_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
target_model = DQN(env.observation_space.shape, env.action_space.n).to(device)

logs = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
print(local_model)

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
epsilon = EPSILON_START

optimizer = optim.RMSprop(local_model.parameters(), lr=LEARNING_RATE, momentum=0.95, alpha=0.95, eps=0.01)

total_rewards = []
curr_frame_idx = 0
prev_frame_idx = 0
curr_time = time.time()
start_time = curr_time
best_mean_reward = None

while True:
    curr_frame_idx += 1

    epsilon = max(EPSILON_FINAL, EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL)
                  * pow(math.e, - curr_frame_idx / EPSILON_DECAY_LAST_FRAME))

    reward = agent.step_env(local_model, epsilon, device=device)

    if reward is not None:
        total_rewards.append(reward)
        speed = (curr_frame_idx - prev_frame_idx) / (time.time() - curr_time)
        prev_frame_idx = curr_frame_idx
        curr_time = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s, time till now %.3f s" %
              (curr_frame_idx, len(total_rewards), mean_reward, epsilon, speed, curr_time - start_time))
        logs.add_scalar("epsilon", epsilon, curr_frame_idx)
        logs.add_scalar("speed", speed, curr_frame_idx)
        logs.add_scalar("reward_100", mean_reward, curr_frame_idx)
        logs.add_scalar("reward", reward, curr_frame_idx)

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(local_model.state_dict(),
                       DEFAULT_ENV_NAME + "-best.dat")
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" %
                      (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward
        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % curr_frame_idx)
            break

    if len(buffer) < REPLAY_START_SIZE:
        continue

    if curr_frame_idx % SYNC_TARGET_FRAMES == 0:
        target_model.load_state_dict(local_model.state_dict())

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = compute_loss(batch, local_model, target_model, device=device)
    loss_t.backward()
    optimizer.step()
logs.close()
