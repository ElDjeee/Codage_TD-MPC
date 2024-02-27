import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import torch
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
from wrappers import PixelOnlyObservation, BinarizeObservation
import environments

import bbrl
from bbrl.workspace import Workspace
from bbrl.agents.agent import Agent
from bbrl.agents import Agents, TemporalAgent

class ActionAgent(Agent):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t, **kwargs):
        action = self.action_space.sample()
        action = torch.tensor([action], dtype=torch.int64)

        self.set(("action", t), action)


class EnvAgent(Agent):
    def __init__(self, img_size=25, num_stack=3, threshold=230, invert=True):
        super().__init__()
        self.env = gym.make("CartPole-v2", render_mode="rgb_array")
        self.env = PixelOnlyObservation(self.env)
        self.env = ResizeObservation(self.env, img_size)

        self.env = GrayScaleObservation(env=self.env)
        self.env = BinarizeObservation(env=self.env, threshold=threshold, invert=invert)
        self.env = FrameStack(env=self.env, num_stack=num_stack)

    def reset(self):
        self.obs = self.env.reset()
        if self.obs is None or len(self.obs) != 25:  # Vérification de l'observation
            self.obs = np.zeros((25, 25, 3), dtype=np.uint8)  # Si l'observation est invalide, créer une observation vide ou une observation par défaut
        self.obs = torch.tensor(self.obs, dtype=torch.float32)
        return self.obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if obs is None or len(obs) != 25:  # Vérification de l'observation
            obs = np.zeros((25, 25, 3), dtype=np.uint8)  # Si l'observation est invalide, créer une observation vide ou une observation par défaut
        self.obs = torch.tensor(obs, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        return self.obs, reward, done, info
    
    def forward(self, t, **kwargs):
        if t != 0:
            action = self.get(("action", t-1))
            obs, reward, done, truncated, _ = self.env.step(action.item())

            obs = torch.tensor(obs, dtype=torch.float32)
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)
            truncated = torch.tensor([truncated], dtype=torch.float32)
        else:
            obs = torch.tensor(self.env.reset()[0], dtype=torch.float32)
            reward = torch.tensor([0.0], dtype=torch.float32)
            done = torch.tensor([0.0], dtype=torch.float32)
            truncated = torch.tensor([0.0], dtype=torch.float32)

        self.set(("obs", t), obs)
        self.set(("reward", t), reward)
        self.set(("done", t), done)
        self.set(("truncated", t), truncated)

        if t%10 == 0:
            obs_tensor = obs.permute(1, 2, 0)
            plt.imshow(obs_tensor)
            plt.show()



# env_agent = EnvAgent()
# action_agent = ActionAgent(env_agent.env.action_space)
# composed_agent = Agents(env_agent, action_agent)
# t_agent = TemporalAgent(composed_agent)

# workspace = Workspace()
# t_agent(workspace, t=0, n_steps=100)

# obs, action, reward, done = workspace["obs", "action", "reward", "done"]

# print("obs:", obs)
# print("action:", action)
# print("reward:", reward)
# print("done:", done)