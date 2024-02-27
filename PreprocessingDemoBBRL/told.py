import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gymnasium.envs.classic_control import CartPoleEnv

import bbrl
from bbrl.workspace import Workspace
from bbrl.agents.agent import Agent
from bbrl.agents import Agents, TemporalAgent

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from wrappers import PixelOnlyObservation, BinarizeObservation
import environments

# from demo import ActionAgent, EnvAgent

from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float().div(255.)

class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape

def enc(cfg):
    """Returns a TOLD encoder."""
    C = int(3*cfg.frame_stack)
    layers = [NormalizeImg(),
                nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
    out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
    layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
    return nn.Sequential(*layers)

def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]), act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        nn.Linear(mlp_dim[1], out_dim))

def q(cfg, act_fn=nn.ELU()):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
                         nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
                         nn.Linear(cfg.mlp_dim, 1))

def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)

class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = enc(cfg)
        self._dynamics = mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = q(cfg), q(cfg)
        self.apply(orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)












# Define a mock configuration for the TOLD model
class MockConfig:
    def __init__(self):
        self.frame_stack = 4  # Number of frames to stack for the input
        self.img_size = 84  # Size of the input images
        self.num_channels = 32  # Number of channels in the convolutional layers
        self.latent_dim = 64  # Dimension of the latent representation
        self.action_dim = 2  # Dimension of the action space
        self.mlp_dim = 256  # Dimension of the MLP layers

# Initialize the configuration
cfg = MockConfig()

# Initialize the TOLD model with the mock configuration
told_model = TOLD(cfg)

# Number of iterations to simulate
num_iterations = 10

for i in range(num_iterations):
    print(f"Iteration: {i+1}")

    # Create a dummy observation
    dummy_observation = torch.rand(1, cfg.frame_stack * 3, cfg.img_size, cfg.img_size)

    # Pass the observation through the encoder to get the latent representation
    latent_representation = told_model.h(dummy_observation)

    # Sample an action from the policy
    sampled_action = told_model.pi(latent_representation, std=0.1)

    # Use the latent representation and action to predict the next latent state and reward
    next_latent, reward = told_model.next(latent_representation, sampled_action)

    # Get the Q-values for the current state and action
    q1, q2 = told_model.Q(latent_representation, sampled_action)

    # Print the outputs for inspection
    print("Latent Representation:", latent_representation.shape)
    print("Next Latent State:", next_latent.shape)
    print("Reward:", reward.shape)
    print("Sampled Action:", sampled_action.shape)
    print("Q1 Value:", q1.shape)
    print("Q2 Value:", q2.shape)
    print("\n")

