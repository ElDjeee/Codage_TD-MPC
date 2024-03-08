import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import helper as h  # Assuming this module contains the necessary functions and classes

import bbrl
from bbrl.workspace import Workspace
from bbrl.agents.agent import Agent
from bbrl.agents import Agents, TemporalAgent

class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.)

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

def enc(frame_stack, num_channels, img_size, latent_dim):
    """Returns a TOLD encoder."""
    C = int(3 * frame_stack)  # Adjust this if your input has a different number of channels per frame
    layers = [NormalizeImg(),
              nn.Conv2d(C, num_channels, 3, stride=2, padding=1), nn.ReLU(),
              nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1), nn.ReLU(),
              nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1), nn.ReLU(),
              nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1), nn.ReLU()]
    out_shape = _get_out_shape((C, img_size, img_size), layers)
    layers.extend([Flatten(), nn.Linear(np.prod(out_shape), latent_dim)])

    return nn.Sequential(*layers)

class TOLD_Agent(Agent):
    def __init__(self, cfg, device, frame_stack=3, num_channel=32, img_size=84, latent_dim=50):
        super().__init__()
        self.device = device
        self.frame_stack = frame_stack
        self.num_channel = num_channel
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.encoder = enc(frame_stack, num_channel, img_size, latent_dim).to(self.device)
        self.optim = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.cfg = cfg
        self.observation = None

        self.workspace = Workspace()

        if cfg is not None:
            # Assuming h.mlp and h.q are properly defined in the helper module
            self._dynamics = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
            self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
            self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
            self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
            self.apply(h.orthogonal_init)  # Assuming this correctly initializes the weights
            for m in [self._reward, self._Q1, self._Q2]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)
        else:
            self._dynamics = None
            self._reward = None
            self._pi = None
            self._Q1 = None
            self._Q2 = None

    def set_observation(self, observation):
        """Sets the current observation."""
        self.observation = observation.to(self.device)

    def forward(self, t, **kwargs):
        if self.observation is not None:
            latent = self.encoder(self.observation)
            self.set(("latent", t), latent)
        else:
            print("Observation not set.")

    def backward(self, t, **kwargs):
        self.optim.zero_grad()
        loss = self.get(("loss", t))
        loss.backward()
        self.optim.step()

# Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# agent = TOLD_Agent(cfg=None, device=device)  # Ensure cfg is defined if needed

# # Create a dummy observation with the correct shape: [batch_size, channels, height, width]
# dummy_observation = torch.randn(1, 3 * agent.frame_stack, agent.img_size, agent.img_size).to(device)

# # Set the observation for the agent and run the forward pass
# agent.set_observation(dummy_observation)
# agent.forward(t=0)

# # Retrieve and print the latent representation
# latent_data = agent.get(("latent", 0))
# print("Latent representation shape:", latent_data.shape)