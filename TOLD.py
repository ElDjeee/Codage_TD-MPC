import helper as h

import torch
from torch import nn

class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

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
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)