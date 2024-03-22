import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from bbrl.agents.agent import Agent
from Utils import _get_out_shape
from preprocess import *

__REDUCE__ = lambda b: 'mean' if b else 'none'

def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))

class L1LossAgent(Agent):
	def __init__(self):
		super().__init__()
		
	def forward(self, reduce=False, **kwargs): 
		# d'après l'article sur salina. 
		target = self.get("y")
		pred = self.get("predicted_y")
		loss = l1(pred, target, reduce)
		self.set("loss", loss)
	
def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

class MSELossAgent(Agent):
	def __init__(self):
		super().__init__()

	def forward(self, reduce=False, **kwargs): 
		# d'après l'article sur salina.
		target = self.get("y")
		pred = self.get("predicted_y")
		loss = mse(pred, target, reduce)
		self.set("loss", loss)
		
class TruncatedNormal(pyd.Normal): # que faire?
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

class NormalizeImg(Agent): # DONE
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x, **kwargs):
		return x.div(255.)

class Flatten(Agent): # DONE
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x, **kwargs):
		return x.view(x.size(0), -1)

def enc(cfg):
	"""Returns a TOLD encoder."""
	if cfg.modality == 'pixels':
		C = int(3*cfg.frame_stack)
		layers = [NormalizeImg(),
				  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
	else:
		layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)

class EncoderAgent(Agent):
	def __init__(self, cfg):
		super().__init__()
		# if cfg.modality == 'pixels':
			# récupérer l'env?
			# PixelOnlyObservation()
			# GrayScaleObservation()
			# BinarizeObservation()
			# FrameStack()
		self.fc = enc(cfg)

	def forward(self, t, **kwargs): # ajouter le code de l'étudiant?
		obs = self.get(("obs", t))
		
		latent = self.fc(obs)
		self.set(("latent", t), latent) 

def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

class MLPAgent(Agent): 
	def __init__(self, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
		super().__init__()
		self.fc = mlp(in_dim, mlp_dim, out_dim, act_fn)

	def forward(self, t, **kwargs): 
		obs = self.get(("obs", t))
		res = self.fc(obs)
		self.set(("pred", t), res) 

def q(cfg, act_fn=nn.ELU()): # act_fn non utilisé?
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))

class QFunctionAgent(Agent): # à remplacer avec ContinuousQAgent de BBRL?
	def __init__(self, cfg, act_fn=nn.ELU()):
		self.fc = q(cfg, act_fn)

	def forward(self, t, **kwargs): 
		obs = self.get(("obs", t))
		qfunc = self.fc(obs)
		self.set(("q-func", t), qfunc)
		
class RandomShiftsAug(Agent): # DONE
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x, **kwargs):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)