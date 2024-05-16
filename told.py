import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


from bbrl.agents.agent import Agent

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

# TODO, as agent
class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)
	  
# TODO, as agent
class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)

# TODO, as agent
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



class EncoderAgent(Agent):
	def __init__(self, device, cfg):
		super().__init__()

		if cfg.modality == 'pixels':
			C = int(cfg.frame_stack)
			layers = [NormalizeImg(),
					nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
					#nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()] # Why this doesnt work for cfg.img_size = 25 TODO
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
			
			in_shape = (C, cfg.img_size, cfg.img_size)
			x = torch.randn(*in_shape).unsqueeze(0)

			for i, layer in enumerate(layers):
				x = layer(x)
			
			out_shape = x.shape
			num_features = np.prod(out_shape[1:]).item()

			layers.append(Flatten())
			layers.append(nn.Linear(num_features, cfg.latent_dim))
		else:
			layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
					nn.Linear(cfg.enc_dim, cfg.latent_dim)]
		
		self.device = device
		self.enc = nn.Sequential(*layers).to(self.device)

		orthogonal_init(self.enc)
	
	def forward(self, t, **kwargs):
		obs = self.get(("env/env_obs", t)).to(self.device)
		latent = self.enc(obs)
		self.set(("latent", t), latent)
	
	def predict_latent(self, obs):
		return self.enc(obs)


class PiAgent(Agent):
	def __init__(self, device, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
		super().__init__()

		if isinstance(mlp_dim, int):
			mlp_dim = [mlp_dim, mlp_dim]
		
		self.device = device
		self.net = nn.Sequential(
			nn.Linear(in_dim, mlp_dim[0]), act_fn,
			nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
			nn.Linear(mlp_dim[1], out_dim)).to(self.device)

		orthogonal_init(self.net)

	def forward(self, t, **kwargs):
		latent = self.get(("latent", t)).to(self.device)

		x = self.net(latent)
		action = torch.argmax(x, axis=1)

		self.set(("action", t), action)
	
	def predict_pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self.net(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

class RewardAgent(Agent):
	def __init__(self, device, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
		super().__init__()

		if isinstance(mlp_dim, int):
			mlp_dim = [mlp_dim, mlp_dim]
		
		self.device = device
		self.net = nn.Sequential(
			nn.Linear(in_dim, mlp_dim[0]), act_fn,
			nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
			nn.Linear(mlp_dim[1], out_dim)).to(self.device)

		orthogonal_init(self.net)

		self.net[-1].weight.data.fill_(0)
		self.net[-1].bias.data.fill_(0)
		

	def forward(self, t, **kwargs):
		latent = self.get(("latent", t)).to(self.device)
		action = self.get(("action", t)).to(self.device)

		x = torch.cat([latent, action], dim=1)

		self.set(("reward", t), self.net(x))
	
	def predict_reward(self, latent, action):
		x = torch.cat([latent, action], dim=1)
		return self.net(x)

class DynamicsAgent(Agent):
	def __init__(self, device, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
		super().__init__()

		if isinstance(mlp_dim, int):
			mlp_dim = [mlp_dim, mlp_dim]
		
		self.device = device
		self.net = nn.Sequential(
			nn.Linear(in_dim, mlp_dim[0]), act_fn,
			nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
			nn.Linear(mlp_dim[1], out_dim)).to(self.device)

		orthogonal_init(self.net)

	def forward(self, t, **kwargs):
		latent = self.get(("latent", t)).to(self.device)
		action = self.get(("action", t)).to(self.device)

		x = torch.cat([latent, action], dim=1)
		z = self.net(x)

		self.set(("latent", t+1), z)
	
	def predict_z(self, latent, action):
		x = torch.cat([latent, action], dim=1)
		return self.net(x)


class QAgent(Agent):
	def __init__(self, device, cfg):
		super().__init__()

		self.device = device

		self.q1 = nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
								nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
								nn.Linear(cfg.mlp_dim, 1)).to(self.device)
		self.q2 = nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
								nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
								nn.Linear(cfg.mlp_dim, 1)).to(self.device)
	
		self.q1[-1].weight.data.fill_(0)
		self.q1[-1].bias.data.fill_(0)
		self.q2[-1].weight.data.fill_(0)
		self.q2[-1].bias.data.fill_(0)

		orthogonal_init(self.q1)
		orthogonal_init(self.q2)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self.q1, self.q2]:
			set_requires_grad(m, enable)

	def forward(self, t, **kwargs):
		latent = self.get(("latent", t)).to(self.device)
		action = self.get(("action", t)).to(self.device)

		x = torch.cat([latent, action], dim=1)

		self.set(("q1", t), self.q1(x))
		self.set(("q2", t), self.q2(x))

	def predict_q(self, latent, actions):
		x = torch.cat([latent, actions], dim=1)
		return self.q1(x), self.q2(x)
