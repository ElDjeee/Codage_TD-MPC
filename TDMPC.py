import helper as h

import numpy as np
import torch
from copy import deepcopy

from TOLD import TOLD

from bbrl.agents.agent import Agent
from bbrl.workspace import Workspace

class EstimateValueAgent(Agent):
	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, t, **kwargs):
		if 'horizon' in kwargs:
			horizon = kwargs['horizon']
		else:
			raise ValueError("horizon not in kwargs")
		
		if 'actions' in kwargs:
			actions = kwargs['actions']
		else:
			raise ValueError("actions not in kwargs")
		
		if 'z' in kwargs:
			z = kwargs['z']
		else:
			raise ValueError("z not in kwargs")

		self.set(("G", t), self.estimate_value(self, z, actions, horizon))

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		return G



class PlanningAgent(Agent):
	def __init__(self, cfg, device, model):
		super().__init__()
		self.cfg = cfg
		self.device = device
		self.model = model

	def forward(self, t, **kwargs):
		if 'obs' in kwargs:
			obs = kwargs['obs']
		else:
			raise ValueError("obs not in kwargs")
		
		eval_mode = False
		if 'eval_mode' in kwargs:
			eval_mode = kwargs['eval_mode']
		
		step = None
		if 'step' in kwargs:
			step = kwargs['step']
		
		t0 = True
		if 't0' in kwargs:
			t0 = kwargs['t0']
		
		if 'std' in kwargs:
			std = kwargs['std']
		else:
			raise ValueError("std not in kwargs")

		action = torch.tensor([self.plan(obs, std, eval_mode, step, t0)], dtype=torch.float32)
		self.set(("a", t), action)

	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		return G

	@torch.no_grad()
	def plan(self, obs, std_input, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32).uniform_(-1, 1).to(self.device)

		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]
		
		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			#value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
			value = self.estimate_value(z, actions, horizon).nan_to_num_(0)

			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(std_input, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a



class UpdateAgent(Agent):
	def __init__(self, cfg, model):
		super().__init__()
		self.cfg = cfg
		self.model = model
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(self.cfg)
		self.model_target = deepcopy(self.model)
		self.model_target.eval()

	def forward(self, t, **kwargs):
		if 'replay_buffer' in kwargs:
			replay_buffer = kwargs['replay_buffer']
		else:
			raise ValueError("replay_buffer not in kwargs")
		
		if 'step' in kwargs:
			step = kwargs['step']
		else:
			raise ValueError("step not in kwargs")
		
		results = self.update(replay_buffer, step)

		consistency_loss = torch.tensor([results['consistency_loss']], dtype=torch.float64)
		reward_loss = torch.tensor([results['reward_loss']], dtype=torch.float64)
		value_loss = torch.tensor([results['value_loss']], dtype=torch.float64)
		pi_loss = torch.tensor([results['pi_loss']], dtype=torch.float64)
		total_loss = torch.tensor([results['total_loss']], dtype=torch.float64)
		weighted_loss = torch.tensor([results['weighted_loss']], dtype=torch.float64)
		grad_norm = torch.tensor([results['grad_norm']], dtype=torch.float64)
		
		self.set(("consistency_loss", t), consistency_loss)
		self.set(("reward_loss", t), reward_loss)
		self.set(("value_loss", t), value_loss)
		self.set(("pi_loss", t), pi_loss)
		self.set(("total_loss", t), total_loss)
		self.set(("weighted_loss", t), weighted_loss)
		self.set(("grad_norm", t), grad_norm)

		# TODO : Update std for planning (self.std)
		# self.set(("std", t), results['std'])

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		with open("log.txt", "a") as f:
			f.write("Update\n")
			f.write(str(replay_buffer) + "\n")
			f.write(str(step) + "\n")

		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg.horizon):

			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

		# Optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs)
		if step % self.cfg.update_freq == 0:
			h.ema(self.model, self.model_target, self.cfg.tau)

		self.model.eval()
		return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm),
				'std': self.std}



class TDMPC(Agent):
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
		self.model = TOLD(cfg).to(self.device)
		self.model.eval()
		

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])




