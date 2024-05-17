import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import random
import copy
from functools import partial
from typing import Tuple
import time

import hydra
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

import bbrl_gymnasium as gym
#from gymnasium import register
from gymnasium.wrappers import AutoResetWrapper

from bbrl import get_arguments, get_class
from bbrl.agents import Agents, TemporalAgent
from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl_algos.models.exploration_agents import AddGaussianNoise
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from bbrl_algos.models.critics import ContinuousQAgent
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.loggers import Logger
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.workspace import Workspace

from preprocess import *

from tqdm import tqdm
import re

torch.backends.cudnn.benchmark = True

from told import EncoderAgent, DynamicsAgent, RewardAgent, PiAgent, QAgent, TruncatedNormal

def make_env_TDMPC(env_name, cfg, autoreset=True):
	# cartpole_spec = gym.spec("CartPole-v1")
	# register(
	#     id="CartPole-v2",
	#     entry_point= __name__ + ":CartPoleEnv",
	#     max_episode_steps=cartpole_spec.max_episode_steps,
	#     reward_threshold=cartpole_spec.reward_threshold,
	# )

	env = gym.make("CartPoleContinuous-v1", render_mode="rgb_array")
	env = PixelOnlyObservation(env)
	env = ResizeObservation(env, cfg.img_size)
	env = GrayScaleObservation(env)
	env = BinarizeObservation(env, 230, True)
	env = FrameStack(env, 3)

	if autoreset:
		env = AutoResetWrapper(env)

	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]

	return env

def get_env_agents(cfg, autoreset=True, include_last_state=True):
	train_env_agent = ParallelGymAgent(
		partial(
			make_env_TDMPC, cfg.gym_env.env_name, cfg, autoreset=autoreset
		),
		cfg.algorithm.n_envs,
		include_last_state=include_last_state,
		seed=cfg.algorithm.seed.train,
	)

	# Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
	eval_env_agent = ParallelGymAgent(
		partial(make_env_TDMPC, cfg.gym_env.env_name, cfg),
		cfg.algorithm.nb_evals,
		include_last_state=include_last_state,
		seed=cfg.algorithm.seed.eval,
	)

	return train_env_agent, eval_env_agent

def create_td3_agent(cfg, train_env_agent, eval_env_agent, device):
	encoder = EncoderAgent(device, cfg)
	dynamics = DynamicsAgent(device, cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
	reward = RewardAgent(device, cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
	pi = PiAgent(device, cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
	QValues = QAgent(device, cfg)

	tr_agent = Agents(train_env_agent, encoder, pi, reward, dynamics) # TODO
	ev_agent = Agents(eval_env_agent, encoder, pi, reward, dynamics) # TODO

	train_agent = TemporalAgent(tr_agent)
	eval_agent = TemporalAgent(ev_agent)
	return (
		train_agent,
		eval_agent,
		encoder,
		dynamics,
		reward,
		pi,
		QValues
	)

def setup_optimizers(cfg, parameters):
	optim_optimizer_args = get_arguments(cfg.actor_optimizer)
	optim_optimizer = get_class(cfg.actor_optimizer)(parameters, **optim_optimizer_args)

	pi_optim_optimizer_args = get_arguments(cfg.critic_optimizer)
	pi_optim_optimizer = get_class(cfg.critic_optimizer)(parameters, **pi_optim_optimizer_args)

	return optim_optimizer, pi_optim_optimizer


# TODO TMP FOR TEST PLAN
def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)

# TODO TMP FOR TEST PLAN
def estimate_value(cfg, z, actions, horizon, model, eval=False):
	"""Estimate value of a trajectory starting at latent state z and executing given actions."""
	with torch.no_grad():
		G, discount = 0, 1
		for t in range(horizon):
			z, r = model.dynamics.predict_z(z, actions[t]), model.reward.predict_reward(z, actions[t])
			G += discount * r
			discount *= cfg.discount
		G += discount * torch.min(*model.QValues.predict_q(z, model.pi.predict_pi(z, cfg.min_std)))
		return G

# TODO TMP FOR TEST PLAN
def plan(cfg, device, model, obs, _prev_mean, self_std, eval_mode=False, step=None, t0=True):
	with torch.no_grad():
		# Seed steps
		if step < cfg.seed_steps:
			return torch.empty(cfg.action_dim, dtype=torch.float32, device=device).uniform_(-1, 1), _prev_mean

		# Sample policy trajectories
		obs = torch.as_tensor(obs).clone().detach().to(dtype=torch.float32, device=device)
		horizon = int(min(cfg.horizon, linear_schedule(cfg.horizon_schedule, step)))
		num_pi_trajs = int(cfg.mixture_coef * cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, cfg.action_dim, device=device)
			z = model.encoder.predict_latent(obs).repeat(num_pi_trajs, 1)
			for tbis in range(horizon):
				pi_actions[tbis] = model.pi.predict_pi(z, cfg.min_std)
				z = model.dynamics.predict_z(z, pi_actions[tbis]).to(device)

		# Initialize state and parameters
		z = model.encoder.predict_latent(obs).repeat(cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, cfg.action_dim, device=device)
		std = 2*torch.ones(horizon, cfg.action_dim, device=device)
		if not t0 and _prev_mean is not None:
			mean[:-1] = _prev_mean[1:]

		# Iterate CEM
		for i in range(cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, cfg.num_samples, cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			value = estimate_value(cfg, z, actions, horizon, model, eval_mode).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(self_std, 2)
			mean, std = cfg.momentum * mean + (1 - cfg.momentum) * _mean, _std

		# Outputs
		score = score.squeeze(1).cpu().detach().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		_prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(cfg.action_dim, device=std.device)
		
		return a, _prev_mean

# TODO MAKE IT WORK
class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x):
		x = x[0]

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

# TODO HANDLE MODEL TARGET
def _td_target(cfg, next_obs, curr_reward, model, model_target):
	"""Compute the TD-target from a reward and the observation at the following time step."""
	with torch.no_grad():
		next_z = model.encoder.predict_latent(next_obs)
		td_target = curr_reward + cfg.discount * \
			torch.min(*model_target.QValues.predict_q(next_z, model.pi.predict_pi(next_z, cfg.min_std)))
		return td_target

__REDUCE__ = lambda b: 'mean' if b else 'none'

def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))

def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)

def track_q_grad(QValues, enable=True):
	"""Utility function. Enables/disables gradient tracking of Q-networks."""
	for m in [QValues.q1, QValues.q2]:
		set_requires_grad(m, enable)

def update_pi(zs, cfg, pi_optim, model):
	"""Update policy using a sequence of latent states."""
	pi_optim.zero_grad(set_to_none=True)
	track_q_grad(model.QValues, False)

	# Loss is a weighted sum of Q-values
	pi_loss = 0
	for t,z in enumerate(zs):
		a = model.pi.predict_pi(z, cfg.min_std)
		Q = torch.min(*model.QValues.predict_q(z, a))
		pi_loss += -Q.mean() * (cfg.rho ** t)
		#print(-Q.mean(), end=" ")

	pi_loss.backward()
	torch.nn.utils.clip_grad_norm_(model.pi.net.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
	pi_optim.step()
	track_q_grad(model.QValues, True)
	return pi_loss.item()

def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)

# TODO MODEL TARGET
def update(replay_buffer, step, cfg, device, optim, pi_optim, model, model_target, aug):
	"""Main update function. Corresponds to one iteration of the TOLD model learning."""
	workspace = replay_buffer.get_shuffled(cfg.batch_size)
	obs = workspace.get_full("env/env_obs").to(device)
	next_obses = workspace.get_full("env/env_obs").to(device) # TODO handle diff between obs and next_obses
	action = workspace.get_full("action").to(device)
	curr_reward = workspace.get_full("reward").to(device)

	optim.zero_grad(set_to_none=True)
	self_std = linear_schedule(cfg.std_schedule, step)
	
	model.train()

	# Representation
	z = model.encoder.predict_latent(aug(obs[0][0].unsqueeze(0).unsqueeze(1)))
	zs = [z.detach()]

	consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
	for t in range(cfg.horizon):

		# Predictions
		Q1, Q2 = model.QValues.predict_q(z, action[0][t].unsqueeze(0).unsqueeze(1)) # TODO why do we have Tensor[2, 256] for our workspace elements ? 
		z, reward_pred = model.dynamics.predict_z(z, action[0][t].unsqueeze(0).unsqueeze(1)), model.reward.predict_reward(z, action[0][t].unsqueeze(0).unsqueeze(1))
		with torch.no_grad():
			next_obs = aug(next_obses[0][t].unsqueeze(0).unsqueeze(1))
			next_z = model_target.encoder.predict_latent(next_obs)
			td_target = _td_target(cfg, next_obs, curr_reward[0][t].unsqueeze(0).unsqueeze(1), model, model_target) # TODO why we cannot use curr_reward[t] ?
		zs.append(z.detach())

		# Losses
		rho = (cfg.rho ** t)
		consistency_loss += rho * torch.mean(mse(z, next_z), dim=1, keepdim=True)
		reward_loss += rho * mse(reward_pred, curr_reward[0][t].unsqueeze(0).unsqueeze(1))
		value_loss += rho * (mse(Q1, td_target) + mse(Q2, td_target))
		priority_loss += rho * (l1(Q1, td_target) + l1(Q2, td_target))

	# Optimize model
	total_loss = cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					cfg.value_coef * value_loss.clamp(max=1e4)
	weighted_loss = (total_loss.squeeze(1)).mean()
	weighted_loss.register_hook(lambda grad: grad * (1/cfg.horizon))
	weighted_loss.backward()
	grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
	optim.step()

	# Update policy + target network
	pi_loss = update_pi(zs, cfg, pi_optim, model)
	if step % cfg.update_freq == 0:
		ema(model, model_target, cfg.tau)

	model.eval()

	return {'consistency_loss': float(consistency_loss.mean().item()),
			'reward_loss': float(reward_loss.mean().item()),
			'value_loss': float(value_loss.mean().item()),
			'pi_loss': pi_loss,
			'total_loss': float(total_loss.mean().item()),
			'weighted_loss': float(weighted_loss.mean().item()),
			'grad_norm': float(grad_norm)}, self_std

# TODO TMP FOR TEST EVAL
def evaluate(env, num_episodes, step, cfg, device, model, _prev_mean, self_std):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env._reset(0)['env_obs'], False, 0, 0
		while not done:
			action, _prev_mean = plan(cfg, device, model, obs, _prev_mean, self_std, eval_mode=True, step=step, t0=(step == 0))
			obs, _, _, done, curr_reward, _, _ = env._step(0, action.cpu()).values() # 0 because we have only one env
		
			ep_reward += curr_reward[0].item()
			t += 1
		episode_rewards.append(ep_reward)
	return np.nanmean(episode_rewards), _prev_mean

class TOLD(nn.Module):
	def __init__(self, cfg, encoder, dynamics, reward, pi, QValues, device):
		super().__init__()
		self.cfg = cfg

		self.encoder = encoder
		self.dynamics = dynamics
		self.reward = reward
		self.pi = pi
		self.QValues = QValues

		self.device = device
	
	def parameters(self):
		return (
            list(self.encoder.enc.parameters()) + 
            list(self.dynamics.net.parameters()) + 
            list(self.reward.net.parameters()) + 
            list(self.pi.net.parameters()) +
            list(self.QValues.q1.parameters()) +
            list(self.QValues.q2.parameters())
        )
	
	def train(self):
		self.encoder.enc.train()
		self.dynamics.net.train()
		self.reward.net.train()
		self.pi.net.train()
		self.QValues.q1.train()
		self.QValues.q2.train()
	
	def eval(self):
		self.encoder.enc.eval()
		self.dynamics.net.eval()
		self.reward.net.eval()
		self.pi.net.eval()
		self.QValues.q1.eval()
		self.QValues.q2.eval()



def run_tdmpc(cfg, trial=None):
	# 1)  Build the  logger
	best_reward = float("-inf")
	delta_list = []
	mean = 0

	device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
	#print(device)

	# 2) Create the environment agents
	train_env_agent, eval_env_agent = get_env_agents(cfg)

	# 3) Create the TD3 Agent
	(
		train_agent,
		eval_agent,
		encoder,
		dynamics,
		reward,
		pi,
		QValues
	) = create_td3_agent(cfg, train_env_agent, eval_env_agent, device)

	model = TOLD(cfg, encoder, dynamics, reward, pi, QValues, device).to(device)
	model.eval()

	model_target = copy.deepcopy(model).to(device)
	model_target.eval()

	ag_encoder = TemporalAgent(encoder)
	ag_dynamics = TemporalAgent(dynamics)
	ag_reward = TemporalAgent(reward)
	ag_pi = TemporalAgent(pi)
	ag_QValues = TemporalAgent(QValues)

	train_workspace = Workspace()
	rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

	aug = RandomShiftsAug(cfg)

	# Configure the optimizer
	optim, pi_optim = setup_optimizers(cfg, model.parameters())
	nb_steps = 0
	tmp_steps = 0

	#logger = Logger(cfg)

	# Trick to compute the values of:
	cfg.train_steps = int(int(cfg.train_steps.split("/")[0]) / int(cfg.train_steps.split("/")[1]))
	cfg.episode_length = int(int(cfg.episode_length.split("/")[0]) / int(cfg.episode_length.split("/")[1]))

	episode_idx, start_time = 0, time.time()
	self_std = linear_schedule(cfg.std_schedule, 0)
	for step in tqdm(range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length)):
		"""if step > 0:
			train_workspace.zero_grad()
			train_workspace.copy_n_last_steps(1)
			train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
		else:
			train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

		transition_workspace = train_workspace.get_transitions()
		action = transition_workspace["action"]
		nb_steps += action[0].shape[0]
		if nb_steps > 0 or cfg.algorithm.n_steps_train > 1:
			rb.put(transition_workspace)"""

		_prev_mean = None

		obs = train_env_agent._reset(0)['env_obs'] # 0 because we have only one env
		done = False
		t = 0
		cum_reward = 0
		while done is False:
			action, _prev_mean = plan(cfg, device, model, obs, _prev_mean, self_std, eval_mode=False, step=step, t0=(step == 0))
			obs, _, _, done, curr_reward, _, _ = train_env_agent._step(0, action.cpu()).values() # 0 because we have only one env
		
			train_workspace.set("env/env_obs", t, obs)
			train_workspace.set("action", t, action)
			train_workspace.set("reward", t, curr_reward)
			train_workspace.set("env/done", t, done)

			done = bool(done[0])
			t += 1

			cum_reward += curr_reward[0].item()
			
		transition_workspace = train_workspace.get_transitions()
		action = transition_workspace["action"]
		nb_steps += action[0].shape[0]

		if nb_steps > 0 or cfg.algorithm.n_steps_train > 1:
			rb.put(transition_workspace)

		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				res, self_std = update(rb, step+i, cfg, device, optim, pi_optim, model, model_target, aug)
				train_metrics.update(res)

		# Log training episode
		episode_idx += 1
		env_step = int(step * cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': cum_reward}
		train_metrics.update(common_metrics)
		
		with open("log.txt", "a") as f:
			f.write('train : ' + str(train_metrics) + '\n')
			print("train : ", train_metrics)

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			print()
			print("="*96)
			#torch.cuda.empty_cache()
			common_metrics['episode_reward'], _prev_mean = evaluate(eval_env_agent, cfg.eval_episodes, step, cfg, device, model, _prev_mean, self_std)
			with open("log.txt", "a") as f:
				f.write('eval : ' + str(common_metrics) + '\n')
				print("eval : ", common_metrics)
			print("="*96)
			print()



def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

@hydra.main(
	config_path="./cfgs/tasks",
	config_name="bbrl_env.yaml",
	# config_name="sac_lunar_lander_continuous.yaml",
	# config_name="sac_cartpolecontinuous.yaml",
	# config_name="sac_pendulum.yaml",
	# config_name="sac_swimmer_optuna.yaml",
	# config_name="sac_swimmer.yaml",
	# config_name="sac_walker_test.yaml",
	# config_name="sac_torcs.yaml",
	# version_base="1.3",
)
def main(cfg_raw: DictConfig):
	#torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)
	set_seed(cfg_raw.algorithm.seed.torch)

	if "optuna" in cfg_raw:
		launch_optuna(cfg_raw, run_tdmpc)
	else:
		run_tdmpc(cfg_raw)

if __name__ == "__main__":
	main()
