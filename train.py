import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from main import parse_cfg
from env import make_env
from Utils import *
from Agents import *
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

from TOLD import TOLD
from TDMPC import PlanningAgent, UpdateAgent
# from Agents import LoggerAgent # à changer en logger de BBRL?

from bbrl.agents.agent import Agent
from bbrl.workspace import Workspace

from omegaconf import OmegaConf
from main import parse_cfg

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
class EvaluateAgent(Agent):
	def __init__(self, cfg, device, model):
		super().__init__()
		self.cfg = cfg
		self.device = device
		self.model = model
	
	def forward(self, t, **kwargs):
		if 'env' in kwargs:
			env = kwargs['env']
		else:
			raise ValueError("env not in kwargs")
		
		if 'num_episodes' in kwargs:
			num_episodes = kwargs['num_episodes']
		else:
			raise ValueError("num_episodes not in kwargs")
		
		if 'step' in kwargs:
			step = kwargs['step']
		else:
			raise ValueError("step not in kwargs")
		
		if 'env_step' in kwargs:
			env_step = kwargs['env_step']
		else:
			raise ValueError("env_step not in kwargs")
		
		if 'video' in kwargs:
			video = kwargs['video']
		else:
			raise ValueError("video not in kwargs")
		
		if 'std' in kwargs:
			std = kwargs['std']
		else:
			raise ValueError("std not in kwargs")
		
		score = self.evaluate(env, std, num_episodes, step, env_step, video)

		if np.isnan(score) and t > 0:
			previous_evaluation = self.get(("evaluation", t-1))
			self.set(("evaluation", t), previous_evaluation.clone())
		else:
			score_tensor = torch.tensor([score], dtype=torch.float64)
			self.set(("evaluation", t), score_tensor)


	def evaluate(self, env, std, num_episodes, step, env_step, video):
		"""Evaluate a trained agent and optionally save a video."""

		if env_step % self.cfg.eval_freq != 0:
			return np.nan

		planning_agent = PlanningAgent(self.cfg, self.device, self.model)

		episode_rewards = []
		for i in range(num_episodes):
			obs, done, ep_reward, t = env.reset(), False, 0, 0
			if video: video.init(env, enabled=(i == 0))

			t = 0
			workspace = Workspace()
			while not done:
				planning_agent(workspace, t=t, std=std, obs=obs, eval_mode=True, step=step, t0=(t == 0))
				action = workspace.get("a", t)

				obs, reward, done, _ = env.step(action.cpu().numpy())
				ep_reward += reward
				if video: video.record(env)
				t += 1
			episode_rewards.append(ep_reward)
			if video: video.save(env_step)
		return np.nanmean(episode_rewards)
class TrainAgent(Agent):
	def __init__(self, cfg, device, model):
		super().__init__()
		self.cfg = cfg
		self.device = device
		self.model = model
		self.replay_buffer = ReplayBuffer(self.cfg.replay_buffer_size)
		self.told = TOLD(self.cfg, self.device, self.model)
		self.update_agent = UpdateAgent(self.cfg, self.device, self.model)
		self.logger_agent = LoggerAgent(self.cfg, self.device, self.model)
		self.evaluate_agent = EvaluateAgent(self.cfg, self.device, self.model)
	
	def forward(self, t, **kwargs):
		if 'env' in kwargs:
			env = kwargs['env']
		else:
			raise ValueError("env not in kwargs")
		
		if 'num_episodes' in kwargs:
			num_episodes = kwargs['num_episodes']
		else:
			raise ValueError("num_episodes not in kwargs")
		
		if 'step' in kwargs:
			step = kwargs['step']
		else:
			raise ValueError("step not in kwargs")
		
		if 'env_step' in kwargs:
			env_step = kwargs['env_step']
		else:
			raise ValueError("env_step not in kwargs")
		
		if 'video' in kwargs:
			video = kwargs['video']
		else:
			raise ValueError("video not in kwargs")
		
		if 'std' in kwargs:
			std = kwargs['std']
		else:
			raise ValueError("std not in kwargs")
		
		self.train(env, std, num_episodes, step, env_step, video)

	def train(self, env, std, num_episodes, step, env_step, video):
		"""Train the agent."""

		# Evaluate the agent
		self.evaluate_agent(t=env_step, env=env, std=std, num_episodes=num_episodes, step=step, env_step=env_step, video=video)

		# Train the agent
		episode = Episode(self.cfg, self.device, self.model)
		episode_rewards = []
		for i in range(num_episodes):
			obs, done, ep_reward, t = env.reset(), False, 0, 0
			if video: video.init(env, enabled=(i == 0))

			t = 0
			workspace = Workspace()
			while not done:
				self.told(workspace, t=t, std=std, obs=obs, eval_mode=False, step=step, t0=(t == 0))
				action = workspace.get("a", t)

				obs, reward, done, _ = env.step(action.cpu().numpy())
				ep_reward += reward
				if video: video.record(env)
				t += 1
				episode.add(obs, action, reward, done)
				if len(episode) >= self.cfg.episode_length:
					self.replay_buffer.add(episode)
					episode = Episode(self.cfg, self.device, self.model)
			episode_rewards.append(ep_reward)
			if video: video.save(env_step)

		# Update the agent
		if len(self.replay_buffer) >= self.cfg.batch_size:
			self.update_agent(self.replay_buffer, step=step, env_step=env_step, t=env_step)
		self.logger_agent(self.replay_buffer, step=step, env_step=env_step, t=env_step, episode_rewards=episode_rewards)
  
  
