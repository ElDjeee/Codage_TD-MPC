import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from functools import partial
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
from copy import deepcopy
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC, TOLD
from algorithm.helper import ReplayBuffer, RandomShiftsAug, linear_schedule, mse, l1, ema
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

from bbrl.agents.gymnasium import GymAgent, ParallelGymAgent, _convert_action, _torch_type
from bbrl.agents import Agents, TemporalAgent
from bbrl.workspace import Workspace

import algorithm.agents as ag

from typing import Any, Callable, Dict, Optional, Union
from gymnasium import Env

class ParallelGymAdapterAgent(ParallelGymAgent):
    def __init__(
        self,
        make_env_fn: Callable[[Optional[Dict[str, Any]]], Env],
        num_envs: int,
        make_env_args: Union[Dict[str, Any], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(make_env_fn, num_envs, make_env_args, *args, **kwargs)
    
    def _step(self, k: int, action: torch.Tensor):
        env = self.envs[k]

        action: Union[int, np.ndarray[int]] = _convert_action(action)
        obs, reward, done, info = env.step(action)

        self._timestep[k] += 1
        self.cumulated_reward[k] += reward

        return self._format_obs(
            k, obs, info, done=done, reward=reward
        )
    
    def _format_obs(
        self, k: int, obs, info, *, done=False, reward=0
    ):
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]] = ParallelGymAdapterAgent._format_frame(
            obs
        )

        if done and self.include_last_state:
            # Create a new frame to be inserted after this step,
            # containing the first observation of the next episode
            self._last_frame[k] = {
                **observation,
                "done": torch.tensor([False]),
                "reward": torch.tensor([0]).float(),
                "cumulated_reward": torch.tensor([0]).float(),
                "timestep": torch.tensor([0]),
            }
            # Use the final observation instead
            #observation = ParallelGymAdapterAgent._format_frame(info["final_observation"]) TODO I GUESS IT'S USELESS ?

        ret: Dict[str, torch.Tensor] = {
            **observation,
            "done": torch.tensor([done]),
            "reward": torch.tensor([reward]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self._timestep[k]]),
        }

        # Resets the cumulated reward and timestep
        if done and self._is_autoreset:
            self.cumulated_reward[k] = 0.0
            if self._is_autoreset and self.include_last_state:
                self._timestep[k] = 0
            else:
                self._timestep[k] = 1

        return _torch_type(ret)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_env_agents(cfg, autoreset=True, include_last_state=True):
    train_env_agent = ParallelGymAdapterAgent(
        partial(
            make_env, cfg, autoreset=autoreset
        ),
        1,
        include_last_state=include_last_state,
        seed=cfg.seed,
    )

    # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
    eval_env_agent = ParallelGymAdapterAgent(
        partial(make_env, cfg, autoreset=False),
        1,
        include_last_state=include_last_state,
        seed=cfg.seed,
    )

    return train_env_agent, eval_env_agent


def create_td3_agent(cfg, model, train_env_agent, eval_env_agent, device):
    planning_agent = ag.PlanningAgent(cfg, model, device)

    train_agent = Agents(train_env_agent, planning_agent)
    eval_agent = Agents(eval_env_agent, planning_agent)

    train_agent = TemporalAgent(train_agent)
    eval_agent = TemporalAgent(eval_agent)
    return (
        train_agent,
        eval_agent,
        planning_agent
    )

def _td_target(cfg, model, model_target, next_obs, reward):
    with torch.no_grad():
        next_z = model.h(next_obs)
        td_target = reward + cfg.discount * \
            torch.min(*model_target.Q(next_z, model.pi(next_z, cfg.min_std)))
        return td_target

def update_pi(cfg, model, pi_optim, zs):
    pi_optim.zero_grad(set_to_none=True)
    model.track_q_grad(False)

    # Loss is a weighted sum of Q-values
    pi_loss = 0
    for t,z in enumerate(zs):
        a = model.pi(z, cfg.min_std)
        Q = torch.min(*model.Q(z, a))
        pi_loss += -Q.mean() * (cfg.rho ** t)

    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(model._pi.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
    pi_optim.step()
    model.track_q_grad(True)
    return pi_loss.item()

def update(cfg, buffer, step, model, model_target, optim, pi_optim, aug):
    obs, next_obses, action, reward, idxs, weights = buffer.sample()

    optim.zero_grad(set_to_none=True)
    std = linear_schedule(cfg.std_schedule, step)
    model.train()

    z = model.h(aug(obs))
    zs = [z.detach()]

    consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0

    for t in range(cfg.horizon):
        # Predictions
        Q1, Q2 = model.Q(z, action[t])
        z, reward_pred = model.next(z, action[t])
        with torch.no_grad():
            next_obs = aug(next_obses[t])
            next_z = model_target.h(next_obs)
            td_target = _td_target(cfg, model, model_target, next_obs, reward[t])
        zs.append(z.detach())

        # Losses
        rho = (cfg.rho ** t)
        consistency_loss += rho * torch.mean(mse(z, next_z), dim=1, keepdim=True)
        reward_loss += rho * mse(reward_pred, reward[t])
        value_loss += rho * (mse(Q1, td_target) + mse(Q2, td_target))
        priority_loss += rho * (l1(Q1, td_target) + l1(Q2, td_target))

    # Optimize model
    total_loss = cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                    cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                    cfg.value_coef * value_loss.clamp(max=1e4)
    weighted_loss = (total_loss.squeeze(1) * weights).mean()
    weighted_loss.register_hook(lambda grad: grad * (1/cfg.horizon))
    weighted_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
    optim.step()
    buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

    # Update policy + target network
    pi_loss = update_pi(cfg, model, pi_optim, zs)
    if step % cfg.update_freq == 0:
        ema(model, model_target, cfg.tau)

    model.eval()

    return {'consistency_loss': float(consistency_loss.mean().item()),
            'reward_loss': float(reward_loss.mean().item()),
            'value_loss': float(value_loss.mean().item()),
            'pi_loss': pi_loss,
            'total_loss': float(total_loss.mean().item()),
            'weighted_loss': float(weighted_loss.mean().item()),
            'grad_norm': float(grad_norm)}, std

# TODO check for video
def evaluate(eval_agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        eval_workspace = Workspace()
        eval_agent(eval_workspace, t=0, step=step, stop_variable="env/done", std=0, first_episode=True)
        episode_rewards.append(eval_workspace['env/cumulated_reward'][-1].float())
    return np.nanmean(episode_rewards)


def train(cfg, device):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    #assert torch.cuda.is_available()

    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    buffer = ReplayBuffer(cfg, device)

    train_workspace = Workspace()

    std = linear_schedule(cfg.std_schedule, 0)
    model = TOLD(cfg).to(device)
    model_target = deepcopy(model).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pi_optim = torch.optim.Adam(model._pi.parameters(), lr=cfg.lr)
    aug = RandomShiftsAug(cfg)
    model.eval()
    model_target.eval()

    (
        train_agent,
        eval_agent,
        planning_agent
    ) = create_td3_agent(cfg, model, train_env_agent, eval_env_agent, device)

    ag_planning = TemporalAgent(planning_agent)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    step = 0
    while step < cfg.train_steps:
        if step > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)

            train_agent(train_workspace, t=1, step=step, stop_variable="env/done", std=std, first_episode=False)
        else:
            train_agent(train_workspace, t=0, step=step, stop_variable="env/done", std=std, first_episode=True)

        transition_workspace = train_workspace.get_transitions()
        if step > 0 or cfg.train_steps > 1:
            buffer.add(transition_workspace)

        # # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                ret, std = update(cfg, buffer, step+i, model, model_target, optim, pi_optim, aug)
                train_metrics.update(ret)

        # Log training episode
        episode_idx += 1
        env_step = int(step*cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': transition_workspace['env/cumulated_reward'][0][-1].float()}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(eval_agent, cfg.eval_episodes, step, env_step, L.video)
            L.log(common_metrics, category='eval')

        step += int(cfg.episode_length)

    L.finish(model, model_target)
    print('Training completed successfully')


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device in ['cuda', 'mps'], "This training script requires a CUDA-compatible or MPS-compatible device."

    train(parse_cfg(Path().cwd() / __CONFIG__, device=device), device=device)
