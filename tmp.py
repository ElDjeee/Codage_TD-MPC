import copy
from functools import partial
from typing import Tuple

import hydra
import torch
from torch import nn
from omegaconf import DictConfig

import gymnasium as gym
from gymnasium import register
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

from told import EncoderAgent, DynamicsAgent, RewardAgent, PiAgent, QAgent

def make_env_TDMPC(env_name, cfg, autoreset=True):
    cartpole_spec = gym.spec("CartPole-v1")
    register(
        id="CartPole-v2",
        entry_point= __name__ + ":CartPoleEnv",
        max_episode_steps=cartpole_spec.max_episode_steps,
        reward_threshold=cartpole_spec.reward_threshold,
    )

    env = gym.make("CartPole-v2", render_mode="rgb_array")
    env = PixelOnlyObservation(env)
    env = ResizeObservation(env, cfg.img_size)
    env = GrayScaleObservation(env)
    env = BinarizeObservation(env, 230, True)
    env = FrameStack(env, 3)

    if autoreset:
        env = AutoResetWrapper(env)

    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    #cfg.action_dim = env.action_space.shape[0]
    cfg.action_dim = 2 # it is supposed to be 2

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
    Qvalues = QAgent(device, cfg)

    tr_agent = Agents(train_env_agent, encoder, pi, reward, dynamics)
    ev_agent = Agents(eval_env_agent, encoder, pi, reward, dynamics)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return (
        train_agent,
        eval_agent,
        encoder,
        dynamics,
        reward,
        pi,
        Qvalues
    )

def setup_optimizers(cfg, encoder, Qvalues):
    encoder_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = encoder.parameters()
    encoder_optimizer = get_class(cfg.actor_optimizer)(parameters, **encoder_optimizer_args)

    Qvalues_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = Qvalues.parameters()
    Qvalues_optimizer = get_class(cfg.critic_optimizer)(parameters, **Qvalues_optimizer_args)

    return encoder_optimizer, Qvalues_optimizer

def run_tdmpc(cfg, trial=None):
    # 1)  Build the  logger
    best_reward = float("-inf")
    delta_list = []
    mean = 0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 2) Create the environment agents
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the TD3 Agent
    (
        train_agent, # TODO: make him Collect trajectory
        eval_agent,
        encoder,
        dynamics,
        reward,
        pi,
        Qvalues
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent, device)
    ag_encoder = TemporalAgent(encoder)
    ag_dynamics = TemporalAgent(dynamics)
    ag_reward = TemporalAgent(reward)
    ag_pi = TemporalAgent(pi)
    ag_Qvalues = TemporalAgent(Qvalues)

    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    encoder_optimizer, Qvalues_optimizer = setup_optimizers(cfg, encoder, Qvalues)
    nb_steps = 0
    tmp_steps = 0

    logger = Logger(cfg)

    # Trick to compute the values of:
    cfg.train_steps = int(int(cfg.train_steps.split("/")[0]) / int(cfg.train_steps.split("/")[1]))
    cfg.episode_length = int(int(cfg.episode_length.split("/")[0]) / int(cfg.episode_length.split("/")[1]))

    # TMP
    cfg.algorithm.n_steps_train = 50

    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        print(f"Step {step}")

        if step > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        if nb_steps > 0 or cfg.algorithm.n_steps_train > 1:
            rb.put(transition_workspace)

@hydra.main(
    config_path="./cfgs/tasks",
    config_name="TMP.yaml",
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
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_tdmpc)
    else:
        run_tdmpc(cfg_raw)

if __name__ == "__main__":
    main()