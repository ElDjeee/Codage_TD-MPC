import copy
from functools import partial
from typing import Tuple

import hydra
import torch
from torch import nn
from omegaconf import DictConfig

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



def get_env_agents(cfg, *, autoreset=True, include_last_state=True):
    if "wrappers" in cfg.gym_env:
        print("using wrappers:", cfg.gym_env.wrappers)
        # wrappers_name_list = cfg.gym_env.wrappers.split(',')
        wrappers_list = []
        wr = get_class(cfg.gym_env.wrappers)
        # for i in range(len(wrappers_name_list)):
        wrappers_list.append(wr)
        wrappers = wrappers_list
        print(wrappers)
    else:
        wrappers = []

    train_env_agent = ParallelGymAgent(
        partial(
            make_env, cfg.gym_env.env_name, autoreset=autoreset, wrappers=wrappers
        ),
        cfg.algorithm.n_envs,
        include_last_state=include_last_state,
        seed=cfg.algorithm.seed.train,
    )

    # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, wrappers=wrappers),
        cfg.algorithm.nb_evals,
        include_last_state=include_last_state,
        seed=cfg.algorithm.seed.eval,
    )

    return train_env_agent, eval_env_agent

def create_td3_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    actor = ContinuousDeterministicActor(
        obs_size,
        cfg.algorithm.architecture.actor_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.act,
    )
    # target_actor = copy.deepcopy(actor)
    noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
    tr_agent = Agents(train_env_agent, actor, noise_agent)
    ev_agent = Agents(eval_env_agent, actor)

    critic_1 = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.q,
    )
    target_critic_1 = copy.deepcopy(critic_1).set_name("target-critic1")
    critic_2 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_2 = copy.deepcopy(critic_2).set_name("target-critic2")

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )

def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer

def run_tdmpc(cfg, logger, trial=None):
    # 1)  Build the  logger
    best_reward = float("-inf")
    delta_list = []
    mean = 0

    # 2) Create the environment agents
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the TD3 Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent_1 = TemporalAgent(critic_1)
    target_q_agent_1 = TemporalAgent(target_critic_1)
    q_agent_2 = TemporalAgent(critic_2)
    target_q_agent_2 = TemporalAgent(target_critic_2)
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    nb_steps = 0
    tmp_steps = 0

    

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
        logger = Logger(cfg_raw)
        run_tdmpc(cfg_raw, logger)

if __name__ == "__main__":
    main()