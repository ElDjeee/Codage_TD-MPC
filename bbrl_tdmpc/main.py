import copy
import torch

import os
import re
import hydra
from omegaconf import DictConfig
from pathlib import Path
from logger import *

from bbrl_algos.models.envs import get_env_agents

from agents_tdmpc import *
from utils_tdmpc import *
from told import *

# ---------------------------------

@hydra.main(version_base='1.1', config_path="../cfgs/configs/", config_name="default_cartpole.yaml")


def main(cfg: DictConfig):
   
    best_reward = run_tdmpc(cfg, logger=None)
    print(f"Best reward: {best_reward}")


    # # 2) Create the environment agents (pourquoi get_env_agents?)
    # train_env_agent, eval_env_agent = get_env_agents(cfg)

    # # 2) Create the TOLD agents
    # (
    #     train_agent,
    #     eval_agent,
    #     told_agent,
    #     target_told_agent,
    #     RandomShiftsAug_agent
    # ) = create_told_agent(cfg, train_env_agent, eval_env_agent)
    
    # print("TOLD agent: ", told_agent)
    # print("Target TOLD agent: ", target_told_agent)
    # print("Random Shifts Augmentation agent: ", RandomShiftsAug_agent)
    # print("Train agent: ", train_agent)
    # print("Eval agent: ", eval_agent)

if __name__ == "__main__":
    main()