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

__CONFIG__, __LOGS__ = 'cfgs', 'logs'

def parse_cfg(cfg_path: str) -> OmegaConf:
	"""Parses a config file and returns an OmegaConf object."""
	base = OmegaConf.load(cfg_path / 'default.yaml')
	cli = OmegaConf.from_cli()
	for k,v in cli.items():
		if v == None:
			cli[k] = True
	base.merge_with(cli)

	# Modality config
	if cli.get('modality', base.modality) not in {'state', 'pixels'}:
		raise ValueError('Invalid modality: {}'.format(cli.get('modality', base.modality)))
	modality = cli.get('modality', base.modality)
	if modality != 'state':
		mode = OmegaConf.load(cfg_path / f'{modality}.yaml')
		base.merge_with(mode, cli)

	# Task config
	try:
		domain, task = base.task.split('-', 1)
	except:
		raise ValueError(f'Invalid task name: {base.task}')
	domain_path = cfg_path / 'tasks' / f'{domain}.yaml'
	if not os.path.exists(domain_path):
		domain_path = cfg_path / 'tasks' / 'default.yaml'
	domain_cfg = OmegaConf.load(domain_path)
	base.merge_with(domain_cfg, cli)

	# Algebraic expressions
	for k,v in base.items():
		if isinstance(v, str):
			match = re.match(r'(\d+)([+\-*/])(\d+)', v)
			if match:
				base[k] = eval(match.group(1) + match.group(2) + match.group(3))
				if isinstance(base[k], float) and base[k].is_integer():
					base[k] = int(base[k])

	# Convenience
	base.task_title = base.task.replace('-', ' ').title()
	base.device = 'cuda' if base.modality == 'state' else 'cpu'
	base.exp_name = str(base.get('exp_name', 'default'))

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