import copy
import torch
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.agent import Agent
from bbrl.workspace import Workspace

from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl_algos.models.envs import get_env_agents

from agents import *
from utils import *

class TOLD(Agent):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = EncoderAgent(cfg)
        self._dynamics = MLPAgent(
            "dynamics", cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = MLPAgent("reward", cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = ActorAgent(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = CriticAgent(cfg), CriticAgent(cfg)
        self.apply(orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)

    def forward(self, t, **kwargs): pass

#------------------------------------------------------------------------
# We already made Train(Agent) and Evaluate(Agent) classes in train.py file ( without gymnasium) 
# making train and evaluate agents ( using gymnasium). Inspired from bbrl examples  :

def create_TOLD_agent(cfg, train_env_agent, eval_env_agent):
    # inspiré de create_td3_agent de bbrl
    # The t agent executing on the rb_workspace workspace
    # t_agent(workspace, t=0)

    # TOLD 
    encoder = EncoderAgent(cfg)
    dynamics_model = MLPAgent(
        cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
    reward_model = MLPAgent(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
    # policy
    actor = ActorAgent(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim, "actor")
    # Q-function
    critic_1, critic_2= CriticAgent(cfg), CriticAgent(cfg)
    TOLD_agent = Agents(encoder, dynamics_model, reward_model,
                        actor, critic_1, critic_2)
    
    # ajout de noise?
    # Question
    # https://github.com/osigaud/bbrl_algos/blob/095d849b6b77e068a6c38b3ce200982ffbbeecd4/src/bbrl_algos/algos/td3/td3.py#L48
    t_agent = TemporalAgent(TOLD_agent)
    tr_agent = Agents(train_env_agent, t_agent)  # , PrintAgent())
    ev_agent = Agents(eval_env_agent, t_agent)

    target_critic_1 = copy.deepcopy(critic_1).set_name("target-critic1")
    target_critic_2 = copy.deepcopy(critic_2).set_name("target-critic2")
    
    # agents that are executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    
    return (train_agent, eval_agent, t_agent, actor, critic_1, critic_2, target_critic_1, target_critic_2)

def compute_TOLD_loss(cfg, current_horizon, z, next_z, reward_pred, current_reward, critic_1, critic_2):
    """
    n'est pas fini
    https://github.com/nicklashansen/tdmpc/blob/f4d85eca7419039b71bab2234ffc8aca378dd313/src/algorithm/tdmpc.py#L203
    """
    # with torch.no_grad(): # à faire
    #     next_obs = aug(next_obses[t])
    #     next_z = model_target.h(next_obs)
    #     td_target = _td_target(next_obs, current_reward)
    # Losses
    rho = (cfg.rho ** current_horizon)
    consistency_loss = rho * torch.mean(mse(z, next_z), dim=1, keepdim=True)
    reward_loss = rho * mse(reward_pred, current_reward)
    # value_loss = rho * (mse(critic_1, td_target) + mse(critic_2, td_target))
    # priority_loss = rho * (l1(critic_1, td_target) + l1(critic_2, td_target))
    return consistency_loss, reward_loss, # value_loss, priority_loss

def compute_actor_loss(): pass

def run_tdmpc(cfg, logger, trial=None):
    
    # 1) Do we need to create the logger ? 
    best_reward = float("-inf")
    
    # 2) Create the environment agents
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 2) Create the TOLD agents
    (train_agent, eval_agent, told_agent ) = create_TOLD_agent(cfg, train_env_agent, eval_env_agent)
    ag_told = TemporalAgent(told_agent) 

    # 3) Create the training workspace
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # The t_agent executing on  workspace.
    # workspace=Workspace()
    # t_agent(workspace, t=0)

    
    # 6) Define the steps counters and the train loop 
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)

            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

        
        transition_workspace = train_workspace.get_transitions(filter_key="env/done")
        terminated, reward, action = transition_workspace["env/terminated","env/reward","action"]
        
        # nb_steps += action[0].shape[0]
        
        if nb_steps > 0 :
            rb.put(transition_workspace)
       
        for _ in range(cfg.algorithm.optim_n_updates):
            if nb_steps > cfg.algorithm.learning_starts:
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

                # Collect episode from TDMPC   
                # TOLD Update   
                # Compute TOLD Loss
                # Store the loss
        

        # Evaluate the agent
        if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
            tmp_steps_eval = nb_steps
            eval_workspace = Workspace()  
            eval_agent(eval_workspace,t=0)
            
            rewards = eval_workspace["env/cumulated_reward"][-1]
            # logger.log_reward_losses(rewards, nb_steps)
            mean = rewards.mean()
            
            if mean > best_reward:
                best_reward = mean
            print(f"nb_steps: {nb_steps}, reward , best")

            # Is the trial done
            # Save/log the best rewards
    return best_reward
