import copy
import torch
from bbrl.agents import Agents, TemporalAgent
from bbrl.workspace import Workspace

from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl_algos.models.envs import get_env_agents

from agents_tdmpc import *
from utils_tdmpc import *

# ------------------------------------------------------------------------
# We already made Train(Agent) and Evaluate(Agent) classes in train.py file ( without gymnasium)
# making train and evaluate agents ( using gymnasium). Inspired from bbrl examples  :


def create_told_agent(cfg, train_env_agent, eval_env_agent):  # orthogonal_init?
    """
    inspiré de create_td3_agent de bbrl
    ajout de noise? https://github.com/osigaud/bbrl_algos/blob/095d849b6b77e068a6c38b3ce200982ffbbeecd4/src/bbrl_algos/algos/td3/td3.py#L48
    """

    encoder = EncoderAgent(cfg)
    dynamics_model = MLPAgent(
        name="dynamics",
        in_dim=cfg.latent_dim + cfg.action_dim, 
        mlp_dim=cfg.mlp_dim, 
        out_dim=cfg.latent_dim
    )
    reward_model = MLPAgent(
        name="reward",
        in_dim=cfg.latent_dim + cfg.action_dim, 
        mlp_dim=cfg.mlp_dim, 
        out_dim=1
    )
    actor = ActorAgent(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)  # policy
    critic_1, critic_2 = CriticAgent(cfg, name="critic1"), CriticAgent(cfg, name="critic2")  # Q-functions
    told_agent = Agents(encoder, dynamics_model, reward_model, actor, critic_1, critic_2) # agent TOLD
    
    # -------- changement --------
    target_told_agent = copy.deepcopy(told_agent) # est-ce qu'on en a besoin?
    target_told_agent[4] = target_told_agent[4].set_name("target-critic1")
    target_told_agent[5] = target_told_agent[5].set_name("target-critic2")
    # -------- changement --------
    
    RandomShiftsAug_agent = RandomShiftsAug(cfg)

    tr_agent = Agents(train_env_agent, told_agent) #, target_told_agent) # doit-on ajouter le target agent?
    ev_agent = Agents(eval_env_agent, told_agent)

    # agents that are executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    
    #train_agent.seed(cfg.algorithm.seed)
    return (
        train_agent,
        eval_agent,
        told_agent,
        target_told_agent,
        RandomShiftsAug_agent
    )

def optimizers(cfg, told_agent, optim_agent, consistency_loss, reward_loss, value_loss): 
    """
    inspiré de 
    https://github.com/nicklashansen/tdmpc/blob/f4d85eca7419039b71bab2234ffc8aca378dd313/src/algorithm/tdmpc.py#L209
    """
    total_loss = cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
        cfg.reward_coef * reward_loss.clamp(max=1e4) + \
        cfg.value_coef * value_loss.clamp(max=1e4)
    weighted_loss = total_loss.squeeze(1).mean()
    weighted_loss.register_hook(lambda grad: grad * (1/cfg.horizon))
    weighted_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        told_agent.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
    optim_agent.step()
    return total_loss, weighted_loss, grad_norm

def compute_actor_loss(cfg, pi_optim_agent, told_agent, zs):  # DONE
    """
    https://github.com/nicklashansen/tdmpc/blob/f4d85eca7419039b71bab2234ffc8aca378dd313/src/algorithm/tdmpc.py#L153
    """
    pi = told_agent[3]
    critic1 = told_agent[4]
    critic2 = told_agent[5]
    
    pi_optim_agent.zero_grad(set_to_none=True)
    set_requires_grad(critic1, False)
    set_requires_grad(critic2, False)
    
    pi_loss = 0
    for t, z in enumerate(zs):
        mu = torch.tanh(pi.predict_action(z))
        std = cfg.min_std
        if std > 0:
            std = torch.ones_like(mu) * std
            mu = TruncatedNormal(mu, std).sample(clip=0.3)
        a = mu
        Q = torch.min(critic1.predict_value(
            z, a), critic2.predict_value(z, a))
        pi_loss += -Q.mean() * (cfg.rho ** t)
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        pi.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
    pi_optim_agent.step() 
    set_requires_grad(critic1, True)
    set_requires_grad(critic2, True)
    return pi_loss.item()


def run_tdmpc(cfg, logger, trial=None):
    # 1) Do we need to create the logger ?
    best_reward = float("-inf")

    # 2) Create the environment agents (pourquoi get_env_agents?)
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 2) Create the TOLD agents
    (
        train_agent,
        eval_agent,
        told_agent,
        target_told_agent,
        RandomShiftsAug_agent
    ) = create_told_agent(cfg, train_env_agent, eval_env_agent)
    ag_told = TemporalAgent(told_agent)
    ag_target = TemporalAgent(target_told_agent)
    ag_rsa = TemporalAgent(RandomShiftsAug_agent)

    # 3) Create the training workspace
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # setup the optimizers
    # possible?, inspiré de https://github.com/nicklashansen/tdmpc/blob/f4d85eca7419039b71bab2234ffc8aca378dd313/src/algorithm/tdmpc.py#L59
    optim_agent = torch.optim.Adam(told_agent.parameters(), lr=cfg.lr)
    pi_optim_agent = torch.optim.Adam(told_agent[3].parameters(), lr=cfg.lr)
    # total_loss, weighted_loss, grad_norm = optimizers(cfg, told_agent, optim_agent, 0, 0, 0) # weights?

    # 6) Define the steps counters and the train loop
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        # follow the steps of the algorithm
        print(f"Step {step}")
        
        # Execute the agent in the workspace
        if step > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions() 
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        if step > 0 or cfg.algorithm.n_steps > 1:
            rb.put(transition_workspace)

        for _ in range(cfg.algorithm.n_updates):
            # num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            if nb_steps > cfg.algorithm.learning_starts:
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)
                # Collect episode from TDMPC
                done, truncated, reward = rb_workspace["env/done", "env/truncated", "env/reward"] # avec env/ ou sans?
                optim_agent.zero_grad(set_to_none=True)
                std = linear_schedule(cfg.std_schedule, step)
                must_bootstrap = torch.logical_or(~done[1], truncated[1])
                # TOLD Update
                # here, agent collects the current state-action pairs and their associated Q-values
                zs = []
                consistency_loss, reward_loss, value_loss = 0, 0, 0 #priority_loss = 0, 0, 0, 0
                for t in range(cfg.horizon):
                    ag_rsa(rb_workspace, t=t, n_steps=1)
                    ag_told(rb_workspace, t=t, n_steps=1)  # n_step?                  
                    z, reward_pred, q_values_rb_1, q_values_rb_2 = rb_workspace["latent", "reward", "critic1/q_values", "critic2/q_values"]
                    zs.append(z.detach())
                    with torch.no_grad():
                        ag_rsa(rb_workspace, t=t+1, n_steps=1)
                        ag_target(rb_workspace, t=t+1, n_steps=1)
                        next_z, reward, q_target_rb_1, q_target_rb_2 = rb_workspace["latent", "reward", "target-critic1/q_values", "target-critic2/q_values"]
                        td_target = reward + cfg.discount * \
                                torch.min(q_target_rb_1, q_target_rb_2)
                    zs.append(z.detach())
                    
                    # Compute TOLD Loss
                    rho = (cfg.rho ** t)
                    consistency_loss += rho * torch.mean(mse(z, next_z), dim=1, keepdim=True)
                    reward_loss += rho * mse(reward_pred, reward)
                    value_loss += rho * (mse(q_values_rb_1, td_target) + mse(q_values_rb_2, td_target))
                    # priority_loss += rho * (l1(q_values_rb_1, td_target) + l1(q_values_rb_2, td_target)) ???????????????
                    # Store the loss

                # optimize model
                total_loss, weighted_loss, grad_norm = optimizers(cfg, told_agent, optim_agent, consistency_loss, reward_loss, value_loss, weights=0.3)

                # Update policy + target network
                pi_loss = compute_actor_loss(cfg, pi_optim_agent, told_agent, zs)
                if step % cfg.update_freq == 0:
                    ema(ag_told, ag_target, cfg.tau) # does it work like this?



        # Evaluate the agent
        if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
            tmp_steps_eval = nb_steps
            eval_workspace = Workspace()
            eval_agent(eval_workspace, t=0, stop_variable="env/done")
            
            rewards = eval_workspace["env/cumulated_reward"][-1]
            ag_told(eval_workspace, t=0, stop_variable="env/done")
            q_values_1 = eval_workspace["critic/q_values"].squeeze()  #.squeeze()? this function is used when we want to remove single-dimensional entries from the shape of an array. 
            delta = q_values_1 - rewards
            maxi_delta = delta.max(axis=0)[0].detach().numpy()
            delta_list.append(maxi_delta)
            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)
            if mean > best_reward:
                best_reward = mean
            print(f"nb_steps: {nb_steps}, reward: {mean:.2f}, best: {best_reward:.2f}")

            # logger ??????
            # logger.log_reward_losses(rewards, nb_steps)


            if mean > best_reward:
                best_reward = mean
            print(f"nb_steps: {nb_steps}, reward , best")

            # Is the trial done
            if trial is not None:
                print("trial is not None")
                # reste du code
            
            if cfg.save_best and best_reward == mean:   
                print("best reward == mean")
                # reste du code
            
    
            # Save/log the best rewards
    
    return best_reward


