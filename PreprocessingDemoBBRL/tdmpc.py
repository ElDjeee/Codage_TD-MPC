import hydra
from omegaconf import DictConfig
import torch
from torch import distributions as pyd
import bbrl
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.agents import Agent
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
import helper as h
from copy import deepcopy
from told2 import TOLD_Agent

# for test (TMP)
from demo import EnvAgent, ActionAgent

class TDMPC_Agent(Agent):
    def __init__(self, agent_cfg):
        super().__init__()
        self.device = torch.device(agent_cfg.device)
        self.model = TOLD_Agent(cfg=agent_cfg.model_cfg, device=self.device)
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=agent_cfg.lr)
        self.aug = h.RandomShiftsAug(agent_cfg.model_cfg)
        


    def update(self):
        batch = self.replay_buffer.sample(self.agent_cfg["batch_size"])
        obs, actions, rewards, next_obs, dones = batch
        
        # Conversion des tenseurs si nécessaire
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        # Calcul des cibles pour la récompense et la valeur
        with torch.no_grad():
            next_latent = self.model.h(next_obs)
            target_value = rewards + (1 - dones) * self.agent_cfg["gamma"] * self.model_target.Q(next_latent, self.model_target.pi(next_latent))
        
        # Mise à jour du réseau principal
        latent = self.model.h(obs)
        predicted_value = self.model.Q(latent, actions)
        value_loss = h.mse(predicted_value, target_value)

        # Optimisation
        self.optim.zero_grad()
        value_loss.backward()
        self.optim.step()

        # Mise à jour du réseau cible
        h.ema(self.model, self.model_target, self.agent_cfg["tau"])
    
    
    def estimate_value(self, z, actions, horizon):
        G = torch.zeros(z.size(0), device=self.device)
        discount = 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.agent_cfg["gamma"]
        return G

    def plan(self, obs, eval_mode=False, step=0, t0=True):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.agent_cfg["model_cfg"]["horizon"], step))  # Assurez-vous d'avoir défini horizon dans votre config
        num_samples = self.agent_cfg["model_cfg"]["num_samples"]  # Assurez-vous d'avoir défini num_samples dans votre config

        # Initialisation des états latents
        z = self.model.h(obs).repeat(num_samples, 1)
        mean = torch.zeros(horizon, self.agent_cfg["model_cfg"]["action_dim"], device=self.device)
        std = torch.ones(horizon, self.agent_cfg["model_cfg"]["action_dim"], device=self.device) * 2

        # Boucle de l'optimisation par l'algorithme Cross-Entropy Method (CEM)
        for i in range(self.agent_cfg["cem_iterations"]):  # Assurez-vous d'avoir défini cem_iterations dans votre config
            actions = pyd.Normal(mean, std).sample([num_samples])
            actions = actions.clamp(-1, 1)  # Assurez-vous que cette limite correspond à votre espace d'actions

            values = self.estimate_value(z, actions, horizon)
            top_values, top_indices = values.topk(int(num_samples * self.agent_cfg["elite_frac"]))  # Assurez-vous d'avoir défini elite_frac dans votre config
            elite_actions = actions[top_indices]

            mean, std = elite_actions.mean(0), elite_actions.std(0)
            mean = mean.clamp(-1, 1)  # Assurez-vous que cette limite correspond à votre espace d'actions
            std = std.clamp(min=0.1)  # Évitez que std devienne trop petit

        # Sélectionnez la première action de la séquence d'actions moyenne
        selected_action = mean[0]
        return selected_action.cpu().numpy()






@hydra.main(config_path="config", config_name="default")
def main(cfg: DictConfig):
    # Initialisation de l'agent TD-MPC
    tdmpc_agent = TDMPC_Agent(cfg)

    # Création de l'environnement et de l'agent d'action
    env_agent = EnvAgent(img_size=84, num_stack=3, threshold=230, invert=True)
    action_agent = ActionAgent(env_agent.env.action_space)

    # Composition des agents
    composed_agent = Agents(env_agent, tdmpc_agent, action_agent)
    temporal_agent = TemporalAgent(composed_agent)

    # Exécution dans l'espace de travail
    workspace = Workspace()
    n_steps = 100  # Définissez le nombre de pas de temps par épisode
    temporal_agent(workspace, t=0, n_steps=n_steps)

    # Récupération et affichage des résultats
    obs, actions, rewards, done = workspace["obs", "action", "reward", "done"]
    print("Récompenses recueillies:", rewards)

if __name__ == "__main__":
    main()
