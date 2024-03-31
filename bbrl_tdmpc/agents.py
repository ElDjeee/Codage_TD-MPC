import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bbrl.agents.agent import Agent
from bbrl_algos.models.critics import NamedCritic
from bbrl_algos.models.actors import BaseActor

from utils import _get_out_shape
import preprocess
import logger

class NormalizeImg(Agent):  # DONE
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t)) # obs ou latent?
        return obs.div(255.)

class Flatten(Agent):  # DONE
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t)) # obs ou latent?
        return obs.view(obs.size(0), -1)

def enc(cfg):
    """Returns a TOLD encoder."""
    if cfg.modality == 'pixels':
        C = int(3*cfg.frame_stack)
        layers = [NormalizeImg(),
                  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels,
                            5, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels,
                            3, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(
            np.prod(out_shape), cfg.latent_dim)])
    else:
        layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
                  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
    return nn.Sequential(*layers)

class EncoderAgent(Agent):
    def __init__(self, cfg):
        super().__init__()
        # if cfg.modality == 'pixels': à faire dans le main
        # 	# récupérer l'env?
        # 	PixelOnlyObservation()
        # 	GrayScaleObservation()
        # 	BinarizeObservation()
        # 	FrameStack()
        self.net = enc(cfg)

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        latent = self.net(obs)
        self.set(("latent", t), latent)

    def _encoder(self, obs):
        return self.net(obs)

def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]), act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        nn.Linear(mlp_dim[1], out_dim))

class MLPAgent(Agent):
    def __init__(self, name, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
        super().__init__()
        self.net = mlp(in_dim, mlp_dim, out_dim, act_fn)
        self.name = name

    def forward(self, t, **kwargs):
        # obs = self.get(("env/env_obs", t))
        obs = self.get(("latent", t))
        res = self.net(obs)
        self.set((f"{self.name}", t), res)

    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)

class ActorAgent(BaseActor): # actor = policy
    """
    inspiré de https://github.com/osigaud/bbrl_algos/blob/095d849b6b77e068a6c38b3ce200982ffbbeecd4/src/bbrl_algos/models/actors.py#L56
    """
    def __init__(self, in_dim, mlp_dim, out_dim, act_fn=nn.ELU(), *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.net = mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU())

    def forward(self, t, **kwargs):
        # obs = self.get(("env/env_obs", t))
        obs = self.get(("latent", t))
        action = self.net(obs)
        self.set(("action", t), action)

    def predict_action(self, obs, stochastic=False):
        """Predict just one action (without using the workspace)"""
        assert (
            not stochastic
        ), "ContinuousDeterministicActor cannot provide stochastic predictions"
        return self.net(obs)

def q(cfg, act_fn=nn.ELU()):  # act_fn non utilisé?
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
                         nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
                         nn.Linear(cfg.mlp_dim, 1))

class CriticAgent(NamedCritic): # critic = Q-function
    """
    inspiré de https://github.com/osigaud/bbrl_algos/blob/095d849b6b77e068a6c38b3ce200982ffbbeecd4/src/bbrl_algos/models/critics.py#L24
    """
    def __init__(self, cfg, act_fn=nn.ELU(), name="critic", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.net = q(cfg, act_fn)
        self.is_q_function = True

    def forward(self, t, detach_actions=False, **kwargs):
        # obs = self.get(("env/env_obs", t))
        obs = self.get(("latent", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        obs_act = torch.cat((obs, action), dim=1)
        q_value = self.net(obs_act)
        self.set((f"{self.name}/q_values", t), q_value)

    def predict_value(self, obs, action): # fait le travail de la méthode Q de TOLD
        # obs_act = torch.cat((obs, action), dim=0)
        obs_act = torch.cat([obs, action], dim=-1)
        q_value = self.net(obs_act)
        return q_value

class RandomShiftsAug(Agent): # DONE
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, cfg):
        super().__init__()
        self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

    def forward(self, t, **kwargs):
        """
        méthode de tdmpc, changements : première et dernière ligne
        """
        x = self.get(("env/env_obs", t)) # x = obs, obs or next_obs?
        if not self.pad:
            return x
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h +
                                2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        next_obs = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        self.set(("next_obs", t), next_obs)

    def predict(self, x):
        if not self.pad:
            return x
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h +
                                2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class LoggerAgent(Agent):
    def __init__(self, work_dir, cfg):
        super().__init__()
        self.logger = logger.Logger(work_dir, cfg)
        # Ajout pour suivre les métriques au fil du temps
        self.metrics_history = {}

    def forward(self, workspace, t, **kwargs):
        # Récupération et enregistrement des métriques
        if 'metrics' in kwargs and 'category' in kwargs:
            metrics = kwargs['metrics']
            category = kwargs['category']
            self.logger.log(metrics, category)
            # Stockage des métriques pour une utilisation future
            for key, value in metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
            # Mise à jour de workspace avec des informations potentiellement utiles pour d'autres agents
            workspace.set(f"{category}_metrics", t, metrics)

    def get_metrics_history(self, metric_name):
        return self.metrics_history.get(metric_name, [])

    def isVideoEnable(self):
        return self.logger.video is not None
