import algorithm.helper as h

from copy import deepcopy
import torch
import numpy as np

from bbrl.agents.agent import Agent

class RepresentationAgent(Agent):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._encoder = h.enc(cfg)

    def __call__(self, x, **kwargs):
        return self._encoder(x)

    def forward(self, t, **kwargs):
        pass



class LatentDynamicAgent(Agent):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)

    def __call__(self, x, **kwargs):
        return self._dynamics(x)

    def forward(self, t, **kwargs):
        pass



class RewardAgent(Agent):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)

    def __call__(self, x, **kwargs):
        return self._reward(x)

    def forward(self, t, **kwargs):
        pass



class ValueAgent(Agent):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)

    def __call__(self, x, **kwargs):
        return self._Q1(x), self._Q2(x)

    def forward(self, t, **kwargs):
        pass



class PolicyAgent(Agent):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)

    def __call__(self, x, **kwargs):
        return self._pi(x)

    def forward(self, t, **kwargs):
        pass


# ----------------------------------------------------------------------

class PlanningAgent(Agent):
    def __init__(self, cfg, model, device):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def plan(self, obs, self_std, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = h.estimate_value(self.model, self.cfg, z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self_std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a

    def forward(self, t, step, **kwargs):
        obs = self.get(("env/env_obs", t)).to(self.device)

        assert 'std' in kwargs, "std must be passed as a keyword argument"
        std = kwargs['std']

        assert 'first_episode' in kwargs, "first_episode must be passed as a keyword argument"
        first_episode = kwargs['std']

        t0 = (first_episode and t==0) or t == 1

        action = self.plan(obs, std, step=step, t0=t0)
        action = action.view(1, action.shape[0])

        self.set(("action", t), action)
