

# import torch
# import torch.nn as nn
# import numpy as np

# # Mocking necessary components from the provided code
# class Agent:
#     def __init__(self):
#         pass

#     def get(self, key):
#         # Dummy implementation
#         pass

#     def set(self, key, value):
#         # Dummy implementation
#         pass

# class NormalizeImg(nn.Module):
#     """Normalizes pixel observations to [0,1) range."""
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x.div(255.)

# class Flatten(nn.Module):
#     """Flattens its input to a (batched) vector."""
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x.view(x.size(0), -1)

# def _get_out_shape(in_shape, layers):
#     """Utility function. Returns the output shape of a network for a given input shape."""
#     x = torch.randn(*in_shape).unsqueeze(0)
#     return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape

# def enc(cfg):
#     """Returns a TOLD encoder."""
#     if cfg['modality'] == 'pixels':
#         C = int(3*cfg['frame_stack'])
#         layers = [NormalizeImg(),
#                   nn.Conv2d(C, cfg['num_channels'], 7, stride=2), nn.ReLU(),
#                   nn.Conv2d(cfg['num_channels'], cfg['num_channels'], 5, stride=2), nn.ReLU(),
#                   nn.Conv2d(cfg['num_channels'], cfg['num_channels'], 3, stride=2), nn.ReLU(),
#                   nn.Conv2d(cfg['num_channels'], cfg['num_channels'], 3, stride=2), nn.ReLU()]
#         out_shape = _get_out_shape((C, cfg['img_size'], cfg['img_size']), layers)
#         layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg['latent_dim'])])
#     else:
#         layers = [nn.Linear(cfg['obs_shape'][0], cfg['enc_dim']), nn.ELU(),
#                   nn.Linear(cfg['enc_dim'], cfg['latent_dim'])]
#     return nn.Sequential(*layers)

# class EncoderAgent(Agent):
#     def __init__(self, cfg):
#         super().__init__()
#         self.fc = enc(cfg)

#     def forward(self, t, **kwargs): 
#         obs = kwargs['obs']
#         latent = self.fc(obs)
#         return latent

# # Testing the EncoderAgent
# cfg = {
#     'modality': 'pixels',  # Testing with pixel modality
#     'frame_stack': 4,
#     'num_channels': 32,
#     'img_size': 64,
#     'latent_dim': 128
# }
# encoder_agent = EncoderAgent(cfg)

# # Creating a dummy observation to test the encoder
# dummy_obs = torch.rand(1, 3*cfg['frame_stack'], cfg['img_size'], cfg['img_size'])  # Shape: (batch, channels, height, width)

# # Forward pass through the encoder
# latent_representation = encoder_agent.forward(0, obs=dummy_obs)
# latent_representation.shape  # Expecting the output shape to match the latent dimension defined in cfg

# print("Output shape of the encoder:", latent_representation.shape)



# Since we cannot directly run the TOLD class due to dependencies on BBRL, Gymnasium, and other components,
# we will create a minimal runnable example to simulate its functionality.



####################################################


# import torch
# from torch import nn

# # Adapting the provided TOLD class structure for a minimal runnable example

# class EncoderAgent(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, latent_dim)

#     def forward(self, x):
#         return torch.relu(self.fc(x))

# class MLPAgent(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.fc(x)

# class TOLD(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim, action_dim):
#         super().__init__()
#         self.encoder = EncoderAgent(input_dim, latent_dim)
#         self.dynamics = MLPAgent(latent_dim + action_dim, hidden_dim, latent_dim)
#         self.reward_model = MLPAgent(latent_dim + action_dim, hidden_dim, 1)
#         self.policy_model = MLPAgent(latent_dim, hidden_dim, action_dim)

#     def h(self, obs):
#         return self.encoder(obs)

#     def next(self, z, a):
#         x = torch.cat([z, a], dim=-1)
#         return self.dynamics(x), self.reward_model(x)

#     def pi(self, z):
#         return torch.tanh(self.policy_model(z))

# # Configuration for testing
# input_dim = 100
# latent_dim = 64
# hidden_dim = 128
# action_dim = 10

# # Creating a TOLD instance for testing
# told = TOLD(input_dim, latent_dim, hidden_dim, action_dim)

# # Testing with dummy data
# obs = torch.randn(5, input_dim)  # Batch of 5 observations
# z = told.h(obs)
# a = torch.randn(5, action_dim)  # Batch of 5 actions

# next_z, reward = told.next(z, a)
# predicted_action = told.pi(z)

# print("Shapes of outputs:")
# print("Next latent state (z):", next_z.shape)
# print("Predicted reward:", reward.shape)
# print("Predicted action:", predicted_action.shape)


from TOLD import TOLD
import torch

class Config:
    def __init__(self):
        self.latent_dim = 64
        self.action_dim = 10
        self.mlp_dim = 128
        self.num_channels = 32
        self.img_size = 64
        self.frame_stack = 4
        self.modality = 'pixels'  # Assuming pixel-based observation for this test

cfg = Config()


def make_env(cfg):
    # Mock function to simulate environment creation
    return None

def get_env_agents(cfg):
    # Mock function to simulate getting environment agents
    return None, None



def test_told(cfg):
    told = TOLD(cfg)
    dummy_obs = torch.rand(1, 3*cfg.frame_stack, cfg.img_size, cfg.img_size)  # Dummy observation

    # Testing core functionalities
    z = told.h(dummy_obs)
    print(f"Latent representation shape: {z.shape}")

    dummy_action = torch.rand(1, cfg.action_dim)  # Dummy action
    next_z, reward = told.next(z, dummy_action)
    print(f"Next latent state shape: {next_z.shape}, Reward shape: {reward.shape}")

    pi_action = told.pi(z)
    print(f"Sampled action shape: {pi_action.shape}")

    q1, q2 = told.Q(z, dummy_action)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")

# Run the test
test_told(cfg)
