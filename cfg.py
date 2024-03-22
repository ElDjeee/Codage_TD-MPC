import os
import re
import hydra
from omegaconf import DictConfig
from pathlib import Path

@hydra.main(config_path="configs/", config_name="default_cartpole.yaml")
def main(cfg: DictConfig):
    
    told_model = TOLD(cfg)

    num_iterations = 10

    for i in range(num_iterations):
        print(f"Iteration: {i+1}")

        # Create a dummy observation
        dummy_observation = torch.rand(1, cfg.frame_stack * 3, cfg.img_size, cfg.img_size)

        # Pass the observation through the encoder to get the latent representation
        latent_representation = told_model.h(dummy_observation)

        # Sample an action from the policy
        sampled_action = told_model.pi(latent_representation, std=0.1)

        # Use the latent representation and action to predict the next latent state and reward
        next_latent, reward = told_model.next(latent_representation, sampled_action)

        # Get the Q-values for the current state and action
        q1, q2 = told_model.Q(latent_representation, sampled_action)

        # Print the outputs for inspection
        print("Latent Representation:", latent_representation.shape)
        print("Next Latent State:", next_latent.shape)
        print("Reward:", reward.shape)
        print("Sampled Action:", sampled_action.shape)
        print("Q1 Value:", q1.shape)
        print("Q2 Value:", q2.shape)
        print("\n")


if __name__ == "__main__":
    main()
