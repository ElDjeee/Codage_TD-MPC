#  Projet Logiciel DAC ― M1 ― Mathis Koroglu
#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import gymnasium
from gymnasium import register

cartpole_spec = gymnasium.spec("CartPole-v1")
register(
    id="CartPole-v2",
    entry_point="environments.cartpole:CartPoleEnv",
    max_episode_steps=cartpole_spec.max_episode_steps,
    reward_threshold=cartpole_spec.reward_threshold,
)
