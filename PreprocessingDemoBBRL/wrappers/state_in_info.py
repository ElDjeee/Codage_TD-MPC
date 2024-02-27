#  Projet Logiciel DAC ― M1 ― Mathis Koroglu
#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#


import gymnasium
import gymnasium as gym


class StateInInfo(gymnasium.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            label: str = "state",
    ) -> None:
        super().__init__(env)
        self.label = label

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info[self.label] = observation
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        info[self.label] = observation
        return observation, info
