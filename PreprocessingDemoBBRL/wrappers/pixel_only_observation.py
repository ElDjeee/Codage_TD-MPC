#  Projet Logiciel DAC ― M1 ― Mathis Koroglu
#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Any

import gymnasium as gym
import gymnasium.vector.utils as g_utils
from gymnasium import Space
from gymnasium.core import ObsType, WrapperObsType


class PixelOnlyObservation(gym.wrappers.PixelObservationWrapper):
    """Keep only observations of pixel values."""

    pixel_key: str = "pixels"
    "Add our custom key if the signature of the parent class changes."

    def __init__(
        self,
        env: gym.Env,
        render_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a new pixel only Wrapper."""

        super().__init__(
            env,
            pixels_only=True,
            render_kwargs={self.pixel_key: render_kwargs}
            if render_kwargs is not None
            else None,
            pixel_keys=(self.pixel_key,),
        )

        if not isinstance(self.observation_space, g_utils.spaces.Dict):
            raise TypeError(
                "Internal observation space must be a spaces.Dict."
                "A change probably happened in gymnasium PixelObservationWrapper behaviour."
            )

        self.observation_space: Space[ObsType] | Space[
            WrapperObsType
        ] = self.observation_space[self.pixel_key]


    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        obs = super().observation(observation)
        return obs[self.pixel_key]
