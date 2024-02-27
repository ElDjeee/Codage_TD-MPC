#  Projet Logiciel DAC ― M1 ― Mathis Koroglu
#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

"""A wrapper to binarize observations."""
from gymnasium.core import ObsType

import gymnasium as gym
from gymnasium.spaces import Box


class BinarizeObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Binarize the image observation."""

    def __init__(self, env: gym.Env, threshold: int, invert: bool = False) -> None:
        """Binarize image observations given the threshold`.
        In the case of color images, each color (channel) is processed separately.
        If you want black and white images, convert them to grayscale first.

        Args:
            env: The environment to apply the wrapper
            threshold: The threshold pixel value to binarize the image
            invert: Invert the binarized image.
        """
        import cv2

        gym.ObservationWrapper.__init__(self, env)
        gym.utils.RecordConstructorArgs.__init__(self, threshold=threshold, invert=invert)

        self.invert = invert
        self._thresh_type = (
            cv2.THRESH_BINARY if not self.invert else cv2.THRESH_BINARY_INV
        )

        if not isinstance(self.observation_space, Box):
            raise ValueError(
                "Observation space must be a Box, got {}".format(self.observation_space)
            )

        if len(self.observation_space.shape) > 3:
            raise TypeError(
                "Observation space is not 3 dimensional, got {}".format(
                    self.observation_space
                )
            )

        if self.invert:
            self.threshold = int(threshold)
        else:
            self.threshold = int(self.observation_space.high.max() - threshold)

        obs_shape = self.observation_space.shape
        self.observation_space = Box(shape=obs_shape, high=1, low=0)

    def observation(self, observation: ObsType) -> ObsType:
        """Updates the observations by binarizing the observation`.

        Args:
            observation: The observation to binarize

        Returns:
            The binarized observations between 0 and 1
        """
        import cv2

        th, observation = cv2.threshold(
            observation, thresh=self.threshold, maxval=1, type=self._thresh_type
        )
        return observation
