#  Projet Logiciel DAC ― M1 ― Mathis Koroglu
#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
import os
import numpy as np
import math
from typing import Any

import gymnasium as gym
from gymnasium import register
import gymnasium.vector.utils as g_utils
from gymnasium import Space, logger, spaces
from gymnasium.core import ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation


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


class CartPoleEnv(gym.Env[np.ndarray, int | np.ndarray]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 80
        self.screen_height = 80
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 3.0
        polelen = scale * (7 * self.length)
        cartwidth = 10
        cartheight = 10.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 10  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

if __name__ == "__main__":
    cartpole_spec = gym.spec("CartPole-v1")
    register(
        id="CartPole-v2",
        entry_point= __name__ + ":CartPoleEnv",
        max_episode_steps=cartpole_spec.max_episode_steps,
        reward_threshold=cartpole_spec.reward_threshold,
    )

    env = gym.make("CartPole-v2", render_mode="rgb_array")
    env = PixelOnlyObservation(env)
    env = ResizeObservation(env, 25) # (not needed with custom CartPoleEnv-v2)

    env_gray = GrayScaleObservation(env=env)
    env_bin_bw = BinarizeObservation(env=env_gray, threshold=230, invert=True)
    env_bin_color = FrameStack(env=env_bin_bw, num_stack=3)

    obs_bin_color, _ = env_bin_color.reset()
    for i in range(4):
        obs_bin_color, _, _, _, _ = env_bin_color.step(0)

    obs = np.array(obs_bin_color)
    obs = np.moveaxis(obs, 0, -1)
    save_dir = "outputs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    name = save_dir + "preprocess_" + str(obs.shape)
    data = obs * 255
    plt.imshow(data)
    plt.imsave(name + ".png", data)
    plt.show()