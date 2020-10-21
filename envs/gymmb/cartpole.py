import gym
import numpy as np
import math
from gym.envs.classic_control.cartpole import CartPoleEnv

from envs.task import Task

_env = CartPoleEnv()
_X_THRESHOLD = _env.x_threshold
_THETA_THRESHOLD = _env.theta_threshold_radians
del _env


class CartPoleBalanceTask(Task):
    def __call__(self, states, actions, next_states):
        next_dones = GYMMB_ContinuousCartPole.is_done(next_states)
        dones = GYMMB_ContinuousCartPole.is_done(states)
        rewards = 1.0 - next_dones.float() * dones.float()  # basically, you get always 1.0 unless you exceed the is_done termination criteria
        return rewards


class CartPoleSpeedyBalanceTask(Task):
    def __call__(self, states, actions, next_states):
        rewards = states[:, 1].abs()  # more rewards for abs velocity. No free points
        return rewards


class GYMMB_ContinuousCartPole(CartPoleEnv):
    """
    A continuous version.
    Observation space:
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
    """

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1., high=+1., shape=(1,))

    def step(self, action):
        assert self.action_space.contains(action)
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # Update
        next_state = np.array([x, x_dot, theta, theta_dot])
        self.state = next_state
        return next_state, None, None, {}

    @staticmethod
    def tasks():
        return dict(
            standard=CartPoleBalanceTask(),
        #    speedy_balance=MagellanCartPoleSpeedyBalanceTask()
        )

    @staticmethod
    def is_done(states):
        cart_positions, _, pole_angles, _ = states.unbind(dim=1)

        # Check if cart positions and pole angles are within established bounds
        in_bounds = (-_X_THRESHOLD <= cart_positions) & (cart_positions <= _X_THRESHOLD) & \
                    (-_THETA_THRESHOLD <= pole_angles) & (pole_angles <= _THETA_THRESHOLD)
        dones = ~in_bounds
        return dones
