import torch
import numpy as np
from gym import Env
from gym.envs.classic_control import PendulumEnv

from envs.task import Task


class StandardTask(Task):
    @staticmethod
    def normalize_angle(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def __call__(self, states, actions, next_states):
        max_torque = 2.

        torque = torch.clamp(actions, min=-max_torque, max=max_torque)[0]

        costheta, sintheta, thetadot = states[:, 0], states[:, 1], states[:, 2]
        theta = self.normalize_angle(torch.atan2(sintheta, costheta))
        cost = theta.pow(2) + .1 * thetadot.pow(2) + .001 * torque.pow(2)
        return -cost


class GYMMB_Pendulum(Env):
    metadata = PendulumEnv.metadata

    def __init__(self):
        self.env = PendulumEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        ob, _, _, info = self.env.step(action)
        return ob, None, None, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    @staticmethod
    def tasks():
        return dict(standard=StandardTask(), poplin=StandardTask())

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
