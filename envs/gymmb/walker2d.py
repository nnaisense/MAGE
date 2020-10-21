import numpy as np
from gym.envs.mujoco import Walker2dEnv
import torch

from envs.task import Task


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        delta_pos = next_states[:, 0]
        alive_bonus = 1.0
        reward = delta_pos + alive_bonus - 1e-3 * actions.pow(2).sum(dim=1)
        return reward


class GYMMB_Walker2d(Walker2dEnv):
    def __init__(self):
        self.prev = None
        super().__init__()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        if self.prev is not None:  # Defensive coding. First reset_model needs to be called.
            self.prev = self.get_additional_obs()
        return ob, None, None, {}

    def _get_obs(self):
        # _get_obs is called also in init to setup the observation space dim. We can ignore that
        curr = self.get_additional_obs()
        delta = curr - self.prev if self.prev is not None else np.zeros_like(curr)

        return np.concatenate([
            delta / self.dt,
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.prev = self.get_additional_obs()
        return self._get_obs()

    def get_additional_obs(self):
        return np.array([self.sim.data.qpos[0]])

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):
        notdone = (states[:, 1] > 0.8) & (states[:, 1] < 2.) & (states[:, 2] > -1) & (states[:, 2] < 1)
        return ~notdone
