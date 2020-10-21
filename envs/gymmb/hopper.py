import numpy as np
from gym.envs.mujoco import HopperEnv
import torch

from envs.task import Task


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        delta_pos = next_states[:, 0]
        alive_bonus = 1.0
        reward = delta_pos + alive_bonus - 1e-3 * actions.pow(2).sum(dim=1)
        return reward


class GYMMB_Hopper(HopperEnv):
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
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev = self.get_additional_obs()
        return self._get_obs()

    def get_additional_obs(self):
        return np.array([self.sim.data.qpos[0]])

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):
        # A small difference: I check whether clipped states are finite instead of raw states
        # Also: states[0] is `delta qpos[0] / dt` instead of qpos[0]
        notdone = torch.all(torch.isfinite(states), dim=1) & (states[:, 2:].abs() < 100).all(dim=1) \
                  & (states[:, 1] > .7) & (states[:, 2].abs() < .2)
        return ~notdone
