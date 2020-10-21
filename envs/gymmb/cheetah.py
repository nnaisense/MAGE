import torch
from gym.envs.mujoco import HalfCheetahEnv
import numpy as np

from envs.task import Task


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        reward_ctrl = -0.1 * actions.pow(2).sum(dim=1)
        reward_run = next_states[:, 0]
        return reward_run + reward_ctrl


class GYMMB_HalfCheetah(HalfCheetahEnv):
    def __init__(self):
        self.prev_qpos_0 = None
        super().__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if self.prev_qpos_0 is not None:  # Defensive coding. First reset_model needs to be called.
            self.prev_qpos_0 = self.sim.data.qpos.flat[0]
        return ob, None, None, {}

    def _get_obs(self):
        # _get_obs is called also in init to setup the observation space dim. We can ignore that
        qpos_0_delta = self.sim.data.qpos.flat[0] - self.prev_qpos_0 if self.prev_qpos_0 is not None else 0

        return np.concatenate([
            [qpos_0_delta / self.dt],
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_qpos_0 = self.sim.data.qpos.flat[0]
        return self._get_obs()

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
