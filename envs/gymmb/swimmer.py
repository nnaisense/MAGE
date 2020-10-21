import numpy as np
from gym.envs.mujoco import SwimmerEnv
from envs.task import Task
import torch


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        dt = 0.01 * 4  # timestep x frameskip, whatever they are
        ctrl_cost_coeff = 0.0001
        xposbefore = states[:, 0]
        xposafter = next_states[:, 0]

        reward_fwd = (xposafter - xposbefore) / dt
        reward_ctrl = - ctrl_cost_coeff * (actions**2).sum()
        reward = reward_fwd + reward_ctrl
        return reward


class GYMMB_Swimmer(SwimmerEnv):
    def __init__(self):
        self.prev = None
        super().__init__()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, None, None, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        # Don't exclude the current position from the observation
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False
