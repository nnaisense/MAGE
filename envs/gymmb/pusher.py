import numpy as np
import gym
#from envs.gymmb.envcopy import PusherEnvCopy
from gym.envs.mujoco import PusherEnv
import torch

from envs.task import Task

_qpos_len = 11
_qvel_len = 11

_snapshot_qpos_offset = 0
_snapshot_qvel_offset = _snapshot_qpos_offset + _qpos_len


class StandardTask(Task):
    def __call__(self, states, actions, next_states):
        """
        states and next_states are actually observations returned by _get_obs() and step() and
        not those quantities with which get_snapshot(), restore_from_snapshot() operate.
        """
        body_com_tips_arm = states[:, 14:14+3]
        body_com_object = states[:, 14+3:14+6]
        body_com_goal = states[:, 14+6:14+9]
        vec_1 = body_com_object - body_com_tips_arm
        vec_2 = body_com_object - body_com_goal

        reward_near = - vec_1.norm(dim=1) # np.linalg.norm(vec_1)
        reward_dist = - vec_2.norm(dim=1)
        reward_ctrl = - actions.pow(2).sum(dim=1)

        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        return reward


# wraper - goal is to provide just interface that magellan needs
class GYMMB_Pusher(gym.Wrapper):
    # but we need to be also gym env in order to register in gym and also inicialization goes throught gym factory method

    def __init__(self):
        # super().__init__(PusherEnvCopy()
        super().__init__(PusherEnv())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(len(self._get_obs()),), dtype=np.float64)

    # from the gym view we are acting like bare environment not as an wrapper
    spec = None

    @property
    def unwrapped(self):
        return self

    def get_snapshot(self):
        """
        Returns :
          returns a snapshot, which can be used by restore_from_snapshot(snapshot)
        """
        act = self.env.sim.data.act
        if act is not None:
            print("act = {}".format(act))
            raise RuntimeError("act is not None")
        return np.concatenate([
            self.env.sim.data.qpos.flat,
            self.env.sim.data.qvel.flat
        ])

    def restore_from_snapshot(self, snapshot):  # set_state(self,state) :
        """
        It sets the state of the environment to snapshot argument. It should be expepected that two instances
        set to the same state generate the same state sequences ater set time. But it is unfortunatelly not true here,
        because of warm conditioning of mujoco state optimization. Thus it should be expected that even if two instances are set to
        the same state, the state sequences diverge after couple of steps (e.g. 20).
        """
        qpos = snapshot[_snapshot_qpos_offset: _snapshot_qpos_offset + _qpos_len]
        qvel = snapshot[_snapshot_qvel_offset: _snapshot_qvel_offset + _qvel_len]
        self.env.set_state(qpos, qvel)

    def _get_obs(self):
        return np.concatenate([
            self.env.sim.data.qpos.flat[:7],
            self.env.sim.data.qvel.flat[:7],
            self.env.get_body_com("tips_arm"),
            self.env.get_body_com("object"),
            self.env.get_body_com("goal"),
        ])

    def step(self, action):
        self.env.do_simulation(action, self.frame_skip)
        return self._get_obs(), None, None, {}

    @staticmethod
    def is_done(states):
        done = torch.zeros((len(states),), dtype=torch.bool,
                           device=states.device)
        return done

    @staticmethod
    def tasks():
        return dict(standard=StandardTask())

    def reset(self, **kwargs):
        # unlike reset_model(self), this also reset simulation
        self.env.reset(**kwargs)
        return self._get_obs()
