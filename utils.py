from collections import defaultdict

import torch
import logging

log = logging.getLogger('max.utils')


def to_torch(x):
    x = torch.from_numpy(x).float()
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    return x


def to_np(x):
    x = x.detach().cpu().numpy()
    if len(x.shape) >= 1:
        x = x.squeeze(0)
    return x


class EpisodeStats:
    """
        Computes rewards fore at each time step in the episode. When episode ends (done==True) it
        logs the total return and episode length

        Args:
            tasks: a list of tasks
    """
    def __init__(self, tasks):
        self.tasks = tasks
        self.curr_episode_rewards = defaultdict(list)
        self.ep_returns = defaultdict(list)
        self.ep_lengths = defaultdict(list)
        self.last_reward = defaultdict(float)

    def add(self, state, action, next_state, done):
        for task_name, task in self.tasks.items():
            with torch.no_grad():
                step_reward = task(state, action, next_state).item()
            self.curr_episode_rewards[task_name].append(step_reward)
            self.last_reward[task_name] = step_reward
            if done:
                self.ep_returns[task_name].append(sum(self.curr_episode_rewards[task_name]))
                self.ep_lengths[task_name].append(len(self.curr_episode_rewards[task_name]))
                self.curr_episode_rewards[task_name].clear()

    def get_recent_reward(self, task_name):
        return self.last_reward[task_name]

