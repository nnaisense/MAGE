import sys
import torch
from utils import to_torch, to_np


class EnvLoop:
    """
        Wraps an environment to easily and safely manage environment resetting,
        running multiple steps / single episode and easy episode recording

        Note: This class is written in a episode-stateless way to make it possible checkpointing determinism so please
        refactor with care. In particular, note that _env is created from scratch only when we ask for a state
    """

    def __init__(self, get_env, render, record, video_file_base, run, torch_np_conversion=True):
        """ Initializes the environment iterator. Resets the environment """
        self._get_env = get_env
        self._env = None
        self._state = None
        self.torch_np_conversion = torch_np_conversion

        self._record_next_episode = False
        self._record_in_queue = False  # Indicates there is a pending recording in the queue, which must be recorded on the next episode
        self._step_i = 0

        self._render = render
        self._record = record
        self._video_file_base = video_file_base
        self._run = run

    def _reset(self):
        self._env = self._get_env()  # Create new env every episode. This is a way to prevent checkpointing from having to save env state
        if self.torch_np_conversion:
            self._state = to_torch(self._env.reset())
        else:
            self._state = self._env.reset()
        self._step_i = 0

    @property
    def env(self):
        return self._env

    @property
    def state(self):
        """ state the environment is currently in. shape  [d_state] """
        if self._state is None:
            self._reset()
        return self._state

    def step(self, action, video_file_suffix=None):
        """ Performs a single step in the environment
            Args:
                action (numpy[d_action])
                video_file_suffix (string, optional, default=None): Suffix added to the end of the video file name

            Returns:
                 (s, s', done) transition

                old_state (torch Tensor[1, d_state])
                next_states (torch Tensor[1, d_state])
                dones (boolean[1]): indicates if episode has terminated (either do to termination condition (is_done) or time limit)
        """
        if self._state is None:
            self._reset()

        if self._record:
            video_file_full_path = self._video_file_base.format(video_file_suffix)
            next_state, _, done, info = self.env.step(action, filename=video_file_full_path, record_episode=self._record_next_episode)
        else:
            next_state, _, done, info = self.env.step(action)

        if self.torch_np_conversion:
            next_state = to_torch(next_state)
        old_state = self._state.detach()
        self._state = next_state.detach()  # For more safety (not required, in principle)
        self._step_i += 1

        if self._render:
            self.env.render()

        # Note: at the end of the episode next_state != self.state. The former is the part of the
        # transition while the latter is the current state of the environment (after reset)
        if done:
            self.env.close()
            self._state = None

            if self._record:
                self._run.add_artifact(video_file_full_path)  # save video to sacred DB  # TODO WJ: This is not nice. Why EnvLoop should know about _run?
                self._record_next_episode = self._record_in_queue  # if there's a pending recording in the queue, record it on the next episode
                self._record_in_queue = False

        return old_state, next_state, done

    def episode(self, agent, video_file_suffix=None):
        return self.multi_step(agent, single_episode=True, video_file_suffix=video_file_suffix)

    def multi_step(self, agent, n_steps=None, single_episode=False, video_file_suffix=None):
        """
            Performs multiple steps (either n_steps or a single episode) in the environment
            and returns tensors with all the (s, a, ns) transitions. Either n_steps or single_episode
            must be specified.

            Args:
                agent (object): agent with get_action(state) method returning an action for the agent
                n_steps (int, optional, default=None): number of steps to take in the environment
                single_episode (boolean, optional, default=False): whether to perform only one episode
                video_file_suffix (string, optional, default=None): Suffix added to the end of the video file name

            Returns:
                (s, a, s') staked transitions

                staked all_old_states (torch Tensor[n_steps, d_state])
                staked all_actions (torch Tensor[n_steps, d_action])
                staked all_next_states (torch Tensor[n_steps, d_state])
        """
        assert (n_steps is None) ^ (single_episode is False)

        all_old_states = []
        all_next_states = []
        all_actions = []

        if single_episode:
            n_steps = sys.maxsize

        for i in range(1, n_steps + 1):
            # FIXME: are the view-numel things necessary?
            action = agent.get_action(self.state.view(1, self.state.numel())).to('cpu')
            if self.torch_np_conversion:
                state, next_state, done = self.step(to_np(action), video_file_suffix=video_file_suffix)
            else:
                state, next_state, done = self.step(action, video_file_suffix=video_file_suffix)

            all_old_states.extend(state.view(1, state.numel()))
            all_actions.extend(action.view(1, action.numel()))
            all_next_states.extend(next_state.view(1, next_state.numel()))

            if single_episode and done:
                break
        return torch.stack(all_old_states), torch.stack(all_actions), torch.stack(all_next_states)

    def record_next_episode(self):
        """
            Enables recording the next episode. If the environment is at the start of episode it will record
            the (immediate) next episode. If the episode hast already started it will queue the recording for
            the next available episode.
        """
        if self._step_i == 0 and self._record:
            self._record_next_episode = True
        elif self._step_i != 0 and self._record:  # if episode has already started, queue recording for next episode
            self._record_in_queue = True  # TODO WJ: Don't we need just this + checking step_counter==0 in step()?

    def close(self):
        if self.env is not None:
            self.env.close()
