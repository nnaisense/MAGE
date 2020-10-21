#!/usr/bin/env python
import logging
import warnings

warnings.filterwarnings(action='ignore', module='importlib', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', module='dotmap', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', module='tensorflow', category=FutureWarning)
warnings.filterwarnings(action='ignore', module='tensorboard', category=FutureWarning)
warnings.filterwarnings(action='ignore', module='gym', category=FutureWarning)

from time import perf_counter
from dotmap import DotMap
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import os
import sacred
import gym
import neptune
import sys
from env_loop import EnvLoop
from datetime import datetime
from logger import configure_logger
from metriclogger import MetricLogger

# noinspection PyUnresolvedReferences
import envs  # Register custom gym envs
# noinspection PyUnresolvedReferences
import envs.gymmb  # Register custom gym envs
# noinspection PyUnresolvedReferences
import sacred_utils  # For a custom mongodb flag

from radam import RAdam
from reward_model import RewardModel
from td3 import TD3
from ddpg import DDPG
from wrappers import BoundedActionsEnv, IsDoneEnv, MuJoCoCloseFixWrapper, RecordedEnv
from buffer import Buffer
from models import Model
from normalizer import TransitionNormalizer
from imagination import SingleStepImagination
from utils import to_np, EpisodeStats


ex = sacred.Experiment()

configure_logger()
logger = logging.getLogger(__file__)


""" Experiment Configuration """


# noinspection PyUnusedLocal
@ex.config
def main_config():
    n_total_steps = 100000                          # total number of steps in real environment (including warm up)
    n_warm_up_steps = 1000                          # number of steps on real MDP to populate the initial buffer, actions selected by random agent

    normalize_data = True                           # normalize states, actions, next states to zero mean and unit variance (both for model training and policy training)


# noinspection PyUnusedLocal
@ex.config
def eval_config():
    eval_freq = 1000                                # interval in steps for testing in exploitations the trained models and agents, can be None
    n_eval_episodes_per_policy = 10                 # number of episodes each eval agent is evaluated for each task


# noinspection PyUnusedLocal
@ex.config
def env_config(n_total_steps):
    env_name = 'GYMMB_HalfCheetah-v2'               # environment name: GYMMB_* or Magellan*
    task_name = 'standard'                          # Name of task to perform within environment e.g. in half cheetah env. either 'running' or 'flipping'  # TODO: We could accept a combined task e.g. flipping+running+renyi

    # Misc
    render = False                                  # render the environment visually (warning: could open too many windows)
    record = False                                  # record videos of episodes (warning: could be slower and use up disk space)
    record_freq = int(n_total_steps * 0.05)         # frequency in steps to record a complete episode (while the agent is training)

    env = gym.make(env_name)
    d_state = env.observation_space.shape[0]        # state dimensionality
    d_action = env.action_space.shape[0]            # action dimensionality
    del env


# noinspection PyUnusedLocal
@ex.config
def model_arch_config(env_name):
    model_ensemble_size = 8                         # number of models in the bootstrap ensemble
    model_n_units = 512                             # number of hidden units in each hidden layer (hidden layer size)
    model_n_layers = 4                              # number of hidden layers in the model (at least 2)
    model_activation = 'swish'                      # activation function (see models.py for options)

    train_reward = False                            # Whether to train the reward function (True) or use the hand-designed one (False)
    reward_n_units = 256
    reward_n_layers = 3
    reward_activation = 'swish'


# noinspection PyUnusedLocal
@ex.config
def model_training_config():
    model_training_freq = 25                        # interval in steps between model trainings
    model_training_n_batches = 120                  # number of batches every model training (1200 corresponds to 15 epochs for 20k buffer)

    model_training_grad_clip = 5                    # gradient clipping to train model
    model_lr = 1e-4                                 # learning rate for training models
    model_weight_decay = 1e-4                       # L2 weight decay on model parameters (good: 1e-5, default: 0)
    model_batch_size = 256                          # batch size for training models

    model_sampling_type = 'ensemble'                # Procedure to use when sampling from ensemble of models, 'ensemble' or 'DS'


# TODO: For training/evaluation active/reactive consider using hierarchical dicts
# noinspection PyUnusedLocal
@ex.config
def policy_training_config(env_name):
    discount = 0.99                                # discount factor

    policy_training_freq = 1                       # interval (in real steps) between subsequent policy trainings
    policy_training_n_iters = 10                   # number of policy update iterations (img data + policy update)
    policy_training_n_updates_per_iter = 1         # number of on-policy optimizations steps

    policy_actors = 1024                           # number of parallel actors in imagination MDP


# noinspection PyUnusedLocal
@ex.config
def policy_arch_config():
    # policy function
    policy_n_layers = 2                             # number of hidden layers (>=1)
    policy_n_units = 384                            # number of units in each hidden layer
    policy_activation = 'swish'
    policy_lr = 1e-4                                # learning rate

    # value function
    value_n_layers = 2                              # number of hidden layers (>=1)
    value_n_units = 384                             # number of units in each hidden layer
    value_activation = 'swish'
    value_lr = 1e-4
    value_tau = 0.005                               # soft target network update mixing factor
    value_loss = 'huber'                            # 'huber' or 'mse'

    # common for value and policy
    agent_grad_clip = 5
    agent_alg = 'td3'                               # td3 or ddpg

    # Parameters for TD3
    td3_policy_delay = 2
    td3_expl_noise = 0.1

    # TD-Gradient parameters
    tdg_error_weight = 5.                           # weight to be used for value gradient td learning (in the paper we use alpha=1/tdg_error_weight=0.2)
    td_error_weight = 1.                            # weight for the standard td error for q learning


# noinspection PyUnusedLocal
@ex.config
def infra_config():
    use_cuda = True                                 # if true use CUDA
    gpu_id = 0                                      # ID of GPU to use (by default use GPU 0)
    print_config = True                             # Set False if you don't want that (e.g. for regression tests)

    if use_cuda and 'CUDA_VISIBLE_DEVICES' not in os.environ:  # gpu_id is used only if CUDA_VISIBLE_DEVICES was not set
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    self_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = '__default__'                        # Set dump_dir=None if you don't want to be create dump_dir
    if dump_dir == '__default__':
        dump_dir = os.path.join(self_dir, 'logs', f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
    neptune_project = None                          # e.g. yourlogin/sandbox

    omp_num_threads = 1                             # 1 is usually the correct choice, especially when using GPU


@ex.capture
def setup(seed, dump_dir, omp_num_threads, print_config, _run):
    if print_config:
        ex.commands["print_config"]()
        print('Shell command:', sacred_utils.get_shell_command())
    if dump_dir is not None:
        sacred_utils.dump_shell_command(dump_dir)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(omp_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)


""" Initialization Helpers (Get) """


@ex.capture
def get_env(env_name, record):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    env = IsDoneEnv(env)
    env = MuJoCoCloseFixWrapper(env)
    if record:
        env = RecordedEnv(env)

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.action_space, 'seed'):  # Only for more recent gym
        env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.observation_space, 'seed'):  # Only for more recent gym
        env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))

    return env


@ex.capture
def get_agent(mode, *, agent_alg):
    logger.debug(f"{ex.step_i:6d} | {mode} | getting fresh agent ...")
    if agent_alg == 'td3':
        return get_td3_agent()
    if agent_alg == 'ddpg':
        return get_ddpg_agent()
    raise ValueError(f'Unknown agent alg {agent_alg}')


@ex.capture
def get_td3_agent(*, d_state, d_action, discount, device, value_tau, value_loss, policy_lr,
                  value_lr, policy_n_units, value_n_units, policy_n_layers, value_n_layers, policy_activation,
                  value_activation, agent_grad_clip, td3_policy_delay, tdg_error_weight, td_error_weight, td3_expl_noise):
    return TD3(d_state=d_state, d_action=d_action, device=device, gamma=discount, tau=value_tau,
               value_loss=value_loss, policy_lr=policy_lr, value_lr=value_lr,
               policy_n_layers=policy_n_layers, value_n_layers=value_n_layers, value_n_units=value_n_units,
               policy_n_units=policy_n_units, policy_activation=policy_activation, value_activation=value_activation,
               grad_clip=agent_grad_clip, policy_delay=td3_policy_delay,
               tdg_error_weight=tdg_error_weight, td_error_weight=td_error_weight, expl_noise=td3_expl_noise)


@ex.capture
def get_ddpg_agent(*, d_state, d_action, discount, device, value_tau, value_loss, policy_lr,
                   value_lr, policy_n_units, value_n_units, policy_n_layers, value_n_layers, policy_activation,
                   value_activation, agent_grad_clip, tdg_error_weight, td_error_weight):
    return DDPG(d_state=d_state, d_action=d_action, device=device, gamma=discount, tau=value_tau,
                value_loss=value_loss, policy_lr=policy_lr, value_lr=value_lr,
                policy_n_layers=policy_n_layers, value_n_layers=value_n_layers, value_n_units=value_n_units,
                policy_n_units=policy_n_units, policy_activation=policy_activation, value_activation=value_activation,
                grad_clip=agent_grad_clip, tdg_error_weight=tdg_error_weight, td_error_weight=td_error_weight)


@ex.capture
def get_random_agent(d_action, device):
    class RandomAgent:
        # noinspection PyUnusedLocal
        @staticmethod
        def get_action(states, deterministic=False):
            # This is not so nice since we hard-code [-1, +1] range but this is the only way to be compatible with the agent interface
            return torch.rand(size=(states.shape[0], d_action), device=device) * 2 - 1
    return RandomAgent()


def to_deterministic_agent(agent):
    class DeterministicAgent:
        @staticmethod
        def get_action(states):
            return agent.get_action(states, deterministic=True)

    return DeterministicAgent()


@ex.capture
def get_model(d_action, d_state, model_ensemble_size, model_n_units, model_n_layers, model_activation, device, _run):
    logger.debug(f"{ex.step_i:6d} | getting fresh model ...")
    model = Model(d_action=d_action, d_state=d_state, ensemble_size=model_ensemble_size,
                  n_units=model_n_units, n_layers=model_n_layers, activation=model_activation,
                  device=device)
    return model


@ex.capture
def get_reward_model(d_action, d_state, reward_n_units, reward_n_layers, reward_activation, device, _run):
    logger.debug(f"{ex.step_i:6d} | getting fresh reward model ...")
    model = RewardModel(d_action=d_action, d_state=d_state,
                        n_units=reward_n_units, n_layers=reward_n_layers, activation=reward_activation,
                        device=device)
    return model


@ex.capture
def get_imagination(model, initial_states, *, model_sampling_type, policy_actors):
    return SingleStepImagination(model, initial_states, n_actors=policy_actors, model_sampling_type=model_sampling_type)


@ex.capture
def get_model_optimizer(params, *, model_lr, model_weight_decay):
    return RAdam(params, lr=model_lr, weight_decay=model_weight_decay)


@ex.capture
def get_reward_model_optimizer(params, *, model_lr, model_weight_decay):
    return RAdam(params, lr=model_lr, weight_decay=model_weight_decay)


@ex.capture
def get_buffer(d_state, d_action, n_total_steps, normalize_data, device):
    data_buffer_size = n_total_steps
    buffer = Buffer(d_action=d_action, d_state=d_state, size=data_buffer_size)
    if normalize_data:
        buffer.setup_normalizer(TransitionNormalizer(d_state, d_action, device))
    return buffer


""" Agent Training """


class BufferTransitionsProvider:
    def __init__(self, buffer, task, is_done, device, policy_actors):
        self.buffer = buffer
        self.task = task
        self.is_done = is_done
        self.device = device
        self.policy_actors = policy_actors

    def get_training_transitions(self, agent):
        states, actions, next_states, _ = self.buffer.view()
        idx = torch.randint(len(self.buffer), size=[self.policy_actors])
        states, actions, next_states = [x[idx].to(self.device) for x in [states, actions, next_states]]
        rewards = self.task(states, actions, next_states)
        dones = self.is_done(next_states)
        logps = torch.ones(actions.shape[0], device=self.device) * np.inf
        return states, actions, logps, next_states, rewards, dones


class ImaginationTransitionsProvider:
    def __init__(self, imagination, task, is_done):
        self.imagination = imagination
        self.task = task
        self.is_done = is_done
        self.imagination.reset()

    def get_training_transitions(self, agent):
        states, actions, logps, next_states = self.imagination.many_steps(agent)
        rewards = self.task(states, actions, next_states)
        dones = self.is_done(next_states)
        return states, actions, logps, next_states, rewards, dones


@ex.capture
def get_training_data_provider(model, buffer, is_done, task):
    initial_states, _, _, _ = buffer.view()
    imagination = get_imagination(model, initial_states)
    return ImaginationTransitionsProvider(imagination=imagination, task=task, is_done=is_done)


@ex.capture
def train_agent(agent, model, reward_model, buffer, task, task_name, is_done, mode, context_i, *, _run, device,
                policy_training_n_updates_per_iter, agent_alg, train_reward, policy_training_n_iters):
    data_provider = get_training_data_provider(model, buffer, is_done, task)

    q_loss, pi_loss = np.nan, np.nan
    for img_step_i in range(1, policy_training_n_iters + 1):
        states, actions, logps, next_states, rewards, dones = data_provider.get_training_transitions(agent)
        if train_reward:
            rewards = reward_model(states, actions, next_states).squeeze(1)

        if len(states) == 0:
            continue

        for img_update_i in range(1, policy_training_n_updates_per_iter + 1):
            raw_action, q_loss, q_grad_loss, pi_loss = agent.update(states, actions, logps, rewards, next_states, masks=~dones)
            # This is rare but can still happen
            if agent.catastrophic_divergence(q_loss, pi_loss):
                logger.info("Catastrophic divergence detected. Agent reset.")
                agent = get_agent('train')
                agent.setup_normalizer(model.normalizer)

    mode_task = f'{mode} | t:{task_name}'
    logger.debug(f"{ex.step_i:6d} | {mode_task} | pi_loss: {pi_loss:6.3f}; q_loss: {q_loss:6.3f}")

    if 'rewards' in locals():
        log_dict = dict(pi_loss=pi_loss, q_loss=q_loss, q_grad_loss=q_grad_loss,
                        img_rewards_avg=rewards.mean().item(),
                        img_rewards_max=rewards.max().item(),
                        raw_action=raw_action,
                        update_batch_size=states.size()[0])
        ex.mlog.add_scalars(f"{mode}/{task_name}/mb_final", log_dict, **context_i)

    return agent


""" Model Training """


@ex.capture
def model_train_epoch(model, buffer, optimizer, model_batch_size, model_training_grad_clip):
    losses = []  # stores loss after each minibatch gradient update
    for states, actions, state_deltas in buffer.train_batches(ensemble_size=model.ensemble_size, batch_size=model_batch_size):
        optimizer.zero_grad()
        loss = model.loss(states, actions, state_deltas)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=model_training_grad_clip)
        optimizer.step()

    return losses


@ex.capture
def train_model(model, optimizer, buffer, mode, model_training_n_batches, *, _run):
    logger.debug(f"{ex.step_i:6d} | {mode} | training model...")
    n_target_batches = model_training_n_batches

    loss = np.nan
    batch_i = 0
    while batch_i < n_target_batches:
        losses = model_train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        batch_i += len(losses)
        loss = np.mean(losses)
        logger.log(5, f'{ex.step_i:6d} | {mode} | batch {batch_i:3d} | model training loss: {loss:.2f}')
        ex.mlog.add_scalar(f"{mode}/model/train_loss", loss, batch_i=batch_i)

    logger.debug(f"{ex.step_i:6d} | {mode} | model training final loss : {loss:.3f}")
    ex.mlog.add_scalar(f"{mode}/model/training_loss_final", loss)


@ex.capture
def reward_model_train_epoch(reward_model, buffer, optimizer, task, model_batch_size, model_training_grad_clip):
    losses = []  # stores loss after each minibatch gradient update
    for states, actions, state_deltas in buffer.train_batches(ensemble_size=1, batch_size=model_batch_size):
        next_states = states + state_deltas
        states, actions, next_states = states.squeeze(0), actions.squeeze(0), next_states.squeeze(0)
        rewards = task(states, actions, next_states)
        optimizer.zero_grad()
        loss = reward_model.loss(states, actions, next_states, rewards)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(reward_model.parameters(), clip_value=model_training_grad_clip)
        optimizer.step()

    return losses


@ex.capture
def train_reward_model(reward_model, optimizer, buffer, mode, model_training_n_batches, task, *, _run):
    logger.debug(f"{ex.step_i:6d} | {mode} | training reward model...")
    n_target_batches = model_training_n_batches

    loss = np.nan
    batch_i = 0
    while batch_i < n_target_batches:
        losses = reward_model_train_epoch(reward_model=reward_model, buffer=buffer, task=task, optimizer=optimizer)
        batch_i += len(losses)
        loss = np.mean(losses)
        logger.log(5, f'{ex.step_i:6d} | {mode} | batch {batch_i:3d} | reward model training loss: {loss:.2f}')
        ex.mlog.add_scalar(f"{mode}/reward_model/train_loss", loss, batch_i=batch_i)

    logger.debug(f"{ex.step_i:6d} | {mode} | reward model training final loss : {loss:.3f}")
    ex.mlog.add_scalar(f"{mode}/reward_model/training_loss_final", loss)


""" Evaluation method (testing model/buffer on task) """


@ex.capture
def evaluate_on_task(agent, model, buffer, task, task_name, context, *,  _run,
                     n_eval_episodes_per_policy, render, record, dump_dir):
    """ Evaluate agent or model & agent """
    episode_returns, episode_lengths = [], []

    video_file_base = dump_dir + f'/{context}_step_{ex.step_i}_task_{task_name}_episode' + '_{}.mp4' if dump_dir is not None else None
    env_loop = EnvLoop(get_env, render=render, record=record, video_file_base=video_file_base, run=_run)
    agent = to_deterministic_agent(agent)

    # Test agent on real environment by running an episode
    for ep_i in range(1, n_eval_episodes_per_policy + 1):
        if ep_i == 1:  # We record only the first episode
            env_loop.record_next_episode()
        with torch.no_grad():
            states, actions, next_states = env_loop.episode(agent, video_file_suffix=ep_i)
            rewards = task(states, actions, next_states)

        ep_return = rewards.sum().item()
        ep_len = len(rewards)
        logger.log(15, f"{ex.step_i:6d} | {context} | t:{task_name} | episode {ep_i} | score: {ep_return:5.2f}")
        ex.mlog.add_scalars(f'{context}/{task_name}/episode/', {'return': ep_return, 'length': ep_len},
                            ep_i=ep_i)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
    env_loop.close()

    return episode_returns, episode_lengths


def evaluate_on_tasks(agent, model, buffer, task_name, context):
    logger.info(f"{ex.step_i:6d} | {context} | evaluating model for tasks...")
    env = get_env(record=False)
    task = env.unwrapped.tasks()[task_name]
    env.close()

    ep_returns, ep_lengths = evaluate_on_task(agent, model, buffer, task, task_name, context)
    avg_ep_return = np.mean(ep_returns)
    std_ep_return = np.std(ep_returns)
    avg_ep_length = np.mean(ep_lengths)

    logger.info(f"{ex.step_i:6d} | {context} | t:{task_name} | avg. return: {avg_ep_return:5.2f}+-{std_ep_return:5.2f}")
    ex.mlog.add_scalars(f'{context}/{task_name}/avg_episode/', {'return': avg_ep_return, 'length': avg_ep_length})

    return avg_ep_return


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        ex.mlog.add_scalar(f'duration/{func.__name__}', end - start)
        return result
    return wrapper


""" Main Methods (sacred commands) """


@ex.capture
def log_last_episode(stats, *, _run):
    for task_name, task in stats.tasks.items():
        last_ep_return = stats.ep_returns[task_name][-1]
        last_ep_len = stats.ep_lengths[task_name][-1]
        logger.info(f'{ex.step_i:6d} | train | t:{task_name} | return: {last_ep_return:5.1f} ({last_ep_len:3d} steps)')
        ex.mlog.add_scalars(f"train/{task_name}/episode", {'return': last_ep_return, 'length': last_ep_len})


@ex.capture
def get_neptune_ex(*, neptune_project, _run):
    if neptune_project is None:
        return None

    return neptune.init(neptune_project).create_experiment(
        name=_run.experiment_info['name'],
        description=None,
        params=sacred_utils.flatten_dict(_run.config),
        tags=None,
        upload_source_files=sacred_utils.get_filepaths(dirpath=_run.experiment_info['base_dir'],
                                                       extensions=['.py', '.yaml', '.yml']),
        logger=logging.getLogger(),
        send_hardware_metrics=True,
        git_info=neptune.utils.get_git_info(_run.experiment_info['base_dir']),
    )


class MainTrainingLoop:
    """ Resembles ray.Trainable """
    @ex.capture
    def __init__(self, *, task_name):
        logger.info(f"Executing training...")

        tmp_env = get_env(record=False)
        self.is_done = tmp_env.unwrapped.is_done
        self.eval_tasks = {task_name: tmp_env.tasks()[task_name]}
        self.exploitation_task = tmp_env.tasks()[task_name]
        del tmp_env

        # Constitute the state of Trainable
        ex.step_i = 0
        self.model = get_model()
        self.reward_model = get_reward_model()
        self.model_optimizer = get_model_optimizer(self.model.parameters())
        self.reward_model_optimizer = get_reward_model_optimizer(self.reward_model.parameters())
        self.buffer = get_buffer()
        self.agent = get_agent(mode='train')
        self.agent.setup_normalizer(self.buffer.normalizer)
        self.stats = EpisodeStats(self.eval_tasks)
        self.last_avg_eval_score = None
        self.neptune_ex = None
        ex.mlog = None

        # Not considered part of the state
        self.new_experiment = True  # I need to know if I had to create a new experiment (neptune) or continue an old one
        self.random_agent = get_random_agent()

        self._common_setup()

    @ex.capture
    def _common_setup(self, *, render, record, dump_dir, _run):
        """ Called in __init__ but needs also to be called after restore (due to reinitialized randomness) """
        video_file_base = dump_dir + "/max_exploitation_step_{}.mp4" if dump_dir is not None else None
        self.env_loop = EnvLoop(get_env, render=render, record=record, video_file_base=video_file_base, run=_run)

    def _setup_if_new(self):
        """ Executed for a new experiment only. This is a workaround for Trainable. """
        if self.new_experiment:
            self.new_experiment = False
            self.neptune_ex = get_neptune_ex()
            ex.mlog = MetricLogger(ex, self.neptune_ex)

    @ex.capture
    def train(self, *, device, n_total_steps, n_warm_up_steps, record_freq, record,
              model_training_freq, policy_training_freq, eval_freq,
              task_name, model_training_n_batches, train_reward):
        """ A single step of interaction with the environment. """
        self._setup_if_new()

        ex.step_i += 1

        behavioral_agent = self.random_agent if ex.step_i <= n_warm_up_steps else self.agent
        with torch.no_grad():
            action = behavioral_agent.get_action(self.env_loop.state, deterministic=False).to('cpu')
        prev_state = self.env_loop.state.clone().to(device)
        if record and (ex.step_i == 1 or ex.step_i % record_freq == 0):
            self.env_loop.record_next_episode()
        state, next_state, done = self.env_loop.step(to_np(action), video_file_suffix=ex.step_i)
        reward = self.exploitation_task(state, action, next_state).item()
        self.buffer.add(state, action, next_state, torch.from_numpy(np.array([[reward]], dtype=np.float)))
        self.stats.add(state, action, next_state, done)
        if done:
            log_last_episode(self.stats)

        tasks_rewards = {f'{task_name}': self.stats.get_recent_reward(task_name) for task_name in self.eval_tasks}
        step_stats = dict(
            step=ex.step_i,
            done=done,
            action_abs_mean=action.abs().mean().item(),
            reward=self.exploitation_task(state, action, next_state).item(),
            action_value=self.agent.get_action_value(prev_state, action).item(),
        )
        ex.mlog.add_scalars('main_loop', {**step_stats, **tasks_rewards})

        # (Re)train the model on the current buffer
        if model_training_freq is not None and model_training_n_batches > 0 and ex.step_i % model_training_freq == 0:
            self.model.setup_normalizer(self.buffer.normalizer)
            self.reward_model.setup_normalizer(self.buffer.normalizer)
            timed(train_model)(self.model, self.model_optimizer, self.buffer, mode='train')
            if train_reward:
                task = self.exploitation_task
                timed(train_reward_model)(self.reward_model, self.reward_model_optimizer, self.buffer, mode='train', task=task)

        # (Re)train the policy using current buffer and model
        if ex.step_i >= n_warm_up_steps and ex.step_i % policy_training_freq == 0:
            task = self.exploitation_task
            self.agent.setup_normalizer(self.buffer.normalizer)
            self.agent = timed(train_agent)(self.agent, self.model, self.reward_model, self.buffer, task=task, task_name=task_name, is_done=self.is_done,
                                            mode='train', context_i={})

        # Evaluate the agent
        if eval_freq is not None and ex.step_i % eval_freq == 0:
            self.last_avg_eval_score = evaluate_on_tasks(agent=self.agent, model=self.model, buffer=self.buffer, task_name=task_name, context='eval')

        experiment_finished = ex.step_i >= n_total_steps
        return DotMap(
            done=experiment_finished,
            avg_eval_score=self.last_avg_eval_score,
            action_abs_mean=action.abs().mean().item(),  # This is just for regression tests
            step_i=ex.step_i)

    def stop(self):
        self.env_loop.close()
        if ex.mlog is not None:
            ex.mlog.save_artifacts()
            if ex.mlog.neptune_ex is not None:
                logger.info("Stopping neptune...")
                ex.mlog.neptune_ex.stop()


@ex.automain
def train():
    setup()
    training = MainTrainingLoop()

    res = DotMap(done=False)
    while not res.done:
        res = training.train()

    training.stop()

    return res.get('avg_eval_score'), res.get('action_abs_mean')
