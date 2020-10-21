import numpy as np
import math

import copy
import torch
import torch.nn.functional as F
from torch import nn as nn

from torch.distributions import Normal
import torch.jit


def linear(x):
    return x


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


# FIXME
class Swish(nn.Module):
    def forward(self, input):
        return swish(input)


def get_activation(activation):
    if activation == 'swish':
        return Swish()
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'tanh':
        return nn.Tanh()
    # TODO: I should also initialize depending on the activation
    raise NotImplementedError(f"Unknown activation {activation}")


def init_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size, non_linearity):
        """
        linear + activation Layer
        there are `ensemble_size` layers
        computation is done using batch matrix multiplication
        hence forward pass through all models in the ensemble can be done in one call

        weights initialized with xavier normal for leaky relu and linear, xavier uniform for swish
        biases are always initialized to zeros

        Args:
            n_in (int): size of input vector
            n_out (int): size of output vector
            ensemble_size (int): number of models in the ensemble
            non_linearity (str): 'linear', 'swish' or 'leaky_relu'
        """

        super().__init__()

        weights = torch.zeros(ensemble_size, n_in, n_out).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()

        for weight in weights:
            weight.transpose_(1, 0)

            if non_linearity == 'swish':
                nn.init.xavier_uniform_(weight)
            elif non_linearity == 'leaky_relu':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'tanh':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'linear':
                nn.init.xavier_normal_(weight)

            weight.transpose_(1, 0)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        if non_linearity == 'swish':
            self.non_linearity = swish
        elif non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif non_linearity == 'linear':
            self.non_linearity = linear

    def forward(self, inp):
        return self.non_linearity(torch.baddbmm(self.biases, inp, self.weights))


# TODO: This is the same as EnsembleDenseLayer. Combine them.
class ParallelLinear(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size):
        super().__init__()

        weights = torch.zeros(ensemble_size, n_in, n_out).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()

        for weight in weights:
            weight.transpose_(1, 0)
            nn.init.xavier_uniform_(weight)
            weight.transpose_(1, 0)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, inp):
        return torch.baddbmm(self.biases, inp, self.weights)


class Model(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, d_action, d_state, n_units, n_layers, ensemble_size, activation, device):
        """
        state space forward model.
        predicts mean and variance of next state given state and action i.e independent gaussians for each dimension of next state.

        using state and  action, delta of state is computed.
        the mean of the delta is added to current state to get the mean of next state.

        there is a soft threshold on the output variance, forcing it to be in the same range as the variance of the training data.
        the thresholds are learnt in the form of bounds on variance and a small penalty is used to contract the distance between the lower and upper bounds.

        loss components:
            1. minimize negative log-likelihood of data
            2. (small weight) try to contract lower and upper bounds of variance

        Args:
            d_action (int): dimensionality of action
            d_state (int): dimensionality of state
            n_units (int): size or width of hidden layers
            n_layers (int): number of hidden layers (number of non-lineatities). should be >= 2
            ensemble_size (int): number of models in the ensemble
            activation (str): 'linear', 'swish' or 'leaky_relu'
            device (str): device of the model
        """

        assert n_layers >= 2, "minimum depth of model is 2"

        super().__init__()

        layers = []
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                lyr = EnsembleDenseLayer(d_action + d_state, n_units, ensemble_size, non_linearity=activation)
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleDenseLayer(n_units, n_units, ensemble_size, non_linearity=activation)
            else:  # lyr_idx == n_layers:
                lyr = EnsembleDenseLayer(n_units, d_state + d_state, ensemble_size, non_linearity='linear')
            layers.append(lyr)

        self.layers = nn.Sequential(*layers)

        self.to(device)

        self.normalizer = None

        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_units
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size
        self.device = device

    def setup_normalizer(self, normalizer):
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is not None:
            state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, var):
        # denormalize to return in raw state space
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            var = self.normalizer.denormalize_state_delta_vars(var)
        return delta_mean, var

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.layers(inp)
        delta_mean, log_var = torch.split(op, self.d_state, dim=2)

        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(log_var)

        return delta_mean, log_var.exp()

    def forward(self, states, actions):
        """
        predict next state mean and variance.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])

        Returns:
            next state means (torch Tensor[ensemble_size, batch size, dim_state])
            next state variances (torch Tensor[ensemble_size, batch size, dim_state])
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        normalized_states, normalized_actions = self._pre_process_model_inputs(states, actions)
        normalized_delta_mean, normalized_var = self._propagate_network(normalized_states, normalized_actions)
        delta_mean, var = self._post_process_model_outputs(normalized_delta_mean, normalized_var)
        next_state_mean = delta_mean + states
        return next_state_mean, var

    def forward_all(self, states, actions):
        """
        predict next state mean and variance of a batch of states and actions for all models in the ensemble.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])

        Returns:
            next state means (torch Tensor[batch size, ensemble_size, dim_state])
            next state variances (torch Tensor[batch size, ensemble_size, dim_state])
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        next_state_means, next_state_vars = self(states, actions)
        return next_state_means.transpose(0, 1), next_state_vars.transpose(0, 1)

    def random_ensemble(self, states, actions):
        """ Returns a distribution for a single model in the ensemble (selected at random) """
        batch_size = states.shape[0]
        # Get next state distribution for all components in the ensemble
        next_state_means, next_state_vars = self.forward_all(states, actions)  # shape: (batch_size, ensemble_size, d_state)

        i = torch.arange(batch_size, device=self.device)
        j = torch.randint(self.ensemble_size, size=(batch_size,), device=self.device)
        mean = next_state_means[i, j]
        var = next_state_vars[i, j]

        return Normal(mean, var.sqrt())

    def ds(self, states, actions):
        """ Create a single Gaussian out of Gaussian ensemble """
        # Get next state distribution for all components in the ensemble
        next_state_means, next_state_vars = self.forward_all(states, actions)  # shape: (batch_size, ensemble_size, d_state)
        next_state_means, next_state_vars = next_state_means.double(), next_state_vars.double()  # to prevent numerical errors (large means, small vars)

        mean = torch.mean(next_state_means, dim=1)  # shape: (batch_size, d_state)
        mean_var = torch.mean(next_state_vars, dim=1)  # shape: (batch_size, d_state)
        var = torch.mean(next_state_means ** 2, dim=1) - mean ** 2 + mean_var  # shape: (batch_size, d_state) 

        # A safety bound to prevent some unexpected numerical issues. The variance cannot be smaller then mean_var since
        # the sum of the other terms needs to be always positive (convexity)
        var = torch.max(var, mean_var)

        return Normal(mean.float(), var.sqrt().float())  # expects inputs shaped: (batch_size, d_state)

    def posterior(self, states, actions, sampling_type):
        assert sampling_type in ['ensemble', 'DS']
        if sampling_type == 'ensemble':
            return self.random_ensemble(states, actions)
        elif sampling_type == 'DS':
            return self.ds(states, actions)
        else:
            raise ValueError(f'Model sampling method {sampling_type} is not supported')

    def sample(self, states, actions, sampling_type, reparam_trick=True):
        """
        sample next states given current states and actions according to the sampling_type

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])

        Returns:
            next state (torch Tensor[batch size, dim_state])
        """
        pdf = self.posterior(states, actions, sampling_type)
        return pdf.rsample() if reparam_trick else pdf.sample()

    def loss(self, states, actions, state_deltas):
        """
        compute loss given states, actions and state_deltas

        the loss is actually computed between predicted state delta and actual state delta, both in normalized space

        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])
            state_deltas (torch Tensor[ensemble_size, batch size, dim_state])

        Returns:
            loss (torch 0-dim Tensor, scalar): `.backward()` can be called on it to compute gradients
        """

        states, actions = self._pre_process_model_inputs(states, actions)
        targets = self._pre_process_model_targets(state_deltas)

        mu, var = self._propagate_network(states, actions)      # delta and variance

        # negative log likelihood
        loss = (mu - targets) ** 2 / var + torch.log(var)
        return torch.mean(loss)

    def mean_likelihood(self, states, actions, next_states):
        """
        Get likelihood for a batch of data by computing the likelihood on each member
        of the ensemble and then getting the average likelihood across all ensembles.
        Input raw (un-normalized) states, actions and state_deltas

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])
            next_states (torch Tensor[batch size, dim_state]

        Returns:
            likelihood (torch Tensor[batch size])
        """

        next_states = next_states.to(self.device)

        with torch.no_grad():
            mu, var = self.forward_all(states, actions)  # next state and variance. shape: (batch_size, ensemble_size, dim_state)

        pdf = Normal(mu, torch.sqrt(var))  # PDF for each member of the ensemble, expecting inputs shaped (batch_size, ensemble_size, dim_state)
        log_likelihood = pdf.log_prob(next_states.unsqueeze(1).repeat(1, self.ensemble_size, 1))  # shape: (batch_size, ensemble_size, dim_state)
        log_likelihood = log_likelihood.mean(dim=2)  # mean over all dimensions
        log_likelihood = log_likelihood.exp().mean(dim=1).log()  # mean over all models (shape: batch_size)

        return log_likelihood

    def gaussian_likelihood(self, states, actions, next_states):
        """
        Get likelihood for a batch of data by treating the ensemble of models as a single
        *unimodal Gaussian distribution* (who's moments match the ensembles moments) and
        then computing the likelihood of each datapoint under this single Gaussian.
        Input raw (un-normalized) states, actions and state_deltas

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])
            next_states (torch Tensor[batch size, dim_state]

        Returns:
            likelihood (torch Tensor[batch size])
        """
        next_states = next_states.to(self.device)

        with torch.no_grad():
            pdf = self.ds(states, actions)  # PDF for single Gaussian, expecting inputs shaped (batch_size, dim_state)
        log_likelihood = pdf.log_prob(next_states)  # shape: (batch_size, dim_state)
        log_likelihood = log_likelihood.mean(dim=1)  # mean over all state dimensions. shape: (batch_size)

        return log_likelihood

