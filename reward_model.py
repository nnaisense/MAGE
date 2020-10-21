import copy

import torch
from torch import nn
from torch.nn.functional import mse_loss

from models import EnsembleDenseLayer


class RewardModel(nn.Module):
    def __init__(self, d_state, d_action, n_units, n_layers, activation, device):
        assert n_layers >= 1, "minimum depth of model is 1"

        super().__init__()

        layers = []
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                lyr = EnsembleDenseLayer(d_state + d_action + d_state, n_units, ensemble_size=1, non_linearity=activation)
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleDenseLayer(n_units, n_units, ensemble_size=1, non_linearity=activation)
            else:  # lyr_idx == n_layers:
                lyr = EnsembleDenseLayer(n_units, 1, ensemble_size=1, non_linearity='linear')
            layers.append(lyr)

        self.layers = nn.Sequential(*layers)

        self.to(device)

        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_units
        self.n_layers = n_layers
        self.ensemble_size = 1
        self.normalizer = None
        self.device = device
        # Don't need init_params() because EnsembleDenseLayer is init by default

    def init_params(self):
        for layer in self.layers:
            layer.init_params()

    def setup_normalizer(self, normalizer):
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)

    def forward(self, states, actions, next_states):
        """
        Args:
            states (torch Tensor[batch size, d_state])
            actions (torch Tensor[batch size, d_action])
            next_states (torch Tensor[batch size, d_state])
        """
        states, actions, next_states = [x.to(self.device) for x in [states, actions, next_states]]
        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        next_states = next_states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(states)

        returns = self.layers(torch.cat((states, actions, next_states), dim=2)).squeeze(0)
        if self.normalizer is not None:
            returns = self.normalizer.denormalize_rewards(returns)
        return returns.squeeze(0)

    def loss(self, states, actions, next_states, target_rewards):
        # Clamping to stabilize gradients when we get very precise. Most probably, this is not crucial.
        target_rewards = target_rewards.to(self.device)
        return mse_loss(self(states, actions, next_states).squeeze(1), target_rewards).mean()
