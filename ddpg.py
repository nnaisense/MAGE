import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_activation, init_weights
from radam import RAdam
import numpy as np


class ActionValueFunction(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super().__init__()
        assert n_layers >= 1, "# of hidden layers"

        layers = [nn.Linear(d_state + d_action, n_units), get_activation(activation)]
        for lyr_idx in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), get_activation(activation)]
        layers += [nn.Linear(n_units, 1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.layers(x)


class Actor(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super().__init__()

        assert n_layers >= 1

        layers = [nn.Linear(d_state, n_units), get_activation(activation)]
        for _ in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), get_activation(activation)]
        layers += [nn.Linear(n_units, d_action)]

        [init_weights(layer) for layer in layers if isinstance(layer, nn.Linear)]

        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return torch.tanh(self.layers(state))  # Bound to -1,1


class DDPG(nn.Module):
    def __init__(
            self,
            d_state,
            d_action,
            device,
            gamma,
            tau,
            policy_lr,
            value_lr,
            value_loss,
            value_n_layers,
            value_n_units,
            value_activation,
            policy_n_layers,
            policy_n_units,
            policy_activation,
            grad_clip,
            policy_noise=0.2,
            noise_clip=0.5,
            expl_noise=0.1,
            tdg_error_weight=0,
            td_error_weight=1,
    ):
        super().__init__()

        self.actor = Actor(d_state, d_action, policy_n_layers, policy_n_units, policy_activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = RAdam(self.actor.parameters(), lr=policy_lr)

        self.critic = ActionValueFunction(d_state, d_action, value_n_layers, value_n_units, value_activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = RAdam(self.critic.parameters(), lr=value_lr)

        self.discount = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.normalizer = None
        self.value_loss = value_loss
        self.grad_clip = grad_clip
        self.device = device

        self.tdg_error_weight = tdg_error_weight
        self.td_error_weight = td_error_weight
        self.step_counter = 0

    def setup_normalizer(self, normalizer):
        self.normalizer = copy.deepcopy(normalizer)

    def get_action(self, states, deterministic=False):
        states = states.to(self.device)
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        actions = self.actor(states)
        if not deterministic:
            actions = actions + torch.randn_like(actions) * self.expl_noise
        return actions.clamp(-1, +1)

    def get_action_with_logp(self, states):
        states = states.to(self.device)
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        a = self.actor(states)
        return a, torch.ones(a.shape[0], device=a.device) * np.inf  # inf: should not be used

    def get_action_value(self, states, actions):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        with torch.no_grad():
            states = states.to(self.device)
            actions = actions.to(self.device)
            return self.critic(states, actions)[0]  # just q1

    def update(self, states, actions, logps, rewards, next_states, masks):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)
        self.step_counter += 1

        # Select action according to policy and add clipped noise
        noise = (
                torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)
        raw_next_actions = self.actor_target(next_states)
        next_actions = (raw_next_actions + noise).clamp(-1, 1)

        # Compute the target Q value
        next_Q = self.critic_target(next_states, next_actions)
        q_target = rewards.unsqueeze(1) + self.discount * masks.float().unsqueeze(1) * next_Q
        zero_targets = torch.zeros_like(q_target, device=self.device)

        q = self.critic(states, actions)  # Q(s,a)
        q_td_error = q_target - q
        critic_loss, standard_loss, gradient_loss = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        if self.td_error_weight > 0:
            if self.value_loss == 'huber':
                standard_loss = 0.5 * F.smooth_l1_loss(q_td_error, zero_targets)
            elif self.value_loss == 'mse':
                standard_loss = 0.5 * F.mse_loss(q_td_error, zero_targets)
            critic_loss = critic_loss + self.td_error_weight * standard_loss
        if self.tdg_error_weight > 0:
            gradients_error_norms = torch.autograd.grad(outputs=q_td_error, inputs=actions,
                                                        grad_outputs=torch.ones(q_td_error.size(), device=self.device),
                                                        retain_graph=True, create_graph=True,
                                                        only_inputs=True)[0].flatten(start_dim=1).norm(dim=1, keepdim=True)
            if self.value_loss == 'huber':
                gradient_loss = 0.5 * F.smooth_l1_loss(gradients_error_norms, zero_targets)
            elif self.value_loss == 'mse':
                gradient_loss = 0.5 * F.mse_loss(gradients_error_norms, zero_targets)
            critic_loss = critic_loss + self.tdg_error_weight * gradient_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Compute actor loss
        q = self.critic(states, self.actor(states))  # Q(s,pi(s))
        actor_loss = -q.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return raw_next_actions[0, 0].item(), self.td_error_weight * standard_loss.item(), self.tdg_error_weight * gradient_loss.item(), actor_loss.item()

    @staticmethod
    def catastrophic_divergence(q_loss, pi_loss):
        return q_loss > 1e2 or (pi_loss is not None and abs(pi_loss) > 1e5)
