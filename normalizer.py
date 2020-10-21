import torch


def fix_stdev(stdev):
    # In case of a very small stdev, normalization would fail with division by 0. This way we handle the extreme case
    # where all the state values are the same (e.g. constant observation)
    stdev[stdev < 1e-6] = 1.0
    return stdev


def normalize(x, mean, stdev):
    return (x - mean) / stdev


def denormalize(norm_x, mean, stdev):
    return norm_x * stdev + mean


class DummyNormalizer:
    def add(self, state, action, state_delta, reward):
        pass

    def normalize_states(self, states):
        return states

    def normalize_rewards(self, rewards):
        return rewards

    def normalize_actions(self, actions):
        return actions

    def normalize_state_deltas(self, state_deltas):
        return state_deltas

    def denormalize_state_delta_means(self, norm_state_deltas_means):
        return norm_state_deltas_means

    def denormalize_rewards(self, norm_rewards):
        return norm_rewards

    def denormalize_state_delta_vars(self, state_delta_vars):
        return state_delta_vars

    def renormalize_state_delta_means(self, state_deltas_means):
        return state_deltas_means

    def renormalize_state_delta_vars(self, state_delta_vars):
        return state_delta_vars


class TransitionNormalizer:
    def __init__(self, d_state, d_action, device):
        """
        Maintain moving mean and standard deviation of state, action and state_delta
        for the formulas see: https://www.johndcook.com/blog/standard_deviation/
        """

        self.state_mean = torch.zeros(d_state, device=device)
        self.state_sk = torch.zeros(d_state, device=device)
        self.state_stdev = torch.ones(d_state, device=device)
        self.action_mean = torch.zeros(d_action, device=device)
        self.action_sk = torch.zeros(d_action, device=device)
        self.action_stdev = torch.ones(d_action, device=device)
        self.state_delta_mean = torch.zeros(d_state, device=device)
        self.state_delta_sk = torch.zeros(d_state, device=device)
        self.state_delta_stdev = torch.ones(d_state, device=device)
        self.reward_mean = torch.zeros(1, device=device)
        self.reward_sk = torch.zeros(1, device=device)
        self.reward_stdev = torch.ones(1, device=device)

        self.count = torch.scalar_tensor(0, device=device)
        self.device = device

    @staticmethod
    def update_mean(mu_old, addendum, n):
        mu_new = mu_old + (addendum - mu_old) / n
        return mu_new

    @staticmethod
    def update_sk(sk_old, mu_old, mu_new, addendum):
        sk_new = sk_old + (addendum - mu_old) * (addendum - mu_new)
        return sk_new

    def add(self, state, action, state_delta, reward):
        reward = reward.float()
        with torch.no_grad():
            state = state.to(self.device)
            action = action.to(self.device)
            state_delta = state_delta.to(self.device)
            reward = reward.to(self.device)

            self.count += 1

            if self.count == 1:
                # first element, initialize
                self.state_mean = state.clone()
                self.action_mean = action.clone()
                self.state_delta_mean = state_delta.clone()
                self.reward_mean = reward.clone()
                return

            state_mean_old = self.state_mean.clone()
            action_mean_old = self.action_mean.clone()
            state_delta_mean_old = self.state_delta_mean.clone()
            reward_mean_old = self.reward_mean.clone()

            self.state_mean = self.update_mean(self.state_mean, state, self.count)
            self.action_mean = self.update_mean(self.action_mean, action, self.count)
            self.state_delta_mean = self.update_mean(self.state_delta_mean, state_delta, self.count)
            self.reward_mean = self.update_mean(self.reward_mean, reward, self.count)

            self.state_sk = self.update_sk(self.state_sk, state_mean_old, self.state_mean, state)
            self.action_sk = self.update_sk(self.action_sk, action_mean_old, self.action_mean, action)
            self.state_delta_sk = self.update_sk(self.state_delta_sk, state_delta_mean_old, self.state_delta_mean, state_delta)
            self.reward_sk = self.update_sk(self.reward_sk, reward_mean_old, self.reward_mean, reward)

            self.state_stdev = fix_stdev(torch.sqrt(self.state_sk / self.count))
            self.action_stdev = fix_stdev(torch.sqrt(self.action_sk / self.count))
            self.state_delta_stdev = fix_stdev(torch.sqrt(self.state_delta_sk / self.count))
            self.reward_stdev = fix_stdev(torch.sqrt(self.reward_sk / self.count))

    def normalize_states(self, states):
        return normalize(states, self.state_mean, self.state_stdev)

    def normalize_rewards(self, rewards):
        return normalize(rewards, self.reward_mean, self.reward_stdev)

    def normalize_actions(self, actions):
        return normalize(actions, self.action_mean, self.action_stdev)

    def normalize_state_deltas(self, state_deltas):
        return normalize(state_deltas, self.state_delta_mean, self.state_delta_stdev)

    def denormalize_state_delta_means(self, norm_state_deltas_means):
        return denormalize(norm_state_deltas_means, self.state_delta_mean, self.state_delta_stdev)

    def denormalize_rewards(self, norm_rewards):
        return denormalize(norm_rewards, self.reward_mean, self.reward_stdev)

    def denormalize_state_delta_vars(self, state_delta_vars):
        mean, stdev = self.state_delta_mean, self.state_delta_stdev
        return state_delta_vars * (stdev ** 2)

    def renormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.state_delta_mean, self.state_delta_stdev
        return normalize(state_deltas_means, mean, stdev)

    def renormalize_state_delta_vars(self, state_delta_vars):
        mean, stdev = self.state_delta_mean, self.state_delta_stdev
        return state_delta_vars / (stdev ** 2)
