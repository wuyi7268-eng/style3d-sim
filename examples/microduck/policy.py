"""Optional feed-forward history policy and its shared-statistics normalization."""

import weakref

import torch
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from torch import nn

from .observations import HISTORY_DIM, HISTORY_LENGTH, HISTORY_SIZE


class HistoryActorCritic(ActorCritic):
    """Encode past actor-visible frames independently for the actor and critic."""

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        actor_dim = num_actor_obs - HISTORY_SIZE
        critic_dim = num_critic_obs - HISTORY_SIZE
        if actor_dim != 61 or critic_dim <= 0 or num_actions != 14:
            raise ValueError("History policy requires current observations + 10x48 history and 14 actions")
        super().__init__(actor_dim + 64, critic_dim + 64, num_actions, **kwargs)
        self.actor_dim, self.critic_dim = actor_dim, critic_dim
        self.actor_history_encoder = nn.Sequential(nn.Linear(HISTORY_SIZE, 128), nn.ELU(), nn.Linear(128, 64))
        self.critic_history_encoder = nn.Sequential(nn.Linear(HISTORY_SIZE, 128), nn.ELU(), nn.Linear(128, 64))
        print(f"Actor history encoder: {self.actor_history_encoder}")
        print(f"Critic history encoder: {self.critic_history_encoder}")

    def actor_input(self, observations):
        current, history = observations.split((self.actor_dim, HISTORY_SIZE), dim=-1)
        return torch.cat((current, self.actor_history_encoder(history)), dim=-1)

    def update_distribution(self, observations):
        super().update_distribution(self.actor_input(observations))

    def act_inference(self, observations):
        return self.actor(self.actor_input(observations))

    def evaluate(self, critic_observations, **kwargs):
        current, history = critic_observations.split((self.critic_dim, HISTORY_SIZE), dim=-1)
        return self.critic(torch.cat((current, self.critic_history_encoder(history)), dim=-1))


class HistoryNormalization(nn.Module):
    """Update statistics from current frames only; reuse actor statistics for all past frames."""

    def __init__(self, current_dim, actor_normalizer=None):
        super().__init__()
        self.current_dim = current_dim
        self.current = EmpiricalNormalization(current_dim, until=1.0e8)
        # The critic borrows actor statistics without registering/saving them twice.
        self._actor_stats = weakref.ref(actor_normalizer.current) if actor_normalizer is not None else None

    def forward(self, observations):
        current, history = observations.split((self.current_dim, HISTORY_SIZE), dim=-1)
        current = self.current(current)
        stats = self.current if self._actor_stats is None else self._actor_stats()
        history = history.reshape(-1, HISTORY_LENGTH, HISTORY_DIM)
        history = (history - stats._mean[:, None, :HISTORY_DIM]) / (stats._std[:, None, :HISTORY_DIM] + stats.eps)
        return torch.cat((current, history.flatten(1)), dim=-1)
