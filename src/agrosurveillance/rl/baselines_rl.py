"""Lightweight RL baselines for the Table 7-style convergence comparison:
tabular Q-Learning, DQN (discrete), and DDPG (continuous).

These are compact reference implementations, not tuned to match the paper's
exact numbers - they exist so `training/train_rl.py` can produce a relative
convergence/stability comparison against SAC the way Table 7 does. PPO and
TD3 are documented as straightforward extensions (e.g. via `stable-
baselines3`) rather than re-implemented here, to keep this module focused.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sac import mlp


def discretize_action_space(max_action: float, n_levels: int = 5) -> np.ndarray:
    """Discretizes [0, max_action] into `n_levels` uniform irrigation levels
    (Eq. 48): zero, low, moderate, medium-high, maximum irrigation."""
    return np.linspace(0.0, max_action, n_levels)


class QLearningAgent:
    """Tabular Q-learning over a coarsely binned state space."""

    def __init__(self, state_dim: int, action_levels: np.ndarray, bins: int = 6,
                 lr: float = 0.1, gamma: float = 0.97, epsilon: float = 0.2, seed: int = 0):
        self.action_levels = action_levels
        self.bins = bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple, np.ndarray] = {}
        self.rng = np.random.RandomState(seed)
        self.state_dim = state_dim

    def _discretize(self, state: np.ndarray) -> tuple:
        clipped = np.clip(state, -5, 5)
        return tuple(np.round(clipped, 1).tolist())

    def _q_row(self, key: tuple) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.action_levels))
        return self.q_table[key]

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        key = self._discretize(state)
        row = self._q_row(key)
        if not deterministic and self.rng.rand() < self.epsilon:
            idx = self.rng.randint(len(self.action_levels))
        else:
            idx = int(np.argmax(row))
        return np.array([self.action_levels[idx]], dtype=np.float32)

    def update_step(self, state, action_value, reward, next_state, done) -> None:
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        idx = int(np.argmin(np.abs(self.action_levels - action_value)))
        row = self._q_row(key)
        next_row = self._q_row(next_key)
        target = reward + (0.0 if done else self.gamma * next_row.max())
        row[idx] += self.lr * (target - row[idx])


class DQNNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, n_actions, n_hidden=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Standard DQN with a target network and epsilon-greedy exploration."""

    def __init__(self, state_dim: int, action_levels: np.ndarray, hidden_dim: int = 64,
                 lr: float = 1e-3, gamma: float = 0.97, epsilon: float = 0.2,
                 target_update_every: int = 20, device: str = "cpu"):
        self.action_levels = action_levels
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_every = target_update_every
        self._step_count = 0

        self.q_net = DQNNet(state_dim, len(action_levels), hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and random.random() < self.epsilon:
            idx = random.randrange(len(self.action_levels))
        else:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                idx = int(self.q_net(s).argmax(dim=-1).item())
        return np.array([self.action_levels[idx]], dtype=np.float32)

    def update(self, batch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        action_idx = np.array(
            [int(np.argmin(np.abs(self.action_levels - a[0]))) for a in actions]
        )
        action_idx = torch.as_tensor(action_idx, dtype=torch.long, device=self.device).unsqueeze(-1)

        q_values = self.q_net(states).gather(1, action_idx)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=-1, keepdim=True).values
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self._step_count += 1
        if self._step_count % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": float(loss.item())}


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_high: float):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, action_dim, n_hidden=2)
        self.action_high = action_high

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(self.net(state)) + 1.0) * 0.5 * self.action_high


class DDPGCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden_dim, 1, n_hidden=2)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    """Deep Deterministic Policy Gradient - a simpler continuous-control
    baseline than SAC (deterministic policy, no entropy regularization)."""

    def __init__(self, state_dim: int, action_dim: int = 1, action_high: float = 10.0,
                 hidden_dim: int = 64, gamma: float = 0.97, tau: float = 0.01,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 noise_std: float = 0.5, device: str = "cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_high = action_high
        self.noise_std = noise_std

        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim, action_high).to(self.device)
        self.critic = DDPGCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            action = action + np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action, 0.0, self.action_high).astype(np.float32)

    def update(self, batch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_action)
            target = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        with torch.no_grad():
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}
