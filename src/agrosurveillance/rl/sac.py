"""Soft Actor-Critic adaptive irrigation controller (Section 3.9, Eq. 36-41).

    J(pi) = E[ sum_t r_t + beta * H(pi(.|s_t)) ]     (Eq. 36/37)
    theta <- theta - eta * grad_theta J(pi_theta)     (Eq. 40)
    pi*   = argmax_pi J(pi)                           (Eq. 41)

Standard SAC with twin Q-critics, a tanh-squashed Gaussian policy bounded to
the irrigation action range [0, a_max] (Eq. 47), target networks with Polyak
averaging, and optional automatic entropy-coefficient tuning.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(input_dim: int, hidden_dim: int, output_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy, action rescaled to [0, action_high]."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_high: float):
        super().__init__()
        self.trunk = mlp(state_dim, hidden_dim, hidden_dim, n_hidden=1)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.action_high = action_high

    def forward(self, state: torch.Tensor):
        h = F.relu(self.trunk(state))
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        tanh_z = torch.tanh(z)
        # Rescale tanh output in [-1, 1] to the irrigation action range [0, a_max].
        action = 0.5 * (tanh_z + 1.0) * self.action_high

        log_prob = dist.log_prob(z) - torch.log(1.0 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Jacobian correction for the additional affine rescale (constant, but
        # included for correctness): d(action)/d(tanh_z) = action_high / 2.
        log_prob = log_prob - torch.log(torch.tensor(self.action_high / 2.0 + 1e-6))

        deterministic_action = 0.5 * (torch.tanh(mean) + 1.0) * self.action_high
        return action, log_prob, deterministic_action


class QCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden_dim, 1, n_hidden=2)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        action_high: float = 10.0,
        hidden_dim: int = 128,
        gamma: float = 0.97,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        auto_entropy_tuning: bool = True,
        init_alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_high = action_high

        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim, action_high).to(self.device)
        self.critic_1 = QCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = QCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        for p in list(self.critic_1_target.parameters()) + list(self.critic_2_target.parameters()):
            p.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr
        )

        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, det_action = self.actor.sample(state_t)
        chosen = det_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy()

    def update(self, batch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # ---- Critic update: soft Bellman backup (Eq. 36/37) ----
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_states)
            target_q1 = self.critic_1_target(next_states, next_action)
            target_q2 = self.critic_2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor update: maximize Q - alpha * log(pi) ----
        new_action, log_prob, _ = self.actor.sample(states)
        q1_new = self.critic_1(states, new_action)
        q2_new = self.critic_2(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Entropy coefficient update ----
        alpha_loss_value = 0.0
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_loss_value = float(alpha_loss.item())

        # ---- Polyak-averaged target update (Eq. 40 analog for targets) ----
        with torch.no_grad():
            for target_p, p in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                target_p.mul_(1 - self.tau).add_(self.tau * p)
            for target_p, p in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
                target_p.mul_(1 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.item()),
        }
