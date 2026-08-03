"""Train the SAC adaptive irrigation controller (and lightweight RL
baselines) inside the artificial agro-environment, reproducing the shape of
the paper's Tables 4, 5 and 7.

Usage:
    python -m agrosurveillance.training.train_rl --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ..data.dataset import build_zone_sequences, causal_split
from ..rl.baselines_rl import DDPGAgent, DQNAgent, QLearningAgent, discretize_action_space
from ..rl.environment import AgroIrrigationEnv, zone_sequence_to_env_data
from ..rl.replay_buffer import ReplayBuffer
from ..rl.sac import SACAgent
from ..utils.config import load_config, resolve_device
from ..utils.seed import set_seed
from .data_loading import load_dataframe


@dataclass
class EpisodeResult:
    total_reward: float
    total_water: float
    initial_vulnerability: float
    final_vulnerability: float


def make_envs(sequences, cfg, seed: int) -> list[AgroIrrigationEnv]:
    envs = []
    for i, seq in enumerate(sequences):
        zdata = zone_sequence_to_env_data(seq)
        envs.append(
            AgroIrrigationEnv(
                zdata,
                max_irrigation=cfg.rl.max_irrigation,
                alpha=cfg.rl.reward_alpha,
                beta=cfg.rl.reward_beta,
                gamma=cfg.rl.reward_gamma,
                delayed_horizon=cfg.rl.delayed_reward_horizon,
                seed=seed + i,
            )
        )
    return envs


def run_episode(env: AgroIrrigationEnv, policy_fn, max_steps: int) -> EpisodeResult:
    obs, _ = env.reset()
    initial_vuln = env._vulnerability
    total_reward, total_water = 0.0, 0.0
    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        total_water += info["irrigation"]
        if term or trunc:
            break
    return EpisodeResult(total_reward, total_water, initial_vuln, env._vulnerability)


# ---------------------------------------------------------------- baselines
def no_control_policy(obs: np.ndarray) -> np.ndarray:
    return np.array([0.0], dtype=np.float32)


def make_threshold_policy(max_irrigation: float, moisture_threshold: float = 45.0):
    def policy(obs: np.ndarray) -> np.ndarray:
        moisture = obs[1]
        return np.array([max_irrigation if moisture < moisture_threshold else 0.0], dtype=np.float32)

    return policy


# ---------------------------------------------------------------- learned agents
def train_off_policy_agent(agent, envs: list[AgroIrrigationEnv], cfg, max_steps: int,
                            buffer_capacity: int, batch_size: int, warmup_steps: int,
                            is_qlearning: bool = False) -> list[float]:
    state_dim = envs[0].observation_space.shape[0]
    buf = None if is_qlearning else ReplayBuffer(buffer_capacity, state_dim, 1)
    rewards_per_episode = []
    total_steps = 0

    for ep in range(cfg.rl.episodes):
        env = envs[ep % len(envs)]
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            if is_qlearning:
                agent.update_step(obs, float(action[0]), reward, next_obs, term or trunc)
            else:
                buf.add(obs, action, reward, next_obs, term or trunc)
                total_steps += 1
                if len(buf) > batch_size and total_steps > warmup_steps:
                    agent.update(buf.sample(batch_size))
            obs = next_obs
            ep_reward += reward
            if term or trunc:
                break
        rewards_per_episode.append(ep_reward)

    return rewards_per_episode


def evaluate_agent(agent, envs: list[AgroIrrigationEnv], max_steps: int, deterministic: bool = True):
    results = [run_episode(env, lambda o: agent.select_action(o, deterministic=deterministic), max_steps) for env in envs]
    return summarize(results)


def evaluate_policy_fn(policy_fn, envs: list[AgroIrrigationEnv], max_steps: int):
    results = [run_episode(env, policy_fn, max_steps) for env in envs]
    return summarize(results)


def summarize(results: list[EpisodeResult]) -> dict[str, float]:
    init_v = np.mean([r.initial_vulnerability for r in results])
    final_v = np.mean([r.final_vulnerability for r in results])
    water = np.mean([r.total_water for r in results])
    reward = np.mean([r.total_reward for r in results])
    reduction_pct = 100.0 * (init_v - final_v) / init_v if init_v > 0 else 0.0
    return {
        "initial_vulnerability": float(init_v),
        "final_vulnerability": float(final_v),
        "vulnerability_reduction_pct": float(reduction_pct),
        "water_used": float(water),
        "avg_reward": float(reward),
    }


def run(cfg) -> dict[str, dict[str, float]]:
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.training.device)

    df = load_dataframe(cfg)
    sequences = build_zone_sequences(df, sequence_length=cfg.data.sequence_length, baseline_window=cfg.data.baseline_window)
    train_seqs, test_seqs = causal_split(sequences, train_fraction=cfg.data.train_fraction)
    if not test_seqs:
        test_seqs = train_seqs

    max_steps = cfg.rl.max_steps_per_episode
    train_envs = make_envs(train_seqs, cfg, seed=cfg.get("seed", 42))
    test_envs = make_envs(test_seqs, cfg, seed=cfg.get("seed", 42) + 1000)
    state_dim = train_envs[0].observation_space.shape[0]

    results: dict[str, dict[str, float]] = {}

    # ---- Non-learned baselines ----
    results["No Control"] = evaluate_policy_fn(no_control_policy, test_envs, max_steps)
    results["Threshold Irrigation"] = evaluate_policy_fn(
        make_threshold_policy(cfg.rl.max_irrigation), test_envs, max_steps
    )

    action_levels = discretize_action_space(cfg.rl.max_irrigation, n_levels=5)

    # ---- Q-Learning ----
    ql_agent = QLearningAgent(state_dim, action_levels, seed=cfg.get("seed", 42))
    train_off_policy_agent(ql_agent, train_envs, cfg, max_steps, cfg.rl.replay_capacity,
                            cfg.rl.batch_size, cfg.rl.warmup_steps, is_qlearning=True)
    results["Q-Learning"] = evaluate_agent(ql_agent, test_envs, max_steps)

    # ---- DQN ----
    dqn_agent = DQNAgent(state_dim, action_levels, hidden_dim=cfg.rl.hidden_dim, device=device)
    train_off_policy_agent(dqn_agent, train_envs, cfg, max_steps, cfg.rl.replay_capacity,
                            cfg.rl.batch_size, cfg.rl.warmup_steps)
    results["DQN"] = evaluate_agent(dqn_agent, test_envs, max_steps)

    # ---- DDPG ----
    ddpg_agent = DDPGAgent(state_dim, action_high=cfg.rl.max_irrigation, hidden_dim=cfg.rl.hidden_dim, device=device)
    train_off_policy_agent(ddpg_agent, train_envs, cfg, max_steps, cfg.rl.replay_capacity,
                            cfg.rl.batch_size, cfg.rl.warmup_steps)
    results["DDPG"] = evaluate_agent(ddpg_agent, test_envs, max_steps)

    # ---- SAC (proposed) ----
    sac_agent = SACAgent(
        state_dim=state_dim,
        action_high=cfg.rl.max_irrigation,
        hidden_dim=cfg.rl.hidden_dim,
        gamma=cfg.rl.gamma,
        tau=cfg.rl.tau,
        actor_lr=cfg.rl.actor_lr,
        critic_lr=cfg.rl.critic_lr,
        alpha_lr=cfg.rl.alpha_lr,
        auto_entropy_tuning=cfg.rl.auto_entropy_tuning,
        device=device,
    )
    train_off_policy_agent(sac_agent, train_envs, cfg, max_steps, cfg.rl.replay_capacity,
                            cfg.rl.batch_size, cfg.rl.warmup_steps)
    results["Proposed (SAC)"] = evaluate_agent(sac_agent, test_envs, max_steps)

    return results


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    header = f"{'Method':22s} {'Init V':>8s} {'Final V':>8s} {'Reduction%':>11s} {'Water':>8s} {'AvgReward':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:22s} {r['initial_vulnerability']:8.3f} {r['final_vulnerability']:8.3f} "
              f"{r['vulnerability_reduction_pct']:11.1f} {r['water_used']:8.1f} {r['avg_reward']:10.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = run(cfg)
    print_results_table(results)


if __name__ == "__main__":
    main()
