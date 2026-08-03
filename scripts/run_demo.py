#!/usr/bin/env python3
"""End-to-end smoke test / quickstart: runs a scaled-down version of the
full pipeline (data -> GeoSpatio-TRiNet -> SAC irrigation control) on the
synthetic data fallback, in well under a minute on CPU.

This is meant to prove the stack wires together correctly, NOT to reproduce
the paper's reported numbers - use a full config (more epochs/episodes) and
the real Kaggle dataset for that (see scripts/download_dataset.py).

Usage:
    python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrosurveillance.training import ablation, train_forecasting, train_rl  # noqa: E402
from agrosurveillance.utils.config import Config  # noqa: E402


def demo_config() -> Config:
    return Config(
        {
            "seed": 42,
            "data": {
                "raw_csv": "data/raw/crop_health_and_environmental_stress.csv",
                "synthetic_fallback": True,
                "zone_group_col": "Field_Boundaries",
                "sequence_length": 12,
                "baseline_window": 4,
                "train_fraction": 0.7,
                "target_col": "Crop_Stress_Indicator",
                "health_label_col": "health_label",
            },
            "model": {
                "feature_dim": 32,
                "spatial_heads": 4,
                "temporal_hidden": 32,
                "temporal_layers": 1,
                "diffusion_steps": 1,
                "diffusion_lambda": 0.35,
                "dropout": 0.1,
                "vulnerability_eta1": 0.6,
                "vulnerability_eta2": 0.4,
                "vulnerability_decay": 0.85,
                "vulnerability_horizon": 3,
            },
            "training": {
                "epochs": 3,
                "batch_size": 8,
                "lr": 1e-3,
                "weight_decay": 1e-5,
                "early_stopping_patience": 3,
                "device": "cpu",
            },
            "rl": {
                "algorithm": "sac",
                "episodes": 6,
                "max_steps_per_episode": 11,
                "gamma": 0.97,
                "tau": 0.01,
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "alpha_lr": 3e-4,
                "auto_entropy_tuning": True,
                "hidden_dim": 32,
                "replay_capacity": 2000,
                "batch_size": 16,
                "warmup_steps": 20,
                "reward_alpha": 1.0,
                "reward_beta": 0.4,
                "reward_gamma": 0.6,
                "delayed_reward_horizon": 2,
                "max_irrigation": 10.0,
            },
            "ablation": {
                "configs": [
                    {"name": "full", "disable": []},
                    {"name": "without_diffusion", "disable": ["diffusion"]},
                ]
            },
        }
    )


def main():
    cfg = demo_config()

    print("=" * 70)
    print("STEP 1/3: Stress forecasting (GeoSpatio-TRiNet vs. baselines)")
    print("=" * 70)
    forecasting_results = train_forecasting.run(cfg)
    train_forecasting.print_results_table(forecasting_results)

    print("\n" + "=" * 70)
    print("STEP 2/3: Adaptive irrigation control (SAC vs. baselines)")
    print("=" * 70)
    rl_results = train_rl.run(cfg)
    train_rl.print_results_table(rl_results)

    print("\n" + "=" * 70)
    print("STEP 3/3: Ablation study (subset)")
    print("=" * 70)
    ablation_results = ablation.run(cfg)
    ablation.print_results_table(ablation_results)

    print("\nDemo complete. All pipeline stages ran successfully.")


if __name__ == "__main__":
    main()
