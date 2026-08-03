"""Ablation study runner, reproducing the structure of the paper's Table 8:
each configuration disables one architectural component (adaptive zoning /
spatial attention, diffusion, temporal encoding, stress-aware normalization,
or the RL controller entirely) and reports forecast accuracy plus, where
applicable, vulnerability reduction and water saving.

Note on what each ablation actually moves: the RL environment's exogenous
stress driver comes from the *observed* dataset trajectory, not from
GeoSpatio-TRiNet's predictions (this mirrors the paper's own description of
an "artificial agro-environment created on the basis of the dataset",
Section 4). Consequently the "spatial"/"diffusion"/"temporal" architectural
ablations only move the forecast-accuracy column - the control-quality
columns (vulnerability reduction / water used) only change for the
"without_rl" and "without_stress_aware_norm" arms, which is expected rather
than a bug. Wiring the SAC state through the model's live predictions
instead of ground truth is a natural extension (see README "Limitations").

Usage:
    python -m agrosurveillance.training.ablation --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from ..data.dataset import AgroStressDataset, build_zone_sequences, causal_split, collate_zone_batch
from ..data.normalization import global_normalize
from ..evaluation.metrics import classification_report
from ..models.geospatio_trinet import GeoSpatioTriNet
from ..rl.environment import zone_sequence_to_env_data, AgroIrrigationEnv
from ..rl.sac import SACAgent
from ..utils.config import load_config, resolve_device
from ..utils.seed import set_seed
from .data_loading import load_dataframe
from .train_forecasting import predict_torch_model, train_torch_model
from .train_rl import evaluate_agent, evaluate_policy_fn, make_threshold_policy, train_off_policy_agent

# Maps the human-readable ablation-config disable flags (configs/default.yaml)
# onto the GeoSpatioTriNet component names it actually accepts.
ZONING_TO_SPATIAL = {"zoning": "spatial", "diffusion": "diffusion", "temporal": "temporal"}


def run_single_ablation(cfg, ablation_cfg, device: str) -> dict[str, float]:
    disable_flags = set(ablation_cfg.get("disable", []))
    model_disable = tuple(ZONING_TO_SPATIAL[f] for f in disable_flags if f in ZONING_TO_SPATIAL)
    use_global_norm = "stress_aware_norm" in disable_flags
    skip_rl = "rl" in disable_flags

    df = load_dataframe(cfg)
    normalize_fn = global_normalize if use_global_norm else None
    sequences = build_zone_sequences(
        df,
        sequence_length=cfg.data.sequence_length,
        baseline_window=cfg.data.baseline_window,
        normalize_fn=normalize_fn,
    )
    train_seqs, test_seqs = causal_split(sequences, train_fraction=cfg.data.train_fraction)

    train_loader = DataLoader(AgroStressDataset(train_seqs), batch_size=cfg.training.batch_size,
                               shuffle=True, collate_fn=collate_zone_batch)
    test_loader = DataLoader(AgroStressDataset(test_seqs), batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_zone_batch)
    input_dim = next(iter(train_loader))["features"].shape[-1]

    model_kwargs = {k: v for k, v in cfg.model.items()}
    model = GeoSpatioTriNet(input_dim=input_dim, disable=model_disable, **model_kwargs)
    model = train_torch_model(model, train_loader, cfg.training.epochs, cfg.training.lr, device, is_geospatio=True)
    probs, labels = predict_torch_model(model, test_loader, device, is_geospatio=True)
    report = classification_report(labels, probs)

    # ---- Downstream control quality ----
    max_steps = cfg.rl.max_steps_per_episode
    test_envs = [
        AgroIrrigationEnv(
            zone_sequence_to_env_data(seq), max_irrigation=cfg.rl.max_irrigation,
            alpha=cfg.rl.reward_alpha, beta=cfg.rl.reward_beta, gamma=cfg.rl.reward_gamma,
            delayed_horizon=cfg.rl.delayed_reward_horizon, seed=cfg.get("seed", 42),
        )
        for seq in (test_seqs or train_seqs)
    ]

    if skip_rl:
        # "Without RL": irrigate reactively via a fixed threshold policy
        # instead of a learned SAC controller.
        control_result = evaluate_policy_fn(make_threshold_policy(cfg.rl.max_irrigation), test_envs, max_steps)
    else:
        train_envs = [
            AgroIrrigationEnv(
                zone_sequence_to_env_data(seq), max_irrigation=cfg.rl.max_irrigation,
                alpha=cfg.rl.reward_alpha, beta=cfg.rl.reward_beta, gamma=cfg.rl.reward_gamma,
                delayed_horizon=cfg.rl.delayed_reward_horizon, seed=cfg.get("seed", 42) + 500,
            )
            for seq in train_seqs
        ]
        sac_agent = SACAgent(
            state_dim=test_envs[0].observation_space.shape[0],
            action_high=cfg.rl.max_irrigation,
            hidden_dim=cfg.rl.hidden_dim,
            gamma=cfg.rl.gamma,
            device=device,
        )
        train_off_policy_agent(sac_agent, train_envs, cfg, max_steps, cfg.rl.replay_capacity,
                                cfg.rl.batch_size, cfg.rl.warmup_steps)
        control_result = evaluate_agent(sac_agent, test_envs, max_steps)

    return {
        "accuracy": report["accuracy"],
        "vulnerability_reduction_pct": control_result["vulnerability_reduction_pct"],
        "water_used": control_result["water_used"],
    }


def run(cfg) -> dict[str, dict[str, float]]:
    device = resolve_device(cfg.training.device)
    results = {}
    for ablation_cfg in cfg.ablation.configs:
        name = ablation_cfg["name"]
        set_seed(cfg.get("seed", 42))
        print(f"[ablation] running '{name}' (disable={ablation_cfg.get('disable', [])}) ...")
        results[name] = run_single_ablation(cfg, ablation_cfg, device)
        print(f"[ablation] '{name}' -> {results[name]}")
    return results


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    header = f"{'Configuration':28s} {'Accuracy%':>10s} {'VulnReduction%':>15s} {'WaterUsed':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:28s} {r['accuracy']:10.1f} {r['vulnerability_reduction_pct']:15.1f} {r['water_used']:10.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = run(cfg)
    print_results_table(results)


if __name__ == "__main__":
    main()
