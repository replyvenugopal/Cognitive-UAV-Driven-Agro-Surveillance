"""Train & evaluate GeoSpatio-TRiNet against the classical/deep baselines,
reproducing the structure of the paper's Table 2.

Usage:
    python -m agrosurveillance.training.train_forecasting --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import AgroStressDataset, build_zone_sequences, causal_split, collate_zone_batch
from ..evaluation.metrics import classification_report
from ..models.baselines import (
    TORCH_BASELINES,
    build_decision_tree,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)
from ..models.geospatio_trinet import GeoSpatioTriNet
from ..utils.config import load_config, resolve_device
from ..utils.seed import set_seed
from .data_loading import load_dataframe


def batch_to_numpy(loader: DataLoader):
    feats, labels = [], []
    for batch in loader:
        feats.append(batch["features"].numpy())
        labels.append(batch["health_label"].numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def train_torch_model(model: nn.Module, train_loader, epochs: int, lr: float, device: str,
                       is_geospatio: bool = False) -> nn.Module:
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["health_label"].float().to(device)
            valid_mask = batch["valid_mask"].to(device)

            optim.zero_grad()
            if is_geospatio:
                out = model(features, valid_mask=valid_mask)
                logits = out["stress_logits"]
            else:
                logits = model(features)

            loss = (loss_fn(logits, labels) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            loss.backward()
            optim.step()
            total_loss += loss.item()
    return model


@torch.no_grad()
def predict_torch_model(model: nn.Module, loader: DataLoader, device: str, is_geospatio: bool = False):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        labels = batch["health_label"].numpy()

        if is_geospatio:
            out = model(features, valid_mask=valid_mask)
            probs = out["stress_prob"]
        else:
            probs = torch.sigmoid(model(features))

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels)
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def run(cfg) -> dict[str, dict[str, float]]:
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.training.device)

    df = load_dataframe(cfg)
    sequences = build_zone_sequences(
        df,
        sequence_length=cfg.data.sequence_length,
        baseline_window=cfg.data.baseline_window,
    )
    train_seqs, test_seqs = causal_split(sequences, train_fraction=cfg.data.train_fraction)
    print(f"[data] {len(sequences)} zone sequences -> {len(train_seqs)} train / {len(test_seqs)} test")

    train_loader = DataLoader(
        AgroStressDataset(train_seqs), batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_zone_batch
    )
    test_loader = DataLoader(
        AgroStressDataset(test_seqs), batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_zone_batch
    )

    input_dim = next(iter(train_loader))["features"].shape[-1]
    results: dict[str, dict[str, float]] = {}

    # ---- Classical ML baselines (flattened, no temporal modeling) ----
    train_x, train_y = batch_to_numpy(train_loader)
    test_x, test_y = batch_to_numpy(test_loader)

    classical = {
        "Logistic Regression": build_logistic_regression(),
        "Decision Tree": build_decision_tree(),
        "Random Forest": build_random_forest(),
    }
    xgb_model = build_xgboost()
    if xgb_model is not None:
        classical["XGBoost"] = xgb_model

    for name, clf in classical.items():
        t0 = time.time()
        clf.fit(train_x, train_y)
        probs = clf.predict_proba(test_x)
        results[name] = classification_report(test_y, probs)
        print(f"[baseline] {name} done in {time.time() - t0:.1f}s -> acc={results[name]['accuracy']:.1f}")

    # ---- Torch baselines (CNN / CNN-LSTM / Temporal Transformer) ----
    for name, cls in TORCH_BASELINES.items():
        t0 = time.time()
        model = cls(input_dim=input_dim)
        model = train_torch_model(model, train_loader, cfg.training.epochs, cfg.training.lr, device)
        probs, labels = predict_torch_model(model, test_loader, device)
        results[name] = classification_report(labels, probs)
        print(f"[baseline] {name} done in {time.time() - t0:.1f}s -> acc={results[name]['accuracy']:.1f}")

    # ---- GeoSpatio-TRiNet (proposed) ----
    t0 = time.time()
    model = GeoSpatioTriNet(input_dim=input_dim, **{k: v for k, v in cfg.model.items()})
    model = train_torch_model(model, train_loader, cfg.training.epochs, cfg.training.lr, device, is_geospatio=True)
    probs, labels = predict_torch_model(model, test_loader, device, is_geospatio=True)
    results["GeoSpatio-TRiNet (proposed)"] = classification_report(labels, probs)
    print(f"[proposed] GeoSpatio-TRiNet done in {time.time() - t0:.1f}s -> "
          f"acc={results['GeoSpatio-TRiNet (proposed)']['accuracy']:.1f}")

    return results


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    header = f"{'Model':30s} {'Acc%':>7s} {'Prec%':>7s} {'Rec%':>7s} {'F1%':>7s} {'ROC-AUC':>8s} {'PR-AUC':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:30s} {r['accuracy']:7.1f} {r['precision']:7.1f} {r['recall']:7.1f} "
              f"{r['f1']:7.1f} {r['roc_auc']:8.1f} {r['pr_auc']:8.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = run(cfg)
    print_results_table(results)


if __name__ == "__main__":
    main()
