"""Zone-time sequence construction and PyTorch Dataset (Section 3.1, Eq. 1-4).

Restructures the flat Kaggle table into field-zone-wise time sequences:

    D = { (X_z^t, S_z^t, E^t, Y_z^t) }_{z=1..Z, t=1..T}     (Eq. 1)
    X_z,t = [U_RGB, U_MS, U_TH, S_z,t, E_z,t]                (Eq. 2)

and applies a strict zone-level + chronological train/test split so that no
zone leaks across splits and no future observation leaks into training
(Eq. 3-4):

    train: t <= t_train
    test:  t >  t_train
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import stress_aware_normalize, thermal_stress_deviation
from .schema import ColumnSchema, DEFAULT_SCHEMA, derive_health_label
from .spatial_zoning import cluster_into_zones, zone_adjacency
from .temporal_alignment import event_indicator, temporal_gradient


@dataclass
class ZoneSequence:
    zone_id: int
    uav: np.ndarray          # (T, D_uav) - stress-aware normalized
    soil: np.ndarray         # (T, D_soil)
    env: np.ndarray          # (T, D_env)
    thermal_deviation: np.ndarray  # (T,)
    features: np.ndarray     # (T, D) fused [uav, soil, env, thermal_dev] - Eq. 18
    stress_gradient: np.ndarray    # (T, D)  Eq. 19, computed from `features`
    event_mask: np.ndarray   # (T,) 1 = stress event detected
    stress_label: np.ndarray  # (T,) 0-100 continuous indicator
    health_label: np.ndarray  # (T,) 0/1


def load_raw_table(csv_path: str, schema: ColumnSchema = DEFAULT_SCHEMA) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in schema.all_numeric_feature_cols() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw dataset is missing expected columns {missing}. "
            "See docs/dataset.md for the schema this loader expects."
        )
    df[schema.health_label_col] = derive_health_label(df[schema.stress_indicator_col])
    return df


def build_zone_sequences(
    df: pd.DataFrame,
    schema: ColumnSchema = DEFAULT_SCHEMA,
    sequence_length: int = 12,
    baseline_window: int = 4,
    n_zones: int | None = None,
    normalize_fn=None,
) -> list[ZoneSequence]:
    """Reconstruct field-zone-wise time sequences from the flat table.

    Since the public dataset has no explicit timestamp, rows are treated as
    chronologically ordered observations within their `zone_group_col` group
    (this mirrors how the paper's Eq. 3/4 causal split is meant to operate:
    strictly on row/observation order, not on a wall-clock date). If
    `zone_group_col` produces coarser groups than desired, rows within each
    group are further split into fixed-length windows of `sequence_length`.

    `normalize_fn(values, window) -> array` defaults to the stress-aware
    normalization (Eq. 5/7-9). Pass `agrosurveillance.data.normalization.
    global_normalize` (which ignores `window`) to reproduce the "Without
    Stress-Aware Norm" ablation arm from Table 8.
    """
    if normalize_fn is None:
        normalize_fn = stress_aware_normalize
    sequences: list[ZoneSequence] = []
    uav_cols = list(
        schema.uav_rgb_proxy_cols + schema.uav_multispectral_proxy_cols + schema.uav_thermal_proxy_cols
    )
    soil_cols = list(schema.soil_cols)
    env_cols = list(schema.environment_cols)

    zone_counter = 0
    group_col = schema.zone_group_col if schema.zone_group_col in df.columns else None
    groups = df.groupby(group_col) if group_col else [(0, df)]

    for _, group_df in groups:
        group_df = group_df.reset_index(drop=True)
        n_windows = max(1, len(group_df) // sequence_length)
        for w in range(n_windows):
            start, end = w * sequence_length, (w + 1) * sequence_length
            window = group_df.iloc[start:end]
            if len(window) < 2:
                continue

            uav = window[uav_cols].to_numpy(dtype=float)
            soil = window[soil_cols].to_numpy(dtype=float)
            env = window[env_cols].to_numpy(dtype=float)

            uav_norm = normalize_fn(uav, window=baseline_window)
            soil_norm = normalize_fn(soil, window=baseline_window)

            # Thermal proxy: first UAV thermal-proxy column vs. ambient Temperature.
            thermal_col = schema.uav_thermal_proxy_cols[0]
            ambient_col = "Temperature" if "Temperature" in schema.environment_cols else env_cols[0]
            thermal_dev = thermal_stress_deviation(window[thermal_col].to_numpy(dtype=float),
                                                     window[ambient_col].to_numpy(dtype=float))

            moisture_col = "Soil_Moisture" if "Soil_Moisture" in schema.soil_cols else soil_cols[0]
            events = event_indicator(window[moisture_col].to_numpy(dtype=float), thermal_dev)

            # Fused multi-modal feature vector F_z^t = [X_z^t, S_z^t, E^t, dT_z^t] (Eq. 18)
            fused = np.concatenate([uav_norm, soil_norm, env, thermal_dev[:, None]], axis=1)
            grad = temporal_gradient(fused)  # G_z^t = F_z^t - F_z^{t-1} (Eq. 19)

            sequences.append(
                ZoneSequence(
                    zone_id=zone_counter,
                    uav=uav_norm,
                    soil=soil_norm,
                    env=env,
                    thermal_deviation=thermal_dev,
                    features=fused,
                    stress_gradient=grad,
                    event_mask=events,
                    stress_label=window[schema.stress_indicator_col].to_numpy(dtype=float),
                    health_label=window[schema.health_label_col].to_numpy(dtype=int),
                )
            )
            zone_counter += 1

    if n_zones is not None and len(sequences) > n_zones:
        # Re-cluster on mean stress-feature profile to collapse to n_zones
        # cognitive zones (Eq. 17), then keep the adjacency for diffusion.
        centroids = np.array([seq.uav.mean(axis=0) for seq in sequences])
        assignments = cluster_into_zones(centroids, n_zones=n_zones)
        for seq, zid in zip(sequences, assignments):
            seq.zone_id = int(zid)

    return sequences


def causal_split(sequences: list[ZoneSequence], train_fraction: float = 0.7):
    """Zone-exclusive + chronological split (Eq. 3-4).

    Zones are partitioned exclusively into train/test (no zone appears in
    both), and within each retained zone only the leading `train_fraction`
    of timesteps are used for training features while evaluation always
    looks at the trailing, strictly-future segment.
    """
    zone_ids = sorted({seq.zone_id for seq in sequences})
    rng = np.random.RandomState(42)
    rng.shuffle(zone_ids)
    split_point = max(1, int(len(zone_ids) * train_fraction))
    train_zones = set(zone_ids[:split_point])
    test_zones = set(zone_ids[split_point:]) or set(zone_ids[-1:])

    train_seqs = [s for s in sequences if s.zone_id in train_zones]
    test_seqs = [s for s in sequences if s.zone_id in test_zones]
    return train_seqs, test_seqs


class AgroStressDataset(Dataset):
    """PyTorch Dataset yielding fused zone-time feature tensors and labels."""

    def __init__(self, sequences: list[ZoneSequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        return {
            "zone_id": seq.zone_id,
            "features": torch.tensor(seq.features, dtype=torch.float32),
            "gradient": torch.tensor(seq.stress_gradient, dtype=torch.float32),
            "event_mask": torch.tensor(seq.event_mask, dtype=torch.float32),
            "stress_label": torch.tensor(seq.stress_label, dtype=torch.float32),
            "health_label": torch.tensor(seq.health_label, dtype=torch.long),
        }


def collate_zone_batch(batch: list[dict]):
    """Pads variable-length zone sequences to the batch max length."""
    max_len = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]

    features = torch.zeros(len(batch), max_len, feat_dim)
    gradient = torch.zeros(len(batch), max_len, feat_dim)
    event_mask = torch.zeros(len(batch), max_len)
    stress_label = torch.zeros(len(batch), max_len)
    health_label = torch.zeros(len(batch), max_len, dtype=torch.long)
    valid_mask = torch.zeros(len(batch), max_len)
    zone_ids = []

    for i, item in enumerate(batch):
        t_len = item["features"].shape[0]
        features[i, :t_len] = item["features"]
        gradient[i, :t_len] = item["gradient"]
        event_mask[i, :t_len] = item["event_mask"]
        stress_label[i, :t_len] = item["stress_label"]
        health_label[i, :t_len] = item["health_label"]
        valid_mask[i, :t_len] = 1.0
        zone_ids.append(item["zone_id"])

    return {
        "zone_ids": torch.tensor(zone_ids, dtype=torch.long),
        "features": features,
        "gradient": gradient,
        "event_mask": event_mask,
        "stress_label": stress_label,
        "health_label": health_label,
        "valid_mask": valid_mask,
    }
