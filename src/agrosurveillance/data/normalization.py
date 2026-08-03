"""Stress-aware normalization (Section 3.2, Eq. 5, 7-9, 12-13).

Implements zone-specific normalization against a rolling historical baseline
instead of global dataset statistics, so that subtle, locally-early stress
signals are not washed out by dataset-wide min-max / z-score scaling.
"""
from __future__ import annotations

import numpy as np


def rolling_zone_baseline(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean/std per zone-sequence (Eq. 8-9).

    Parameters
    ----------
    values: array of shape (T,) or (T, D) - a single zone's time series.
    window: L, the length of the historical baseline window.

    Returns
    -------
    (mu, sigma) each of shape matching `values`, where mu[t]/sigma[t] are
    computed from the preceding `window` timesteps (Eq. 8-9). The first
    `window` steps fall back to an expanding window so no NaNs are produced.
    """
    values = np.asarray(values, dtype=float)
    t_len = values.shape[0]
    mu = np.zeros_like(values)
    sigma = np.zeros_like(values)
    for t in range(t_len):
        lo = max(0, t - window)
        hist = values[lo:t] if t > 0 else values[0:1]
        mu[t] = hist.mean(axis=0)
        sigma[t] = hist.std(axis=0)
    return mu, sigma


def stress_aware_normalize(values: np.ndarray, window: int, eps: float = 1e-6) -> np.ndarray:
    """Zone-specific normalization relative to a local environmental baseline.

    Implements Eq. 5 / Eq. 7 / Eq. 12:
        x_hat[z,t] = (x[z,t] - mu_z) / (sigma_z + eps)
    with mu_z, sigma_z drawn from a trailing window (Eq. 8-9), instead of a
    single global mean/std over the whole dataset.
    """
    mu, sigma = rolling_zone_baseline(values, window)
    return (values - mu) / (sigma + eps)


def thermal_stress_deviation(canopy_temp: np.ndarray, ambient_temp: np.ndarray) -> np.ndarray:
    """Canopy-minus-ambient temperature deviation (Eq. 6 / Eq. 13).

    The public dataset has no direct canopy/ambient thermal channel pair, so
    callers typically pass a thermal-proxy feature (e.g. Chlorophyll_Content
    or a derived index) as `canopy_temp` and the recorded air Temperature
    column as `ambient_temp`; see schema.py for the mapping used by
    dataset.py. This keeps the equation faithful even though the underlying
    dataset substitutes a proxy channel for true thermal imagery.
    """
    return np.asarray(canopy_temp, dtype=float) - np.asarray(ambient_temp, dtype=float)


def global_normalize(values: np.ndarray, window: int | None = None, eps: float = 1e-6) -> np.ndarray:
    """Traditional dataset-wide min-max/z-score normalization, kept only for
    the "Without Stress-Aware Norm" ablation arm (Table 8).

    Accepts (and ignores) a `window` argument purely so it is interchangeable
    with `stress_aware_normalize` at call sites such as `dataset.build_zone_sequences`.
    """
    values = np.asarray(values, dtype=float)
    mu = values.mean(axis=0, keepdims=True)
    sigma = values.std(axis=0, keepdims=True)
    return (values - mu) / (sigma + eps)
