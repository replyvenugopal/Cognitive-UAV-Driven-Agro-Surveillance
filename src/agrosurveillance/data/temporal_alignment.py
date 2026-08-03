"""Event-anchored temporal alignment (Section 3.3, Eq. 10-11, 14, 16).

Rather than aligning multimodal streams purely on timestamp proximity, the
paper anchors synchronization to detected stress-triggering events (rapid
soil-moisture loss, sudden temperature swings). This module implements the
event indicator, the alignment operator, and the event-anchor selection.
"""
from __future__ import annotations

import numpy as np


def event_indicator(
    moisture: np.ndarray,
    canopy_temp: np.ndarray,
    theta_m: float = 0.05,
    theta_t: float = 1.0,
) -> np.ndarray:
    """Binary stress-event indicator per timestep (Eq. 10).

    e[t] = 1 if |dM/dt| > theta_m or |dT/dt| > theta_t else 0.
    """
    moisture = np.asarray(moisture, dtype=float)
    canopy_temp = np.asarray(canopy_temp, dtype=float)
    dm = np.gradient(moisture)
    dt = np.gradient(canopy_temp)
    return ((np.abs(dm) > theta_m) | (np.abs(dt) > theta_t)).astype(int)


def event_anchor(soil_moisture_series: np.ndarray) -> int:
    """Select the strongest stress-triggering timestep (Eq. 14 / Eq. 16).

    tau* = argmax_tau |dS/dt|
    Returns the index of the largest-magnitude soil-moisture gradient, used
    as the anchor around which multimodal observations are aligned.
    """
    grad = np.gradient(np.asarray(soil_moisture_series, dtype=float))
    return int(np.argmax(np.abs(grad)))


def align_to_events(
    values: np.ndarray,
    events: np.ndarray,
    max_gap: int = 2,
) -> np.ndarray:
    """Event-anchored alignment operator A(.) (Eq. 11).

    For each timestep, if it is not itself an event but lies within
    `max_gap` steps of one, linearly interpolate towards the nearest event
    observation; longer gaps are left untouched (masked) rather than
    force-interpolated, matching the paper's description of masking large
    gaps during temporal encoding to avoid artificial continuity.
    """
    values = np.asarray(values, dtype=float).copy()
    event_idx = np.flatnonzero(events)
    if event_idx.size == 0:
        return values

    t_len = values.shape[0]
    aligned = values.copy()
    for t in range(t_len):
        if events[t]:
            continue
        distances = np.abs(event_idx - t)
        nearest = event_idx[np.argmin(distances)]
        gap = abs(nearest - t)
        if gap <= max_gap and gap > 0:
            frac = gap / max_gap
            aligned[t] = (1 - frac) * values[nearest] + frac * values[t]
        # else: gap too large -> leave as-is (equivalent to masking at the
        # encoder level; dataset.py additionally emits a validity mask).
    return aligned


def temporal_gradient(features: np.ndarray) -> np.ndarray:
    """Temporal stress gradient G_z^t = F_z^t - F_z^{t-1} (Eq. 19)."""
    features = np.asarray(features, dtype=float)
    grad = np.zeros_like(features)
    grad[1:] = features[1:] - features[:-1]
    return grad
