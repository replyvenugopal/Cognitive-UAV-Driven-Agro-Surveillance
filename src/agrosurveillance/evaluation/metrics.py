"""Evaluation metrics mirroring the paper's Tables 2-9: classification
metrics (Eq. 45 F1, plus ROC-AUC/PR-AUC/RMSE), and control-quality metrics
(vulnerability reduction %, water-use efficiency %, early-detection lead
time) used to assess the SAC irrigation controller.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy / precision / recall / F1 (Eq. 45) / ROC-AUC / PR-AUC.

    y_true: binary health labels, y_prob: predicted probability of the
    positive ("healthy") class, both flattened to 1D.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    report = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }
    # ROC-AUC/PR-AUC need both classes present.
    if len(np.unique(y_true)) > 1:
        report["roc_auc"] = roc_auc_score(y_true, y_prob) * 100
        report["pr_auc"] = average_precision_score(y_true, y_prob) * 100
    else:
        report["roc_auc"] = float("nan")
        report["pr_auc"] = float("nan")
    return report


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_std_over_seeds(values: list[float]) -> tuple[float, float]:
    """mu +/- sigma over independent-seed runs, as reported in Section 4."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def vulnerability_reduction(initial_vulnerability: float, final_vulnerability: float) -> float:
    """Percentage reduction in yield vulnerability (Table 4 style metric)."""
    if initial_vulnerability <= 0:
        return 0.0
    return 100.0 * (initial_vulnerability - final_vulnerability) / initial_vulnerability


def water_use_efficiency_gain(baseline_water: float, water_used: float) -> float:
    """Percentage water saving relative to a baseline strategy (Table 5)."""
    if baseline_water <= 0:
        return 0.0
    return 100.0 * (baseline_water - water_used) / baseline_water


def early_detection_lead_time(event_step: int, first_alarm_step: int | None, step_duration_days: float = 1.0) -> float:
    """Days of lead time between the first raised alarm and the actual
    stress event (Table 6). Returns 0 if no alarm preceded the event."""
    if first_alarm_step is None or first_alarm_step > event_step:
        return 0.0
    return (event_step - first_alarm_step) * step_duration_days


def false_alarm_rate(alarms: np.ndarray, events: np.ndarray) -> float:
    """Fraction of raised alarms that did not precede a real stress event
    within the same window (Table 6)."""
    alarms = np.asarray(alarms).astype(bool)
    events = np.asarray(events).astype(bool)
    if alarms.sum() == 0:
        return 0.0
    false_alarms = alarms & ~events
    return 100.0 * false_alarms.sum() / alarms.sum()
