"""Baseline stress-forecasting models used for the Table 2-style comparison:
Logistic Regression, Decision Tree, Random Forest, (optionally) XGBoost, a
plain CNN, a CNN-LSTM hybrid, and a temporal Transformer encoder.

These are intentionally simple, standard implementations - the paper's
contribution is GeoSpatio-TRiNet (models/geospatio_trinet.py); the baselines
exist purely so `training/train_forecasting.py` can produce a comparable
leaderboard the way Table 2 in the manuscript does.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False


def flatten_sequences(features: np.ndarray, labels: np.ndarray):
    """(B, T, D), (B, T) -> (B*T, D), (B*T,) for classical ML baselines that
    treat every timestep as an i.i.d. observation (no temporal modeling)."""
    b, t, d = features.shape
    return features.reshape(b * t, d), labels.reshape(b * t)


class SklearnBaseline:
    """Thin wrapper so classical sklearn/xgboost models share a .fit/.predict_proba
    interface with the torch baselines below."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, features: np.ndarray, labels: np.ndarray):
        x, y = flatten_sequences(features, labels)
        self.estimator.fit(x, y)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        b, t, d = features.shape
        x = features.reshape(b * t, d)
        proba = self.estimator.predict_proba(x)[:, 1]
        return proba.reshape(b, t)


def build_logistic_regression() -> SklearnBaseline:
    from sklearn.linear_model import LogisticRegression

    return SklearnBaseline(LogisticRegression(max_iter=1000))


def build_decision_tree() -> SklearnBaseline:
    from sklearn.tree import DecisionTreeClassifier

    return SklearnBaseline(DecisionTreeClassifier(max_depth=8, random_state=42))


def build_random_forest() -> SklearnBaseline:
    from sklearn.ensemble import RandomForestClassifier

    return SklearnBaseline(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))


def build_xgboost() -> SklearnBaseline | None:
    if not HAS_XGBOOST:
        return None
    return SklearnBaseline(
        xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss")
    )


class CNNBaseline(nn.Module):
    """1D-CNN over the temporal axis - spatial (feature) patterns only, no
    explicit long-range temporal modeling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T) for Conv1d -> back to (B, T)
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.head(h).squeeze(-1)


class CNNLSTMBaseline(nn.Module):
    """CNN feature extractor followed by an LSTM for temporal modeling."""

    def __init__(self, input_dim: int, cnn_hidden: int = 64, lstm_hidden: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.head(h).squeeze(-1)


class TemporalTransformerBaseline(nn.Module):
    """A standard Transformer encoder over the time axis - no explicit
    inter-zone spatial attention or diffusion, unlike GeoSpatio-TRiNet."""

    def __init__(self, input_dim: int, model_dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=model_dim * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


TORCH_BASELINES = {
    "cnn": CNNBaseline,
    "cnn_lstm": CNNLSTMBaseline,
    "temporal_transformer": TemporalTransformerBaseline,
}
