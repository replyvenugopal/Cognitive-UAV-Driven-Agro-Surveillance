"""GeoSpatio-TRiNet: the paper's central predictive model (Section 3.6).

Pipeline: input projection -> spatial attention (Eq. 20-23) -> temporal
encoder (Eq. 24-25) -> diffusion module (Eq. 26-27) -> stress + vulnerability
heads (Eq. 28, 30). Each stage can be individually disabled to reproduce the
ablation study in Table 8 (see `disable` argument).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DiffusionModule, SpatialAttention, TemporalEncoder
from .vulnerability import vulnerability_persistence, vulnerability_trajectory

ABLATION_COMPONENTS = ("spatial", "temporal", "diffusion")


class GeoSpatioTriNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 64,
        spatial_heads: int = 4,
        temporal_hidden: int = 128,
        temporal_layers: int = 2,
        diffusion_steps: int = 2,
        diffusion_lambda: float = 0.35,
        dropout: float = 0.1,
        vulnerability_eta1: float = 0.6,
        vulnerability_eta2: float = 0.4,
        vulnerability_decay: float = 0.85,
        vulnerability_horizon: int = 5,
        disable: tuple[str, ...] = (),
    ):
        super().__init__()
        for name in disable:
            if name not in ABLATION_COMPONENTS:
                raise ValueError(f"Unknown ablation component '{name}', expected one of {ABLATION_COMPONENTS}")
        self.disable = set(disable)

        self.input_proj = nn.Linear(input_dim, feature_dim)
        self.spatial_attn = SpatialAttention(feature_dim, heads=spatial_heads, dropout=dropout)
        self.temporal_encoder = TemporalEncoder(feature_dim, temporal_hidden, temporal_layers, dropout)

        diffusion_input_dim = temporal_hidden if "temporal" not in self.disable else feature_dim
        self.diffusion = DiffusionModule(diffusion_input_dim, steps=diffusion_steps, diffusion_lambda=diffusion_lambda)

        head_input_dim = diffusion_input_dim
        self.stress_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim // 2, 1),
        )

        self.vuln_eta1 = vulnerability_eta1
        self.vuln_eta2 = vulnerability_eta2
        self.vuln_decay = vulnerability_decay
        self.vuln_horizon = vulnerability_horizon

    def forward(
        self,
        features: torch.Tensor,
        adjacency_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """features: (B_zones, T, D_in). Returns a dict of model outputs.

        NOTE: B_zones here is the batch dimension, and doubles as the "zone"
        axis that spatial attention / diffusion operate over (see layers.py
        docstring for the neighbor-set modeling assumption).
        """
        h = self.input_proj(features)  # (B, T, feature_dim)

        if "spatial" not in self.disable:
            h_spatial = self.spatial_attn(h, adjacency_mask)
            h = h + h_spatial  # residual spatial aggregation (Eq. 22/23)

        if "temporal" not in self.disable:
            h_temporal = self.temporal_encoder(h)  # (B, T, temporal_hidden), Eq. 24/25
        else:
            h_temporal = h

        if "diffusion" not in self.disable:
            diffusion_state = self.diffusion(h_temporal, adjacency_mask)  # Eq. 26/27
        else:
            diffusion_state = h_temporal

        stress_logits = self.stress_head(diffusion_state).squeeze(-1)  # (B, T)
        stress_prob = torch.sigmoid(stress_logits)

        vulnerability = vulnerability_trajectory(
            stress_prob, self.vuln_eta1, self.vuln_eta2, self.vuln_decay, self.vuln_horizon
        )  # Eq. 28
        persistence = vulnerability_persistence(vulnerability, self.vuln_decay, self.vuln_horizon)  # Eq. 30

        if valid_mask is not None:
            stress_logits = stress_logits * valid_mask
            stress_prob = stress_prob * valid_mask
            vulnerability = vulnerability * valid_mask
            persistence = persistence * valid_mask

        return {
            "stress_logits": stress_logits,
            "stress_prob": stress_prob,
            "vulnerability": vulnerability,
            "vulnerability_persistence": persistence,
            "diffusion_state": diffusion_state,
            "temporal_state": h_temporal,
        }
