"""Building blocks for GeoSpatio-TRiNet: spatial attention, temporal encoder,
and diffusion-aware stress propagation (Section 3.6, Eq. 20-27).

Modeling note on "neighboring zones" N(z): the public tabular dataset has no
GPS coordinates, so there is no ground-truth spatial adjacency to condition
on. We treat all other zones present in the same training batch as the
candidate neighborhood by default (an unrestricted attention over zones,
which the model itself learns to weight via Eq. 20/21), and optionally accept
an explicit adjacency mask (e.g. from `spatial_zoning.zone_adjacency`) to
restrict attention to a fixed neighbor set when one is available. This keeps
Eq. 22/23 (spatial aggregation) and Eq. 26/27 (diffusion) faithful to the
paper's formulation while remaining runnable on the real dataset's schema.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Inter-zone spatial attention (Eq. 20-23).

    alpha_ij = softmax_j( (W_q h_i)^T (W_k h_j) )     (Eq. 20)
    H_i = sum_{j in N(i)} alpha_ij (W_v h_j)           (Eq. 22/23)

    Operates independently at every timestep across the zone (batch) axis.
    """

    def __init__(self, feature_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % heads == 0, "feature_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.w_q = nn.Linear(feature_dim, feature_dim)
        self.w_k = nn.Linear(feature_dim, feature_dim)
        self.w_v = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adjacency_mask: torch.Tensor | None = None) -> torch.Tensor:
        """h: (B_zones, T, D). adjacency_mask: optional (B_zones, B_zones) with
        1 where zone j is a neighbor of zone i, 0 otherwise (self excluded).
        Returns spatially-aggregated features of shape (B_zones, T, D).
        """
        b, t, d = h.shape
        q = self.w_q(h).view(b, t, self.heads, self.head_dim)
        k = self.w_k(h).view(b, t, self.heads, self.head_dim)
        v = self.w_v(h).view(b, t, self.heads, self.head_dim)

        # (heads, T, B, head_dim) so attention is computed across the zone axis.
        q = q.permute(2, 1, 0, 3)
        k = k.permute(2, 1, 0, 3)
        v = v.permute(2, 1, 0, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (heads, T, B, B)

        if adjacency_mask is not None:
            mask = adjacency_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            # Exclude self-loops (a zone is not its own "neighbor" in Eq. 22).
            eye = torch.eye(b, device=h.device, dtype=torch.bool)
            scores = scores.masked_fill(eye.unsqueeze(0).unsqueeze(0), float("-inf"))

        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha)  # rows with no valid neighbor -> all -inf -> softmax nan
        alpha = self.dropout(alpha)

        agg = torch.matmul(alpha, v)                    # (heads, T, B, head_dim)
        agg = agg.permute(2, 1, 0, 3).reshape(b, t, d)   # (B, T, D)
        return self.out_proj(agg)


class TemporalEncoder(nn.Module):
    """Long-range temporal stress evolution encoder (Eq. 24-25).

    H_t^temp = f(W_t X_t + U_t H_{t-1} + b_t), implemented with a GRU stack
    which is the standard gated realization of that recurrence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_in) -> (B, T, hidden_dim)."""
        out, _ = self.gru(x)
        return out


class DiffusionModule(nn.Module):
    """Diffusion-aware stress propagation across neighboring zones (Eq. 26-27).

    D_i^{t+1} = D_i^t + lambda * sum_{j in N(i)} w_ij (D_j^t - D_i^t)

    Learnable pairwise weights w_ij are derived from a similarity projection
    of each zone's temporal state, and the update is applied for a fixed
    number of diffusion steps at every timestep independently.
    """

    def __init__(self, hidden_dim: int, steps: int = 2, diffusion_lambda: float = 0.35):
        super().__init__()
        self.steps = steps
        self.diffusion_lambda = diffusion_lambda
        self.weight_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, temporal_state: torch.Tensor, adjacency_mask: torch.Tensor | None = None) -> torch.Tensor:
        """temporal_state: (B_zones, T, D). Returns diffused state, same shape."""
        b, t, d = temporal_state.shape
        d_state = temporal_state

        proj = self.weight_proj(temporal_state)  # (B, T, D)
        # Pairwise similarity -> softmax-normalized diffusion weights w_ij (per timestep).
        sim = torch.einsum("btd,jtd->btj", proj, proj) / (d ** 0.5)  # (B, T, B)

        if adjacency_mask is not None:
            mask = adjacency_mask.to(dtype=torch.bool)
            sim = sim.masked_fill(~mask.unsqueeze(1), float("-inf"))
        else:
            eye = torch.eye(b, device=temporal_state.device, dtype=torch.bool)
            sim = sim.masked_fill(eye.unsqueeze(1), float("-inf"))

        w = F.softmax(sim, dim=-1)
        w = torch.nan_to_num(w)  # (B, T, B_neighbor)

        for _ in range(self.steps):
            # sum_j w_ij (D_j - D_i) for every zone i, at every timestep.
            neighbor_state = torch.einsum("btj,jtd->btd", w, d_state)
            neighbor_avg_weight = w.sum(dim=-1, keepdim=True)  # (B, T, 1)
            delta = neighbor_state - neighbor_avg_weight * d_state
            d_state = d_state + self.diffusion_lambda * delta

        return d_state
