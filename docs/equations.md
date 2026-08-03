# Equation-to-code map

Maps every numbered equation in the manuscript to the module/function that
implements it. Section numbers refer to the paper's Section 3 (Methodology).

| Eq. | Description | Code |
|---|---|---|
| 1 | Unified zone-time dataset `D` | `data/dataset.py::build_zone_sequences` |
| 2 | Fused per-zone-timestep input `X_z,t` | `data/dataset.py::build_zone_sequences` (fused array) |
| 3-4 | Causal, zone-exclusive train/test split | `data/dataset.py::causal_split` |
| 5, 7, 12 | Stress-aware normalization (zone-specific baseline) | `data/normalization.py::stress_aware_normalize` |
| 6, 13 | Thermal stress deviation `T_canopy - T_ambient` | `data/normalization.py::thermal_stress_deviation` |
| 8-9 | Rolling zone-specific mean/std baseline | `data/normalization.py::rolling_zone_baseline` |
| 10 | Stress-event indicator | `data/temporal_alignment.py::event_indicator` |
| 11 | Event-anchored alignment operator | `data/temporal_alignment.py::align_to_events` |
| 14, 16 | Event-anchor selection (argmax gradient) | `data/temporal_alignment.py::event_anchor` |
| 15 | Affine spatial registration | *(documented, not modeled - no raw imagery to register; see docs/dataset.md #1)* |
| 17 | Adaptive spatial zoning / clustering | `data/spatial_zoning.py::cluster_into_zones` |
| 18 | Multi-modal fused feature vector `F_z^t` | `data/dataset.py::build_zone_sequences` (`fused`) |
| 19 | Temporal stress gradient `G_z^t` | `data/temporal_alignment.py::temporal_gradient` |
| 20-23 | Spatial attention + aggregation | `models/layers.py::SpatialAttention` |
| 24-25 | Temporal encoder (recurrent stress evolution) | `models/layers.py::TemporalEncoder` |
| 26-27 | Diffusion-aware stress propagation | `models/layers.py::DiffusionModule` |
| 28 | Vulnerability trajectory `V_t` | `models/vulnerability.py::vulnerability_trajectory` |
| 29, 31, 33, 46 | RL cognitive state `s_t` | `rl/environment.py::AgroIrrigationEnv._state_vector` |
| 30 | Vulnerability persistence `Vbar_t` | `models/vulnerability.py::vulnerability_persistence` |
| 32, 34, 35, 44 | SAC reward (vulnerability / water / delayed term) | `rl/environment.py::AgroIrrigationEnv.step` |
| 36-37 | SAC maximum-entropy objective `J(pi)` | `rl/sac.py::SACAgent.update` (critic/actor loss) |
| 38-39 | Environment state transition | `rl/environment.py::AgroIrrigationEnv.step` |
| 40-41 | Policy gradient update / optimal policy | `rl/sac.py::SACAgent.update` (`actor_optim.step()`) |
| 42-43 | Yield sustainability score | `evaluation/metrics.py::vulnerability_reduction` (analogous summary stat) |
| 45 | F1 score | `evaluation/metrics.py::classification_report` |
| 46-48 | Shared RL state / action space (fair baseline comparison) | `rl/environment.py`, `rl/baselines_rl.py::discretize_action_space` |

## Notes on the manuscript's equation numbering

While implementing this, we found the manuscript's own equation numbering to
be internally inconsistent (documented here for transparency, since this
repo necessarily had to pick *one* interpretation per concept):

- The reward formula `R_t = -alpha*V_t - beta*W_t + gamma*(V_{t-1} - V_{t+k})`
  is stated three times as Eq. (32), (35) and (44) with no difference between
  them. This repo implements it once, in `AgroIrrigationEnv.step`, using a
  causal reformulation (see `docs/dataset.md` item 5 and the docstring in
  `rl/environment.py`) since an online agent cannot observe `V_{t+k}` at
  decision time `t`.
- The RL state vector `s_t` is defined three separate times - Eq. (29), (31)
  and (33) - with slightly different variable lists; Eq. (46) gives a fourth,
  fuller version `[V_t, M_t, T_t, E_t, P_t]`. This repo standardizes on the
  Eq. (46) form since it is the version actually used for the fair
  cross-algorithm RL comparison described in Section 4.
- Text preceding Eq. (20) says "Equation (8) defines spatial attention
  weighting", but Eq. (8) is the normalization baseline formula defined
  earlier in Section 3.2 - the attention weighting is actually Eq. (20).
  Eq. (20) and (21) also both define the same attention coefficient with
  different notation (projected `W_q h_i`/`W_k h_j` vs. raw dot product
  `F_i . F_j`). This repo implements the projected form (Eq. 20), since it is
  the learnable version consistent with the rest of the architecture.

None of this blocks a working implementation - it just means "faithful to
the paper" required choosing one internally-consistent reading per equation
rather than a single unambiguous source of truth. See the manuscript review
notes shared earlier in this project for the full list.
