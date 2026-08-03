"""Artificial agro-environment simulator for adaptive irrigation control
(Section 3.8-3.9, Eq. 29, 31-35, 38-39, 44, 46-47).

The paper states the SAC agent "is interfaced with an artificial
agro-environment that is created on the basis of the dataset" (Section 4) -
this module is that simulator. Each episode replays one zone's observed
stress/moisture/thermal/environmental trajectory as an exogenous stress
driver, while the agent's irrigation actions perturb soil moisture and, in
turn, the simulated vulnerability trajectory.

State (Eq. 46):  s_t = [V_t, M_t, T_t, E_t, P_t]
    V_t - current vulnerability, M_t - soil moisture, T_t - thermal
    deviation, E_t - environmental vector, P_t - vulnerability persistence.

Action (Eq. 47): a_t in [0, a_max], continuous irrigation intensity.

Reward (Eq. 44): R_t = -alpha*V_t - beta*W_t + gamma*(V_{t-k} - V_t)
    A causal reformulation is used here: since an online agent cannot see
    V_{t+k} at decision time t, the "delayed vulnerability reduction" term
    is computed retrospectively as the drop in vulnerability over the last
    k steps (V_{t-k} - V_t), which rewards sustained improvement rather than
    instantaneous vulnerability dips - matching the paper's stated intent
    of discouraging oscillatory, reflexive irrigation (Section 3.9).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


@dataclass
class ZoneEnvironmentData:
    """Exogenous trajectory driving one episode (from a real or synthetic
    ZoneSequence). All arrays share shape (T,) except `env` which is (T, D_env)."""

    base_stress: np.ndarray          # 0-1 normalized exogenous stress pressure
    soil_moisture: np.ndarray        # raw soil moisture level (0-100 scale)
    thermal_deviation: np.ndarray    # canopy - ambient temperature deviation
    env: np.ndarray                  # (T, D_env) weather/environment vector
    rainfall: np.ndarray = field(default=None)  # optional, mm per step


class AgroIrrigationEnv(gym.Env if gym is not None else object):
    """Single-zone adaptive irrigation control environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        zone_data: ZoneEnvironmentData,
        max_irrigation: float = 10.0,
        alpha: float = 1.0,
        beta: float = 0.4,
        gamma: float = 0.6,
        delayed_horizon: int = 3,
        moisture_target: float = 55.0,
        irrigation_efficiency: float = 0.9,
        evapotranspiration_coef: float = 0.05,
        vuln_eta1: float = 0.6,
        vuln_eta2: float = 0.4,
        vuln_decay: float = 0.85,
        seed: int | None = None,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for AgroIrrigationEnv")
        super().__init__()
        self.zone_data = zone_data
        self.t_len = len(zone_data.base_stress)
        self.max_irrigation = max_irrigation
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.delayed_horizon = delayed_horizon
        self.moisture_target = moisture_target
        self.irrigation_efficiency = irrigation_efficiency
        self.evapotranspiration_coef = evapotranspiration_coef
        self.vuln_eta1, self.vuln_eta2, self.vuln_decay = vuln_eta1, vuln_eta2, vuln_decay

        env_dim = zone_data.env.shape[1]
        state_dim = 1 + 1 + 1 + env_dim + 1  # V, M, T, E, P (Eq. 46)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=max_irrigation, shape=(1,), dtype=np.float32)

        self._rng = np.random.RandomState(seed)
        self.reset()

    # ------------------------------------------------------------------
    def _compute_vulnerability(self, moisture: float, base_stress: float) -> float:
        deficit = max(0.0, self.moisture_target - moisture) / max(self.moisture_target, 1e-6)
        instantaneous = np.clip(self.vuln_eta1 * base_stress + self.vuln_eta2 * deficit, 0.0, 1.0)
        return float(instantaneous)

    def _state_vector(self) -> np.ndarray:
        t = self.t
        persistence = float(np.mean(self._vuln_history)) if self._vuln_history else 0.0
        env_vec = self.zone_data.env[t]
        return np.concatenate(
            [
                np.array([self._vulnerability, self._moisture, self.zone_data.thermal_deviation[t]], dtype=np.float32),
                env_vec.astype(np.float32),
                np.array([persistence], dtype=np.float32),
            ]
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.t = 0
        self._moisture = float(self.zone_data.soil_moisture[0])
        self._vulnerability = self._compute_vulnerability(self._moisture, float(self.zone_data.base_stress[0]))
        self._vuln_history = deque(maxlen=self.delayed_horizon + 1)
        self._vuln_history.append(self._vulnerability)
        obs = self._state_vector()
        return obs, {}

    def step(self, action):
        action = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, self.max_irrigation))
        t = self.t

        rainfall = float(self.zone_data.rainfall[t]) if self.zone_data.rainfall is not None else 0.0
        evapotranspiration = self.evapotranspiration_coef * (25.0 + self.zone_data.thermal_deviation[t])

        next_moisture = self._moisture + self.irrigation_efficiency * action - evapotranspiration + 0.1 * rainfall
        next_moisture = float(np.clip(next_moisture, 0.0, 100.0))

        next_t = min(t + 1, self.t_len - 1)
        next_base_stress = float(self.zone_data.base_stress[next_t])
        next_vulnerability = self._compute_vulnerability(next_moisture, next_base_stress)

        vuln_before_horizon = self._vuln_history[0]  # V_{t-k} (or earliest available)
        delayed_term = vuln_before_horizon - next_vulnerability

        reward = -self.alpha * next_vulnerability - self.beta * action + self.gamma * delayed_term

        self._moisture = next_moisture
        self._vulnerability = next_vulnerability
        self._vuln_history.append(next_vulnerability)
        self.t = next_t

        terminated = False
        truncated = self.t >= self.t_len - 1
        obs = self._state_vector()
        info = {
            "vulnerability": next_vulnerability,
            "moisture": next_moisture,
            "irrigation": action,
        }
        return obs, float(reward), terminated, truncated, info


def zone_sequence_to_env_data(seq) -> ZoneEnvironmentData:
    """Adapts a `agrosurveillance.data.dataset.ZoneSequence` into the
    exogenous driver format expected by `AgroIrrigationEnv`."""
    base_stress = np.clip(seq.stress_label / 100.0, 0.0, 1.0)
    soil_moisture = seq.soil[:, 0] if seq.soil.ndim > 1 else seq.soil
    # `seq.soil` is stress-aware normalized; rescale to an approximate 0-100
    # moisture percentage so the simulator's physical bounds are meaningful.
    soil_moisture = np.clip(50.0 + 15.0 * soil_moisture, 0.0, 100.0)
    return ZoneEnvironmentData(
        base_stress=base_stress,
        soil_moisture=soil_moisture,
        thermal_deviation=seq.thermal_deviation,
        env=seq.env,
    )
