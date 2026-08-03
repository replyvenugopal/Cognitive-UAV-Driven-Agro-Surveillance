import numpy as np

from agrosurveillance.data.dataset import build_zone_sequences
from agrosurveillance.data.schema import DEFAULT_SCHEMA, derive_health_label
from agrosurveillance.data.synthetic import generate_synthetic_dataset
from agrosurveillance.rl.environment import AgroIrrigationEnv, zone_sequence_to_env_data


def _make_env():
    df = generate_synthetic_dataset(n_fields=2, rows_per_field=40, seed=3)
    df[DEFAULT_SCHEMA.health_label_col] = derive_health_label(df[DEFAULT_SCHEMA.stress_indicator_col])
    seqs = build_zone_sequences(df, sequence_length=16, baseline_window=4)
    zdata = zone_sequence_to_env_data(seqs[0])
    return AgroIrrigationEnv(zdata, max_irrigation=10.0, seed=0)


def test_env_reset_and_step():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape

    next_obs, reward, term, trunc, info = env.step(np.array([5.0]))
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert "vulnerability" in info


def test_more_irrigation_raises_moisture():
    env = _make_env()
    env.reset()
    _, _, _, _, info_low = env.step(np.array([0.0]))
    env.reset()
    _, _, _, _, info_high = env.step(np.array([10.0]))
    assert info_high["moisture"] >= info_low["moisture"]


def test_episode_truncates_at_sequence_end():
    env = _make_env()
    env.reset()
    truncated = False
    for _ in range(env.t_len + 5):
        _, _, terminated, truncated, _ = env.step(np.array([1.0]))
        if terminated or truncated:
            break
    assert truncated
