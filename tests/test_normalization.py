import numpy as np

from agrosurveillance.data.normalization import (
    global_normalize,
    rolling_zone_baseline,
    stress_aware_normalize,
    thermal_stress_deviation,
)


def test_rolling_zone_baseline_shapes():
    values = np.random.RandomState(0).randn(10, 3)
    mu, sigma = rolling_zone_baseline(values, window=4)
    assert mu.shape == values.shape
    assert sigma.shape == values.shape
    assert np.all(sigma >= 0)


def test_stress_aware_normalize_no_nans():
    values = np.random.RandomState(1).randn(20, 2) * 5 + 10
    out = stress_aware_normalize(values, window=5)
    assert out.shape == values.shape
    assert not np.isnan(out).any()


def test_global_normalize_zero_mean_unit_ish_std():
    values = np.random.RandomState(2).randn(500, 4) * 3 + 7
    out = global_normalize(values)
    assert abs(out.mean()) < 0.1
    assert abs(out.std() - 1.0) < 0.1


def test_global_normalize_accepts_window_kwarg():
    values = np.random.RandomState(3).randn(10, 2)
    # Must be interchangeable with stress_aware_normalize at call sites.
    out = global_normalize(values, window=4)
    assert out.shape == values.shape


def test_thermal_stress_deviation():
    canopy = np.array([30.0, 32.0, 28.0])
    ambient = np.array([25.0, 25.0, 25.0])
    dev = thermal_stress_deviation(canopy, ambient)
    np.testing.assert_allclose(dev, [5.0, 7.0, 3.0])
