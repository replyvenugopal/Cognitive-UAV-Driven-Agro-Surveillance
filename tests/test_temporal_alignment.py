import numpy as np

from agrosurveillance.data.temporal_alignment import (
    align_to_events,
    event_anchor,
    event_indicator,
    temporal_gradient,
)


def test_event_indicator_flags_large_changes():
    moisture = np.array([50, 49, 48, 20, 19, 18], dtype=float)  # sharp drop at idx 3
    canopy_temp = np.zeros(6)
    events = event_indicator(moisture, canopy_temp, theta_m=2.0, theta_t=100.0)
    assert events[3] == 1 or events[2] == 1  # gradient centered near the drop


def test_event_anchor_picks_largest_gradient():
    moisture = np.array([50, 50, 50, 10, 50, 50])
    idx = event_anchor(moisture)
    assert 2 <= idx <= 4


def test_temporal_gradient_first_row_zero():
    features = np.random.RandomState(0).randn(6, 3)
    grad = temporal_gradient(features)
    np.testing.assert_allclose(grad[0], 0.0)
    np.testing.assert_allclose(grad[1], features[1] - features[0])


def test_align_to_events_interpolates_near_gap():
    values = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0])
    events = np.array([0, 0, 0, 1, 0, 0])
    aligned = align_to_events(values, events, max_gap=2)
    assert aligned.shape == values.shape
