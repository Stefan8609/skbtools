"""Helpers behind the numbers printed on the residual and delta-c figures."""

import numpy as np
import pytest

from plot_tests import _figure_common as fc


def test_window_steps_mean_and_sd():
    draws = np.array([[0.0, 1.0, 3.0], [0.0, 3.0, 3.0]])
    mean, sd = fc.window_steps(draws)
    np.testing.assert_allclose(mean, [2.0, 1.0])
    np.testing.assert_allclose(sd, [1.0, 1.0])


def test_window_steps_rejects_single_window():
    with pytest.raises(ValueError, match="n_draws"):
        fc.window_steps(np.zeros((5, 1)))


def test_level_shifts_finds_one_step():
    times = np.arange(600.0)
    values = np.where(times < 300, 0.0, 0.5)
    shifts, event_times, event_shifts = fc.level_shifts(times, values, 60.0, 0.2)
    np.testing.assert_allclose(event_shifts, [0.5])
    assert 295 <= event_times[0] <= 305
    assert np.isnan(shifts[:20]).all() and shifts[100] == 0.0


def test_level_shifts_needs_samples_on_both_sides():
    times = np.arange(100.0)
    values = np.where(times < 50, 0.0, 1.0)
    _, event_times, _ = fc.level_shifts(times, values, 60.0, 0.2, min_count=60)
    assert event_times.size == 0


def test_level_shifts_separates_two_events_and_signs():
    times = np.arange(2000.0)
    values = np.select([times < 500, times < 1500], [0.0, -0.4], 0.3)
    _, event_times, event_shifts = fc.level_shifts(times, values, 60.0, 0.2)
    np.testing.assert_allclose(event_shifts, [-0.4, 0.7])
    assert event_times[0] < 600 < 1400 < event_times[1]


def test_level_shifts_rejects_unsorted_times():
    with pytest.raises(ValueError, match="increasing"):
        fc.level_shifts([0.0, 2.0, 1.0], [0.0, 0.0, 0.0], 1.0, 0.1)


def test_antenna4_height_anomaly_cancels_ship_motion():
    grid = np.array(
        [[0, 0, 0], [-2.4, -4.2, 0.0], [-12.1, -0.9, 0.0], [-8.7, 5.2, 0.0]]
    )
    angle = np.deg2rad(30.0)  # yaw plus a tilt about x
    yaw = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    c, s = np.cos(0.1), np.sin(0.1)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    rotation = yaw @ tilt
    epoch = grid @ rotation.T + np.array([100.0, -50.0, 7.0])
    up = np.array([0.0, 0.0, 1.0])
    moved = epoch.copy()
    moved[3] += 0.3 * up
    anomaly = fc.antenna4_height_anomaly(np.stack([epoch, moved]), grid, up)
    np.testing.assert_allclose(anomaly, [0.0, 0.3], atol=1e-9)


def test_antenna4_height_anomaly_rejects_three_receivers():
    with pytest.raises(ValueError, match="receivers"):
        fc.antenna4_height_anomaly(np.zeros((2, 3, 3)), np.zeros((4, 3)), [0, 0, 1])
