"""Helpers of plot_tests/_figure_common.py (vessel frame, intervals, ranging sd)."""

import numpy as np
import pytest

from plot_tests import _figure_common as fc


def test_receivers_in_body_frame_undoes_a_rigid_motion():
    grid = np.array(
        [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.03], [-12.1, -0.9, 0.0], [-8.7, 5.2, 0.02]]
    )
    angle = np.deg2rad(40.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shift = np.array([1.9e6, -5.0e6, 3.3e6])
    epochs = np.stack([grid @ rotation.T + shift, grid @ rotation + 2 * shift])
    body = fc.receivers_in_body_frame(epochs, grid)
    np.testing.assert_allclose(body, np.stack([grid, grid]), atol=1e-6)
    with pytest.raises(ValueError, match="need"):
        fc.receivers_in_body_frame(epochs[:, :3], grid)


def test_central_interval_of_a_uniform_grid():
    mean, low, high = fc.central_interval(np.arange(101.0), 0.9)
    assert mean == pytest.approx(50.0)
    assert (low, high) == pytest.approx((5.0, 95.0))


def test_mean_with_ci_matches_normal_theory():
    values = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0], [7.0, 10.0]])
    mean, half = fc.mean_with_ci(values, 0.95)
    np.testing.assert_allclose(mean, [4.0, 10.0])
    sd = np.std([1.0, 3.0, 5.0, 7.0], ddof=1)
    np.testing.assert_allclose(half, [1.959964 * sd / 2.0, 0.0], rtol=1e-6)
    with pytest.raises(ValueError, match="NaN"):
        fc.mean_with_ci([1.0, np.nan, 2.0])


def test_b36_range_sd_limits_and_gain():
    assert fc.b36_range_sd_cm(0.0, 20e-6) == pytest.approx(20e-6 * 1515.0 * 100.0)
    assert fc.b36_range_sd_cm(0.02, 0.0) == pytest.approx(2.0)
    assert fc.b36_range_sd_cm(0.02, 0.0, gain=1.6) == pytest.approx(3.2)
