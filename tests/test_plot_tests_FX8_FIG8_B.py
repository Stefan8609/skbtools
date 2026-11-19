"""Helpers of FX-8, FIG-8 and appendix B figures (plot_tests/), plus FIG-8's plot."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot_tests import _figure_common as fc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plot_tests"))
import fig_FIG8_lever_per_dog as fig8  # noqa: E402


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


def test_fig8_lever_figure_from_tiny_fake_posteriors():
    rng = np.random.default_rng(0)
    centres = {
        1: [-12.77, 9.58, -12.07],
        3: [-12.89, 9.45, -12.28],
        4: [-13.4, 8.7, -13],
    }
    levers = {
        dog: rng.normal(centre, 0.01, (200, 3)) for dog, centre in centres.items()
    }
    joint = rng.normal([-12.99, 9.27, -14.13], 0.01, (200, 3))
    prior_center = np.array([-13.1, 9.7, -14.0])
    prior_sd = np.array([0.5, 0.5, 2.0])
    # paper_style.apply() changes global rcParams; keep them out of other tests.
    with matplotlib.rc_context():
        fig, spread_cm = fig8.lever_figure(levers, joint, prior_center, prior_sd)
        assert len(fig.axes) == 3
        plt.close(fig)
    expected = [
        100 * (np.ptp([levers[d][:, k].mean() for d in centres])) for k in range(3)
    ]
    np.testing.assert_allclose(spread_cm, expected)
