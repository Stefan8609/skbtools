"""Helpers of plot_tests/_figure_common.py behind FIG-3 (clock offset) and FX-4."""

import numpy as np
import pytest
from pymap3d import geodetic2ecef

from plot_tests import _figure_common as fc


def test_wrap_to_period_maps_into_half_open_interval():
    wrapped = fc.wrap_to_period([0.2, 0.7, -0.6, 3.0, 0.5])
    np.testing.assert_allclose(wrapped, [0.2, -0.3, 0.4, 0.0, -0.5], atol=1e-12)


def synthetic_pings(true_dt_s=1234.25, n=200):
    """1 Hz emissions, travel time 4 s +- 0.5 s, arrivals tagged on the DOG clock."""
    gps = np.arange(n, dtype=float)
    travel = 4.0 + 0.5 * np.sin(2.0 * np.pi * gps / 60.0)
    arrival = gps + travel + true_dt_s
    tags = np.column_stack([np.floor(arrival), arrival - np.floor(arrival)])
    return tags, gps, travel


def test_alignment_scan_circular_variance_ignores_fractional_dt():
    tags, gps, travel = synthetic_pings()
    trial = 1234.25 + np.array([-0.3, 0.0, 0.3, 1.0])
    cv, rmse_s, n_pairs = fc.alignment_scan(trial, tags, gps, travel)
    # A fractional error shifts every residual equally: same circular variance.
    np.testing.assert_allclose(cv[:3], cv[1], atol=1e-12)
    np.testing.assert_allclose(rmse_s[:3], [0.3, 0.0, 0.3], atol=1e-9)
    # One whole second pairs each arrival with the next pulse: residuals spread.
    assert cv[3] > cv[1] + 1e-4
    assert n_pairs[1] == 200


def test_alignment_scan_rejects_wrong_shapes():
    tags, gps, travel = synthetic_pings()
    with pytest.raises(ValueError, match="CDOG_data must be"):
        fc.alignment_scan([0.0], tags[:, 0], gps, travel)
    with pytest.raises(ValueError, match="must be equal 1-D"):
        fc.alignment_scan([0.0], tags, gps, travel[:-1])


def test_line_fit_recovers_slope_and_skips_nan():
    x = np.array([0.0, 1.0, 2.0, np.nan, 4.0])
    slope, intercept, n = fc.line_fit(x, 2.0 * np.nan_to_num(x) - 1.0)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(-1.0)
    assert n == 4


def test_line_fit_needs_two_points():
    with pytest.raises(ValueError, match="two finite"):
        fc.line_fit([1.0, np.nan], [1.0, 2.0])


def test_vertical_travel_time_of_point_above_dog():
    dog = np.array(geodetic2ecef(31.45, -68.70, -5000.0))
    above = np.array(geodetic2ecef(31.45, -68.70, -1970.0))
    np.testing.assert_allclose(
        fc.vertical_travel_time_s(dog, above[None, :], 1515.0), [2.0], rtol=1e-9
    )
