"""Pure helpers of plot_tests/_figure_common.py (numbers printed on the figures)."""

import numpy as np
import pytest

from plot_tests import _figure_common as fc


def test_signed_uses_unicode_minus():
    assert fc.signed(-0.4712) == "−0.47"
    assert fc.signed(0.4712) == "+0.47"


def test_ellipse_points_lie_on_the_probability_contour():
    cov = np.array([[4.0, 1.2], [1.2, 1.0]])
    mean = np.array([1.0, -2.0])
    boundary = fc.ellipse_points(mean, cov, 0.95, n=50)
    d = boundary - mean
    mahalanobis2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)
    np.testing.assert_allclose(mahalanobis2, -2.0 * np.log(0.05), rtol=1e-10)


def test_ellipse_points_rejects_wrong_shape():
    with pytest.raises(ValueError, match="cov must be"):
        fc.ellipse_points([0, 0], np.eye(3), 0.68)


def test_correlation_of_linear_relation():
    x = np.arange(10.0)
    assert fc.correlation(x, -3 * x + 2) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        fc.correlation(x, x[:5])


def test_autocorrelation_of_alternating_series():
    x = np.tile([1.0, -1.0], 50)
    acf = fc.autocorrelation(x, 2)
    assert acf[0] == pytest.approx(1.0)
    assert acf[1] == pytest.approx(-0.99)  # biased estimator: (n - 1) / n
    assert acf[2] == pytest.approx(0.98)


def test_coverage_fraction_counts_inside_interval():
    errors = np.array([0.5, -1.5, 2.5, -0.1])
    sds = np.ones(4)
    assert fc.coverage_fraction(errors, sds, 0.68) == pytest.approx(0.5)
    assert fc.coverage_fraction(errors, sds, 0.95) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="positive"):
        fc.coverage_fraction(errors, np.zeros(4), 0.68)


def test_binned_percentiles_and_empty_bins():
    x = np.array([0.1, 0.2, 0.3, 2.5])
    y = np.array([1.0, 2.0, 3.0, 10.0])
    centres, values, counts = fc.binned_percentiles(x, y, [0, 1, 2, 3], (50.0,))
    np.testing.assert_allclose(centres, [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(counts, [3, 0, 1])
    assert values[0, 0] == 2.0 and np.isnan(values[1, 0]) and values[2, 0] == 10.0


def test_circular_variance_aligned_and_uniform():
    assert fc.circular_variance([0.3, 1.3, -0.7]) == pytest.approx(0.0, abs=1e-12)
    assert fc.circular_variance(np.arange(10) / 10) == pytest.approx(1.0)


def test_rejection_counts_any_receiver_rule():
    masks = np.array([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=bool)
    per_receiver, combined = fc.rejection_counts(masks)
    np.testing.assert_array_equal(per_receiver, [2, 2])
    assert combined == 3


def test_elevation_straight_up_and_horizontal():
    origin = np.array([6378137.0, 0.0, 0.0])  # equator, lon 0: up is +x
    assert fc.elevation_deg(origin, origin + [100.0, 0, 0])[0] == pytest.approx(90.0)
    assert fc.elevation_deg(origin, origin + [0, 100.0, 0])[0] == pytest.approx(0.0)
