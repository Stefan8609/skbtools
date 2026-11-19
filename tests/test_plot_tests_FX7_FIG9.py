"""Helpers of plot_tests/_figure_common.py used by FX-7 (corner) and FIG-9 (MCMC)."""

import numpy as np
import pytest

from plot_tests import _figure_common as fc


def test_linear_gaussian_prior_of_a_difference():
    # q = 2 theta_0 - theta_2 + 1; theta_1 has no Gaussian prior and is unused.
    mean, sd = fc.linear_gaussian_prior(
        [2.0, 0.0, -1.0], [1.0, np.nan, 3.0], [0.5, np.nan, 2.0], constant=1.0
    )
    assert mean == pytest.approx(2.0 * 1.0 - 3.0 + 1.0)
    assert sd == pytest.approx(np.sqrt(1.0**2 + 2.0**2))


def test_linear_gaussian_prior_rejects_parameter_without_prior():
    with pytest.raises(ValueError, match="without a Gaussian prior"):
        fc.linear_gaussian_prior([1.0, 1.0], [0.0, np.nan], [1.0, np.nan])


def test_linear_gaussian_prior_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="equal 1-D"):
        fc.linear_gaussian_prior([1.0, 1.0], [0.0], [1.0])


def test_density_levels_hold_the_requested_mass():
    density = np.array([[5.0, 3.0], [1.0, 1.0]])  # total 10
    level_50, level_80, level_95 = fc.density_levels(density, [0.5, 0.8, 0.95])
    assert level_50 == 5.0  # {5} holds 50 %
    assert level_80 == 3.0  # {5, 3} holds 80 %
    assert level_95 == 1.0  # needs every cell


def test_density_levels_rejects_bad_input():
    with pytest.raises(ValueError, match="non-negative"):
        fc.density_levels(np.array([1.0, -1.0]), [0.5])
    with pytest.raises(ValueError, match="probability"):
        fc.density_levels(np.ones(3), [1.0])
