import numpy as np
import pytest

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion import Numba_xAline_Geiger_bias
from Inversion_Workflow.Inversion.Numba_xAline import find_int_offset
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    gauss_newton_offset_search,
    lever_basis,
    lever_bias_geiger,
    misfit_rms,
)
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned

# Same smooth synthetic ESV table and geometry as test_synthetic_workflow.
DZ = np.arange(5100.0, 5400.0, 1.0)
ANGLE = np.arange(0.0, 90.01, 1.0)
ESV = 1500.0 + 0.02 * (DZ[:, None] - 5200.0) + 0.5 * ANGLE[None, :] / 90.0
OTHERS = np.array(
    [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.03], [-12.1, -0.95, 0.004], [-8.7, 5.17, 0.025]]
)
LEVER = np.array([-12.5, 9.6, -13.3])
OFFSET, TIME_BIAS = 3141.5926, 0.2
# Clock offset dt recovered by the inversion.
TRUE_DT = OFFSET + TIME_BIAS


def _generate(time_noise, position_noise, seed=0):
    np.random.seed(seed)
    return generateUnaligned(
        2000,
        time_noise,
        position_noise,
        OFFSET,
        0.0,
        TIME_BIAS,
        DZ,
        ANGLE,
        ESV,
        gps1_to_others=OTHERS,
        gps1_to_transponder=LEVER,
    )


def test_lever_basis_reproduces_find_transponder():
    _, _, GPS_Coordinates, _, _ = _generate(0.0, 0.02)
    lever_origin, lever_rotation = lever_basis(GPS_Coordinates, OTHERS)
    lever = np.array([3.0, -7.0, -20.0])
    np.testing.assert_allclose(
        lever_origin + lever_rotation @ lever,
        findTransponder(GPS_Coordinates, OTHERS, lever),
        atol=1e-8,
    )


def test_lever_bias_geiger_recovers_lever_at_the_true_offset():
    """Noiseless, lever 3 m and position 40 m off, fractional offset 0.1 s off."""
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(0.0, 0.0)
    lever_origin, lever_rotation = lever_basis(GPS_Coordinates, OTHERS)
    estimate, lever, rms, n_pairs = lever_bias_geiger(
        CDOG + np.array([30.0, -20.0, 20.0]),
        CDOG_mat,
        GPS_time,
        lever_origin,
        lever_rotation,
        LEVER + np.array([2.0, -2.0, 1.0]),
        OFFSET + 0.1,
        DZ,
        ANGLE,
        ESV,
    )
    assert n_pairs > 1000
    assert rms < 1e-6
    np.testing.assert_allclose(lever, LEVER, atol=0.01)
    np.testing.assert_allclose(estimate[:3], CDOG, atol=0.01)
    # time_bias absorbs the 0.1 s misalignment: offset - time_bias is dt.
    assert abs(OFFSET + 0.1 - estimate[3] - TRUE_DT) < 1e-6


@pytest.mark.slow
def test_offset_search_recovers_the_integer_second_with_a_wrong_lever():
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(2e-5, 0.02)
    lever, offset, estimate, diagnostics = gauss_newton_offset_search(
        CDOG_mat,
        GPS_time,
        GPS_Coordinates,
        OTHERS,
        CDOG + np.array([30.0, -20.0, 35.0]),
        LEVER + np.array([3.0, -3.0, 2.0]),
        DZ,
        ANGLE,
        ESV,
    )
    assert estimate[3] == 0.0
    assert abs(offset - TRUE_DT) < 1e-3
    assert np.linalg.norm(lever - LEVER) < 0.05
    assert diagnostics["delta_chi2"] > 1000


@pytest.mark.slow
def test_offset_search_recentres_when_the_hint_is_seconds_off():
    """The window (+-2 s) excludes the truth at first; recentring reaches it."""
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(2e-5, 0.02)
    _, offset, _, _ = gauss_newton_offset_search(
        CDOG_mat,
        GPS_time,
        GPS_Coordinates,
        OTHERS,
        CDOG + np.array([30.0, -20.0, 35.0]),
        LEVER,
        DZ,
        ANGLE,
        ESV,
        initial_offset=np.round(TRUE_DT) + 5.0,
        halfwindow=2,
    )
    assert abs(offset - TRUE_DT) < 1e-3


def test_find_int_offset_default_window_reaches_negative_offsets():
    """Puerto Rico offsets are negative (-19 s, -7459 s); Bermuda's are positive."""
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(0.0, 0.0)
    transponder = findTransponder(GPS_Coordinates, OTHERS, LEVER)
    times, esv = calculateTimesRayTracing_Bias(CDOG, transponder, 0.0, DZ, ANGLE, ESV)
    shift = 10000.0
    shifted = CDOG_mat.copy()
    shifted[:, 0] -= shift
    offset = find_int_offset(shifted, GPS_time, times, transponder, esv)
    # The coarse scan only has to land within half a second of the offset.
    assert abs(offset - (TRUE_DT - shift)) < 0.5


def test_offset_search_raises_when_the_best_second_fits_poorly(monkeypatch):
    """A clear best second is still rejected when its misfit RMS is implausible."""
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(2e-5, 0.0)
    monkeypatch.setattr(Numba_xAline_Geiger_bias, "MAX_OFFSET_RMS_SECONDS", 1e-6)
    with pytest.raises(ValueError, match="fits too poorly"):
        gauss_newton_offset_search(
            CDOG_mat,
            GPS_time,
            GPS_Coordinates,
            OTHERS,
            CDOG,
            LEVER,
            DZ,
            ANGLE,
            ESV,
            initial_offset=np.round(TRUE_DT),
            halfwindow=1,
        )


def test_offset_search_raises_when_the_ship_does_not_move():
    """A stationary ship gives every integer second the same misfit."""
    _, CDOG, GPS_Coordinates, _, _ = _generate(0.0, 0.0)
    n = 300
    GPS_Coordinates = np.repeat(GPS_Coordinates[:1], n, axis=0)
    GPS_time = np.arange(n, dtype=np.float64)
    transponder = findTransponder(GPS_Coordinates, OTHERS, LEVER)
    times, _ = calculateTimesRayTracing_Bias(CDOG, transponder, 0.0, DZ, ANGLE, ESV)
    rng = np.random.default_rng(0)
    arrival = GPS_time + times + TRUE_DT + rng.normal(0.0, 2e-5, n)
    fraction, integer = np.modf(arrival)
    CDOG_mat = np.column_stack((integer, fraction))
    with pytest.raises(ValueError, match="ambiguous"):
        gauss_newton_offset_search(
            CDOG_mat,
            GPS_time,
            GPS_Coordinates,
            OTHERS,
            CDOG + np.array([5.0, -5.0, 5.0]),
            LEVER,
            DZ,
            ANGLE,
            ESV,
            initial_offset=np.round(TRUE_DT),
            halfwindow=2,
        )


@pytest.mark.slow
def test_annealing_does_not_beat_the_gauss_newton_search():
    """Validation: annealing at the found offset finds no lower misfit.

    Annealing is no longer in the workflow; if it ever fits better than the
    Gauss-Newton search, that search is missing the optimum.
    """
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(2e-5, 0.02)
    initial_guess = CDOG + np.array([30.0, -20.0, 35.0])
    initial_lever = LEVER + np.array([1.5, -1.5, 1.0])
    lever, offset, estimate, _ = gauss_newton_offset_search(
        CDOG_mat,
        GPS_time,
        GPS_Coordinates,
        OTHERS,
        initial_guess,
        initial_lever,
        DZ,
        ANGLE,
        ESV,
    )
    gauss_newton_rms = misfit_rms(
        estimate,
        offset,
        CDOG_mat,
        GPS_time,
        findTransponder(GPS_Coordinates, OTHERS, lever),
        DZ,
        ANGLE,
        ESV,
        False,
    )

    np.random.seed(1)
    annealed_lever, annealed_offset, annealed_estimate = simulated_annealing_bias(
        300,
        CDOG_mat,
        GPS_time,
        GPS_Coordinates,
        OTHERS,
        initial_guess,
        initial_lever,
        DZ,
        ANGLE,
        ESV,
        initial_offset=offset,
    )
    annealed_estimate = final_bias_geiger(
        annealed_estimate[:3],
        CDOG_mat,
        GPS_time,
        findTransponder(GPS_Coordinates, OTHERS, annealed_lever),
        annealed_offset,
        annealed_estimate[4],
        annealed_estimate[3],
        DZ,
        ANGLE,
        ESV,
    )[0]
    annealed_rms = misfit_rms(
        annealed_estimate,
        annealed_offset,
        CDOG_mat,
        GPS_time,
        findTransponder(GPS_Coordinates, OTHERS, annealed_lever),
        DZ,
        ANGLE,
        ESV,
        False,
    )
    assert annealed_rms > 0.99 * gauss_newton_rms
    assert abs(annealed_offset - annealed_estimate[3] - offset) < 1e-4
