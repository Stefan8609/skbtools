import numpy as np
import pytest
import scipy.io as sio

from bermuda_data import gps_data_path
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Synthetic import Realistic_Synthetic as rs

TABLE_PATH = gps_data_path("ESV_Tables/global_table_esv_synth_ref.mat")
needs_data = pytest.mark.skipif(
    not TABLE_PATH.exists(), reason="Bermuda data and synthetic ESV tables not present"
)


def _table():
    table = sio.loadmat(TABLE_PATH)
    return table["distance"].flatten(), table["angle"].flatten(), table["matrice"]


def test_current_towards_the_dog_shortens_the_travel_time():
    """dT = -(u . dx_h) / c_eff^2: 5 cm/s along a 5 km horizontal path, 1500 m/s."""
    dog = np.array([0.0, 0.0, -5000.0])
    transponder = np.array([[-5000.0, 0.0, 0.0], [5000.0, 0.0, 0.0]])
    slant = np.hypot(5000.0, 5000.0)
    times = np.full(2, slant / 1500.0)
    current = np.array([[0.05, 0.0], [0.05, 0.0]])  # eastward
    delay = rs.current_delay_s(transponder, dog, times, current)
    expected = 0.05 * 5000.0 / 1500.0**2
    np.testing.assert_allclose(delay, [-expected, expected], rtol=1e-12)


def test_gradient_is_the_value_at_the_ray_midpoint():
    dog = np.array([2000.0, 0.0, -5000.0])
    transponder = np.array([[0.0, 0.0, 0.0], [6000.0, 4000.0, 0.0]])
    dc = rs.dc_gradient_mps(transponder, dog, (0.005, 0.002))
    np.testing.assert_allclose(dc, [0.005 * 1.0, 0.005 * 4.0 + 0.002 * 2.0])


def test_dc_series_random_walk_size_and_seed():
    t = 20000.0 + np.arange(0.0, 14 * 3600.0, 60.0)
    knobs = {
        **rs.ZERO_KNOBS,
        "dc_series": "random_walk",
        "dc_random_walk_mps_per_sqrt_h": 0.05,
    }
    walks = np.array([rs.dc_series_mps(t, knobs, seed) for seed in range(200)])
    hourly = walks[:, 60] - walks[:, 0]  # one hour apart
    assert walks[:, 0].max() == 0.0
    assert abs(hourly.std() / 0.05 - 1) < 0.15
    np.testing.assert_array_equal(walks[3], rs.dc_series_mps(t, knobs, 3))


def test_dc_series_observed_and_sinusoid():
    t = np.array([20116.0, 23716.0 + 1.0, 70399.0])
    observed = rs.dc_series_mps(t, {**rs.ZERO_KNOBS, "dc_series": "observed"}, 0)
    np.testing.assert_allclose(observed, rs.OBSERVED_DC_MPS[[0, 1, -1]])
    sine = {**rs.ZERO_KNOBS, "dc_sinusoid_mps": 0.01, "dc_sinusoid_phase_rad": 0.0}
    t = 20000.0 + 3600.0 * np.array([0.0, rs.M2_PERIOD_H / 2])
    np.testing.assert_allclose(rs.dc_series_mps(t, sine, 0), [0.01, -0.01])


def test_m2_current_rotates_clockwise_with_constant_speed():
    t = 20000.0 + 3600.0 * np.array([0.0, rs.M2_PERIOD_H / 4])
    knobs = {**rs.ZERO_KNOBS, "current_m2_mps": 0.02, "current_m2_phase_rad": 0.0}
    current = rs.current_en_mps(t, knobs, 0)
    np.testing.assert_allclose(current, [[0.02, 0.0], [0.0, -0.02]], atol=1e-15)


def test_gauss_markov_sd_and_correlation_time():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 200000.0)
    x = rs.gauss_markov(t, [0.02, 0.05], 600.0, rng)
    np.testing.assert_allclose(x.std(axis=0), [0.02, 0.05], rtol=0.15)
    lagged = np.corrcoef(x[:-600, 0], x[600:, 0])[0, 1]
    assert abs(lagged - np.exp(-1)) < 0.1


def test_random_dropout_fraction():
    keep = rs.random_dropout_mask(50000, 0.2, np.random.default_rng(1))
    assert 0.2 <= 1 - keep.mean() <= 0.2 + 500 / 50000


def test_tracks_keep_speed_and_radius():
    centre = np.array([100.0, -200.0])
    circle, _ = rs.track_en_m("circle", 3000, centre)
    radius = np.linalg.norm(circle - centre, axis=1)
    np.testing.assert_allclose(radius, rs.CIRCLE_RADIUS_M, rtol=1e-4)
    grid, heading = rs.track_en_m("grid", 3000, centre)
    step = np.linalg.norm(np.diff(grid, axis=0), axis=1)
    np.testing.assert_allclose(np.median(step), rs.TRACK_SPEED_MPS)
    assert set(np.round(np.degrees(heading)).astype(int)) <= {90, -90, 0, 180}


@pytest.mark.parametrize(
    "knobs",
    [
        {"no_such_knob": 1.0},
        {"dc_series": "spline"},
        {"current_mps": (0.1, 0.0, 0.0)},
        {"timing_noise_s": -1e-6},
        {"dropout_fraction": 1.0},
        {"gnss_common_tau_s": 0.0},
    ],
)
def test_invalid_knobs_raise(knobs):
    with pytest.raises(ValueError):
        rs.check_knobs(knobs)


@needs_data
def test_zero_knobs_reproduce_bermuda_trajectory():
    from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
        bermuda_trajectory,
    )

    table = _table()
    knobs = {"timing_noise_s": 2e-5, "position_noise_m": 0.02}
    out = rs.realistic_synthetic(3, knobs, 7, *table)
    np.random.seed([7, 3])
    expected = bermuda_trajectory(2e-5, 0.02, 0.0, 0.0, *table, DOG_num=3)
    for got, want in zip(out[:5], expected, strict=True):
        np.testing.assert_array_equal(got, want)
    assert out[5]["delay_s"] == {}


@needs_data
@pytest.mark.slow
def test_knobs_act_as_documented():
    """Transmit delay is added to the arrivals, common GNSS error moves the
    transducer 1:1, and the real dropout pattern keeps DOG 1's paired fraction."""
    table = _table()
    base = rs.realistic_synthetic(1, {}, 2, *table)
    knobs = {
        "transmit_delay_s": 30e-6,
        "gnss_common_sd_m": (0.02, 0.02, 0.05),
        "dropouts": "real_pattern",
    }
    CDOG_data, _, GPS_Coordinates, _, transponder, truth = rs.realistic_synthetic(
        1, knobs, 2, *table
    )
    keep = truth["keep"]
    arrival = CDOG_data[:, 0] + CDOG_data[:, 1]
    base_arrival = (base[0][:, 0] + base[0][:, 1])[keep]
    np.testing.assert_allclose(arrival - base_arrival, 30e-6, atol=1e-9)

    observed = findTransponder(GPS_Coordinates, rs.GPS1_TO_OTHERS, truth["lever"])
    moved_enu = (observed - transponder) @ rs.enu_rotation().T
    np.testing.assert_allclose(moved_enu, truth["gnss_common_enu_m"], atol=1e-6)

    # 22 824 of 29 595 GPS epochs have a DOG 1 pair in the real data.
    assert abs(keep.mean() - 22824 / 29595) < 0.01
