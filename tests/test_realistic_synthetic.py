import numpy as np
import pytest

from bermuda_data import gps_data_path
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Synthetic import Realistic_Synthetic as rs
from Inversion_Workflow.Synthetic import Realistic_Synthetic_Ensemble as ens

TABLE_PATH = gps_data_path("ESV_Tables/global_table_esv_synth_ref.mat")
needs_data = pytest.mark.skipif(
    not TABLE_PATH.exists(), reason="Bermuda data and synthetic ESV tables not present"
)


def _table():
    return load_esv_table(TABLE_PATH)


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
    t = np.array([20111.0, 21911.0 + 1.0, 70399.0])
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


def test_hourglass_keeps_speed_and_balance():
    centre = np.array([100.0, -200.0])
    hourglass, _ = rs.track_en_m("hourglass", 50401, centre)  # the Bermuda length
    step = np.linalg.norm(np.diff(hourglass, axis=0), axis=1)
    np.testing.assert_allclose(np.median(step), rs.TRACK_SPEED_MPS)
    offset = hourglass - centre
    assert np.all(np.abs(offset) <= rs.HOURGLASS_HALF_SIZE_M + 1e-6)
    # Two whole loops: every quadrant gets the same share of the survey.
    quadrant_counts = np.unique(np.sign(offset), axis=0, return_counts=True)[1]
    assert quadrant_counts.min() > 0.99 * quadrant_counts.max()


def test_geometry_cells_and_summary_table(tmp_path):
    cells = [name for name, _, _ in ens.experiment_cells("geometry")]
    assert cells == [
        f"{track}_{lever}"
        for track in ("bermuda", "hourglass")
        for lever in ("lever_fixed", "lever_free")
    ]
    rows = {
        "cell": ["hourglass_lever_fixed"] * 3,
        "DOG": [3, 3, 3],
        "failed": ["", "", "diverged"],
        "wrong_second": [0, 0, 0],
        "E_err_cm": [3.0, 1.0, 50.0],
        "N_err_cm": [4.0, 0.0, 50.0],
        "U_err_cm": [-2.0, 4.0, 50.0],
        "E_sd_ar1_cm": [1.0, 1.0, 1.0],
        "N_sd_ar1_cm": [1.0, 1.0, 1.0],
        "U_sd_ar1_cm": [1.0, 3.0, 1.0],
    }
    path = tmp_path / "geometry.csv"
    ens.pd.DataFrame(rows).to_csv(path, index=False)
    row = ens.summary_table(path).iloc[0]
    assert (row["n"], row["n_excluded"]) == (2, 1)
    assert row["horizontal_bias_cm"] == pytest.approx(np.hypot(2.0, 2.0))
    assert row["vertical_bias_cm"] == pytest.approx(1.0)
    assert row["horizontal_rms_cm"] == pytest.approx(np.sqrt((25 + 1) / 2))
    assert row["horizontal_coverage_95"] == pytest.approx(0.5)
    assert row["vertical_coverage_95"] == pytest.approx(0.5)
    assert (tmp_path / "geometry_summary.csv").exists()


def test_merge_task_csvs_joins_tasks_and_rejects_repeats(tmp_path, monkeypatch):
    monkeypatch.setattr(ens, "OUTPUT_DIR", tmp_path)
    row = {"cell": "hourglass_lever_fixed", "DOG": 3, "E_err_cm": 1.0}
    ens.pd.DataFrame([{**row, "seed": 0}]).to_csv(tmp_path / "geometry.csv")
    ens.pd.DataFrame([{**row, "seed": 1}]).to_csv(tmp_path / "geometry_seed0001.csv")
    merged = ens.pd.read_csv(ens.merge_task_csvs("geometry"))
    assert list(merged["seed"]) == [0, 1]
    ens.pd.DataFrame([{**row, "seed": 0}]).to_csv(tmp_path / "geometry_seed0002.csv")
    with pytest.raises(ValueError):
        ens.merge_task_csvs("geometry")


@pytest.mark.parametrize(
    "knobs",
    [
        {"no_such_knob": 1.0},
        {"dc_series": "spline"},
        {"current_mps": (0.1, 0.0, 0.0)},
        {"timing_noise_s": -1e-6},
        {"dropouts": "random"},
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

    # 38 073 of 49 552 GPS epochs have a DOG 1 pair in the real data.
    assert abs(keep.mean() - 38073 / 49552) < 0.01


@pytest.mark.parametrize(
    "amplitudes, speed, gradient, m2",
    [
        ("guess", 0.03, 0.005, 0.02),
        ("models_low", 0.03, 0.0026, 0.009),
        ("models_high", 0.11, 0.008, 0.009),
    ],
)
def test_amplitude_sets_set_current_gradient_and_tide(amplitudes, speed, gradient, m2):
    knobs = ens.with_random_directions(ens.full_preset("random", amplitudes), seed=3)
    assert set(knobs) <= set(rs.ZERO_KNOBS)
    assert np.linalg.norm(knobs["current_mps"]) == pytest.approx(speed)
    assert np.linalg.norm(knobs["dc_gradient_mps_per_km"]) == pytest.approx(gradient)
    assert knobs["current_m2_mps"] == m2


def test_amplitude_cells_are_labelled_and_budget_groups_follow_the_set():
    guess = ens.experiment_cells("budget")
    high = ens.experiment_cells("budget", "models_high")
    assert [name + "_models_high" for name, _, _ in guess] == [n for n, _, _ in high]
    cells = {name: knobs for name, knobs, _ in high}
    only_current = cells["random_only_current_steady_models_high"]
    knobs = ens.with_random_directions(only_current, seed=0)
    assert np.linalg.norm(knobs["current_mps"]) == pytest.approx(0.11)
    with pytest.raises(KeyError):
        ens.experiment_cells("budget", "models_middle")


def test_synthetic_mcmc_budget_knobs(monkeypatch):
    """Budget runs: the floor plus one group, same directions as the full preset."""
    from Inversion_Workflow.Synthetic import Realistic_Synthetic_MCMC as sm

    monkeypatch.setattr(sm, "ONLY", None)
    full = sm.survey_knobs(3)
    monkeypatch.setattr(sm, "ONLY", "current_steady")
    knobs = sm.survey_knobs(3)
    assert knobs["current_mps"] == full["current_mps"]
    assert knobs["dropouts"] == "real_pattern" and knobs["timing_noise_s"] > 0
    assert "dc_series" not in knobs
    monkeypatch.setattr(sm, "ONLY", "esv_profile")
    assert set(sm.survey_knobs(3)) == {"timing_noise_s", "dropouts", "track"}
    monkeypatch.setattr(sm, "ONLY", "transmit")
    with pytest.raises(ValueError, match="SYN_ONLY"):
        sm.survey_knobs(3)
