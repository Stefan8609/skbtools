import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from plotting.Atlantic_Explorer.moon_pool_lever import moon_pool_center_body_frame

ROOT = Path(__file__).resolve().parents[1]


def synthetic_problem(num_esv_splits=1, n_arrivals=60):
    """One DOG 5 km below a vessel circling at 1 km radius, constant 1500 m/s ESV.

    Sixty 1 s epochs with a 100 s gap after epoch 40 (an AR(1) segment break) and
    3 cm of range noise; the DOG records the first n_arrivals of them. Returns
    the configuration and the built problem.
    """
    CDOG_reference = np.array(ms.CONFIG["CDOG_reference"])
    up = CDOG_reference / np.linalg.norm(CDOG_reference)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    grid = np.array(ms.CONFIG["gps1_to_others"])
    lever = np.array([-13.0, 9.5, -14.5])

    GPS_data = 1000.0 + np.concatenate([np.arange(40.0), 140.0 + np.arange(20.0)])
    angles = 2 * np.pi * (GPS_data - GPS_data[0]) / 600.0
    GPS_Coordinates = np.empty((GPS_data.size, 4, 3))
    transducer = np.empty((GPS_data.size, 3))
    for i, a in enumerate(angles):
        gps1 = (
            CDOG_reference
            + 5000.0 * up
            + 1000.0 * (np.cos(a) * east + np.sin(a) * north)
        )
        heading = -np.sin(a) * east + np.cos(a) * north
        body_to_ecef = np.column_stack([heading, np.cross(up, heading), up])
        GPS_Coordinates[i] = gps1 + grid @ body_to_ecef.T
        transducer[i] = gps1 + body_to_ecef @ lever

    dz_array = np.arange(4000.0, 6001.0, 100.0)
    angle_array = np.linspace(0.0, 90.0, 91)
    esv_matrix = np.full((dz_array.size, angle_array.size), 1500.0)
    times = ms.dog_travel_times(
        CDOG_reference, transducer, np.array([0.0]), dz_array, angle_array, esv_matrix
    )
    noise_s = np.random.default_rng(7).normal(0.0, 2e-5, GPS_data.size)
    arrival = GPS_data + times + 100.0 + noise_s
    arrival = arrival[:n_arrivals]
    CDOG_data = np.column_stack([np.floor(arrival), arrival - np.floor(arrival)])

    config = {
        **ms.CONFIG,
        "run_name": "synthetic",
        "likelihood": "ar1",
        "clock_drift": True,
        "dogs": [1],
        "num_esv_splits": num_esv_splits,
        "offsets_s": [100.0],
        "outlier_mad_factor": None,
        "start_lever": (lever + 0.05).tolist(),
        "start_CDOG_aug": [[0.1, -0.1, 0.05]],
        "start_esv_bias": [0.0],
        "prior_CDOG_aug_center_m": [[0.0, 0.0, 0.0]],
        "n_burn": 60,
        "n_samples": 60,
        "thin": 2,
        "first_adapt_window": 10,
        "checkpoint_every": 25,
        "log_every": 1000,
    }
    problem = ms.build_problem(
        config,
        GPS_Coordinates,
        GPS_data,
        [CDOG_data],
        dz_array,
        angle_array,
        esv_matrix,
    )
    return config, problem


def test_iid_log_likelihood_matches_hand_computation():
    residuals = np.array([10.0, -20.0, 30.0])
    expected = -0.5 * (100.0 + 400.0 + 900.0) / 400.0 - 3 * np.log(20.0)

    value = ms.ar1_log_likelihood(
        residuals,
        np.array([True, False, False]),
        np.zeros(3, dtype=int),
        np.array([20.0]),
        np.array([0.0]),
    )

    assert value == pytest.approx(expected, rel=1e-14)


def test_ar1_log_likelihood_matches_hand_computation():
    # Two segments: [1, 2, -1] and [3]; sigma = 2 cm, phi = 0.5.
    residuals = np.array([1.0, 2.0, -1.0, 3.0])
    segment_start = np.array([True, False, False, True])
    innovation_sd = 2.0 * np.sqrt(1.0 - 0.5**2)
    whitened = np.array(
        [
            1.0 / 2.0,
            (2.0 - 0.5 * 1.0) / innovation_sd,
            (-1.0 - 0.5 * 2.0) / innovation_sd,
            3.0 / 2.0,
        ]
    )
    expected = -0.5 * np.sum(whitened**2) - 2 * np.log(2.0) - 2 * np.log(innovation_sd)

    value = ms.ar1_log_likelihood(
        residuals,
        segment_start,
        np.zeros(4, dtype=int),
        np.array([2.0]),
        np.array([0.5]),
    )

    assert value == pytest.approx(expected, rel=1e-14)


def test_ar1_with_zero_phi_equals_iid():
    rng = np.random.default_rng(1)
    residuals = rng.normal(0.0, 15.0, 50)
    pair_dog = np.repeat([0, 1], 25)
    segment_start = np.zeros(50, dtype=bool)
    segment_start[[0, 10, 25, 40]] = True
    sigma = np.array([12.0, 18.0])
    iid = -0.5 * np.sum((residuals / sigma[pair_dog]) ** 2) - np.sum(
        np.log(sigma[pair_dog])
    )

    value = ms.ar1_log_likelihood(
        residuals, segment_start, pair_dog, sigma, np.zeros(2)
    )

    assert value == pytest.approx(iid, rel=1e-13)


def test_prior_matches_configuration():
    model = ms.build_model(ms.run_config("ar1_drift"))
    layout = model["layout"]
    center = model["prior_center"]
    sd = model["prior_sd"]

    assert center[layout["lever"]].tolist() == [-13.1, 9.7, -14.0]
    assert sd[layout["lever"]].tolist() == [0.5, 0.5, 2.0]
    assert np.all(sd[layout["CDOG_aug"]] == 5.0)
    assert np.all(center[layout["esv_bias"]] == 0.0)
    assert np.all(sd[layout["esv_bias"]] == 2.0)
    assert np.all(center[layout["time_bias"]] == 0.0)
    assert np.all(sd[layout["time_bias"]] == 0.01)

    theta = np.where(np.isfinite(center), center, 0.5)  # phi = 0.5 is in bounds
    assert ms.log_prior(theta, model) == 0.0
    one_sd_moves = [
        (layout["lever"].start + 2, 2.0),
        (layout["lever"].start, 0.5),
        (layout["CDOG_aug"].start + 4, 5.0),
        (layout["esv_bias"].start, 2.0),
        (layout["time_bias"].start, 0.01),
        (layout["drift_rate"].start, 1e-6),
        (layout["log_sigma"].start, 1.0),
    ]
    for index, step in one_sd_moves:
        moved = theta.copy()
        moved[index] += step
        assert ms.log_prior(moved, model) == pytest.approx(-0.5)

    out_of_bounds = theta.copy()
    out_of_bounds[layout["phi"].start] = 0.9995
    assert ms.log_prior(out_of_bounds, model) == -np.inf


def test_moon_pool_helper_reproduces_lever_prior_center():
    center = moon_pool_center_body_frame()

    assert np.round(center, 1).tolist() == ms.PRIOR_LEVER_CENTER_M[:2]


def test_same_seed_gives_same_chain(tmp_path):
    config, problem = synthetic_problem()
    runs = {}
    for name, chain_index in (("a", 0), ("b", 0), ("c", 1)):
        ms.run_chain(
            chain_index, ms.build_model(config), problem, tmp_path / f"{name}.npz"
        )
        runs[name] = np.load(tmp_path / f"{name}.npz")

    assert np.array_equal(runs["a"]["theta_samples"], runs["b"]["theta_samples"])
    assert np.array_equal(runs["a"]["burnin_theta"], runs["b"]["burnin_theta"])
    assert not np.array_equal(runs["a"]["start_theta"], runs["c"]["start_theta"])


def test_resume_from_checkpoint_matches_uninterrupted_run(tmp_path):
    config, problem = synthetic_problem()
    ms.run_chain(0, ms.build_model(config), problem, tmp_path / "full.npz")
    # Stop inside an adaptation window, then inside the sampling phase, then finish.
    for stop_at in (35, 85, None):
        ms.run_chain(
            0,
            ms.build_model(config),
            problem,
            tmp_path / "resumed.npz",
            stop_at=stop_at,
        )
    full = np.load(tmp_path / "full.npz")
    resumed = np.load(tmp_path / "resumed.npz")

    assert int(resumed["n_done"]) == 120
    for key in (
        "theta_samples",
        "logpost",
        "loglike",
        "burnin_theta",
        "dt",
        "state_proposal_cov",
        "state_theta",
        "window_acceptance",
    ):
        np.testing.assert_array_equal(full[key], resumed[key], err_msg=key)
    assert str(full["state_rng"]) == str(resumed["state_rng"])


def test_resume_refuses_a_changed_configuration(tmp_path):
    config, problem = synthetic_problem()
    ms.run_chain(0, ms.build_model(config), problem, tmp_path / "chain.npz", stop_at=20)

    changed = {**config, "prior_lever_sd_m": [1.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match="different configuration"):
        ms.run_chain(0, ms.build_model(changed), problem, tmp_path / "chain.npz")


REFERENCE = ROOT / "revision/reference_outputs/development/mcmc_likelihood_prior.npz"
FIG7_CHAIN = ROOT / "Bermuda_Data/Output/mcmc_chain_1_22_new_MCMC_long.npz"


@pytest.mark.skipif(
    not (REFERENCE.exists() and FIG7_CHAIN.exists()),
    reason="needs the local reference snapshot and Bermuda data",
)
def test_iid_fixed_reproduces_reference_log_likelihood():
    # Runs with Numba compiled (the reference was), so in a subprocess: conftest
    # disables the JIT for this process.
    script = textwrap.dedent(f"""
        import numpy as np
        from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
        from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder

        reference = np.load("{REFERENCE}")
        init = np.load("{FIG7_CHAIN}")
        config = ms.run_config("iid_fixed", {{
            "outlier_mad_factor": None,
            "gps1_to_others": init["init_gps_grid"].tolist(),
            "start_lever": init["init_lever"].tolist(),
            "start_CDOG_aug": init["init_CDOG_aug"].tolist(),
            "start_esv_bias": init["init_esv_bias"][:, 0].tolist(),
        }})
        model, problem = ms.setup_bermuda(config)
        assert np.array_equal(problem["pair_counts"], reference["pair_counts"])
        inputs = ms.load_bermuda_inputs(config)
        shifts = {{"init": (0.0, 0.0, 0.0, 0.0),
                   "shifted": (np.array([0.02, -0.03, 0.1]), 0.05, 0.01, 1e-4)}}
        for name, (d_lever, d_aug, d_esv, d_time) in shifts.items():
            lever = init["init_lever"] + d_lever
            trans = findTransponder(inputs["GPS_Coordinates"], init["init_gps_grid"],
                                    lever)
            value, _ = ms.compute_log_likelihood(
                trans, init["init_CDOG_aug"] + d_aug, init["init_esv_bias"] + d_esv,
                init["init_time_bias"] + d_time, problem["CDOG_reference"],
                problem["gps_pair_indices"], problem["pair_time_base_2d"],
                problem["pair_counts"], problem["dz_array"], problem["angle_array"],
                problem["esv_matrix"])
            assert value == float(reference["loglike_" + name]), name

            theta = model["start_theta"].copy()
            layout = model["layout"]
            theta[layout["lever"]] = lever
            theta[layout["CDOG_aug"]] += d_aug
            theta[layout["esv_bias"]] += d_esv
            theta[layout["time_bias"]] = init["init_time_bias"] + d_time
            sampler_value = ms.log_likelihood(theta, model, problem)
            # Transponders from precomputed rotations differ by < 1e-9 m.
            assert abs(sampler_value / value - 1) < 1e-12, name
        print("ok")
    """)
    env = {
        **os.environ,
        "NUMBA_DISABLE_JIT": "0",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(ROOT / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT / "src",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_residual_jacobian_time_bias_column_is_the_range_scale():
    config, problem = synthetic_problem()
    model = ms.build_model(config)

    jacobian = ms.residual_jacobian(model["start_theta"], model, problem)

    time_bias_column = jacobian[:, model["layout"]["time_bias"].start]
    np.testing.assert_allclose(time_bias_column, ms.RANGE_CM_PER_S, rtol=1e-6)


def test_ar1_white_likelihood_matches_dense_gaussian():
    # Two segments ([0:4] and [4:6]); AR(1) sd 3 cm, phi 0.7, white-noise sd 1.5 cm.
    residuals = np.array([1.0, 2.5, -0.5, 0.8, 3.0, 1.0])
    segment_start = np.array([True, False, False, False, True, False])
    problem = {"segment_start": segment_start, "pair_dog": np.zeros(6, dtype=np.int64)}
    sigma, phi, white = np.array([3.0]), np.array([0.7]), np.array([1.5])
    cov = np.zeros((6, 6))
    for block in (range(0, 4), range(4, 6)):
        for i in block:
            for k in block:
                cov[i, k] = sigma[0] ** 2 * phi[0] ** abs(i - k)
    cov += white[0] ** 2 * np.eye(6)
    expected = -0.5 * (
        np.linalg.slogdet(cov)[1] + residuals @ np.linalg.solve(cov, residuals)
    )

    whitened, innovation_sd = ms.whiten_residuals(residuals, problem, sigma, phi, white)
    value = -0.5 * np.dot(whitened, whitened) - np.sum(np.log(innovation_sd))

    assert value == pytest.approx(expected, rel=1e-12)


def test_ar1_white_reduces_to_ar1_and_to_iid():
    rng = np.random.default_rng(3)
    residuals = rng.normal(0.0, 10.0, 40)
    pair_dog = np.repeat([0, 1], 20).astype(np.int64)
    segment_start = np.zeros(40, dtype=bool)
    segment_start[[0, 12, 20]] = True
    problem = {"segment_start": segment_start, "pair_dog": pair_dog}
    sigma, phi = np.array([8.0, 12.0]), np.array([0.6, 0.9])

    def loglike(w, sd):
        return -0.5 * np.dot(w, w) - np.sum(np.log(sd))

    no_white = loglike(*ms.whiten_residuals(residuals, problem, sigma, phi, 0 * sigma))
    ar1 = ms.ar1_log_likelihood(residuals, segment_start, pair_dog, sigma, phi)
    assert no_white == pytest.approx(ar1, rel=1e-12)

    white = np.array([3.0, 4.0])
    no_phi = loglike(*ms.whiten_residuals(residuals, problem, sigma, 0 * phi, white))
    total_sd = np.sqrt(sigma**2 + white**2)[pair_dog]
    iid = -0.5 * np.sum((residuals / total_sd) ** 2) - np.sum(np.log(total_sd))
    assert no_phi == pytest.approx(iid, rel=1e-12)


def test_ar1_white_and_split_variants_leave_older_runs_unchanged():
    for variant in ("iid_fixed", "iid", "ar1", "ar1_drift"):
        assert "prior_log_sigma_white_center" not in ms.run_config(variant)
        assert ms.run_config(variant)["num_esv_splits"] == 4

    layout, names = ms.parameter_layout(ms.run_config("ar1_white_splits8"))
    assert layout["esv_bias"].stop - layout["esv_bias"].start == 3 * 8
    assert names[layout["log_sigma_white"]][0] == "log_sigma_white_cm_DOG1"
    assert ms.run_config("ar1_splits16")["n_samples"] == 400000

    for variant, splits in (
        ("ar1_white_drift", 4),
        ("ar1_white_drift_splits1", 1),
        ("ar1_white_drift_splits8", 8),
    ):
        config = ms.run_config(variant)
        assert config["likelihood"] == "ar1_white" and config["clock_drift"]
        assert config["num_esv_splits"] == splits
        assert config["n_burn"] == 40000


def test_ar1_white_chain_runs_and_resumes(tmp_path):
    config, problem = synthetic_problem()
    config = {**config, **ms.RUN_VARIANTS["ar1_white"], "num_esv_splits": 1}
    ms.run_chain(0, ms.build_model(config), problem, tmp_path / "full.npz")
    for stop_at in (35, None):
        ms.run_chain(
            0,
            ms.build_model(config),
            problem,
            tmp_path / "resumed.npz",
            stop_at=stop_at,
        )
    full = np.load(tmp_path / "full.npz")
    resumed = np.load(tmp_path / "resumed.npz")

    assert full["sigma_white_cm"].shape == (30, 1)
    assert np.all(np.isfinite(full["logpost"]))
    np.testing.assert_array_equal(full["theta_samples"], resumed["theta_samples"])


def test_atanh_phi_prior_is_uniform_in_phi():
    model = ms.build_model(ms.run_config("ar1_white"))
    index = model["layout"]["atanh_phi"].start
    theta = np.where(np.isfinite(model["prior_center"]), model["prior_center"], 0.0)

    def log_prior_at(phi):
        moved = theta.copy()
        moved[index] = np.arctanh(phi)
        return ms.log_prior(moved, model)

    # Density in atanh(phi) = uniform density in phi times d phi / d atanh(phi).
    assert log_prior_at(0.99) - log_prior_at(0.5) == pytest.approx(
        np.log(1 - 0.99**2) - np.log(1 - 0.5**2)
    )
    assert log_prior_at(0.99995) == -np.inf
