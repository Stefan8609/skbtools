from functools import lru_cache

import numpy as np

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_Geiger_bias import compute_Jacobian_biased
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    drift_step,
    final_bias_geiger,
    initial_bias_geiger,
)
from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned

# Same smooth synthetic ESV table and geometry as test_synthetic_workflow.
DZ = np.arange(5100.0, 5400.0, 1.0)
ANGLE = np.arange(0.0, 90.01, 1.0)
ESV = 1500.0 + 0.02 * (DZ[:, None] - 5200.0) + 0.5 * ANGLE[None, :] / 90.0
OTHERS = np.array(
    [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.03], [-12.1, -0.95, 0.004], [-8.7, 5.17, 0.025]]
)
LEVER = np.array([-12.5, 9.6, -13.3])
OFFSET, TIME_BIAS, DRIFT_RATE = 3141.5926, 0.2, 5e-10


@lru_cache(maxsize=None)
def _invert_unaligned(drift_rate, clock_drift):
    """Noiseless 8000 s unaligned synthetic, inverted from 50 m off (cached)."""
    np.random.seed(0)
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = generateUnaligned(
        8000,
        0.0,
        0.0,
        OFFSET,
        0.0,
        TIME_BIAS,
        DZ,
        ANGLE,
        ESV,
        gps1_to_others=OTHERS,
        gps1_to_transponder=LEVER,
        drift_rate=drift_rate,
    )
    transponder = findTransponder(GPS_Coordinates, OTHERS, LEVER)
    estimate, offset = initial_bias_geiger(
        CDOG + np.array([30.0, -20.0, 35.0]),
        CDOG_mat,
        GPS_time,
        transponder,
        DZ,
        ANGLE,
        ESV,
        real_data=False,
        clock_drift=clock_drift,
    )
    result = final_bias_geiger(
        estimate[:3],
        CDOG_mat,
        GPS_time,
        transponder,
        offset,
        estimate[4],
        estimate[3],
        DZ,
        ANGLE,
        ESV,
        real_data=False,
        clock_drift=clock_drift,
        drift_rate=estimate[5] if clock_drift else 0.0,
    )
    return result, offset, CDOG


def test_noiseless_drift_is_recovered():
    (estimate, CDOG_full, GPS_full, _, _), offset, CDOG = _invert_unaligned(
        DRIFT_RATE, clock_drift=True
    )
    assert estimate.shape == (6,)
    assert abs(estimate[5] - DRIFT_RATE) < 1e-12
    assert np.linalg.norm(estimate[:3] - CDOG) < 1e-4
    assert abs(estimate[4]) < 1e-4
    # The generator's t_mid is the GPS mid-span, the inversion's the paired one;
    # the difference (a few s at most) times the drift is far below 1e-7 s.
    assert abs((offset - estimate[3]) - (OFFSET + TIME_BIAS)) < 1e-7
    assert np.sqrt(np.mean((GPS_full - CDOG_full) ** 2)) < 1e-8


def test_ignoring_drift_leaves_a_larger_misfit():
    """Documents the bias when drift is present but not modelled."""
    (on, on_CDOG, on_GPS, _, _), on_offset, CDOG = _invert_unaligned(
        DRIFT_RATE, clock_drift=True
    )
    (off, off_CDOG, off_GPS, _, _), off_offset, _ = _invert_unaligned(
        DRIFT_RATE, clock_drift=False
    )
    print(
        "drift off: position error (m)",
        off[:3] - CDOG,
        "dc (m/s)",
        off[4],
        "dt0 error (s)",
        off_offset - off[3] - (OFFSET + TIME_BIAS),
    )
    rms_on = np.sqrt(np.mean((on_GPS - on_CDOG) ** 2))
    rms_off = np.sqrt(np.mean((off_GPS - off_CDOG) ** 2))
    assert rms_off > 100 * rms_on


def test_drift_off_ignores_the_drift_start_value():
    np.random.seed(0)
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = generateUnaligned(
        2000, 0.0, 0.0, OFFSET, 0.0, TIME_BIAS, DZ, ANGLE, ESV,
        gps1_to_others=OTHERS, gps1_to_transponder=LEVER,
    )  # fmt: skip
    transponder = findTransponder(GPS_Coordinates, OTHERS, LEVER)
    args = (CDOG + 1.0, CDOG_mat, GPS_time, transponder, OFFSET + TIME_BIAS, 0.0,
            0.0, DZ, ANGLE, ESV)  # fmt: skip
    default = final_bias_geiger(*args, real_data=False)
    ignored = final_bias_geiger(*args, real_data=False, drift_rate=1e-6)
    assert default[0].shape == (5,)
    for a, b in zip(default, ignored, strict=False):
        assert np.array_equal(a, b)


def _mcmc_problem_with_drift():
    """One DOG 5 km below a vessel circling at 1 km, 1500 m/s, 3 cm noise, drift.

    Built with ``MCMC_sampler.build_problem`` so t_mid is the MCMC's own.
    """
    CDOG_reference = np.array(ms.CONFIG["CDOG_reference"])
    up = CDOG_reference / np.linalg.norm(CDOG_reference)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    grid = np.array(ms.CONFIG["gps1_to_others"])
    lever = np.array([-13.0, 9.5, -14.5])

    GPS_data = 1000.0 + 10.0 * np.arange(200.0)
    angles = 2 * np.pi * (GPS_data - GPS_data[0]) / 600.0
    GPS_Coordinates = np.empty((GPS_data.size, 4, 3))
    for i, a in enumerate(angles):
        gps1 = (
            CDOG_reference
            + 5000.0 * up
            + 1000.0 * (np.cos(a) * east + np.sin(a) * north)
        )
        heading = -np.sin(a) * east + np.cos(a) * north
        body_to_ecef = np.column_stack([heading, np.cross(up, heading), up])
        GPS_Coordinates[i] = gps1 + grid @ body_to_ecef.T
    transducer = findTransponder(GPS_Coordinates, grid, lever)

    dz_array = np.arange(4000.0, 6001.0, 100.0)
    angle_array = np.linspace(0.0, 90.0, 91)
    esv_matrix = np.full((dz_array.size, angle_array.size), 1500.0)
    times = ms.dog_travel_times(
        CDOG_reference, transducer, np.array([0.0]), dz_array, angle_array, esv_matrix
    )
    noise_s = np.random.default_rng(7).normal(0.0, 2e-5, GPS_data.size)
    drift_s = 2e-8 * (GPS_data - GPS_data.mean())
    arrival = GPS_data + times + 100.0 + noise_s + drift_s
    CDOG_data = np.column_stack([np.floor(arrival), arrival - np.floor(arrival)])

    config = {
        **ms.CONFIG,
        "likelihood": "iid",
        "clock_drift": True,
        "dogs": [1],
        "num_esv_splits": 1,
        "offsets_s": [100.0],
        "outlier_mad_factor": None,
        "start_lever": lever.tolist(),
        "start_CDOG_aug": [[0.0, 0.0, 0.0]],
        "start_esv_bias": [0.0],
        "prior_CDOG_aug_center_m": [[0.0, 0.0, 0.0]],
    }
    problem = ms.build_problem(
        config, GPS_Coordinates, GPS_data, [CDOG_data], dz_array, angle_array,
        esv_matrix,
    )  # fmt: skip
    gn_inputs = (CDOG_reference, CDOG_data, GPS_data, transducer, dz_array,
                 angle_array, esv_matrix)  # fmt: skip
    return config, problem, gn_inputs


def test_gauss_newton_and_mcmc_residuals_agree_with_drift():
    """At the Gauss-Newton fit, both codes give the same residuals (opposite sign:
    the MCMC uses observed minus modelled, the Gauss-Newton modelled minus observed),
    so they share one drift_rate and t_mid definition."""
    config, problem, gn_inputs = _mcmc_problem_with_drift()
    CDOG_reference, CDOG_data, GPS_data, transducer, dz, angle, esv = gn_inputs
    estimate, CDOG_full, GPS_full, _, _ = final_bias_geiger(
        CDOG_reference + np.array([0.5, -0.3, 0.4]),
        CDOG_data,
        GPS_data,
        transducer,
        100.0,
        0.0,
        0.0,
        dz,
        angle,
        esv,
        real_data=True,
        clock_drift=True,
    )
    # 2e-5 s of noise over +-1000 s gives a drift sd of about 2.4e-9 s/s.
    assert abs(estimate[5] - 2e-8) < 1e-8

    model = ms.build_model(config)
    layout = model["layout"]
    theta = model["start_theta"].copy()
    theta[layout["CDOG_aug"]] = estimate[:3] - CDOG_reference
    theta[layout["esv_bias"]] = estimate[4]
    theta[layout["time_bias"]] = estimate[3]
    theta[layout["drift_rate"]] = estimate[5]
    mcmc_residuals = ms.residuals_cm(theta, model, problem)

    gn_residuals_cm = (GPS_full - CDOG_full) * ms.RANGE_CM_PER_S
    assert mcmc_residuals.shape == gn_residuals_cm.shape
    np.testing.assert_allclose(mcmc_residuals, -gn_residuals_cm, rtol=0, atol=1e-6)


def test_drift_jacobian_column_matches_finite_difference():
    """The Gauss-Newton drift column is +(t - t_mid), s per s/s: minus the MCMC's
    finite-difference column divided by the range scale."""
    config, problem, _ = _mcmc_problem_with_drift()
    model = ms.build_model(config)
    jacobian = ms.residual_jacobian(model["start_theta"], model, problem)
    fd_column = jacobian[:, model["layout"]["drift_rate"].start]
    gn_column = problem["pair_t_from_mid"]
    np.testing.assert_allclose(-fd_column / ms.RANGE_CM_PER_S, gn_column, rtol=1e-6)


def test_drift_step_solves_a_pure_drift_residual_in_one_step():
    """Checks the column scaling: the step undoes a residual of drift alone."""
    rng = np.random.default_rng(3)
    transducer = rng.normal(0.0, 1000.0, (300, 3)) + np.array([0.0, 0.0, 5000.0])
    times = np.linalg.norm(transducer, axis=1) / 1500.0
    J = compute_Jacobian_biased(
        np.zeros(3), transducer, times, np.full(times.size, 1500.0)
    )
    t_from_mid = np.linspace(-4e4, 4e4, times.size)
    delta = drift_step(J, 7e-10 * t_from_mid, t_from_mid)
    np.testing.assert_allclose(delta[5], -7e-10, rtol=1e-8)
    np.testing.assert_allclose(delta[:5], 0.0, atol=1e-9)
