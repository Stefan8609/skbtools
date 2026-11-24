import numpy as np
import pytest
from pymap3d import ecef2enu, ecef2geodetic

from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from Inversion_Workflow.MCMC_Functions.MCMC_diagnostics import (
    ess,
    ess_bulk,
    ess_tail,
    rank_normalized_split_rhat,
)
from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
    SUBMITTED_TABLE1_AUG_M,
    enu_rotation,
    linearized_comparison,
    position_covariance_enu,
    reported_quantities,
)
from test_mcmc_sampler import synthetic_problem


def ar1_chains(phi, n_chains=4, n_draws=4000, seed=0):
    rng = np.random.default_rng(seed)
    draws = np.empty((n_chains, n_draws))
    draws[:, 0] = rng.normal(size=n_chains)
    for t in range(1, n_draws):
        draws[:, t] = phi * draws[:, t - 1] + np.sqrt(1 - phi**2) * rng.normal(
            size=n_chains
        )
    return draws


def test_rhat_is_one_for_mixed_chains_and_large_for_separated_chains():
    mixed = ar1_chains(0.5)
    separated = mixed + np.array([0.0, 0.0, 0.0, 1.0])[:, None]

    assert rank_normalized_split_rhat(mixed) < 1.01
    assert rank_normalized_split_rhat(separated) > 1.1


def test_ess_matches_independent_and_ar1_draws():
    white = ar1_chains(0.0)
    correlated = ar1_chains(0.9)
    n = white.size

    assert ess(white) == pytest.approx(n, rel=0.1)
    # For AR(1) the effective sample size is n (1 - phi) / (1 + phi).
    assert ess(correlated) == pytest.approx(n * 0.1 / 1.9, rel=0.25)
    assert ess_bulk(correlated) == pytest.approx(n * 0.1 / 1.9, rel=0.25)
    assert 0 < ess_tail(correlated) < ess_bulk(white)


def test_enu_rotation_matches_pymap3d():
    CDOG_reference = np.array(ms.CONFIG["CDOG_reference"])
    lat, lon, height = ecef2geodetic(*CDOG_reference)
    offset = np.array([12.0, -7.0, 30.0])

    expected = ecef2enu(*(CDOG_reference + offset), lat, lon, height)

    np.testing.assert_allclose(
        enu_rotation(lat, lon) @ offset, expected, rtol=0, atol=1e-6
    )


def test_reported_quantities_follow_the_paper_conventions():
    config = ms.run_config("ar1_drift")
    model = ms.build_model(config)
    layout = model["layout"]
    offsets = np.array(config["offsets_s"])
    CDOG_reference = np.array(config["CDOG_reference"])
    names, _, A, b = reported_quantities(model, offsets, CDOG_reference)
    theta = model["start_theta"].copy()
    theta[layout["CDOG_aug"]] = np.ravel([SUBMITTED_TABLE1_AUG_M[d] for d in (1, 3, 4)])
    theta[layout["time_bias"]] = [1e-4, 0.0, -2e-4]
    theta[layout["esv_bias"].start : layout["esv_bias"].start + 4] = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    value = dict(zip(names, A @ theta + b, strict=False))

    # dt in the Eq. 6 convention: offset - time_bias.
    assert value["DOG1_dt_Eq6"] == pytest.approx(offsets[0] - 1e-4, abs=1e-12)
    assert value["DOG4_dt_Eq6"] == pytest.approx(offsets[2] + 2e-4, abs=1e-12)
    assert value["DOG1_dc_mean"] == pytest.approx(0.25)
    for axis in "ENU":
        assert value[f"DOG3_{axis}_minus_Table1"] == pytest.approx(0.0, abs=1e-6)
    assert value["lever_z_minus_DOG3_U"] == pytest.approx(theta[2], abs=1e-6)


def test_position_covariance_enu_rotates_each_dog_into_its_own_frame():
    config = ms.run_config("ar1_drift")
    model = ms.build_model(config)
    CDOG_reference = np.array(config["CDOG_reference"])
    aug = model["layout"]["CDOG_aug"]
    rng = np.random.default_rng(0)
    theta = np.tile(model["start_theta"], (4000, 1))
    # DOG 3 moves only along its local up (sd 0.1 m), DOG 4 only east (sd 0.2 m).
    for j, sd, axis in ((1, 0.1, 2), (2, 0.2, 0)):
        start = aug.start + 3 * j
        lat, lon, _ = ecef2geodetic(*(theta[0, start : start + 3] + CDOG_reference))
        direction = enu_rotation(lat, lon)[axis]
        theta[:, start : start + 3] += sd * rng.normal(size=(4000, 1)) * direction

    rows = position_covariance_enu(theta, model, CDOG_reference)
    names = [row["quantity"] for row in rows]
    covariance = np.array([[row[name] for name in names] for row in rows])

    assert names[3:6] == ["DOG3_E", "DOG3_N", "DOG3_U"]
    np.testing.assert_allclose(covariance, covariance.T)
    expected_diagonal = [0, 0, 0, 0, 0, 0.01, 0.04, 0, 0]
    np.testing.assert_allclose(np.diag(covariance), expected_diagonal, atol=2e-3)
    assert covariance[3, 5] == pytest.approx(0.0, abs=1e-8)


def test_old_plot_helpers_read_the_new_prior():
    import json

    import matplotlib as mpl

    with mpl.rc_context():  # MCMC_plots switches on usetex when imported
        from plotting.MCMC_plots import get_init_params_and_prior

    config = ms.run_config("ar1")
    chain = {"prior_center": np.zeros(1), "config_json": np.array(json.dumps(config))}

    centers, scales, proposals = get_init_params_and_prior(chain)

    assert centers["lever"].tolist() == [-13.1, 9.7, -14.0]
    assert centers["CDOG_aug"].shape == (3, 3)
    assert np.all(centers["time_bias"] == 0.0)
    assert scales["lever"].tolist() == [0.5, 0.5, 2.0]
    assert scales["esv_mean"] == pytest.approx(2.0 / np.sqrt(4))
    assert proposals == {}


def test_linearized_comparison_leaves_out_an_esv_split_without_data():
    # The DOG records only the 40 epochs before the gap, so the last of three
    # ESV splits (epochs 40-59) has no pairs and a zero Jacobian column.
    config, problem = synthetic_problem(num_esv_splits=3, n_arrivals=40)
    model = ms.build_model(config)
    rng = np.random.default_rng(0)
    theta = model["start_theta"] + 1e-3 * rng.normal(
        size=(200, model["start_theta"].size)
    )
    logpost = rng.normal(size=200)
    offsets = np.array(config["offsets_s"])
    CDOG_reference = np.array(config["CDOG_reference"])

    # 40 s of a 1 km circle barely separates lever arm from DOG position, so the
    # formal covariance is near-singular; only its NaN pattern is tested here.
    with np.errstate(invalid="ignore"):
        sd_rows, _, condition = linearized_comparison(
            theta, logpost, model, problem, offsets, CDOG_reference
        )

    assert condition["unconstrained_parameters"] == ["esv_bias_DOG1_split2"]
    assert np.isfinite(condition["cond_JTWJ_unit_columns"])
    formal_sd = {row["quantity"]: row["formal_iid_sd"] for row in sd_rows}
    for name in ("DOG1_dc_split2", "DOG1_dc_mean"):
        assert np.isnan(formal_sd[name])
    for name in ("lever_z", "DOG1_U_minus_Table1", "DOG1_dt_Eq6", "DOG1_dc_split0"):
        assert np.isfinite(formal_sd[name])
