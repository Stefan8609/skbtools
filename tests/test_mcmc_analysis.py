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
    reported_quantities,
)


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

    value = dict(zip(names, A @ theta + b))

    # dt in the Eq. 6 convention: offset - time_bias.
    assert value["DOG1_dt_Eq6"] == pytest.approx(offsets[0] - 1e-4, abs=1e-12)
    assert value["DOG4_dt_Eq6"] == pytest.approx(offsets[2] + 2e-4, abs=1e-12)
    assert value["DOG1_dc_mean"] == pytest.approx(0.25)
    for axis in "ENU":
        assert value[f"DOG3_{axis}_minus_Table1"] == pytest.approx(0.0, abs=1e-6)
    assert value["lever_z_minus_DOG3_U"] == pytest.approx(theta[2], abs=1e-6)
