import numpy as np
import pytest

from Inversion_Workflow.PuertoRico.Crossover_Mask import crossover_leg_mask
from Inversion_Workflow.PuertoRico.PuertoRico_MCMC_sampler import save_mcmc_chain
from plotting.MCMC_Correlation import _initial_esv_mean
from plotting.MCMC_plots import get_chain_cdog_reference, get_init_params_and_prior


BERMUDA_MCMC_KEYS = {
    "lever",
    "CDOG_aug",
    "esv_bias",
    "time_bias",
    "loglike",
    "logpost",
    "init_lever",
    "init_gps_grid",
    "init_CDOG_aug",
    "init_esv_bias",
    "init_time_bias",
    "prior_lever",
    "prior_gps_grid",
    "prior_CDOG_aug",
    "prior_esv_bias",
    "prior_time_bias",
    "proposal_lever",
    "proposal_gps_grid",
    "proposal_CDOG_aug",
    "proposal_esv_bias",
    "proposal_time_bias",
}


def test_trajectory_mode_selects_full_or_cross_epochs():
    GPS_hours = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    eligible = np.array([True, True, False, True, True])
    crossover_leg_GPS_bounds = np.array([[1.5, 2.5], [3.5, 4.5]]) * 3600.0

    cross = eligible & crossover_leg_mask(
        GPS_hours * 3600.0,
        crossover_leg_GPS_bounds,
    )

    assert cross.tolist() == [False, True, False, True, False]


def test_correlation_plot_reads_each_surveys_initial_esv_bias():
    initial_esv_bias = np.array([[-0.12, -0.11], [-0.05, -0.04]])

    assert _initial_esv_mean(initial_esv_bias, 0) == pytest.approx(-0.115)
    assert _initial_esv_mean(initial_esv_bias, 1) == pytest.approx(-0.045)


def test_saved_chain_matches_bermuda_schema_and_retains_joint_metadata(tmp_path):
    samples = 3
    shared_aug = np.arange(samples * 3, dtype=float).reshape(samples, 1, 3)
    shared_aug = np.repeat(shared_aug, 2, axis=1)
    chains = (
        np.zeros((samples, 3)),
        shared_aug,
        np.zeros((samples, 2, 1)),
        np.zeros((samples, 2)),
        np.zeros(samples),
        np.zeros(samples),
    )
    initial_values = {
        "lever": np.zeros(3),
        "gps_grid": np.zeros((4, 3)),
        "CDOG_aug": np.zeros((2, 3)),
        "esv_bias": np.zeros((2, 1)),
        "time_bias": np.zeros(2),
    }
    prior_scales = {
        "lever": np.ones(3),
        "gps_grid": 0.1,
        "CDOG_aug": 0.5,
        "esv_bias": 1.0,
        "time_bias": 0.01,
    }
    proposal_scales = {
        "lever": np.ones(3) * 0.01,
        "gps_grid": 0.0,
        "CDOG_aug": 0.003,
        "esv_bias": 0.0006,
        "time_bias": 0.000002,
    }
    output_path = tmp_path / "chain.npz"

    save_mcmc_chain(
        output_path,
        chains,
        initial_values,
        prior_scales,
        proposal_scales,
        metadata={
            "CDOG_reference": np.ones(3),
            "survey_names": np.array(["before", "after"]),
        },
    )

    with np.load(output_path) as chain:
        assert BERMUDA_MCMC_KEYS <= set(chain.files)
        assert chain["CDOG_aug"].shape == (samples, 2, 3)
        assert np.array_equal(chain["CDOG_aug"][:, 0], chain["CDOG_aug"][:, 1])
        assert chain["esv_bias"].shape == (samples, 2, 1)
        assert chain["time_bias"].shape == (samples, 2)
        assert chain["survey_names"].tolist() == ["before", "after"]
        initial, priors, proposals = get_init_params_and_prior(chain)
        assert initial["CDOG_aug"].shape == (2, 3)
        assert priors["time_bias"] == prior_scales["time_bias"]
        assert proposals["esv_bias"] == proposal_scales["esv_bias"]
        assert np.array_equal(get_chain_cdog_reference(chain), np.ones(3))
