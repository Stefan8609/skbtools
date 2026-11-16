import numpy as np

from Inversion_Workflow.PuertoRico.PuertoRico_MCMC_best_sample import (
    load_best_sample,
)


def test_load_best_sample_uses_requested_score(tmp_path):
    chain_path = tmp_path / "chain.npz"
    np.savez(
        chain_path,
        lever=np.arange(9).reshape(3, 3),
        CDOG_aug=np.arange(9).reshape(3, 1, 3),
        esv_bias=np.arange(12).reshape(3, 2, 2),
        time_bias=np.arange(6).reshape(3, 2),
        loglike=np.array([3.0, 1.0, 2.0]),
        logpost=np.array([0.0, 2.0, 1.0]),
        init_lever=np.zeros(3),
        init_CDOG_aug=np.zeros((1, 3)),
        init_esv_bias=np.zeros((2, 2)),
        init_time_bias=np.zeros(2),
    )

    best_posterior = load_best_sample(chain_path)
    best_likelihood = load_best_sample(chain_path, loglike=True)

    np.testing.assert_array_equal(best_posterior["lever"], np.array([3, 4, 5]))
    np.testing.assert_array_equal(best_likelihood["lever"], np.array([0, 1, 2]))
