import numpy as np
from plotting.MCMC_plots import get_init_params_and_prior


def print_MCMC_inputs(npz_path):
    """
    Print the initial parameters and prior scales from an MCMC chain stored in
    a .npz file.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing the MCMC chain.
    """
    chain = np.load(npz_path, allow_pickle=True)
    initial_params, prior_scales, proposal_scales = get_init_params_and_prior(chain)

    print("Initial Parameters:")
    for key, value in initial_params.items():
        print(f"{key}: {value}")

    print("\nPrior Scales:")
    for key, value in prior_scales.items():
        print(f"{key}: {value}")

    print("\n Proposal Scales:")
    for key, value in proposal_scales.items():
        print(f"{key}: {value}")

    return


if __name__ == "__main__":
    from data import gps_output_path

    file_name = "mcmc_chain_8-7.npz"
    npz_path = gps_output_path(file_name)
    print_MCMC_inputs(npz_path)
