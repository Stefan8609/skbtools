import numpy as np

from MCMC_plots import corner_plot, get_init_params_and_prior
from data import gps_output_path


def threshold_posterior(chain, threshold=-50, save=False, timestamp=None):
    initial_params, prior_scales = get_init_params_and_prior(chain)

    mask = chain["logpost"] > threshold

    reduced_chain = {}
    for key in chain.files:
        if key == "prior" or key == "initial":
            continue
        reduced_chain[key] = chain[key][mask]

    print(
        f"Reduced chain size from {len(chain['logpost'])} to"
        f" {len(reduced_chain['logpost'])} with threshold {threshold}"
    )

    corner_plot(
        reduced_chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        downsample=1,
        save=save,
        timestamp=f"threshold_{threshold}_" + timestamp,
        loglike=loglike,
    )
    return


if __name__ == "__main__":
    file_name = "mcmc_chain_moonpool_better.npz"

    loglike = True
    save = True

    if loglike:
        timestamp = file_name
    else:
        timestamp = file_name
    chain = np.load(gps_output_path(file_name))

    threshold_posterior(chain, threshold=-45, save=save, timestamp=timestamp)
