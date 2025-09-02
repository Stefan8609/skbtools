import numpy as np

from MCMC_plots import corner_plot, get_init_params_and_prior
from data import gps_output_path


def threshold_posterior(
    chain, threshold=-50, save=False, chain_name=None, loglike=False
):
    initial_params, prior_scales = get_init_params_and_prior(chain)

    mask = chain["logpost"] > threshold

    # Build a reduced view of the chain by masking only arrays whose shape matches the
    # number of samples. Leave metadata arrays (scalars, constants) untouched.
    reduced_chain = {}
    nmask = mask.size
    for key in chain.files:
        if key in ("prior", "initial"):
            continue
        arr = chain[key]
        # Ensure ndarray
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        # Scalars or 0-d arrays: leave as-is
        if arr.ndim == 0:
            reduced_chain[key] = arr
            continue
        # If any axis matches the sample count, mask along the first matching axis
        if nmask in arr.shape:
            axis = next(i for i, s in enumerate(arr.shape) if s == nmask)
            index = [slice(None)] * arr.ndim
            index[axis] = mask
            reduced_chain[key] = arr[tuple(index)]
        else:
            # Keep arrays that don't have a matching sample dimension
            reduced_chain[key] = arr

    print(
        f"Reduced chain size from {len(chain['logpost'])} to"
        f" {len(reduced_chain['logpost'])} with threshold {threshold}"
    )

    name = (
        f"threshold_{threshold}_{chain_name}"
        if chain_name
        else f"threshold_{threshold}"
    )
    corner_plot(
        reduced_chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        downsample=1,
        save=save,
        chain_name=name,
        loglike=loglike,
    )
    return


if __name__ == "__main__":
    file_name = "mcmc_chain_8-7.npz"

    loglike = True
    save = True

    chain_name = file_name
    chain = np.load(gps_output_path(file_name))

    threshold_posterior(
        chain, threshold=-43, save=save, chain_name=chain_name, loglike=loglike
    )
