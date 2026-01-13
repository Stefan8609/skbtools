import numpy as np
import matplotlib.pyplot as plt
from data import gps_output_path

from plotting.MCMC_plots import get_init_params_and_prior, corner_plot


def plot_mcmc_posterior_histogram(
    chain, percentile, loglike=False, save=False, chain_name=None
):
    arr_cm = chain["loglike"] if loglike else chain["logpost"]
    arr_cm = np.sort(arr_cm)
    threshold_value = np.percentile(arr_cm, percentile)
    print(f"{percentile}th percentile value: {threshold_value}")

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(arr_cm, bins=50, density=True, alpha=0.7, color="blue")
    plt.axvline(threshold_value, color="red", linestyle="dashed", linewidth=1)
    for patch in patches:
        if patch.get_x() < threshold_value:
            patch.set_facecolor("orange")
    plt.xlabel("Log-Likelihood" if loglike else "Log-Posterior")
    plt.ylabel("Density")
    plt.title("MCMC Posterior Histogram")
    if save and chain_name:
        plt.savefig(gps_output_path(f"{chain_name}_histogram.png"))
    plt.show()


def percentile_plots(chain, percentile, loglike=False, save=False, chain_name=None):
    type = "loglike" if loglike else "logpost"
    initial_params, prior_scales, _ = get_init_params_and_prior(chain)
    arr_cm = chain[type]

    threshold = np.percentile(arr_cm, percentile)
    mask = chain[type] > threshold

    reduced_chain = {}
    nmask = mask.size
    for key in chain.files:
        if key in ("prior", "initial"):
            continue
        arr = chain[key]
        # Ensure ndarray
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
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
            reduced_chain[key] = arr
    reduced_length = len(reduced_chain[type])
    downsample = reduced_length // 10000 if reduced_length > 10000 else 1
    print(reduced_length, downsample)
    corner_plot(
        reduced_chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        downsample=downsample,
        save=save,
        chain_name=chain_name,
        loglike=loglike,
    )

    return


if __name__ == "__main__":
    file_name = "mcmc_chain_9_18_no_grid"
    loglike = False
    save = False

    chain_name = ("loglike_" if loglike else "logpost_") + file_name

    chain = np.load(gps_output_path(f"{file_name}.npz"))
    plot_mcmc_posterior_histogram(
        chain, 95, loglike=loglike, save=save, chain_name=chain_name
    )
    percentile_plots(chain, 95, loglike=loglike, save=save, chain_name=chain_name)
