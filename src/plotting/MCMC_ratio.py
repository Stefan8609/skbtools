import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from .save import save_plot


def plot_marginal_density_ratio(
    samples,
    prior_mean,
    prior_sd,
    nbins=200,
    pct_trim=(1, 99),
    min_prior_pdf=1e-8,
    show_densities=False,
    save=False,
    chain_name=None,
    path="Figs",
):
    """
    Plot the ratio of posterior to prior marginal densities for a single parameter,
    trimming extremes and guarding against tiny denominators.

    Parameters
    ----------
    samples : array-like, shape (n_samples,)
        1D array of posterior draws for one parameter.
    prior_mean : float
        Mean of the Normal prior.
    prior_sd : float
        Std. dev. of the Normal prior.
    nbins : int, optional
        Number of grid points (default: 200).
    pct_trim : tuple, optional
        Lower and upper percentiles to trim the grid (default: (1, 99)).
    min_prior_pdf : float, optional
        Minimum allowed prior PDF, below which ratio is set to NaN.
    show_densities : bool, optional
        Overlay posterior KDE and prior PDF if True.
    """
    theta = np.asarray(samples).ravel()

    lo, hi = np.percentile(theta, pct_trim)
    grid = np.linspace(lo, hi, nbins)

    kde_post = gaussian_kde(theta)
    # kde_post.set_bandwidth(kde_post.factor * 1.2)  # example: increase smoothing
    post_pdf = kde_post(grid)
    prior_pdf = norm.pdf(grid, loc=prior_mean, scale=prior_sd)

    safe_prior = np.maximum(prior_pdf, min_prior_pdf)
    ratio = post_pdf / safe_prior

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, ratio, lw=2)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Posterior / Prior density ratio")
    ax.set_title("Marginal Density Ratio")
    ax.grid(True)

    if show_densities:
        ax2 = ax.twinx()
        ax2.plot(grid, post_pdf, ls="--", label="Posterior KDE")
        ax2.plot(grid, prior_pdf, ls=":", label="Prior PDF")
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    if save:
        save_plot(fig, chain_name, "plot_marginal_density_ratio", subdir=path)
    plt.show()

    return grid, ratio


if __name__ == "__main__":
    from data import gps_output_path

    file_name = "mcmc_chain_moonpool_better.npz"
    chain = np.load(gps_output_path(file_name))

    init_lever = np.array([-13.12, 9.72, -15.9])
    init_gps_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.393414, -4.223503, 0.029415],
            [-12.095685, -0.945685, 0.004397],
            [-8.686741, 5.169188, -0.024993],
        ]
    )
    init_aug = np.array(
        [
            [-397.63809, 371.47355, 773.26347],
            [825.31541, -110.93683, -734.15039],
            [236.27742, -1307.44426, -2189.59746],
        ]
    )

    prior_lever = np.array([0.5, 0.5, 0.5])
    prior_gps_grid = 0.1
    prior_CDOG_aug = 0.5

    lever_samples = chain["CDOG_aug"][:, 1, 0]

    plot_marginal_density_ratio(
        lever_samples, init_aug[1, 0], prior_CDOG_aug, show_densities=True
    )
