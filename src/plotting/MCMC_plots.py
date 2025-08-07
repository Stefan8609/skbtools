import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from data import gps_output_path, gps_data_path
from scipy.stats import norm
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse
import itertools


def _save_fig(fig, save, tag, timestamp=None, ext="pdf"):
    """Helper to save `fig` under Figs/MCMC/<timestamp>/ with timestamped filename."""
    if not save:
        return

    # 1) build timestamp and filename
    fname = f"{tag}_{timestamp}.{ext}"

    # 2) build the directory path and ensure it exists
    #    gps_data_path should return the root data directory, e.g. '/Users/…/Data'
    dirpath = gps_data_path(f"Figs/MCMC/{timestamp}")
    os.makedirs(dirpath, exist_ok=True)

    # 3) save into that directory
    fullpath = os.path.join(dirpath, fname)
    fig.savefig(fullpath, format=ext, bbox_inches="tight")


def get_init_params_and_prior(chain):
    """
    Extract prior and initial parameters from the MCMC chain.

    Parameters
    ----------
    chain : dict
        Dictionary containing MCMC chain data.

    Returns
    -------
    tuple
        Initial parameters and prior scales.
    """
    try:
        init_lever = chain["init_lever"]
        init_gps_grid = chain["init_gps_grid"]
        init_aug = chain["init_CDOG_aug"]
        init_ebias = chain["init_esv_bias"]
        init_tbias = chain["init_time_bias"]

        prior_lever = chain["prior_lever"]
        prior_gps_grid = chain["prior_gps_grid"]
        prior_aug = chain["prior_CDOG_aug"]
        prior_esv_bias = chain["prior_esv_bias"]
        prior_time_bias = chain["prior_time_bias"]

    except KeyError:
        print(
            "Using default initial values for lever, GPS grid, "
            "CDOG_aug, ESV bias, and time bias"
        )
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
        init_ebias = np.array([-0.4775, -0.3199, 0.1122])
        init_tbias = np.array([0.01518602, 0.015779, 0.018898])

        prior_lever = np.array([0.3, 0.3, 0.3])
        prior_gps_grid = 0.1
        prior_aug = 0.5
        prior_esv_bias = 1.0
        prior_time_bias = 0.5

    initial_params = {
        "lever": init_lever,
        "gps_grid": init_gps_grid,
        "CDOG_aug": init_aug,
        "esv_bias": init_ebias,
        "time_bias": init_tbias,
    }

    prior_scales = {
        "lever": prior_lever,
        "gps_grid": prior_gps_grid,
        "CDOG_aug": prior_aug,
        "esv_bias": prior_esv_bias,
        "time_bias": prior_time_bias,
    }

    return initial_params, prior_scales


def trace_plot(chain, initial_params=None, downsample=1, save=False, timestamp=None):
    """Plot traces of MCMC parameters, with per-DOG ESV bias
    and time_bias mean-centered, including units."""
    DOG_index_num = {0: 1, 1: 3, 2: 4}

    # Downsample
    lever = chain["lever"][::downsample]  # (n_iter, 3), units: meters
    esv = chain["esv_bias"][
        ::downsample
    ]  # could be (n_iter,), (n_iter, n_splits) or (n_iter, n_dogs, n_splits)
    tb = chain["time_bias"][
        ::downsample
    ]  # (n_iter,) or (n_iter, n_dogs), units: seconds
    rmse = chain["logpost"][::downsample] * -2  # units: ms

    # --- ensure esv is 3D: (n_iter, n_dogs, n_splits) ---
    n_iter = esv.shape[0]
    if esv.ndim == 1:
        # single DOG, single split
        esv = esv.reshape(n_iter, 1, 1)
    elif esv.ndim == 2:
        # assume shape (n_iter, n_splits) → single DOG
        esv = esv.reshape(n_iter, 1, esv.shape[1])
    elif esv.ndim != 3:
        raise ValueError(f"esv_bias must be 1-, 2- or 3-D; got shape {esv.shape}")
    n_dogs, n_splits = esv.shape[1], esv.shape[2]

    # --- ensure tb is 2D: (n_iter, n_tb) ---
    tb = tb.reshape(n_iter, -1)
    n_tb = tb.shape[1]

    # Mean-center ESV and time_bias
    esv_mean = esv.mean(axis=0)
    esv_centered = esv - esv_mean[np.newaxis, ...]
    tb_mean = tb.mean(axis=0)
    tb_centered = tb - tb_mean[np.newaxis, :]

    # Build subplots
    n_rows = 3 + n_dogs + 1 + 1  # lever(3) + esv + time bias + rmse
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 1.5 * n_rows), sharex=True)

    # --- Lever-arm (meters) ---
    axes[0].plot(lever[:, 0])
    axes[0].set_ylabel("lever x (m)")
    axes[1].plot(lever[:, 1])
    axes[1].set_ylabel("lever y (m)")
    axes[2].plot(lever[:, 2])
    axes[2].set_ylabel("lever z (m)")

    # --- ESV bias per DOG (mean-centered, m/s) ---
    for j in range(n_dogs):
        ax = axes[3 + j]
        DOG_num = DOG_index_num.get(j, j + 1)
        for k in range(n_splits):
            ax.plot(esv_centered[:, j, k], linewidth=0.8, label=f"split {k + 1}")
        ax.set_ylabel(f"ESV {DOG_num} (m/s)")
        ax.legend(fontsize="x-small", ncol=min(n_splits, 3), loc="upper right")

    # --- time_bias (mean-centered, s) ---
    ax_tb = axes[3 + n_dogs]
    for j in range(n_tb):
        DOG_num = DOG_index_num.get(j, j + 1)
        ax_tb.plot(tb_centered[:, j], linewidth=0.8, label=f"Time bias {DOG_num}")
    ax_tb.set_ylabel("time bias (s)")
    ax_tb.legend(fontsize="x-small", ncol=min(n_tb, 3), loc="upper right")

    # --- RMSE (ms → shown in cs?) ---
    axes[4 + n_dogs].plot(rmse)
    axes[4 + n_dogs].set_ylabel("RMSE (ms)")

    # Optional initial params (also mean-centered)
    if initial_params:
        # lever horiz lines
        for i, _ in enumerate(["x", "y", "z"]):
            axes[i].axhline(initial_params["lever"][i], color="r", ls="--")

        eb0 = initial_params.get("esv_bias", None)
        if eb0 is not None:
            eb0 = np.asarray(eb0).ravel()

            # compute a single value per DOG by averaging across splits if needed
            if eb0.size == n_dogs * n_splits:
                eb0 = eb0.reshape(n_dogs, n_splits).mean(axis=1)
            elif eb0.size == n_dogs:
                # already one per dog
                eb0 = eb0
            else:
                raise ValueError(
                    f"initial esv_bias has {eb0.size} entries; "
                    f"expected {n_dogs * n_splits} or {n_dogs}"
                )

            # center against the mean ESV per dog
            esv_mean_per_dog = esv_mean.mean(axis=1)
            eb0_centered = eb0 - esv_mean_per_dog

            # draw one line per DOG
            for j in range(n_dogs):
                ax = axes[3 + j]
                ax.axhline(eb0_centered[j], color="r", ls="--", linewidth=0.7)

        # time_bias init
        tb0 = initial_params.get("time_bias", None)
        if tb0 is not None:
            tb0 = np.asarray(tb0).reshape(-1)[:n_tb]
            tb0_centered = tb0 - tb_mean
            for j in range(n_tb):
                ax_tb.axhline(tb0_centered[j], color="r", ls="--", linewidth=0.7)

    for ax in axes:
        ax.margins(x=0)
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()

    _save_fig(fig, save=save, tag="traceplot", timestamp=timestamp)
    plt.show()


def marginal_hists(
    chain, initial_params=None, prior_scales=None, save=False, timestamp=None
):
    """Plot marginal histograms of the MCMC parameters with optional prior overlays."""
    # Define keys and data arrays
    keys = ["lever_x", "lever_y", "lever_z", "CDOG_aug_x", "CDOG_aug_y", "CDOG_aug_z"]
    data = [
        chain["lever"][:, 0],
        chain["lever"][:, 1],
        chain["lever"][:, 2],
        chain["CDOG_aug"][:, 0, 0],
        chain["CDOG_aug"][:, 0, 1],
        chain["CDOG_aug"][:, 0, 2],
    ]

    # Flatten initial parameters for centering priors
    flat_init = {}
    if initial_params is not None:
        flat_init = {
            "lever_x": initial_params["lever"][0],
            "lever_y": initial_params["lever"][1],
            "lever_z": initial_params["lever"][2],
            "CDOG_aug_x": initial_params["CDOG_aug"][0, 0],
            "CDOG_aug_y": initial_params["CDOG_aug"][0, 1],
            "CDOG_aug_z": initial_params["CDOG_aug"][0, 2],
        }

    # Flatten prior scales for easy lookup using direct indexing
    flat_prior = {}
    if prior_scales is not None:
        flat_prior = {
            "lever_x": prior_scales["lever"][0],
            "lever_y": prior_scales["lever"][1],
            "lever_z": prior_scales["lever"][2],
            "CDOG_aug_x": prior_scales["CDOG_aug"],
            "CDOG_aug_y": prior_scales["CDOG_aug"],
            "CDOG_aug_z": prior_scales["CDOG_aug"],
        }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for key, arr, ax in zip(keys, data, axes):
        # Plot normalized histogram
        counts, bins, _ = ax.hist(arr, bins=30, density=True, color="gray", alpha=0.6)
        ax.set_title(key.replace("_", " "))

        # Overlay prior distribution if provided
        if flat_prior:
            scale = flat_prior[key]
            mean = flat_init.get(key, 0)
            xgrid = np.linspace(bins.min(), bins.max(), 200)
            prior_pdf = norm.pdf(xgrid, loc=mean, scale=scale)
            ax.plot(xgrid, prior_pdf, color="blue", lw=1.5, label="Prior")

        # Overlay initial parameter
        if key in flat_init:
            ax.axvline(flat_init[key], color="red", linestyle="--", label="Initial")

        if flat_prior or key in flat_init:
            ax.legend(fontsize="small")

    _save_fig(fig, save=save, tag="marginalhists", timestamp=timestamp)
    plt.tight_layout()
    plt.show()


def corner_plot(
    chain,
    initial_params=None,
    prior_scales=None,
    downsample=1,
    save=False,
    timestamp=None,
    loglike=False,
):
    """Plot a corner plot of the posterior samples with
    ubest-fit points on top and 2D prior contours."""
    # Extract and sort by log-posterior ascending so best (highest) are plotted last
    if loglike:
        raw_logpost = chain["loglike"][::downsample]
    else:
        raw_logpost = chain["logpost"][::downsample]
    order = np.argsort(raw_logpost)
    logpost = raw_logpost[order]

    pars = {
        "lx": chain["lever"][::downsample, 0][order],
        "ly": chain["lever"][::downsample, 1][order],
        "lz": chain["lever"][::downsample, 2][order],
        "augx": chain["CDOG_aug"][::downsample, 0, 0][order],
        "augy": chain["CDOG_aug"][::downsample, 0, 1][order],
        "augz": chain["CDOG_aug"][::downsample, 0, 2][order],
    }

    flat_init = {}
    if initial_params is not None:
        flat_init = {
            "lx": initial_params["lever"][0],
            "ly": initial_params["lever"][1],
            "lz": initial_params["lever"][2],
            "augx": initial_params["CDOG_aug"][0, 0],
            "augy": initial_params["CDOG_aug"][0, 1],
            "augz": initial_params["CDOG_aug"][0, 2],
        }

    flat_prior = {}
    if prior_scales is not None:
        flat_prior = {
            "lx": prior_scales["lever"][0],
            "ly": prior_scales["lever"][1],
            "lz": prior_scales["lever"][2],
            "augx": prior_scales["CDOG_aug"],
            "augy": prior_scales["CDOG_aug"],
            "augz": prior_scales["CDOG_aug"],
        }

    keys = list(pars.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(12, 8))

    # Cap the colorbar at 80
    min_val = -65
    norm_cmap = mpl.colors.Normalize(vmin=min_val, vmax=logpost.max())
    cmap = plt.get_cmap("viridis")

    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]
        key_i, key_j = keys[i], keys[j]
        if i == j:
            arr = pars[key_i]
            ax.hist(arr, bins=30, density=True, color="gray", alpha=0.6)
            if flat_prior:
                sx = flat_prior[key_i]
                mx = flat_init.get(key_i, 0)
                x0, x1 = ax.get_xlim()
                xg = np.linspace(x0, x1, 200)
                ax.plot(
                    xg,
                    norm.pdf(xg, loc=mx, scale=sx),
                    color="blue",
                    lw=1.5,
                    label="Prior",
                )
            if key_i in flat_init:
                ax.axvline(
                    flat_init[key_i], color="red", ls="--", lw=1.2, label="Initial"
                )
            if i == 0 and (flat_prior or key_i in flat_init):
                ax.legend(fontsize="x-small")
        elif j < i:
            # scatter with best-fit on top (cap color)
            sc = ax.scatter(
                pars[key_j],
                pars[key_i],
                c=logpost,
                cmap=cmap,
                norm=norm_cmap,
                s=1,
                alpha=0.7,
            )
            if flat_prior:
                # build 2×2 prior covariance
                sx = flat_prior[key_j]
                sy = flat_prior[key_i]
                mx = flat_init.get(key_j, 0)
                my = flat_init.get(key_i, 0)
                cov = np.array([[sx**2, 0], [0, sy**2]])

                # draw two confidence ellipses (e.g. 68% and 95%)
                for conf in [0.68, 0.95]:
                    ellipse = plot_prior_ellipse(
                        mean=(mx, my), cov=cov, confidence=conf, zorder=3
                    )
                    ax.add_patch(ellipse)

            if key_j in flat_init and key_i in flat_init:
                ax.scatter(
                    flat_init[key_j],
                    flat_init[key_i],
                    color="red",
                    marker="x",
                    s=40,
                    label="Initial",
                )
        else:
            ax.set_visible(False)
        if i == n - 1:
            ax.set_xlabel(key_j)
        if j == 0:
            ax.set_ylabel(key_i)

    label = "log likelihood" if loglike else "log posterior"
    fig.colorbar(
        sc,
        ax=axes[:, :],
        label=label,
        location="right",
        shrink=0.9,
        extend="max",
    )

    _save_fig(fig, save=save, tag="cornerplot", timestamp=timestamp)
    plt.show()


if __name__ == "__main__":
    # Initial Parameters for adding to plot
    file_name = "7_individual_splits_esv_20250806_165630/split_4.npz"
    loglike = True
    save = False

    if loglike:
        timestamp = "loglike_" + file_name
    else:
        timestamp = "logpost_" + file_name

    chain = np.load(gps_output_path(file_name))

    initial_params, prior_scales = get_init_params_and_prior(chain)

    trace_plot(
        chain,
        initial_params=initial_params,
        downsample=1,
        save=save,
        timestamp=timestamp,
    )
    marginal_hists(
        chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        save=save,
        timestamp=timestamp,
    )
    corner_plot(
        chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        downsample=10,
        save=save,
        timestamp=timestamp,
        loglike=loglike,
    )
