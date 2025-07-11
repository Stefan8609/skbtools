import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from data import gps_output_path, gps_data_path
import itertools


def _save_fig(fig, save, tag, timestamp, ext="pdf"):
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


def marginal_hists(chain, initial_params=None, save=False, timestamp=None):
    """Plot marginal histograms of the MCMC parameters."""
    # Marginal Histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    axes[0].hist(chain["lever"][:, 0], bins=30)
    axes[0].set_title("lever x")
    axes[1].hist(chain["lever"][:, 1], bins=30)
    axes[1].set_title("lever y")
    axes[2].hist(chain["lever"][:, 2], bins=30)
    axes[2].set_title("lever z")
    axes[3].hist(chain["CDOG_aug"][:, 0, 0], bins=30)
    axes[3].set_title("CDOG_aug x")
    axes[3].set_xlabel("CDOG Location Augment x (m)")
    axes[4].hist(chain["CDOG_aug"][:, 0, 1], bins=30)
    axes[4].set_title("CDOG_aug y")
    axes[4].set_xlabel("CDOG Location Augment y (m)")
    axes[5].hist(chain["CDOG_aug"][:, 0, 2], bins=30)
    axes[5].set_title("CDOG_aug z")
    axes[5].set_xlabel("CDOG Location Augment z (m)")

    if initial_params:
        axes[0].axvline(
            initial_params["lever"][0], color="red", linestyle="--", label="Initial x"
        )
        axes[1].axvline(
            initial_params["lever"][1], color="red", linestyle="--", label="Initial y"
        )
        axes[2].axvline(
            initial_params["lever"][2], color="red", linestyle="--", label="Initial z"
        )
        axes[3].axvline(
            initial_params["CDOG_aug"][0, 0],
            color="red",
            linestyle="--",
            label="Initial CDOG_aug x",
        )
        axes[4].axvline(
            initial_params["CDOG_aug"][0, 1],
            color="red",
            linestyle="--",
            label="Initial CDOG_aug y",
        )
        axes[5].axvline(
            initial_params["CDOG_aug"][0, 2],
            color="red",
            linestyle="--",
            label="Initial CDOG_aug z",
        )

    _save_fig(fig, save=save, tag="marginalhists", timestamp=timestamp)

    plt.show()


def corner_plot(chain, initial_params=None, downsample=1, save=False, timestamp=None):
    """Plot a corner plot of the posterior samples."""
    # Extract parameter arrays
    pars = {
        "lx": chain["lever"][::downsample, 0],
        "ly": chain["lever"][::downsample, 1],
        "lz": chain["lever"][::downsample, 2],
        "augx": chain["CDOG_aug"][::downsample, 0, 0],
        "augy": chain["CDOG_aug"][::downsample, 0, 1],
        "augz": chain["CDOG_aug"][::downsample, 0, 2],
    }
    if initial_params:
        initial_params = {
            "lx": initial_params["lever"][0],
            "ly": initial_params["lever"][1],
            "lz": initial_params["lever"][2],
            "augx": initial_params["CDOG_aug"][0, 0],
            "augy": initial_params["CDOG_aug"][0, 1],
            "augz": initial_params["CDOG_aug"][0, 2],
        }

    keys = list(pars)
    n = len(keys)

    # Set up figure and axes
    fig, axes = plt.subplots(n, n, figsize=(12, 8))

    # Normalize log-posterior for color mapping
    logpost = chain["logpost"][::downsample]
    norm = mpl.colors.Normalize(vmin=-80, vmax=logpost.max())
    cmap = plt.get_cmap("viridis")

    # Loop over panels
    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]
        if i == j:
            ax.hist(pars[keys[i]], bins=30, color="gray")
            if initial_params:
                ax.axvline(
                    initial_params[keys[i]],
                    color="red",
                    linestyle="--",
                    label="Initial",
                )
        else:
            sc = ax.scatter(
                pars[keys[j]],
                pars[keys[i]],
                c=logpost,
                cmap=cmap,
                norm=norm,
                s=1,
                alpha=0.8,
            )
            if initial_params:
                ax.scatter(
                    initial_params[keys[j]],
                    initial_params[keys[i]],
                    color="red",
                    marker="x",
                    s=50,
                    label="Initial",
                )
        # Labeling
        if i == n - 1:
            ax.set_xlabel(keys[j])
        if j == 0:
            ax.set_ylabel(keys[i])
        # Turn off upper triangle if you only want the lower
        if j > i:
            ax.set_visible(False)

    # Adjust layout and add colorbar
    plt.tight_layout()
    # place colorbar on the right spanning all rows
    cbar = fig.colorbar(sc, ax=axes[:, :], location="right", shrink=0.9)
    cbar.set_label("log posterior")

    _save_fig(fig, save=save, tag="cornerplot", timestamp=timestamp)

    plt.show()


if __name__ == "__main__":
    # Initial Parameters for adding to plot
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    init_lever = np.array([-12.4659, 9.6021, -13.2993])
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

    initial_params = {
        "lever": init_lever,
        "gps_grid": init_gps_grid,
        "CDOG_aug": init_aug,
        "esv_bias": init_ebias,
        "time_bias": init_tbias,
    }

    chain = np.load(gps_output_path("mcmc_chain_adroit_1.npz"))

    # Works for chains saved with either a single or split ESV bias term
    trace_plot(
        chain,
        initial_params=initial_params,
        downsample=1000,
        save=True,
        timestamp=timestamp,
    )
    marginal_hists(chain, initial_params=initial_params, save=True, timestamp=timestamp)
    corner_plot(
        chain,
        initial_params=initial_params,
        downsample=1000,
        save=True,
        timestamp=timestamp,
    )
