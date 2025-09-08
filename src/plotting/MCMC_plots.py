import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
from pymap3d import geodetic2enu
from geometry.ECEF_Geodetic import ECEF_Geodetic
from data import gps_output_path
from scipy.stats import norm
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse
from plotting.save import save_plot
import itertools


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


# === Helper: Convert CDOG_aug samples from ECEF offsets to ENU coordinates ===
def cdog_aug_ecef_to_enu(
    aug_samples,
    prior_mean,
    CDOG_reference=None,
):
    """
    Convert CDOG augment samples from ECEF offsets to ENU coordinates (meters).

    Parameters
    ----------
    aug_samples : (N, 3) array
        Samples of CDOG augment in ECEF (meters) relative to CDOG_reference.
    prior_mean : (3,) array
        Prior mean of the CDOG augment in ECEF (meters) used to define the ENU origin.
    CDOG_reference : (3,) array, optional
        Absolute ECEF position of the CDOG reference point.

    Returns
    -------
    enu : (N, 3) array
        ENU coordinates (E, N, U) in meters for each sample.
    """
    if CDOG_reference is None:
        CDOG_reference = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    aug_samples = np.asarray(aug_samples)
    if aug_samples.ndim != 2 or aug_samples.shape[1] != 3:
        raise ValueError(f"aug_samples must have shape (N, 3); got {aug_samples.shape}")

    # Reference (origin) is the prior-mean location of the aug in geodetic coords
    CDOG_loc = np.asarray(prior_mean) + np.asarray(CDOG_reference)
    CDOG_lat, CDOG_lon, CDOG_height = ECEF_Geodetic(np.array([CDOG_loc]))
    CDOG_lat = float(np.squeeze(CDOG_lat))
    CDOG_lon = float(np.squeeze(CDOG_lon))
    CDOG_height = float(np.squeeze(CDOG_height))

    # Convert each sample to geodetic and then to ENU relative to the reference
    enu = np.zeros_like(aug_samples, dtype=float)
    for i in range(aug_samples.shape[0]):
        ecef_abs = aug_samples[i] + CDOG_reference
        lat, lon, h = ECEF_Geodetic(np.array([ecef_abs]))
        lat = float(np.squeeze(lat))
        lon = float(np.squeeze(lon))
        h = float(np.squeeze(h))
        e, n, u = geodetic2enu(lat, lon, h, CDOG_lat, CDOG_lon, CDOG_height)
        enu[i] = np.array([float(e), float(n), float(u)], dtype=float)

    return enu


def trace_plot(chain, initial_params=None, downsample=1, save=False, chain_name=None):
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

    if save:
        save_plot(fig, chain_name, "trace_plot", f"Figs/MCMC/{chain_name}")
    plt.show()


def marginal_hists(
    chain,
    initial_params=None,
    prior_scales=None,
    downsample=1,
    save=False,
    chain_name=None,
):
    """Plot marginal distributions in cm, centered at the posterior mode, with
    prior overlay, posterior Gaussian overlay, std annotations, and a common
    extent across panels."""
    # Convert CDOG_aug (first DOG) from ECEF offsets to ENU (meters)
    cdog_aug_samples = chain["CDOG_aug"][
        ::downsample, 0, :
    ]  # (N, 3) in meters (ECEF offsets)
    if initial_params is not None:
        prior_mean_aug = initial_params["CDOG_aug"][0]
    else:
        # Fallback: use the mean of samples as the origin if initial not provided
        prior_mean_aug = np.mean(cdog_aug_samples, axis=0)
    cdog_aug_enu = cdog_aug_ecef_to_enu(cdog_aug_samples, prior_mean_aug)

    # Marginals to plot (lever and CDOG_aug in ENU, 3 components each)
    keys = [
        "lever_x",
        "lever_y",
        "lever_z",
        "CDOG_E",
        "CDOG_N",
        "CDOG_U",
    ]
    data = [
        chain["lever"][::downsample, 0],
        chain["lever"][::downsample, 1],
        chain["lever"][::downsample, 2],
        cdog_aug_enu[:, 0],  # East (m)
        cdog_aug_enu[:, 1],  # North (m)
        cdog_aug_enu[:, 2],  # Up (m)
    ]

    # Flatten initial parameters for centering priors
    flat_init = {}
    if initial_params is not None:
        flat_init = {
            "lever_x": initial_params["lever"][0],
            "lever_y": initial_params["lever"][1],
            "lever_z": initial_params["lever"][2],
        }
        # Convert the prior mean of CDOG_aug to ENU for plotting prior mean lines
        enu0 = cdog_aug_ecef_to_enu(
            initial_params["CDOG_aug"][0][None, :], initial_params["CDOG_aug"][0]
        )[0]
        flat_init.update(
            {
                "CDOG_E": enu0[0],
                "CDOG_N": enu0[1],
                "CDOG_U": enu0[2],
            }
        )

    # Flatten prior scales for easy lookup using direct indexing
    flat_prior = {}
    if prior_scales is not None:
        flat_prior = {
            "lever_x": prior_scales["lever"][0],
            "lever_y": prior_scales["lever"][1],
            "lever_z": prior_scales["lever"][2],
        }
        flat_prior.update(
            {
                "CDOG_E": prior_scales["CDOG_aug"],
                "CDOG_N": prior_scales["CDOG_aug"],
                "CDOG_U": prior_scales["CDOG_aug"],
            }
        )

    # Compute statistics for centering and scaling (in cm)
    stats = {}
    all_arr_centered = []
    for k, arr in zip(keys, data):
        arr_cm = np.asarray(arr) * 100.0
        # Center data at an empirical mode estimate (use median as a robust proxy)
        mode_cm = float(np.median(arr_cm))
        arr_centered = arr_cm - mode_cm
        std_cm = float(np.std(arr_centered))
        stats[k] = {"mode_cm": mode_cm, "std_cm": std_cm}
        all_arr_centered.append(arr_centered)

    # Determine a common symmetric extent (max abs deviation across all panels)
    max_abs = max(np.abs(np.concatenate(all_arr_centered)).max(), 1.0)
    common_lim = np.ceil(max_abs / 5.0) * 5.0  # round up to nearest 5 cm

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    # Track a common y-limit across all histograms/curves
    global_ymax = 0.0

    for i, (k, arr_c, ax) in enumerate(zip(keys, all_arr_centered, axes)):
        std_cm = stats[k]["std_cm"]
        # Track max y for uniform y-axis
        local_ymax = 0.0
        # Histogram of samples (posterior sampling distribution)
        counts, bins, _ = ax.hist(
            arr_c,
            bins=40,
            density=True,
            color="gray",
            alpha=0.6,
            label="Posterior (samples)",
        )
        if len(counts):
            local_ymax = max(local_ymax, float(np.max(counts)))

        # x-grid for analytic curves
        xg = np.linspace(-common_lim, common_lim, 800)

        # Posterior Gaussian (centered at 0 with std from samples)
        from scipy.stats import norm as _norm

        post_gauss = _norm.pdf(xg, loc=0.0, scale=std_cm)
        ax.plot(xg, post_gauss, lw=1.6, label="Posterior (Gaussian)")
        local_ymax = max(local_ymax, float(np.max(post_gauss)))

        # Prior overlay (centered at prior mean minus posterior mode)
        if flat_prior:
            mu0 = flat_init.get(k, 0.0) * 100.0 if k in flat_init else 0.0
            s0 = flat_prior.get(k, None)
            if s0 is not None and s0 > 0:
                prior_pdf = _norm.pdf(
                    xg, loc=(mu0 - stats[k]["mode_cm"]), scale=s0 * 100.0
                )
                ax.plot(xg, prior_pdf, color="blue", lw=1.5, label="Prior (Gaussian)")
                local_ymax = max(local_ymax, float(np.max(prior_pdf)))

        # Vertical line for zero (posterior mode)
        ax.axvline(0.0, color="black", ls="--", lw=1.0, label="Posterior mode")
        # Vertical line for prior mean (relative to posterior mode)
        if k in flat_init:
            mu0 = flat_init[k] * 100.0
            ax.axvline(
                mu0 - stats[k]["mode_cm"],
                color="red",
                ls="--",
                lw=1.1,
                label="Prior mean",
            )

        # Annotate std
        ax.text(
            0.02,
            0.98,
            f"σ = {std_cm:.2f} cm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"),
        )

        ax.set_xlim(-common_lim, common_lim)
        ax.set_xlabel("Offset (cm)")
        ax.set_title(k.replace("_", " "))
        # Update global maximum after all curves
        global_ymax = max(global_ymax, local_ymax)
        if i == 0:
            ax.legend(fontsize="small")

    # Apply a uniform y-axis across all panels
    if global_ymax > 0:
        ylim = (0.0, global_ymax * 1.10)
        for ax in axes:
            ax.set_ylim(*ylim)

    if save:
        save_plot(fig, chain_name, "marginal_hists", subdir=f"Figs/MCMC/{chain_name}")
    plt.tight_layout()
    plt.show(block=True)


def corner_plot(
    chain,
    initial_params=None,
    prior_scales=None,
    downsample=1,
    save=False,
    chain_name=None,
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

    # Convert CDOG_aug (first DOG) from ECEF offsets to ENU for plotting
    cdog_aug_samples = chain["CDOG_aug"][::downsample, 0, :]
    if initial_params is not None:
        prior_mean_aug = initial_params["CDOG_aug"][0]
    else:
        prior_mean_aug = np.mean(cdog_aug_samples, axis=0)
    cdog_aug_enu = cdog_aug_ecef_to_enu(cdog_aug_samples, prior_mean_aug)

    # Build parameter dicts in meters, then convert to cm for plotting
    pars_m = {
        "lx": chain["lever"][::downsample, 0][order],
        "ly": chain["lever"][::downsample, 1][order],
        "lz": chain["lever"][::downsample, 2][order],
        "auge": cdog_aug_enu[:, 0][order],  # East (m)
        "augn": cdog_aug_enu[:, 1][order],  # North (m)
        "augu": cdog_aug_enu[:, 2][order],  # Up (m)
    }
    pars = {k: v * 100.0 for k, v in pars_m.items()}  # convert to cm

    flat_init = {}
    if initial_params is not None:
        flat_init_m = {
            "lx": initial_params["lever"][0],
            "ly": initial_params["lever"][1],
            "lz": initial_params["lever"][2],
        }
        # Convert initial CDOG_aug mean to ENU for prior/initial overlays (meters)
        enu0 = cdog_aug_ecef_to_enu(
            initial_params["CDOG_aug"][0][None, :], initial_params["CDOG_aug"][0]
        )[0]
        flat_init_m.update(
            {
                "auge": enu0[0],
                "augn": enu0[1],
                "augu": enu0[2],
            }
        )
        # Store cm for plotting
        flat_init = {k: v * 100.0 for k, v in flat_init_m.items()}

    flat_prior = {}
    if prior_scales is not None:
        flat_prior = {
            "lx": prior_scales["lever"][0],
            "ly": prior_scales["lever"][1],
            "lz": prior_scales["lever"][2],
            "auge": prior_scales["CDOG_aug"],
            "augn": prior_scales["CDOG_aug"],
            "augu": prior_scales["CDOG_aug"],
        }

    # Center each parameter at its posterior mode (use median; units: cm)
    stats = {}
    pars_c = {}
    all_centered_vals = []
    for k, arr_cm in pars.items():
        arr_cm = np.asarray(arr_cm)
        mode_cm = float(np.median(arr_cm))
        centered_cm = arr_cm - mode_cm
        std_cm = float(np.std(centered_cm))
        stats[k] = {"mode_cm": mode_cm, "std_cm": std_cm}
        pars_c[k] = centered_cm
        all_centered_vals.append(centered_cm)

    # Common symmetric extent across all subplots (in cm), reduced by ~0.8
    global_max_abs = (
        np.max(np.abs(np.concatenate(all_centered_vals))) if all_centered_vals else 1.0
    )
    if not np.isfinite(global_max_abs) or global_max_abs <= 0:
        global_max_abs = 1.0
    common_lim = float(np.ceil(global_max_abs / 1.0) * 1.0)
    common_lim *= 0.8  # reduce extent by ~0.8

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
            arr_cm = pars_c[key_i]
            ax.hist(
                arr_cm,
                bins=30,
                density=True,
                color="gray",
                alpha=0.6,
                label="Posterior (samples)",
            )

            # Posterior Gaussian (centered at 0 with sample std, in cm)
            x0, x1 = -common_lim, common_lim
            xg = np.linspace(x0, x1, 400)
            post_pdf = norm.pdf(xg, loc=0.0, scale=max(stats[key_i]["std_cm"], 1e-12))
            ax.plot(xg, post_pdf, lw=1.6, label="Posterior (Gaussian)", color="red")

            # Prior overlay (centered relative to posterior mode, convert σ to cm)
            if flat_prior:
                sx_cm = float(flat_prior[key_i]) * 100.0
                mx_cm = float(flat_init.get(key_i, 0.0))
                prior_loc_cm = mx_cm - stats[key_i]["mode_cm"]
                if sx_cm > 0:
                    prior_pdf = norm.pdf(xg, loc=prior_loc_cm, scale=sx_cm)
                    ax.plot(xg, prior_pdf, color="blue", lw=1.5, label="Prior")

            # Vertical reference lines
            ax.axvline(0.0, color="black", ls="--", lw=1.0, label="Posterior mode")
            if key_i in flat_init:
                ax.axvline(prior_loc_cm, color="red", ls="--", lw=1.1, label="Initial")

            # Std annotation (cm)
            ax.text(
                0.02,
                0.98,
                f"σ = {stats[key_i]['std_cm']:.3g} cm",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"),
            )

            ax.set_xlim(-common_lim, common_lim)
            # Removed per-axes legend here
        elif j < i:
            sc = ax.scatter(
                pars_c[key_j],
                pars_c[key_i],
                c=logpost,
                cmap=cmap,
                norm=norm_cmap,
                s=1,
                alpha=0.7,
                label="Posterior (samples)",
            )

            # 68% PRIOR ellipse (Gaussian, centered at prior mean relative to
            #  posterior modes)
            if flat_prior:
                sx_cm = float(flat_prior[key_j]) * 100.0
                sy_cm = float(flat_prior[key_i]) * 100.0
                mx_cm = float(flat_init.get(key_j, 0.0))
                my_cm = float(flat_init.get(key_i, 0.0))
                mean_xy_cm = (
                    mx_cm - stats[key_j]["mode_cm"],
                    my_cm - stats[key_i]["mode_cm"],
                )  # centered (cm)
                cov_prior_cm = np.array([[sx_cm**2, 0], [0, sy_cm**2]])
                prior68 = plot_prior_ellipse(
                    mean=mean_xy_cm, cov=cov_prior_cm, confidence=0.68, zorder=3
                )
                prior68.set_edgecolor("blue")
                prior68.set_linewidth(1.5)
                prior68.set_facecolor("none")
                prior68.set_label("Prior 68%")
                ax.add_patch(prior68)

            # 68% POSTERIOR Gaussian ellipse (centered at 0, covariance from samples)
            samp_xy = np.vstack([pars_c[key_j], pars_c[key_i]])  # shape (2, N)
            cov_post_cm = np.cov(samp_xy)
            post68 = plot_prior_ellipse(
                mean=(0.0, 0.0), cov=cov_post_cm, confidence=0.68, zorder=4
            )
            post68.set_edgecolor("red")
            post68.set_linewidth(1.5)
            post68.set_facecolor("none")
            post68.set_label("Posterior 68% (Gaussian)")
            ax.add_patch(post68)

            if key_j in flat_init and key_i in flat_init:
                ax.scatter(
                    mx_cm - stats[key_j]["mode_cm"],
                    my_cm - stats[key_i]["mode_cm"],
                    color="red",
                    marker="x",
                    s=40,
                    label="Initial",
                )

            ax.set_xlim(-common_lim, common_lim)
            ax.set_ylim(-common_lim, common_lim)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_visible(False)
        if i == n - 1:
            ax.set_xlabel(f"{key_j} (cm)")
        if j == 0:
            ax.set_ylabel(f"{key_i} (cm)")

    # Only show tick numbers on outside boxes
    for ii in range(n):
        for jj in range(n):
            ax_ij = axes[ii, jj]
            if ii != n - 1:
                ax_ij.set_xticklabels([])
            if jj != 0:
                ax_ij.set_yticklabels([])

    # Build a single, unobtrusive figure-level legend with proxy artists (top-center)
    sample_h = mlines.Line2D(
        [], [], linestyle="none", marker="o", markersize=4, label="Posterior (samples)"
    )
    prior_h = mpatches.Patch(
        edgecolor="blue", facecolor="none", linewidth=1.5, label="Prior 68%"
    )
    post_h = mpatches.Patch(
        edgecolor="red",
        facecolor="none",
        linewidth=1.5,
        label="Posterior 68% (Gaussian)",
    )
    init_h = mlines.Line2D(
        [], [], color="red", marker="x", linestyle="none", markersize=6, label="Initial"
    )

    fig.legend(
        handles=[sample_h, prior_h, post_h, init_h],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        fontsize="small",
        frameon=False,
    )

    label = "log likelihood" if loglike else "log posterior"
    fig.colorbar(
        sc,
        ax=axes[:, :],
        label=label,
        location="right",
        shrink=0.9,
        extend="max",
    )

    if save:
        save_plot(fig, chain_name, "corner_plot", subdir=f"Figs/MCMC/{chain_name}")
    plt.show()


if __name__ == "__main__":
    # Initial Parameters for adding to plot
    file_name = "7_individual_splits_esv_20250806_143356/split_3"
    loglike = False
    save = True

    chain_name = ("loglike_" if loglike else "logpost_") + file_name

    chain = np.load(gps_output_path(f"{file_name}.npz"))

    initial_params, prior_scales = get_init_params_and_prior(chain)

    trace_plot(
        chain,
        initial_params=initial_params,
        downsample=1000,
        save=save,
        chain_name=chain_name,
    )
    marginal_hists(
        chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        downsample=100,
        save=save,
        chain_name=chain_name,
    )
    corner_plot(
        chain,
        initial_params=initial_params,
        prior_scales=prior_scales,
        downsample=1000,
        save=save,
        chain_name=chain_name,
        loglike=loglike,
    )


# Compare locations of the plot_segments lever and receiver location
