import numpy as np
import matplotlib.pyplot as plt
import itertools
from plotting.MCMC_plots import get_init_params_and_prior, cdog_aug_ecef_to_enu
from data import gps_output_path
from plotting.save import save_plot
import matplotlib as mpl
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
from scipy.stats import norm
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse

"""Enable this for paper plots"""
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
        "font.size": 12,
    }
)


def _as_TDS(arr):
    """Force arr to shape (T, D, S)."""
    arr = np.asarray(arr)
    T = arr.shape[0]
    if arr.ndim == 1:
        return arr.reshape(T, 1, 1)
    if arr.ndim == 2:
        return arr.reshape(T, arr.shape[1], 1)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 1D/2D/3D array; got {arr.shape}")


def corner_plot_with_corr(
    chain,
    initial_params=None,
    prior_scales=None,
    DOG_num=1,
    downsample=1,
    save=False,
    chain_name=None,
    loglike=False,
    annotate_corr=True,
    tb_lim_ms=1.0,
    esv_lim_native=0.2,
):
    """
    Corner plot in the SAME style as the original corner_plot, but:
      - lower triangle: corner scatter + ellipses
      - diagonal: hist + posterior Gaussian + prior overlay + sigma box
      - upper triangle: correlation matrix cell with annotated r (white box)

    Parameters included:
      DOG E,N,U ; Lever x,y,z ; ESV mean ; Time bias ; (Lever z - DOG U)

    Adds only:
      - per-parameter axis limits for time-bias and ESV (hard-coded)
    """

    # ----- objective for coloring -----
    raw_logpost = (
        chain["loglike"][::downsample] if loglike else chain["logpost"][::downsample]
    )
    raw_logpost = np.asarray(raw_logpost)
    order = np.argsort(raw_logpost)
    logpost = raw_logpost[order]

    # ----- DOG mapping -----
    DOG_index_map = {1: 0, 3: 1, 4: 2}
    if DOG_num not in DOG_index_map:
        raise ValueError(f"DOG_num must be one of {list(DOG_index_map.keys())}")
    d = DOG_index_map[DOG_num]

    # ----- pull chains -----
    lever = np.asarray(chain["lever"][::downsample])[order]  # (T,3)
    cdog_aug = np.asarray(chain["CDOG_aug"][::downsample])[order]  # (T,D,3)
    esv = _as_TDS(chain["esv_bias"][::downsample])[order]  # (T,D,S)
    tb = np.asarray(chain["time_bias"][::downsample])[order]  # (T,...) dims vary

    # ----- DOG ENU (meters) -----
    cdog_aug_samples = cdog_aug[:, d, :]  # (T,3) ECEF offsets for this DOG
    if initial_params is not None:
        prior_mean_aug = np.asarray(initial_params["CDOG_aug"][d])
    else:
        prior_mean_aug = np.mean(cdog_aug_samples, axis=0)
    cdog_enu_m = cdog_aug_ecef_to_enu(
        cdog_aug_samples, prior_mean_aug
    )  # (T,3) in meters

    # ----- ESV mean (for this DOG) -----
    esv_mean = esv[:, d, :].mean(axis=1)  # (T,)

    # ----- Time bias (for this DOG) -----
    if tb.ndim == 1:
        tb_dog = tb
    elif tb.ndim == 2:
        tb_dog = tb[:, d]
    elif tb.ndim == 3:
        tb_dog = tb[:, d, :].mean(axis=1)
    else:
        raise ValueError(f"time_bias must be 1D/2D/3D; got {tb.shape}")

    # ----- derived -----
    lever_minus_dogu_m = lever[:, 2] - cdog_enu_m[:, 2]

    # ----- parameter keys -----
    key_esv = "mean esv"  # native units
    key_tb = "tb"  # ms

    pars = {
        "DOG E": cdog_enu_m[:, 0] * 100.0,  # cm
        "DOG N": cdog_enu_m[:, 1] * 100.0,  # cm
        "DOG U": cdog_enu_m[:, 2] * 100.0,  # cm
        "lx": lever[:, 0] * 100.0,  # cm
        "ly": lever[:, 1] * 100.0,  # cm
        "lz": lever[:, 2] * 100.0,  # cm
        key_esv: np.asarray(esv_mean),  # native units
        key_tb: np.asarray(tb_dog) * 1000.0,  # ms
        "lz - DOG U": lever_minus_dogu_m * 100.0,  # cm
    }

    # ----- apply a single finite mask across ALL parameters + logpost -----
    keys = list(pars.keys())
    X = np.column_stack([np.asarray(pars[k]) for k in keys])
    finite = np.isfinite(X).all(axis=1) & np.isfinite(logpost)
    logpost = logpost[finite]
    for k in keys:
        pars[k] = np.asarray(pars[k])[finite]

    # ----- initial values (flat_init) in plot units -----
    flat_init = {}
    if initial_params is not None:
        flat_init["lx"] = float(initial_params["lever"][0]) * 100.0
        flat_init["ly"] = float(initial_params["lever"][1]) * 100.0
        flat_init["lz"] = float(initial_params["lever"][2]) * 100.0

        enu0 = cdog_aug_ecef_to_enu(
            initial_params["CDOG_aug"][d][None, :],
            initial_params["CDOG_aug"][d],
        )[0]
        flat_init[f"DOG{DOG_num}_E"] = float(enu0[0]) * 100.0
        flat_init[f"DOG{DOG_num}_N"] = float(enu0[1]) * 100.0
        flat_init[f"DOG{DOG_num}_U"] = float(enu0[2]) * 100.0

        if "time_bias" in initial_params:
            tb0 = np.asarray(initial_params["time_bias"])
            if tb0.ndim == 1:
                flat_init[key_tb] = float(tb0[d] if tb0.size > 1 else tb0[0]) * 1000.0
            else:
                flat_init[key_tb] = float(np.mean(tb0)) * 1000.0

        if "esv_bias" in initial_params:
            esv0 = _as_TDS(initial_params["esv_bias"])[0, d, :].mean()
            flat_init[key_esv] = float(esv0)

        if ("lz" in flat_init) and (f"DOG{DOG_num}_U" in flat_init):
            flat_init[f"lz-DOG{DOG_num}U"] = (
                flat_init["lz"] - flat_init[f"DOG{DOG_num}_U"]
            )

    # ----- prior sigmas (flat_prior) in native prior units like original -----
    flat_prior = {}
    if prior_scales is not None:
        if "lever" in prior_scales:
            flat_prior["lx"] = float(prior_scales["lever"][0])
            flat_prior["ly"] = float(prior_scales["lever"][1])
            flat_prior["lz"] = float(prior_scales["lever"][2])
        if "CDOG_aug" in prior_scales:
            flat_prior[f"DOG{DOG_num}_E"] = float(prior_scales["CDOG_aug"])
            flat_prior[f"DOG{DOG_num}_N"] = float(prior_scales["CDOG_aug"])
            flat_prior[f"DOG{DOG_num}_U"] = float(prior_scales["CDOG_aug"])
        if "time_bias" in prior_scales:
            flat_prior[key_tb] = float(prior_scales["time_bias"])  # seconds
        if "esv_mean" in prior_scales:
            flat_prior[key_esv] = float(prior_scales["esv_mean"])  # native units

        if ("lz" in flat_prior) and (f"DOG{DOG_num}_U" in flat_prior):
            flat_prior[f"lz-DOG{DOG_num}U"] = float(
                np.sqrt(flat_prior["lz"] ** 2 + flat_prior[f"DOG{DOG_num}_U"] ** 2)
            )

    # ----- center each parameter at its posterior median -----
    stats = {}
    pars_c = {}
    all_centered_vals = []
    for k in keys:
        arr = np.asarray(pars[k])
        mode = float(np.median(arr))
        centered = arr - mode
        std = float(np.std(centered))
        stats[k] = {"mode": mode, "std": std}
        pars_c[k] = centered
        all_centered_vals.append(centered)

    # ----- common symmetric extent across all subplots -----
    global_max_abs = (
        np.max(np.abs(np.concatenate(all_centered_vals))) if all_centered_vals else 1.0
    )
    if not np.isfinite(global_max_abs) or global_max_abs <= 0:
        global_max_abs = 1.0
    common_lim = float(np.ceil(global_max_abs / 1.0) * 1.0)
    common_lim *= 0.8

    # ----- NEW: per-key axis limits (centered units) -----
    axis_limits = {
        key_tb: float(tb_lim_ms),
        key_esv: float(esv_lim_native),
    }

    def _get_lim(key: str) -> float:
        return axis_limits.get(key, common_lim)

    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(14, 10))

    # ----- color maps -----
    vmin = float(np.min(logpost))
    vmax = float(np.max(logpost))
    norm_cmap = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    corr = np.corrcoef(np.column_stack([pars_c[k] for k in keys]), rowvar=False)
    corr_cmap = plt.get_cmap("RdBu")

    sc = None
    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]
        key_i, key_j = keys[i], keys[j]

        if i == j:
            arr = pars_c[key_i]
            ax.hist(arr, bins=30, density=True, color="gray", alpha=0.6)

            lim_i = _get_lim(key_i)  # NEW
            x0, x1 = -lim_i, lim_i
            xg = np.linspace(x0, x1, 400)
            post_pdf = norm.pdf(xg, loc=0.0, scale=max(stats[key_i]["std"], 1e-12))
            ax.plot(xg, post_pdf, lw=1.6, color="red")

            # prior overlay if available
            prior_loc = None
            if flat_prior and (key_i in flat_prior) and (key_i in flat_init):
                sig = float(flat_prior[key_i])
                if (
                    key_i in ("lx", "ly", "lz")
                    or key_i.startswith("DOG")
                    or key_i.startswith("lz-DOG")
                ):
                    sig_plot = sig * 100.0
                elif key_i.startswith("tb"):
                    sig_plot = sig * 1000.0
                else:
                    sig_plot = sig

                mx = float(flat_init.get(key_i, 0.0))
                prior_loc = mx - stats[key_i]["mode"]
                if sig_plot > 0:
                    ax.plot(
                        xg,
                        norm.pdf(xg, loc=prior_loc, scale=sig_plot),
                        color="blue",
                        lw=1.5,
                    )

            ax.axvline(0.0, color="black", ls="--", lw=1.0)
            if prior_loc is not None:
                ax.axvline(prior_loc, color="red", ls="--", lw=1.1)

            ax.text(
                0.02,
                0.98,
                rf"$\sigma = {stats[key_i]['std']:.3g}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"),
            )
            ax.set_xlim(-lim_i, lim_i)  # NEW

        elif j < i:
            sc = ax.scatter(
                pars_c[key_j],
                pars_c[key_i],
                c=logpost,
                cmap=cmap,
                norm=norm_cmap,
                s=1,
                alpha=0.7,
            )

            # 68% prior ellipse
            if (
                flat_prior
                and (key_j in flat_prior)
                and (key_i in flat_prior)
                and (key_j in flat_init)
                and (key_i in flat_init)
            ):
                sx = float(flat_prior[key_j])
                sy = float(flat_prior[key_i])
                if (
                    key_j in ("lx", "ly", "lz")
                    or key_j.startswith("DOG")
                    or key_j.startswith("lz-DOG")
                ):
                    sx *= 100.0
                elif key_j.startswith("tb"):
                    sx *= 1000.0
                if (
                    key_i in ("lx", "ly", "lz")
                    or key_i.startswith("DOG")
                    or key_i.startswith("lz-DOG")
                ):
                    sy *= 100.0
                elif key_i.startswith("tb"):
                    sy *= 1000.0

                mx = float(flat_init.get(key_j, 0.0))
                my = float(flat_init.get(key_i, 0.0))
                mean_xy = (mx - stats[key_j]["mode"], my - stats[key_i]["mode"])
                cov_prior = np.array([[sx**2, 0], [0, sy**2]])

                prior68 = plot_prior_ellipse(
                    mean=mean_xy, cov=cov_prior, confidence=0.68, zorder=3
                )
                prior68.set_edgecolor("blue")
                prior68.set_linewidth(1.5)
                prior68.set_facecolor("none")
                ax.add_patch(prior68)

            # 68% posterior ellipse
            samp_xy = np.vstack([pars_c[key_j], pars_c[key_i]])
            cov_post = np.cov(samp_xy)
            post68 = plot_prior_ellipse(
                mean=(0.0, 0.0), cov=cov_post, confidence=0.68, zorder=4
            )
            post68.set_edgecolor("red")
            post68.set_linewidth(1.5)
            post68.set_facecolor("none")
            ax.add_patch(post68)

            # initial marker
            if key_j in flat_init and key_i in flat_init:
                ax.scatter(
                    float(flat_init[key_j]) - stats[key_j]["mode"],
                    float(flat_init[key_i]) - stats[key_i]["mode"],
                    color="red",
                    marker="x",
                    s=40,
                )

            # NEW: per-key x/y limits
            ax.set_xlim(-_get_lim(key_j), _get_lim(key_j))
            ax.set_ylim(-_get_lim(key_i), _get_lim(key_i))

            ax.set_aspect("auto")

        else:
            r = float(corr[i, j])
            ax.imshow(
                np.array([[r]]),
                vmin=-1.0,
                vmax=1.0,
                cmap=corr_cmap,
                aspect="auto",
                interpolation="nearest",
                extent=[0, 1, 0, 1],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            if annotate_corr and np.isfinite(r):
                ax.text(
                    0.5,
                    0.5,
                    f"{r:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        boxstyle="round,pad=0.18",
                        alpha=0.85,
                    ),
                )

        if i == n - 1:
            ax.set_xlabel(f"{key_j}")
        if j == 0:
            ax.set_ylabel(f"{key_i}")

    # Only show tick numbers on outside boxes (same as original)
    for ii in range(n):
        for jj in range(n):
            ax_ij = axes[ii, jj]
            if ii != n - 1:
                ax_ij.set_xticklabels([])
            if jj != 0:
                ax_ij.set_yticklabels([])
    axes[0, 0].set_yticklabels([])

    # Figure-level legend (same style)
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
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        fontsize="small",
        frameon=False,
    )

    label = "log likelihood" if loglike else "log posterior"
    if sc is not None:
        fig.colorbar(
            sc, ax=axes[:, :], label=label, location="right", shrink=0.9, extend="max"
        )

    if save:
        save_plot(
            fig, func_name="corner_plot_with_corr", subdir=f"Figs/MCMC/{chain_name}"
        )
        print("Saved corner plot with correlations with chain name:", chain_name)

    plt.show()
    return corr, keys


if __name__ == "__main__":
    file_name = "mcmc_chain_1_22_new_MCMC_long"
    chain_name = "logpost_" + file_name
    chain = np.load(gps_output_path(f"{file_name}.npz"))
    initial_params, prior_scales, proposal_scales = get_init_params_and_prior(chain)

    corner_plot_with_corr(
        chain,
        initial_params,
        prior_scales=prior_scales,
        DOG_num=1,
        downsample=500,
        save=True,
        chain_name=chain_name,
        loglike=False,
        annotate_corr=True,
    )
