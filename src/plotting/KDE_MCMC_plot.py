import numpy as np
import matplotlib.pyplot as plt
from pymap3d import geodetic2enu
from geometry.ECEF_Geodetic import ECEF_Geodetic
from scipy.stats import gaussian_kde
from data import gps_output_path
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse
from plotting.Ellipses.Error_Ellipse import compute_error_ellipse
from plotting.MCMC_plots import get_init_params_and_prior
from plotting.save import save_plot


def plot_kde_mcmc(
    samples,
    nbins=100,
    cmap="viridis",
    prior_mean=None,
    prior_sd=None,
    conf_level=0.68,
    CDOG_reference=None,
    ellipses=0,
    save=False,
    chain_name=None,
    path="Figs",
):
    if CDOG_reference is None:
        CDOG_reference = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    CDOG_loc = prior_mean + CDOG_reference
    CDOG_lat, CDOG_lon, CDOG_height = ECEF_Geodetic(np.array([CDOG_loc]))
    samples_lat, samples_lon, samples_height = ECEF_Geodetic(samples + CDOG_reference)

    # Convert to ENU coordinates
    num_points = samples.shape[0]
    samples_converted = np.zeros((num_points, 3))
    for i in range(num_points):
        enu = geodetic2enu(
            samples_lat[i],
            samples_lon[i],
            samples_height[i],
            CDOG_lat,
            CDOG_lon,
            CDOG_height,
        )
        samples_converted[i] = np.squeeze(enu)

    samples_converted = samples_converted * 100.0  # m -> cm

    x, y, z = samples_converted.T

    # KDE in x,y
    xy = np.vstack([x, y])
    kde_xy = gaussian_kde(xy)
    lim_xy = max(np.max(np.abs(x)), np.max(np.abs(y)))
    xi = np.linspace(-lim_xy, lim_xy, nbins)
    yi = np.linspace(-lim_xy, lim_xy, nbins)
    X, Y = np.meshgrid(xi, yi)
    Z_xy = kde_xy(np.vstack([X.ravel(), Y.ravel()])).reshape(nbins, nbins)

    # Find Principal Component of x,y
    xy_mean = xy.mean(axis=1, keepdims=True)
    xy_cent = xy - xy_mean
    cov = np.cov(xy_cent)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1_vec = eigvecs[:, np.argmax(eigvals)]
    # Principal component line endpoints for plotting
    t = np.linspace(-lim_xy, lim_xy, 2)
    line_x = xy_mean[0] + t * pc1_vec[0]
    line_y = xy_mean[1] + t * pc1_vec[1]

    # KDE in Principal Component and z
    pcz = np.vstack([pc1_vec.dot(xy_cent), z])
    kde_pcz = gaussian_kde(pcz)
    lim_pcz = max(np.max(np.abs(pcz[0])), np.max(np.abs(pcz[1])))
    pi = np.linspace(-lim_pcz, lim_pcz, nbins)
    zi = np.linspace(-lim_pcz, lim_pcz, nbins)
    P, Z = np.meshgrid(pi, zi)
    Z_pcz = kde_pcz(np.vstack([P.ravel(), Z.ravel()])).reshape(nbins, nbins)

    # === Posterior mode (argmax of KDE) ===
    max_idx_xy = np.unravel_index(np.argmax(Z_xy), Z_xy.shape)
    mode_x_cm = X[max_idx_xy]
    mode_y_cm = Y[max_idx_xy]

    max_idx_pcz = np.unravel_index(np.argmax(Z_pcz), Z_pcz.shape)
    mode_xi_cm = P[max_idx_pcz]
    mode_up_cm = Z[max_idx_pcz]

    # === Posterior standard deviations along axes (in cm) ===
    # Left plot axes: East (x), North (y)
    sigma_E_cm = float(np.std(x, ddof=1))
    sigma_N_cm = float(np.std(y, ddof=1))
    # Right plot axes: PC1 (xi), Up (z)
    sigma_xi_cm = float(np.std(pcz[0], ddof=1))
    sigma_U_cm = float(np.std(z, ddof=1))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_aspect("equal", "box")
    ax2.set_aspect("equal", "box")

    levels_xy = np.linspace(Z_xy.min(), Z_xy.max(), 21)
    cf1 = ax1.contourf(X, Y, Z_xy, levels=levels_xy, cmap=cmap, antialiased=True)
    ax1.set_facecolor(plt.get_cmap(cmap)(0))
    ax1.plot(line_x, line_y, color="red", linestyle="--", linewidth=2, label="PC1")
    # ax1.legend()
    # Posterior mode and std (left plot)
    ax1.plot(
        mode_x_cm,
        mode_y_cm,
        marker="x",
        linestyle="None",
        color="red",
        markersize=8,
        markeredgewidth=2,
        label="Posterior Mode",
        zorder=5,
    )
    ax1.text(
        0.02,
        0.98,
        f"$\\sigma_E$ = {sigma_E_cm:.1f} cm\n$\\sigma_N$ = {sigma_N_cm:.1f} cm",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        color="red",
    )
    ax1.set_xlabel("East (cm)")
    ax1.set_ylabel("North (cm)")
    ax1.set_title("KDE of (East, North)")
    fig.colorbar(cf1, ax=ax1, label="Density")

    if prior_sd is not None and prior_mean is not None:
        prior_sd_cm = prior_sd * 100.0
        prior_cov = np.diag([prior_sd_cm**2, prior_sd_cm**2])
        e_xy = plot_prior_ellipse(
            mean=np.array([0, 0]),
            cov=prior_cov,
            confidence=conf_level,
            zorder=3,
            label="Prior",
        )
        e_xy.set_label("Prior 68%")
        e_pcz = plot_prior_ellipse(
            mean=np.array([0, 0]),
            cov=prior_cov,
            confidence=conf_level,
            zorder=3,
            label="Prior",
        )
        e_pcz.set_label("Prior 68%")
        ax1.add_patch(e_xy)
        ax2.add_patch(e_pcz)

    levels_pcz = np.linspace(Z_pcz.min(), Z_pcz.max(), 21)
    cf2 = ax2.contourf(P, Z, Z_pcz, levels=levels_pcz, cmap=cmap, antialiased=True)
    ax2.set_facecolor(plt.get_cmap(cmap)(0))
    ax2.set_xlabel(r"$\xi$ (cm)")
    ax2.set_ylabel("Up (cm)")
    ax2.set_title("KDE of (PC1, Up)")
    fig.colorbar(cf2, ax=ax2, label="Density")
    # Posterior mode and std (right plot)
    ax2.plot(
        mode_xi_cm,
        mode_up_cm,
        marker="x",
        linestyle="None",
        color="red",
        markersize=8,
        markeredgewidth=2,
        label="Posterior Mode",
        zorder=5,
    )
    ax2.text(
        0.02,
        0.98,
        f"$\\sigma_\\xi$ = {sigma_xi_cm:.1f} cm\n$\\sigma_U$ = {sigma_U_cm:.1f} cm",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        color="red",
    )

    # Draw error ellipses for segmented subsets, if requested
    if isinstance(ellipses, int) and ellipses > 0:
        # Configuration for stability/visibility
        min_segment_size = 20  # require at least this many points per segment

        # split indices so both XY and PCZ use identical segments
        idx_splits = np.array_split(np.arange(num_points), ellipses)

        seg_cmap = plt.get_cmap("tab10")

        for i, idx in enumerate(idx_splits):
            # Guard against tiny segments
            if idx.size < min_segment_size:
                # If too small, try to borrow neighbors by expanding the slice bounds
                # to reach ~min_segment_size where possible
                # Compute a contiguous window around the segment mid-point
                mid = int(idx.mean())
                half = max(min_segment_size // 2, 1)
                lo = max(0, mid - half)
                hi = min(num_points, mid + half)
                idx = np.arange(lo, hi)
                if idx.size < min_segment_size:
                    # Still too small; skip this one
                    continue

            # === (East, North) plane ===
            # xy has shape (2, N); we need data as (n_i, 2) for compute_error_ellipse
            segment_xy = xy[:, idx].T  # shape (n_i, 2)
            if segment_xy.shape[0] < min_segment_size:
                continue

            e_xy, _ = compute_error_ellipse(
                segment_xy, confidence=conf_level, zorder=10
            )
            e_xy.set_fill(False)
            e_xy.set_linewidth(1.5)
            e_xy.set_alpha(0.95)
            e_xy.set_label(f"Segment {i + 1} Posterior")
            edge_col = (
                "red"
                if (isinstance(ellipses, int) and ellipses == 1)
                else seg_cmap(i % 10)
            )
            e_xy.set_edgecolor(edge_col)
            if isinstance(ellipses, int) and ellipses == 1:
                e_xy.set_linewidth(2.0)
            ax1.add_patch(e_xy)

            # === (PC1, Up) plane ===
            segment_pcz = pcz[:, idx].T  # shape (n_i, 2)
            if segment_pcz.shape[0] < min_segment_size:
                continue

            e_pcz, _ = compute_error_ellipse(
                segment_pcz, confidence=conf_level, zorder=10
            )
            e_pcz.set_fill(False)
            e_pcz.set_linewidth(1.5)
            e_pcz.set_alpha(0.95)
            e_pcz.set_label(f"Segment {i + 1} Posterior")
            edge_col = (
                "red"
                if (isinstance(ellipses, int) and ellipses == 1)
                else seg_cmap(i % 10)
            )
            e_pcz.set_edgecolor(edge_col)
            if isinstance(ellipses, int) and ellipses == 1:
                e_pcz.set_linewidth(2.0)
            ax2.add_patch(e_pcz)

    # Set symmetric limits for both subplots
    # ax1.legend()
    # ax2.legend()
    # Build ordered, compact legends
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        priority = {"Prior 68%": 0, "Posterior Mode": 1, "PC1": 2}
        order = sorted(
            range(len(labels)), key=lambda i: (priority.get(labels[i], 3), labels[i])
        )
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            fontsize=8,
            loc="upper right",
            frameon=True,
            framealpha=0.8,
            handlelength=1.0,
            borderpad=0.3,
            labelspacing=0.3,
        )
    lim_all = max(lim_xy, lim_pcz)
    ax1.set_xlim(-lim_all, lim_all)
    ax1.set_ylim(-lim_all, lim_all)
    ax2.set_xlim(-lim_all, lim_all)
    ax2.set_ylim(-lim_all, lim_all)

    fig.suptitle(
        "Red X = posterior mode; Red ellipse = posterior (when 1 ellipse)", fontsize=10
    )

    plt.tight_layout()
    if save:
        save_plot(fig, chain_name, "plot_kde_mcmc", subdir=path)
    plt.show()


if __name__ == "__main__":
    file = "mcmc_chain_8-7"
    chain = np.load(gps_output_path(f"{file}.npz"))
    DOG_num = 0
    sample = chain["CDOG_aug"][::100, DOG_num]

    initial_params, prior_scales = get_init_params_and_prior(chain)
    init_aug = initial_params["CDOG_aug"]
    prior_aug = prior_scales["CDOG_aug"]

    plot_kde_mcmc(
        sample,
        nbins=100,
        cmap="viridis",
        prior_mean=init_aug[DOG_num],
        prior_sd=prior_aug,
        ellipses=1,
        save=True,
        chain_name=file,
    )

    # for i in range(7):
    #     chain = np.load(
    #         gps_output_path(f"7_individual_splits_esv_20250806_165630/split_{i}.npz")
    #     )
    #     DOG_num = 0
    #     segment_samples = chain["CDOG_aug"][::1, DOG_num]
    #     if i == 0:
    #         samples = segment_samples
    #     else:
    #         # Stack samples from each segment
    #         samples = np.vstack((samples, segment_samples))

    # initial_params, prior_scales = get_init_params_and_prior(chain)
    # init_aug = initial_params["CDOG_aug"]
    # prior_aug = prior_scales["CDOG_aug"]
    # print(samples)

    # downsample = 10
    # samples = samples[::downsample]

    # plot_kde_mcmc(
    #     samples,
    #     nbins=100,
    #     cmap="viridis",
    #     prior_mean=init_aug[DOG_num],
    #     prior_sd=prior_aug,
    #     ellipses=7,
    #     save=True,
    #     chain_name = "7_splits_combined",
    # )
