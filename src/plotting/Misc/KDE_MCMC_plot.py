import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    prior_mean=None,
    prior_sd=None,
    conf_level=0.68,
    CDOG_reference=None,
    ellipses=0,
    save=False,
    chain_name=None,
    path="Figs",
    return_axes=False,
    ax1=None,
    ax2=None,
    fig=None,
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

    # Convert 2D KDE density to expected counts per cell (XY)
    N = num_points
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    cell_area_xy = dx * dy
    Z_xy_counts = Z_xy * N * cell_area_xy

    # Principal component of x,y
    xy_mean = xy.mean(axis=1, keepdims=True)
    xy_cent = xy - xy_mean
    cov = np.cov(xy_cent)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1_vec = eigvecs[:, np.argmax(eigvals)]  # unit vector in EN plane

    # KDE in (PC1, z)
    pcz = np.vstack([pc1_vec.dot(xy_cent), z])
    kde_pcz = gaussian_kde(pcz)
    lim_pcz = max(np.max(np.abs(pcz[0])), np.max(np.abs(pcz[1])))
    pi = np.linspace(-lim_pcz, lim_pcz, nbins)
    zi = np.linspace(-lim_pcz, lim_pcz, nbins)
    P, Z = np.meshgrid(pi, zi)
    Z_pcz = kde_pcz(np.vstack([P.ravel(), Z.ravel()])).reshape(nbins, nbins)

    # Convert 2D KDE density to expected counts per cell (PC1–Up)
    dpi = pi[1] - pi[0]
    dzi = zi[1] - zi[0]
    cell_area_pcz = dpi * dzi
    Z_pcz_counts = Z_pcz * N * cell_area_pcz

    # Posterior modes
    max_idx_xy = np.unravel_index(np.argmax(Z_xy_counts), Z_xy_counts.shape)
    mode_x_cm = X[max_idx_xy]
    mode_y_cm = Y[max_idx_xy]
    max_idx_pcz = np.unravel_index(np.argmax(Z_pcz_counts), Z_pcz_counts.shape)
    mode_xi_cm = P[max_idx_pcz]
    mode_up_cm = Z[max_idx_pcz]

    # STD (rounded to 1 decimal place)
    sigma_E_cm = float(np.std(x, ddof=1))
    sigma_N_cm = float(np.std(y, ddof=1))
    sigma_xi_cm = float(np.std(pcz[0], ddof=1))
    sigma_U_cm = float(np.std(z, ddof=1))

    # White->Blue colormap
    new_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

    # Set up figure/axes
    if fig is None and ax1 is None and ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_aspect("equal", "box")
        ax2.set_aspect("equal", "box")

    # Left panel (E,N)
    levels_xy = np.linspace(0, Z_xy_counts.max(), 21)
    cf1 = ax1.contourf(
        X, Y, Z_xy_counts, levels=levels_xy, cmap=new_cmap, antialiased=True
    )
    ax1.set_facecolor(new_cmap(0))
    ax1.plot(
        mode_x_cm,
        mode_y_cm,
        marker="x",
        linestyle="None",
        color="red",
        markersize=4,
        markeredgewidth=2,
        label="Posterior Mode",
        zorder=5,
    )
    # Monospace, aligned stats (1 decimal place)
    ax1.text(
        0.02,
        0.98,
        f"$\\sigma_E$ = {sigma_E_cm:4.1f} cm\n$\\sigma_N$ = {sigma_N_cm:4.1f} cm",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        color="red",
        family="monospace",
    )
    ax1.set_xlabel("East (cm)")
    ax1.set_ylabel("North (cm)")
    ax1.set_title("KDE of (East, North)")
    fig.colorbar(cf1, ax=ax1, label=f"Counts (Total Samples = {num_points})")

    # Prior ellipses (blue) in both panels
    if prior_sd is not None and prior_mean is not None:
        prior_sd_cm = prior_sd * 100.0
        prior_cov = np.diag([prior_sd_cm**2, prior_sd_cm**2])
        e_xy = plot_prior_ellipse(
            mean=np.array([0, 0]),
            cov=prior_cov,
            confidence=conf_level,
            zorder=3,
        )
        e_xy.set_label("Prior 68%")
        e_xy.set_edgecolor("blue")
        e_xy.set_linewidth(1.5)

        e_pcz = plot_prior_ellipse(
            mean=np.array([0, 0]),
            cov=prior_cov,
            confidence=conf_level,
            zorder=3,
        )
        e_pcz.set_label("Prior 68%")
        e_pcz.set_edgecolor("blue")
        e_pcz.set_linewidth(1.5)

        ax1.add_patch(e_xy)
        ax2.add_patch(e_pcz)

    # Right panel (ξ, Up)
    levels_pcz = np.linspace(0, Z_pcz_counts.max(), 21)
    cf2 = ax2.contourf(
        P, Z, Z_pcz_counts, levels=levels_pcz, cmap=new_cmap, antialiased=True
    )
    ax2.set_facecolor(new_cmap(0))
    ax2.set_xlabel(r"$\xi$ (cm)")
    ax2.set_ylabel("Up (cm)")
    ax2.set_title("KDE of (PC1, Up)")
    fig.colorbar(cf2, ax=ax2, label=f"Counts (Total Samples = {num_points})")
    ax2.plot(
        mode_xi_cm,
        mode_up_cm,
        marker="x",
        linestyle="None",
        color="red",
        markersize=4,
        markeredgewidth=2,
        label="Posterior Mode",
        zorder=5,
    )
    ax2.text(
        0.02,
        0.98,
        f"$\\sigma_\\xi\\,$ = {sigma_xi_cm:4.1f} cm \
        \n$\\sigma_U$ = {sigma_U_cm:4.1f} cm",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        color="red",
        family="monospace",
    )

    lim_all = max(lim_xy, lim_pcz)

    # Start/end points in EN for the PC1 arrow
    start_EN = (xy_mean.flatten() - lim_all * pc1_vec).tolist()
    end_EN = (xy_mean.flatten() + lim_all * pc1_vec).tolist()

    # Solid arrow along PC1
    ax1.annotate(
        "",
        xy=(end_EN[0], end_EN[1]),
        xytext=(start_EN[0], start_EN[1]),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        clip_on=False,
        annotation_clip=False,
        zorder=10,
    )

    # "ξ" label next to the arrow, rotated with the arrow direction
    label_pos = (
        xy_mean.flatten() + 0.65 * lim_all * pc1_vec
    )  # slight offset along arrow
    ax1.text(
        label_pos[0] - 35,
        label_pos[1] + 20,
        r"$\xi$",
        color="red",
        fontsize=11,
        ha="left",
        va="bottom",
        zorder=5,
    )

    # === Error ellipses (posterior) in BLACK ===
    if isinstance(ellipses, int) and ellipses > 0:
        min_segment_size = 20
        idx_splits = np.array_split(np.arange(num_points), ellipses)

        for i, idx in enumerate(idx_splits):
            if idx.size < min_segment_size:
                mid = int(idx.mean())
                half = max(min_segment_size // 2, 1)
                lo = max(0, mid - half)
                hi = min(num_points, mid + half)
                idx = np.arange(lo, hi)
                if idx.size < min_segment_size:
                    continue

            # EN ellipse
            segment_xy = xy[:, idx].T
            if segment_xy.shape[0] >= min_segment_size:
                e_xy_seg, _ = compute_error_ellipse(
                    segment_xy, confidence=conf_level, zorder=10
                )
                e_xy_seg.set_fill(False)
                e_xy_seg.set_linewidth(1.5 if ellipses == 1 else 1.2)
                e_xy_seg.set_alpha(0.95)
                e_xy_seg.set_edgecolor("black")  # <-- BLACK
                e_xy_seg.set_label(f"Segment {i + 1} Posterior")
                ax1.add_patch(e_xy_seg)

            # (ξ,Up) ellipse
            segment_pcz = pcz[:, idx].T
            if segment_pcz.shape[0] >= min_segment_size:
                e_pcz_seg, _ = compute_error_ellipse(
                    segment_pcz, confidence=conf_level, zorder=10
                )
                e_pcz_seg.set_fill(False)
                e_pcz_seg.set_linewidth(1.5 if ellipses == 1 else 1.2)
                e_pcz_seg.set_alpha(0.95)
                e_pcz_seg.set_edgecolor("black")  # <-- BLACK
                e_pcz_seg.set_label(f"Segment {i + 1} Posterior")
                ax2.add_patch(e_pcz_seg)

    # Consistent limits on both panels
    ax1.set_xlim(-lim_all, lim_all)
    ax1.set_ylim(-lim_all, lim_all)
    ax2.set_xlim(-lim_all, lim_all)
    ax2.set_ylim(-lim_all, lim_all)

    # Legends (no entry for the ξ arrow)
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        priority = {"Prior 68%": 0, "Posterior Mode": 1}
        order = sorted(
            range(len(labels)), key=lambda i: (priority.get(labels[i], 3), labels[i])
        )
        if order:
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

    plt.tight_layout()
    if save:
        save_plot(fig, chain_name, "plot_kde_mcmc", subdir=path)
    if return_axes:
        return fig, ax1, ax2
    plt.show()


if __name__ == "__main__":
    file = "mcmc_chain_1_20_new_inversion"
    chain = np.load(gps_output_path(f"{file}.npz"))
    DOG_num = 0
    sample = chain["CDOG_aug"][::100, DOG_num]

    initial_params, prior_scales, _ = get_init_params_and_prior(chain)
    init_aug = initial_params["CDOG_aug"]
    prior_aug = prior_scales["CDOG_aug"]

    plot_kde_mcmc(
        sample,
        nbins=100,
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
