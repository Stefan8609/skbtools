import numpy as np
import matplotlib.pyplot as plt
from pymap3d import geodetic2enu
from GeigerMethod.Synthetic.Numba_Functions.ECEF_Geodetic import ECEF_Geodetic
from scipy.stats import gaussian_kde
from data import gps_output_path
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse
from plotting.MCMC_plots import get_init_params_and_prior
from .save import save_plot


def plot_kde_mcmc(
    samples,
    nbins=100,
    cmap="viridis",
    prior_mean=None,
    prior_sd=None,
    confidences=(0.68, 0.95),
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
    pc1 = pc1_vec.dot(xy_cent)

    # KDE in Principal Component and z
    pcz = np.vstack([pc1, z])
    kde_pcz = gaussian_kde(pcz)
    lim_pcz = max(np.max(np.abs(pc1)), np.max(np.abs(z)))
    pi = np.linspace(-lim_pcz, lim_pcz, nbins)
    zi = np.linspace(-lim_pcz, lim_pcz, nbins)
    P, Z = np.meshgrid(pi, zi)
    Z_pcz = kde_pcz(np.vstack([P.ravel(), Z.ravel()])).reshape(nbins, nbins)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cf1 = ax1.contourf(X, Y, Z_xy, levels=20, cmap=cmap, extend="both")
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_title("KDE of (East, North)")
    fig.colorbar(cf1, ax=ax1, label="Density")

    if prior_sd is not None and prior_mean is not None:
        prior_cov = np.diag([prior_sd**2, prior_sd**2])

        # draw 68% then 95% so the smaller sits on top
        for conf in confidences:
            e_xy = plot_prior_ellipse(
                mean=np.array([0, 0]), cov=prior_cov, confidence=conf, zorder=3
            )
            e_pcz = plot_prior_ellipse(
                mean=np.array([0, 0]), cov=prior_cov, confidence=conf, zorder=3
            )
            ax1.add_patch(e_xy)
            ax2.add_patch(e_pcz)

    cf2 = ax2.contourf(P, Z, Z_pcz, levels=20, cmap=cmap, extend="both")
    ax2.set_xlabel(r"$\xi$ (m)")
    ax2.set_ylabel("Up (m)")
    ax2.set_title("KDE of (PC1, Up)")
    fig.colorbar(cf2, ax=ax2, label="Density")

    # if ellipses > 0:
    #     segment_xy_splits = np.array_split(xy, ellipses, axis=1)
    #     segment_pcz_splits = np.array_split(pcz, ellipses, axis=1)
    #
    #     for i in range(ellipses):
    #         print(i)
    #         segment_xy = segment_xy_splits[i]
    #         segment_pcz = segment_pcz_splits[i]
    #
    #         print(segment_pcz)
    #
    #         # Compute error ellipse for xy
    #         cov_xy = np.cov(segment_xy)
    #         print(segment_xy.shape,cov_xy.shape)
    #         e_xy, _ = compute_error_ellipse(cov_xy, confidence=0.68, zorder=2)
    #         ax1.add_patch(e_xy)
    #
    #         print("PCZ shape", segment_pcz.shape)
    #         # Compute error ellipse for pcz
    #         cov_pcz = np.cov(segment_pcz)
    #         print(segment_pcz.shape, cov_pcz.shape)
    #         e_pcz, _ = compute_error_ellipse(cov_pcz, confidence=0.68, zorder=2)
    #         ax2.add_patch(e_pcz)

    # Set limits
    ax1.set_xlim(-lim_xy, lim_xy)
    ax1.set_ylim(-lim_xy, lim_xy)
    ax2.set_xlim(-lim_pcz, lim_pcz)
    ax2.set_ylim(-lim_pcz, lim_pcz)

    plt.tight_layout()
    if save:
        save_plot(fig, chain_name, "plot_kde_mcmc", subdir=path)
    plt.show()


if __name__ == "__main__":
    chain = np.load(gps_output_path("mcmc_chain_8-7.npz"))
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
    #
    # initial_params, prior_scales = get_init_params_and_prior(chain)
    # init_aug = initial_params["CDOG_aug"]
    # prior_aug = prior_scales["CDOG_aug"]
    # print(samples)
    #
    # plot_kde_mcmc(
    #     samples,
    #     nbins=100,
    #     cmap="viridis",
    #     prior_mean=init_aug[DOG_num],
    #     prior_sd=prior_aug,
    #     ellipses=0,
    # )
