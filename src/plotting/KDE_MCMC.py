import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from data import gps_output_path
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse


def plot_kde_mcmc(
    samples,
    nbins=100,
    cmap="viridis",
    prior_mean=None,
    prior_sd=None,
    confidences=(0.68, 0.95),
):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("`samples` must be shape (n_samples, 3)")

    x, y, z = samples.T

    # KDE in x,y
    xy = np.vstack([x, y])
    kde_xy = gaussian_kde(xy)
    xi = np.linspace(x.min(), x.max(), nbins)
    yi = np.linspace(y.min(), y.max(), nbins)
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
    pi = np.linspace(pc1.min(), pc1.max(), nbins)
    zi = np.linspace(z.min(), z.max(), nbins)
    P, Z = np.meshgrid(pi, zi)
    Z_pcz = kde_pcz(np.vstack([P.ravel(), Z.ravel()])).reshape(nbins, nbins)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cf1 = ax1.contourf(X, Y, Z_xy, levels=20, cmap=cmap)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("KDE of (x, y)")
    fig.colorbar(cf1, ax=ax1, label="Density")

    if prior_sd is not None and prior_mean is not None:
        prior_cov = np.diag([prior_sd**2, prior_sd**2])
        mean_xy = prior_mean[:2]
        mean_pcz = [0, prior_mean[2]]

        # draw 68% then 95% so the smaller sits on top
        for conf in confidences:
            e_xy = plot_prior_ellipse(
                mean=mean_xy, cov=prior_cov, confidence=conf, zorder=3
            )
            e_pcz = plot_prior_ellipse(
                mean=mean_pcz, cov=prior_cov, confidence=conf, zorder=3
            )
            ax1.add_patch(e_xy)
            ax2.add_patch(e_pcz)

    cf2 = ax2.contourf(P, Z, Z_pcz, levels=20, cmap=cmap)
    ax2.set_xlabel(r"$\xi$ (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_title("KDE of (PC1, z)")
    fig.colorbar(cf2, ax=ax2, label="Density")

    ax1.set_xlim(xi.min(), xi.max())
    ax1.set_ylim(yi.min(), yi.max())
    ax2.set_xlim(pi.min(), pi.max())
    ax2.set_ylim(zi.min(), zi.max())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    chain = np.load(gps_output_path("mcmc_chain_adroit_5_test_xy_lever.npz"))
    DOG_num = 0
    sample = chain["CDOG_aug"][::100, DOG_num]

    init_aug = np.array(
        [
            [-397.63809, 371.47355, 773.26347],
            [825.31541, -110.93683, -734.15039],
            [236.27742, -1307.44426, -2189.59746],
        ]
    )
    prior_aug = 3.0

    plot_kde_mcmc(
        sample,
        nbins=100,
        cmap="viridis",
        prior_mean=init_aug[DOG_num],
        prior_sd=prior_aug,
    )
