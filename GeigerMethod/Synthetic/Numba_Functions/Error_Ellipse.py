import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from Modular_Synthetic import modular_synthetic
from ECEF_Geodetic import ECEF_Geodetic

from pymap3d import geodetic2enu


def _plot_error_ellipse(
    ax,
    x,
    y,
    n_std=1.0,
    facecolor="none",
    edgecolor="black",
    linestyle="--",
    label=None,
    **ellipse_kwargs,
):
    """
    Plot an n_std‐sigma error ellipse for the 2D points (x, y) onto the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the ellipse on.
    x, y : array‐like, shape (N,)
        Data in the two coordinate directions.
    n_std : float, optional
        The radius of the ellipse in standard deviations (default 1.0).
    facecolor, edgecolor, linestyle, label : passed to Ellipse
    **ellipse_kwargs : other kwargs passed to Ellipse constructor
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mean_x, mean_y = x.mean(), y.mean()
    cov = np.cov(x, y)

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width, height = 2 * n_std * np.sqrt(eigvals)

    vx, vy = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))

    ellipse = Ellipse(
        (mean_x, mean_y),
        width,
        height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
        label=label,
        **ellipse_kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def error_ellipse(num_points, time_noise, position_noise):
    """Plot error ellipses from Monte Carlo samples.

    Parameters
    ----------
    num_points : int
        Number of Monte Carlo iterations to run.
    time_noise : float
        Standard deviation of timing noise.
    position_noise : float
        Standard deviation of position noise in metres.

    Returns
    -------
    None
        Displays plots of the estimated CDOG positions and lever errors.
    """

    # Real Lever
    real_lever = np.array([-12.48862757, 0.22622633, -15.89601934])

    estimate_array = np.zeros((num_points, 3))
    lever_diff_array = np.zeros((num_points, 3))
    RMSE_array = np.zeros(num_points)
    for i in range(num_points):
        (
            inversion_result,
            CDOG_data,
            CDOG_full,
            GPS_data,
            GPS_full,
            CDOG_clock,
            GPS_clock,
            CDOG,
            transponder_coordinates,
            GPS_Coordinates,
            offset,
            lever,
        ) = modular_synthetic(
            time_noise,
            position_noise,
            0,
            0,
            esv1="global_table_esv",
            esv2="global_table_esv_perturbed",
            generate_type=1,
            inversion_type=1,
            plot=False,
        )
        diff_data = (CDOG_full - GPS_full) * 1000
        RMSE = np.sqrt(np.nanmean(diff_data**2)) / 1000 * 1515 * 100

        estimate_array[i] = inversion_result[:3]
        lever_diff_array[i] = lever - real_lever
        RMSE_array[i] = RMSE
        print(f"{i + 1}/{num_points} iterations complete")

    # Convert to geodetic
    CDOG_lat, CDOG_lon, CDOG_height = ECEF_Geodetic(np.array([CDOG]))
    estimate_lat, estimate_lon, estimate_height = ECEF_Geodetic(estimate_array)

    # Convert to ENU coordinates
    estimate_converted = np.zeros((num_points, 3))
    for i in range(num_points):
        enu = geodetic2enu(
            estimate_lat[i],
            estimate_lon[i],
            estimate_height[i],
            CDOG_lat,
            CDOG_lon,
            CDOG_height,
        )
        estimate_converted[i] = np.squeeze(enu)

    # Plot the CDOG estimates and actual
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Easting vs Northing
    axs[0].scatter(
        estimate_converted[:, 0] * 100,
        estimate_converted[:, 1] * 100,
        c=RMSE_array,
        cmap="viridis",
        s=5,
        label=r"CDOG Estimates",
    )
    axs[0].scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    for std in range(4):
        _plot_error_ellipse(
            axs[0],
            estimate_converted[:, 0] * 100,
            estimate_converted[:, 1] * 100,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=None,
        )
    axs[0].set_xlabel("Easting (cm)")
    axs[0].set_ylabel("Northing (cm)")
    axs[0].axis("equal")
    axs[0].grid()

    # Easting vs Elevation
    axs[1].scatter(
        estimate_converted[:, 0] * 100,
        estimate_converted[:, 2] * 100,
        c=RMSE_array,
        cmap="viridis",
        s=5,
        label=r"CDOG Estimates",
    )
    axs[1].scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    for std in range(2):
        _plot_error_ellipse(
            axs[1],
            estimate_converted[:, 0] * 100,
            estimate_converted[:, 2] * 100,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=None,
        )
    axs[1].set_xlabel("Easting (cm)")
    axs[1].set_ylabel("Elevation (cm)")
    axs[1].axis("equal")
    axs[1].grid()

    # Northing vs Elevation
    sc2 = axs[2].scatter(
        estimate_converted[:, 1] * 100,
        estimate_converted[:, 2] * 100,
        c=RMSE_array,
        cmap="viridis",
        s=5,
        label=r"CDOG Estimates",
    )
    cbar2 = fig.colorbar(sc2, ax=axs[2])
    cbar2.set_label("RMSE (cm)")
    axs[2].scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    for std in range(2):
        _plot_error_ellipse(
            axs[2],
            estimate_converted[:, 1] * 100,
            estimate_converted[:, 2] * 100,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=None,
        )
    axs[2].set_xlabel("Northing (cm)")
    axs[2].set_ylabel("Elevation (cm)")
    axs[2].legend(loc="upper right")
    axs[2].axis("equal")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    # Plot the lever error
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # x vs y error
    axs[0].scatter(
        lever_diff_array[:, 0] * 100,
        lever_diff_array[:, 1] * 100,
        c=RMSE_array,
        cmap="viridis",
        s=5,
        label=r"Lever Errors",
    )
    for std in range(4):
        _plot_error_ellipse(
            axs[0],
            lever_diff_array[:, 0] * 100,
            lever_diff_array[:, 1] * 100,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=None,
        )
    axs[0].set_xlabel("x Lever Error (cm)")
    axs[0].set_ylabel("y Lever Error (cm)")
    axs[0].axis("equal")
    axs[0].grid()

    # x vs z error
    axs[1].scatter(
        lever_diff_array[:, 0] * 100,
        lever_diff_array[:, 2] * 100,
        c=RMSE_array,
        cmap="viridis",
        s=5,
        label=r"Lever Errors",
    )
    for std in range(2):
        _plot_error_ellipse(
            axs[1],
            lever_diff_array[:, 0] * 100,
            lever_diff_array[:, 2] * 100,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=None,
        )
    axs[1].set_xlabel("x Lever Error (cm)")
    axs[1].set_ylabel("z Lever Error (cm)")
    axs[1].axis("equal")
    axs[1].grid()

    # y vs z error
    sc5 = axs[2].scatter(
        lever_diff_array[:, 1] * 100,
        lever_diff_array[:, 2] * 100,
        c=RMSE_array,
        cmap="viridis",
        s=5,
        label=r"Lever Errors",
    )
    fig.colorbar(sc5, ax=axs[2])
    for std in range(2):
        _plot_error_ellipse(
            axs[2],
            lever_diff_array[:, 1] * 100,
            lever_diff_array[:, 2] * 100,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            label=None,
        )
    axs[2].set_xlabel("y Lever Error (cm)")
    axs[2].set_ylabel("z Lever Error (cm)")
    axs[2].axis("equal")
    axs[2].grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_points = 2000
    time_noise = 2 * 10**-5
    position_noise = 2 * 10**-2
    error_ellipse(num_points, time_noise, position_noise)
