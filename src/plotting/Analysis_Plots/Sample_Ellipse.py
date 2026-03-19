import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from Inversion_Workflow.Synthetic.Modular_Synthetic import (
    modular_synthetic,
)
from geometry.ECEF_Geodetic import ECEF_Geodetic

from pymap3d import geodetic2enu
from data import gps_output_path


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
    Plot an n_std-sigma error ellipse for the 2D points (x, y) onto the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the ellipse on.
    x, y : array-like, shape (N,)
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


def _add_xy_stats(ax, x, y, xlabel="x", ylabel="y"):
    """Add mean and std of x and y to upper left of axis."""
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)

    ax.text(
        0.02,
        0.98,
        (
            f"{xlabel}: mean={mean_x:.2f}, std={std_x:.2f}\n"
            f"{ylabel}: mean={mean_y:.2f}, std={std_y:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )


def _center_axes_at_zero(ax, x, y, pad_fraction=0.08):
    """Set symmetric x/y limits so the plot is centered at (0, 0)."""
    max_abs_x = np.nanmax(np.abs(x))
    max_abs_y = np.nanmax(np.abs(y))
    max_abs = max(max_abs_x, max_abs_y)

    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    max_abs *= 1.0 + pad_fraction

    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)


def error_ellipse(num_points, time_noise, position_noise, downsample, generate=True):
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
    real_lever = np.array([-12.4659, 9.6021, -13.2993])

    if generate:
        estimate_array = np.zeros((num_points, 3))
        lever_diff_array = np.zeros((num_points, 3))
        RMSE_array = np.zeros(num_points)
        for i in range(num_points):
            try:
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

                if RMSE > 15:
                    print(f"\n{i + 1}/{num_points} iteration failed to converge")
                    estimate_array[i] = np.nan
                    lever_diff_array[i] = np.nan
                    RMSE_array[i] = np.nan
                    continue

                estimate_array[i] = inversion_result[:3]
                lever_diff_array[i] = lever - real_lever
                RMSE_array[i] = RMSE
                print(f"\n{i + 1}/{num_points} iterations complete")
            except Exception as e:
                print(f"\n{i + 1}/{num_points} iteration failed: {e}")
                estimate_array[i] = np.nan
                lever_diff_array[i] = np.nan
                RMSE_array[i] = np.nan

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

    else:
        npzfile = np.load(gps_output_path("synthetic_error_ellipse.npz"))
        estimate_converted = npzfile["estimate_converted"]
        lever_diff_array = npzfile["lever_diff_array"]
        RMSE_array = npzfile["RMSE_array"]

    # Remove NaN values
    valid_indices = (
        ~np.isnan(estimate_converted).any(axis=1)
        & ~np.isnan(lever_diff_array).any(axis=1)
        & ~np.isnan(RMSE_array)
    )
    estimate_converted = estimate_converted[valid_indices]
    lever_diff_array = lever_diff_array[valid_indices]
    RMSE_array = RMSE_array[valid_indices]

    colorbar_max = 7
    colorbar_min = 4

    # ------------------------
    # Plot the CDOG estimates
    # ------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Easting vs Northing
    x0 = estimate_converted[:, 0] * 100
    y0 = estimate_converted[:, 1] * 100
    axs[0].scatter(
        x0,
        y0,
        c=RMSE_array,
        cmap="viridis",
        vmin=colorbar_min,
        vmax=colorbar_max,
        s=5,
        label=r"CDOG Estimates",
    )
    axs[0].scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    for std in range(1, 4):
        _plot_error_ellipse(
            axs[0],
            x0,
            y0,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
        )
    _add_xy_stats(axs[0], x0, y0, "E", "N")
    _center_axes_at_zero(axs[0], x0, y0)
    axs[0].set_xlabel("Easting (cm)")
    axs[0].set_ylabel("Northing (cm)")
    axs[0].axis("equal")
    axs[0].grid()

    # Easting vs Elevation
    x1 = estimate_converted[:, 0] * 100
    y1 = estimate_converted[:, 2] * 100
    axs[1].scatter(
        x1,
        y1,
        c=RMSE_array,
        cmap="viridis",
        vmin=colorbar_min,
        vmax=colorbar_max,
        s=5,
        label=r"CDOG Estimates",
    )
    axs[1].scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    for std in range(1, 3):
        _plot_error_ellipse(
            axs[1],
            x1,
            y1,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
        )
    _add_xy_stats(axs[1], x1, y1, "E", "Z")
    _center_axes_at_zero(axs[1], x1, y1)
    axs[1].set_xlabel("Easting (cm)")
    axs[1].set_ylabel("Elevation (cm)")
    axs[1].legend(loc="upper right")
    axs[1].axis("equal")
    axs[1].grid()

    # Northing vs Elevation
    x2 = estimate_converted[:, 1] * 100
    y2 = estimate_converted[:, 2] * 100
    sc2 = axs[2].scatter(
        x2,
        y2,
        c=RMSE_array,
        cmap="viridis",
        vmin=colorbar_min,
        vmax=colorbar_max,
        s=5,
        label=r"CDOG Estimates",
    )
    cbar2 = fig.colorbar(sc2, ax=axs[2])
    cbar2.set_label("RMSE (cm)")
    axs[2].scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    for std in range(1, 3):
        _plot_error_ellipse(
            axs[2],
            x2,
            y2,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
        )
    _add_xy_stats(axs[2], x2, y2, "N", "Z")
    _center_axes_at_zero(axs[2], x2, y2)
    axs[2].set_xlabel("Northing (cm)")
    axs[2].set_ylabel("Elevation (cm)")
    axs[2].axis("equal")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    # ---------------------
    # Plot the lever errors
    # ---------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # x vs y error
    x3 = lever_diff_array[:, 0] * 100
    y3 = lever_diff_array[:, 1] * 100
    axs[0].scatter(
        x3,
        y3,
        c=RMSE_array,
        cmap="viridis",
        vmin=colorbar_min,
        vmax=colorbar_max,
        s=5,
        label=r"Lever Errors",
    )
    axs[0].scatter(0, 0, s=100, color="red", marker="o", label="Lever Actual")
    for std in range(1, 4):
        _plot_error_ellipse(
            axs[0],
            x3,
            y3,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
        )
    _add_xy_stats(axs[0], x3, y3, "x", "y")
    _center_axes_at_zero(axs[0], x3, y3)
    axs[0].set_xlabel("x Lever Error (cm)")
    axs[0].set_ylabel("y Lever Error (cm)")
    axs[0].axis("equal")
    axs[0].grid()

    # x vs z error
    x4 = lever_diff_array[:, 0] * 100
    y4 = lever_diff_array[:, 2] * 100
    axs[1].scatter(
        x4,
        y4,
        c=RMSE_array,
        cmap="viridis",
        vmin=colorbar_min,
        vmax=colorbar_max,
        s=5,
        label=r"Lever Errors",
    )
    axs[1].scatter(0, 0, s=100, color="red", marker="o", label="Lever Actual")
    for std in range(1, 3):
        _plot_error_ellipse(
            axs[1],
            x4,
            y4,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
        )
    _add_xy_stats(axs[1], x4, y4, "x", "z")
    _center_axes_at_zero(axs[1], x4, y4)
    axs[1].set_xlabel("x Lever Error (cm)")
    axs[1].set_ylabel("z Lever Error (cm)")
    axs[1].legend(loc="upper right")
    axs[1].axis("equal")
    axs[1].grid()

    # y vs z error
    x5 = lever_diff_array[:, 1] * 100
    y5 = lever_diff_array[:, 2] * 100
    sc5 = axs[2].scatter(
        x5,
        y5,
        c=RMSE_array,
        cmap="viridis",
        vmin=colorbar_min,
        vmax=colorbar_max,
        s=5,
        label=r"Lever Errors",
    )
    cbar5 = fig.colorbar(sc5, ax=axs[2])
    cbar5.set_label("RMSE (cm)")
    axs[2].scatter(0, 0, s=100, color="red", marker="o", label="Lever Actual")
    for std in range(1, 3):
        _plot_error_ellipse(
            axs[2],
            x5,
            y5,
            n_std=std,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
        )
    _add_xy_stats(axs[2], x5, y5, "y", "z")
    _center_axes_at_zero(axs[2], x5, y5)
    axs[2].set_xlabel("y Lever Error (cm)")
    axs[2].set_ylabel("z Lever Error (cm)")
    axs[2].axis("equal")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    np.savez(
        gps_output_path("synthetic_error_ellipse"),
        estimate_converted=estimate_converted,
        lever_diff_array=lever_diff_array,
        RMSE_array=RMSE_array,
    )


if __name__ == "__main__":
    num_points = 1000
    time_noise = 2 * 10**-5
    position_noise = 2 * 10**-2
    downsample = 50
    generate = False
    error_ellipse(num_points, time_noise, position_noise, downsample, generate=generate)

    npzfile = np.load(gps_output_path("synthetic_error_ellipse.npz"))
    estimate_converted = npzfile["estimate_converted"]
    lever_diff_array = npzfile["lever_diff_array"]
    RMSE_array = npzfile["RMSE_array"]

    print("Testing loaded data:")
    print("Estimate converted shape:", estimate_converted.shape)
    print("Lever diff array shape:", lever_diff_array.shape)
    print("RMSE array shape:", RMSE_array.shape)
