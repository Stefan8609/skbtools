import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import final_bias_geiger
from geometry.ECEF_Geodetic import ECEF_Geodetic

from pymap3d import geodetic2enu
from bermuda_data import gps_data_path, gps_output_path

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 20,  # change this freely
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)


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
            f"{xlabel}: $\mu$={mean_x:.2f}, $\sigma$={std_x:.2f}\n"
            f"{ylabel}: $\mu$={mean_y:.2f}, $\sigma$={std_y:.2f}"
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


def error_ellipse(
    num_points,
    time_noise,
    position_noise,
    downsample,
    generate=True,
    DOG_num=1,
):
    """Plot error ellipses from Monte Carlo samples.

    Parameters
    ----------
    num_points : int
        Number of Monte Carlo iterations to run.
    time_noise : float
        Standard deviation of timing noise in seconds.
    position_noise : float
        Standard deviation of GNSS position noise in metres.
    downsample : int
        Downsampling interval applied to the clean GNSS trajectory.
    generate : bool, optional
        Generate new Monte Carlo results when True, otherwise load saved results.
    DOG_num : int, optional
        Bermuda DOG data set used for the synthetic trajectory.

    Returns
    -------
    None
        Displays plots of the estimated CDOG positions and lever errors.
    """

    real_lever = np.array([-12.4659, 9.6021, -13.2993])

    if generate:
        gps1_to_others = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.39341409, -4.22350344, 0.02941493],
                [-12.09568416, -0.94568462, 0.0043972],
                [-8.68674054, 5.16918806, 0.02499322],
            ]
        )

        esv_table_generate = sio.loadmat(
            gps_data_path("ESV_Tables/global_table_esv.mat")
        )
        dz_array_generate = esv_table_generate["distance"].flatten()
        angle_array_generate = esv_table_generate["angle"].flatten()
        esv_matrix_generate = esv_table_generate["matrice"]

        esv_table_inversion = sio.loadmat(
            gps_data_path("ESV_Tables/global_table_esv_perturbed.mat")
        )
        dz_array_inversion = esv_table_inversion["distance"].flatten()
        angle_array_inversion = esv_table_inversion["angle"].flatten()
        esv_matrix_inversion = esv_table_inversion["matrice"]

        true_offset = np.random.rand() * 9000 + 1000

        (
            correct_CDOG_data,
            CDOG,
            correct_GPS_Coordinates,
            correct_GPS_data,
            _,
        ) = bermuda_trajectory(
            0,
            0,
            0,
            0,
            dz_array_generate,
            angle_array_generate,
            esv_matrix_generate,
            offset=true_offset,
            DOG_num=DOG_num,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=real_lever,
        )

        leg1 = (correct_GPS_data / 3600 >= 9.0) & (correct_GPS_data / 3600 <= 11.0)
        leg2 = (correct_GPS_data / 3600 >= 12.4) & (correct_GPS_data / 3600 <= 15.0)
        leg_mask = leg1 | leg2

        correct_GPS_Coordinates = correct_GPS_Coordinates[leg_mask][::downsample]
        correct_GPS_data = correct_GPS_data[leg_mask][::downsample]
        # correct_GPS_Coordinates = correct_GPS_Coordinates[::downsample]
        # correct_GPS_data = correct_GPS_data[::downsample]
        correct_CDOG_time = correct_CDOG_data[:, 0] + correct_CDOG_data[:, 1]

        print("True Offset:", true_offset)

        estimate_array = np.zeros((num_points, 3))
        lever_diff_array = np.zeros((num_points, 3))
        RMSE_array = np.zeros(num_points)

        for i in range(num_points):
            try:
                GPS_Coordinates = correct_GPS_Coordinates + np.random.normal(
                    0,
                    position_noise,
                    correct_GPS_Coordinates.shape,
                )
                GPS_data = correct_GPS_data.copy()

                CDOG_time = correct_CDOG_time + np.random.normal(
                    0,
                    time_noise,
                    len(correct_CDOG_time),
                )
                CDOG_remain, CDOG_int = np.modf(CDOG_time)
                CDOG_data = np.column_stack((CDOG_int, CDOG_remain))

                initial_guess = CDOG + np.random.normal(0, 30, 3)
                initial_lever = real_lever - np.random.normal(0, 2, 3)

                lever, offset, inversion_result = simulated_annealing_bias(
                    300,
                    CDOG_data,
                    GPS_data,
                    GPS_Coordinates,
                    gps1_to_others,
                    initial_guess,
                    initial_lever,
                    dz_array_inversion,
                    angle_array_inversion,
                    esv_matrix_inversion,
                    real_data=True,
                    initial_offset=true_offset,
                    z_sample=True,
                    enforce_offset=True,
                )

                inversion_guess = inversion_result[:3]
                time_bias = inversion_result[3]
                esv_bias = inversion_result[4]

                transponder_coordinates = findTransponder(
                    GPS_Coordinates,
                    gps1_to_others,
                    lever,
                )

                (
                    inversion_result,
                    CDOG_full,
                    GPS_full,
                    CDOG_clock,
                    GPS_clock,
                ) = final_bias_geiger(
                    inversion_guess,
                    CDOG_data,
                    GPS_data,
                    transponder_coordinates,
                    offset,
                    esv_bias,
                    time_bias,
                    dz_array_inversion,
                    angle_array_inversion,
                    esv_matrix_inversion,
                    real_data=True,
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
                print(
                    f"\n{i + 1}/{num_points} iterations complete "
                    f"(offset error: {abs(offset - true_offset):.4f} s)"
                )

            except Exception as e:
                print(f"\n{i + 1}/{num_points} iteration failed: {e}")
                estimate_array[i] = np.nan
                lever_diff_array[i] = np.nan
                RMSE_array[i] = np.nan

        CDOG_lat, CDOG_lon, CDOG_height = ECEF_Geodetic(np.array([CDOG]))
        estimate_lat, estimate_lon, estimate_height = ECEF_Geodetic(estimate_array)

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
        npzfile = np.load(gps_output_path("Plot_Data/synthetic_error_ellipse.npz"))
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
    _add_xy_stats(axs[1], x1, y1, "E", "U")
    _center_axes_at_zero(axs[1], x1, y1)
    axs[1].set_xlabel("Easting (cm)")
    axs[1].set_ylabel("Elevation (cm)")
    axs[1].legend(loc="lower right")
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
    cbar2.set_label("$c\sigma_\epsilon$ (cm)")
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
    _add_xy_stats(axs[2], x2, y2, "N", "U")
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
    axs[1].legend(loc="lower right")
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
    cbar5.set_label("$c\sigma_\epsilon$ (cm)")
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
        gps_output_path("Plot_Data/synthetic_error_ellipse"),
        estimate_converted=estimate_converted,
        lever_diff_array=lever_diff_array,
        RMSE_array=RMSE_array,
    )


if __name__ == "__main__":
    num_points = 1000
    time_noise = 2 * 10**-5
    position_noise = 2 * 10**-2
    downsample = 5
    generate = True
    error_ellipse(num_points, time_noise, position_noise, downsample, generate=generate)

    npzfile = np.load(gps_output_path("Plot_Data/synthetic_error_ellipse.npz"))
    estimate_converted = npzfile["estimate_converted"]
    lever_diff_array = npzfile["lever_diff_array"]
    RMSE_array = npzfile["RMSE_array"]

    print("Testing loaded data:")
    print("Estimate converted shape:", estimate_converted.shape)
    print("Lever diff array shape:", lever_diff_array.shape)
    print("RMSE array shape:", RMSE_array.shape)
