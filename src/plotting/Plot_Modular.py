import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from GeigerMethod.Synthetic.Numba_Functions.ECEF_Geodetic import ECEF_Geodetic
from data import gps_data_path

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
    }
)


def _save_fig(fig, save, tag, path, timestamp=None, ext="pdf"):
    """Helper to save `fig` under Figs/ with timestamp and optional tag."""
    if not save:
        return
    # 1) build timestamp and filename
    fname = f"{tag}_{timestamp}.{ext}"

    # 2) build the directory path and ensure it exists
    #    gps_data_path should return the root data directory, e.g. '/Users/…/Data'
    dirpath = gps_data_path(f"Figs/Residual/{timestamp}")
    os.makedirs(dirpath, exist_ok=True)

    fullpath = os.path.join(dirpath, fname)
    fig.savefig(fullpath, format=ext, bbox_inches="tight")


def time_series_plot(
    CDOG_clock,
    CDOG_full,
    GPS_clock,
    GPS_full,
    position_noise=0,
    time_noise=0,
    block=True,
    save=False,
    path="Figs",
    timestamp=None,
):
    """Plot DOG and GPS time series with residuals."""
    difference_data = CDOG_full - GPS_full

    # convert ms residuals to a normal distribution
    mu, std = norm.fit(difference_data * 1000)

    # Get range of times for zoom in
    zoom_region = np.random.randint(min(CDOG_clock), max(CDOG_clock) - 100)
    zoom_idx = (np.abs(CDOG_clock - zoom_region)).argmin()
    zoom_length = 1200

    # Plot axes to return
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1, 4, 2, 1], "height_ratios": [2, 1]},
    )
    axes[0, 0].axis("off")
    axes[0, 3].axis("off")

    # Acoustic vs GNSS plot
    axes[0, 1].scatter(
        CDOG_clock,
        CDOG_full,
        s=5,
        label="Observed Travel Times",
        alpha=0.6,
        marker="o",
        color="b",
        zorder=2,
    )
    axes[0, 1].scatter(
        GPS_clock,
        GPS_full,
        s=10,
        label="Modelled Travel Times",
        alpha=1,
        marker="x",
        color="r",
        zorder=1,
    )
    axes[0, 1].axvline(zoom_region, color="k")
    axes[0, 1].axvline(zoom_region + zoom_length, color="k")
    axes[0, 1].set_ylabel("Travel Time (s)")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_xlim(min(CDOG_clock), max(CDOG_clock))

    axes[0, 2].scatter(
        CDOG_clock[zoom_idx : zoom_idx + zoom_length],
        CDOG_full[zoom_idx : zoom_idx + zoom_length],
        s=5,
        label="Observed Travel Times",
        alpha=0.6,
        marker="o",
        color="b",
        zorder=2,
    )
    axes[0, 2].scatter(
        GPS_clock[zoom_idx : zoom_idx + zoom_length],
        GPS_full[zoom_idx : zoom_idx + zoom_length],
        s=10,
        label="Modelled Travel Times",
        alpha=1,
        marker="x",
        color="r",
        zorder=1,
    )
    axes[0, 2].set_xlim(
        min(CDOG_clock[zoom_idx : zoom_idx + zoom_length]),
        max(CDOG_clock[zoom_idx : zoom_idx + zoom_length]),
    )
    axes[0, 2].legend(loc="upper right")

    # Histogram and normal distributions
    data_ms = difference_data * 1000
    mask = np.abs(data_ms - mu) <= 3 * std
    n, bins, patches = axes[1, 0].hist(
        data_ms[mask],
        orientation="horizontal",
        bins=40,
        alpha=0.5,
        density=True,
    )

    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    axes[1, 0].set_xlim([n.min() * 1.4, n.max() * 1.4])
    axes[1, 0].set_ylim([mu - 3 * std, mu + 3 * std])
    p = norm.pdf(x, mu, std)
    point1, point2 = norm.pdf(np.array([-std, std]), mu, std)
    axes[1, 0].plot(p, x, "k", linewidth=2, label="Normal Distribution of Differences")
    axes[1, 0].scatter([point1, point2], [-std, std], s=10, color="r", zorder=1)

    # add horizontal lines for the noise and uncertainty
    axes[1, 0].axhline(-std, color="r", label="Observed Noise")
    axes[1, 0].axhline(std, color="r")
    axes[1, 0].text(-0.1, std * 1.2, "$\\sigma_p$", va="center", color="r")

    if position_noise != 0:
        axes[1, 0].axhline(
            -position_noise / 1515 * 1000, color="g", label="Input Position Noise"
        )
        axes[1, 0].axhline(position_noise / 1515 * 1000, color="g")
        axes[1, 0].text(
            -0.2,
            position_noise / 1515 * 1000 * 0.5,
            "$\\sigma_x$",
            va="center",
            color="g",
        )

    if time_noise != 0:
        axes[1, 0].axhline(-time_noise * 1000, color="y", label="Input Time Noise")
        axes[1, 0].axhline(time_noise * 1000, color="y")
        axes[1, 0].text(-0.2, time_noise * 1000, "$\\sigma_t$", va="center", color="y")

    # invert axis and plot
    axes[1, 0].set_ylabel(
        f"Difference (ms) \n Std: {np.round(std, 3)} "
        f"ms or {np.round(std * 1515 / 10, 2)} cm"
    )
    axes[1, 0].set_xlabel("Normalized Frequency")
    axes[1, 0].invert_xaxis()

    # Difference plot
    axes[1, 1].scatter(CDOG_clock, difference_data * 1000, s=1)
    axes[1, 1].axvline(zoom_region, color="k")
    axes[1, 1].axvline(zoom_region + zoom_length, color="k")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_ylim([mu - 3 * std, mu + 3 * std])
    axes[1, 1].set_xlim(min(CDOG_clock), max(CDOG_clock))
    axes[1, 1].axhline(-std, color="r", label="Observed Noise")
    axes[1, 1].axhline(std, color="r")

    # Zoom difference plot
    mu_zoom, std_zoom = norm.fit(
        difference_data[zoom_idx : zoom_idx + zoom_length] * 1000
    )
    axes[1, 2].scatter(
        CDOG_clock[zoom_idx : zoom_idx + zoom_length],
        difference_data[zoom_idx : zoom_idx + zoom_length] * 1000,
        s=1,
    )
    axes[1, 2].set_xlabel("Time (ms)")
    axes[1, 2].set_ylim([mu - 3 * std, mu + 3 * std])
    axes[1, 2].set_xlim(
        min(CDOG_clock[zoom_idx : zoom_idx + zoom_length]),
        max(CDOG_clock[zoom_idx : zoom_idx + zoom_length]),
    )
    axes[1, 2].axhline(mu_zoom - std_zoom, color="r", label="Observed Noise")
    axes[1, 2].axhline(mu_zoom + std_zoom, color="r")
    axes[1, 2].yaxis.tick_right()

    # Histogram and normal distributions
    data_zoom_ms = difference_data[zoom_idx : zoom_idx + zoom_length] * 1000
    mask_zoom = np.abs(data_zoom_ms - mu_zoom) <= 3 * std_zoom
    n_zoom, bins_zoom, patches_zoom = axes[1, 3].hist(
        data_zoom_ms[mask_zoom],
        orientation="horizontal",
        bins=40,
        alpha=0.5,
        density=True,
    )

    x_zoom = np.linspace(mu_zoom - 3 * std_zoom, mu_zoom + 3 * std_zoom, 100)
    axes[1, 3].set_xlim([n_zoom.min() * 1.4, n_zoom.max() * 1.4])
    axes[1, 3].set_ylim([mu_zoom - 3 * std_zoom, mu_zoom + 3 * std_zoom])
    p_zoom = norm.pdf(x_zoom, mu_zoom, std_zoom)
    point1, point2 = norm.pdf(np.array([-std_zoom, std_zoom]), mu_zoom, std_zoom)
    axes[1, 3].plot(
        p_zoom, x_zoom, "k", linewidth=2, label="Normal Distribution of Differences"
    )
    axes[1, 3].scatter(
        [point1, point2], [-std_zoom, std_zoom], s=10, color="r", zorder=1
    )

    # add horizontal lines for the noise and uncertainty
    axes[1, 3].axhline(mu_zoom - std_zoom, color="r", label="Observed Noise")
    axes[1, 3].axhline(mu_zoom + std_zoom, color="r")
    axes[1, 3].text(
        -0.1, mu_zoom + std_zoom * 1.2, "$\\sigma_p$", va="center", color="r"
    )

    if position_noise != 0:
        axes[1, 3].axhline(
            -position_noise / 1515 * 1000, color="g", label="Input Position Noise"
        )
        axes[1, 3].axhline(position_noise / 1515 * 1000, color="g")
        axes[1, 3].text(
            -0.2,
            position_noise / 1515 * 1000 * 0.5,
            "$\\sigma_x$",
            va="center",
            color="g",
        )

    if time_noise != 0:
        axes[1, 3].axhline(-time_noise * 1000, color="y", label="Input Time Noise")
        axes[1, 3].axhline(time_noise * 1000, color="y")
        axes[1, 3].text(-0.2, time_noise * 1000, "$\\sigma_t$", va="center", color="y")

    # invert axis and plot
    axes[1, 3].set_ylabel(
        f"Difference (ms) \n Std: {np.round(std_zoom, 3)} "
        f"ms or {np.round(std_zoom * 1515 / 10, 2)} cm"
    )
    axes[1, 3].yaxis.set_label_position("right")
    axes[1, 3].yaxis.tick_right()
    axes[1, 3].set_ylim([mu - 3 * std, mu + 3 * std])
    axes[1, 3].set_xlabel("Normalized Frequency")
    axes[1, 3].invert_xaxis()

    # Adjust spacing between subplots
    plt.tight_layout()

    _save_fig(fig, save=save, tag="timeseries", path=path, timestamp=timestamp)

    plt.show(block=block)


def trajectory_plot(
    coordinates,
    GPS_clock,
    CDOGs,
    block=True,
    save=False,
    path="Figs",
    timestamp=None,
):
    """Plot the surface vessel trajectory in the horizontal plane."""
    # Calculate time values in hours for proper colorbar range
    times_hours = GPS_clock / 3600  # Convert seconds to hours
    min_time = np.min(times_hours)
    max_time = np.max(times_hours)

    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=1,
        c=times_hours,
        cmap="viridis",
        label="Surface Vessel",
    )
    nums = {0: "1", 1: "3", 2: "4"}
    for i in range(len(CDOGs)):
        plt.scatter(
            CDOGs[i, 0],
            CDOGs[i, 1],
            marker="x",
            s=20,
            label=f"CDOG {nums[i]}",
            color="k",
        )
    plt.colorbar(scatter, label="Elapsed Time (hours)")
    plt.clim(min_time, max_time)  # Set the colorbar to actual time range
    plt.title("Plot of Trajectory and CDOG location")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    plt.show(block=block)


def split_trajectory_plot(
    coordinates, GPS_clock, CDOGs, numsplit, save=False, path="Figs", timestamp=None
):
    """
    Plot a 2D trajectory split into `numsplit` contiguous blocks,
    marking CDOG locations, and optionally save the figure.

    Parameters
    ----------
    coordinates : (N,2) array-like
        X,Y coordinates to plot.
    GPS_clock : array-like, unused here but retained for signature.
    CDOGs : (M,2) array-like
        M CDOG X,Y positions.
    numsplit : int
        Number of blocks to split the trajectory into.
    save : bool, optional
        If True, save the figure via `_save_fig`.
    path : str, optional
        Directory under which to save (passed to `_save_fig`).
    timestamp : str or None, optional
        Timestamp string to tag the filename. If None, uses current time.
    """
    # create figure & axis
    fig, ax = plt.subplots(figsize=(8, 6))

    N = coordinates.shape[0]
    base = N // numsplit
    remainder = N % numsplit
    start = 0

    cmap = plt.get_cmap("tab10")
    for n in range(numsplit):
        size = base + 1 if n < remainder else base
        end = start + size
        blk = coordinates[start:end]
        ax.scatter(
            blk[:, 0], blk[:, 1], color=cmap(n % cmap.N), label=f"Block {n + 1}", s=1
        )
        start = end

    # plot CDOGs
    nums = {0: "1", 1: "3", 2: "4"}
    for i in range(len(CDOGs)):
        ax.scatter(
            CDOGs[i, 0],
            CDOGs[i, 1],
            marker="x",
            s=20,
            label=f"CDOG {nums[i]}",
            color="k",
        )

    ax.set_title("Trajectory Split into Blocks with CDOG Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis("equal")
    ax.legend()

    _save_fig(
        fig,
        save=save,
        tag=f"split_trajectory_{numsplit}",
        path=path,
        timestamp=timestamp,
    )

    plt.show()


def range_residual(
    transponder_coordinates,
    ESV,
    CDOG,
    CDOG_full,
    GPS_full,
    GPS_clock,
    save=False,
    path="Figs",
    timestamp=None,
):
    """Plot residual range errors along the track."""
    times_hours = GPS_clock / 3600  # Convert seconds to hours
    range_residuals = (CDOG_full - GPS_full) * ESV * 100  # Convert to cm
    calculated_range = np.linalg.norm(transponder_coordinates - CDOG, axis=1)
    mu_rr, std_rr = norm.fit(range_residuals)

    # Create a scatter plot and histogram of the range residuals
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1.2]}
    )

    # Scatter subplot for range residuals
    scatter = axes[0].scatter(
        calculated_range / 1000, range_residuals, s=1, c=times_hours, cmap="viridis"
    )
    fig.colorbar(scatter, ax=axes[0], label="Elapsed Time (hours)")
    axes[0].set_xlabel("Calculated Slant Range (km)")
    axes[0].set_ylabel("Slant Range Residuals (cm)")
    axes[0].axhline(-std_rr, color="r")
    axes[0].axhline(std_rr, color="r")
    axes[0].set_ylim([mu_rr - 3 * std_rr, mu_rr + 3 * std_rr])

    # Histogram subplot with normal curve
    mask_r = np.abs(range_residuals - mu_rr) <= 3 * std_rr
    n, bins, patches = axes[1].hist(
        range_residuals[mask_r],
        bins=40,
        density=True,
        alpha=0.5,
        color="C0",
        orientation="horizontal",
    )
    x = np.linspace(mu_rr - 3 * std_rr, mu_rr + 3 * std_rr, 100)
    p = norm.pdf(x, mu_rr, std_rr)
    axes[1].plot(
        p, x, "k", linewidth=2, label=f"Normal fit: mean={mu_rr:.2f}, std={std_rr:.2f}"
    )
    axes[1].axhline(-std_rr, color="r")
    axes[1].axhline(std_rr, color="r")
    axes[1].set_ylim([mu_rr - 3 * std_rr, mu_rr + 3 * std_rr])
    axes[1].set_xlabel("Density")
    axes[1].set_ylabel("Slant Range Residuals (cm)")
    axes[1].legend()

    _save_fig(fig, save=save, tag="rangeresidual", path=path, timestamp=timestamp)

    plt.tight_layout()
    plt.show()


def elevation_angle_residual(
    angles, CDOG_full, GPS_full, save=False, path="Figs", timestamp=None
):
    """
    Compute residuals (CDOG_full – GPS_full) and plot them versus grazing angle.

    Parameters
    ----------
    angles : array-like, shape (N,)
        Grazing angles (e.g., in degrees) for each measurement.
    CDOG_full : array-like, shape (N,)
        Observed values (e.g., CDOG‐derived times or angles).
    GPS_full : array-like, shape (N,)
        Predicted values (e.g., GPS‐derived times or angles).
    save : bool, default False
        If True, save the figure to disk.
    path : str, default "Figs"
        Directory in which to save the figure.
    timestamp : str or None, default None
        Timestamp string to include in the filename; if None, current time is used.
    """
    # Compute residuals
    residuals = CDOG_full - GPS_full

    # Create figure & axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot
    ax.scatter(angles, residuals, s=10, alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Grazing Angle (°)")
    ax.set_ylabel("Residual (CDOG – GPS)")
    ax.set_title("Residuals vs. Elevation Angle")
    ax.grid(True)

    _save_fig(fig, save=save, tag="angleresidual", path=path, timestamp=timestamp)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    CDOGs = np.array(
        [
            [-398.16, 371.90, 773.02],
            [825.182985, -111.05670221, -734.10011698],
            [236.428385, -1307.98390221, -2189.21991698],
        ]
    )
    CDOGs += CDOG_guess_base

    CDOGs_lat, CDOGs_lon, CDOGs_height = ECEF_Geodetic(CDOGs)

    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    GPS_lat, GPS_lon, GPS_height = ECEF_Geodetic(GPS_Coordinates[:, 0, :])
    # trajectory_plot(
    #     np.array([GPS_lon, GPS_lat, GPS_height]).T,
    #     GPS_data,
    #     np.array([CDOGs_lon, CDOGs_lat, CDOGs_height]).T,
    # )
    split_trajectory_plot(
        np.array([GPS_lon, GPS_lat, GPS_height]).T,
        GPS_data,
        np.array([CDOGs_lon, CDOGs_lat, CDOGs_height]).T,
        10,
        save=True,
    )
