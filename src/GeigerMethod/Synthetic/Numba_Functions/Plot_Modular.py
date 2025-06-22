import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ECEF_Geodetic import ECEF_Geodetic

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


def time_series_plot(
    CDOG_clock,
    CDOG_full,
    GPS_clock,
    GPS_full,
    position_noise=0,
    time_noise=0,
    block=True,
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
    n, bins, patches = axes[1, 0].hist(
        difference_data * 1000,
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

    n_zoom, bins_zoom, patches_zoom = axes[1, 3].hist(
        difference_data[zoom_idx : zoom_idx + zoom_length] * 1000,
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

    plt.show(block=block)


def trajectory_plot(coordinates, GPS_clock, CDOGs, block=True):
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


def range_residual(transponder_coordinates, ESV, CDOG, CDOG_full, GPS_full, GPS_clock):
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
    n, bins, patches = axes[1].hist(
        range_residuals,
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

    from data import gps_data_path

    data = np.load(gps_data_path("Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    GPS_lat, GPS_lon, GPS_height = ECEF_Geodetic(GPS_Coordinates[:, 0, :])
    trajectory_plot(
        np.array([GPS_lon, GPS_lat, GPS_height]).T,
        GPS_data,
        np.array([CDOGs_lon, CDOGs_lat, CDOGs_height]).T,
    )
