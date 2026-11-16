import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.spatial import cKDTree

from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.PuertoRico.Crossover_Mask import (
    CROSSOVER_LEG_UTC_BOUNDS,
    crossover_leg_mask,
)
from Inversion_Workflow.PuertoRico.Initialize_PuertoRico_Data import (
    CDOG_guess,
    find_valid_receivers,
    load_all_receivers,
    process_gnss,
    receiver_names,
)
from puertorico_data import gps_data_path


def save_plot(fig, name):
    """Save a Puerto Rico plot as a PDF."""
    path = gps_data_path(f"Figs/{name}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")


def filtered_elevation_plots(
    GNSS_start=0,
    GNSS_end=120,
    elevation_limits=None,
    save=False,
    mask=True,
    gridsize=180,
):
    """Plot receiver elevations colored by nearby retained-observation count."""
    if gridsize <= 0:
        raise ValueError(f"gridsize must be positive, received {gridsize}")

    print(f"Loading GPS Data between hours {GNSS_start} and {GNSS_end}")
    GPS_data, GPS_Coordinates, elevations, nsv, gdop = load_all_receivers(
        GNSS_start, GNSS_end
    )
    if mask:
        crossover_mask = np.logical_or.reduce(
            [
                crossover_leg_mask(GPS_data, leg_bounds)
                for leg_bounds in CROSSOVER_LEG_UTC_BOUNDS.values()
            ]
        )
        GPS_data = GPS_data[crossover_mask]
        GPS_Coordinates = GPS_Coordinates[crossover_mask]
        elevations = elevations[crossover_mask]
        nsv = nsv[crossover_mask]
        gdop = gdop[crossover_mask]

    if len(GPS_data) == 0:
        raise ValueError("No GNSS epochs satisfy the requested time and crossover mask")

    used_receivers = find_valid_receivers(GPS_Coordinates, nsv, gdop)
    filtered = used_receivers.copy()

    if elevation_limits is not None:
        elev_lower, elev_upper = elevation_limits
        filtered &= (elevations >= elev_lower) & (elevations <= elev_upper)
    if not np.any(filtered):
        raise ValueError("No receiver elevations satisfy the filtering criteria")

    time_hours = GPS_data / 3600.0
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = axes.ravel()
    density_plots = []
    vertical_bins = max(1, round(gridsize / np.sqrt(3.0)))
    time_radius = np.ptp(time_hours) / gridsize
    elevation_radius = np.ptp(elevations[filtered]) / vertical_bins
    if time_radius <= 0 or elevation_radius <= 0:
        raise ValueError("Density plotting requires varying times and elevations")

    for receiver, ax in enumerate(axes):
        available = np.isfinite(elevations[:, receiver])
        retained = filtered[:, receiver]
        rejected = available & ~retained
        ax.scatter(
            time_hours[rejected],
            elevations[rejected, receiver],
            s=2,
            color="lightgray",
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
            label="Rejected observations",
        )
        retained_time = time_hours[retained]
        retained_elevation = elevations[retained, receiver]
        scaled_points = np.column_stack(
            (retained_time / time_radius, retained_elevation / elevation_radius)
        )
        nearby_counts = cKDTree(scaled_points).query_ball_point(
            scaled_points,
            r=1.0,
            return_length=True,
        )
        plot_order = np.argsort(nearby_counts)
        density_plot = ax.scatter(
            retained_time[plot_order],
            retained_elevation[plot_order],
            c=nearby_counts[plot_order],
            s=2,
            cmap="viridis",
            edgecolors="none",
            rasterized=True,
        )
        density_plots.append(density_plot)

        if elevation_limits is not None:
            ax.axhline(elev_lower, color="tab:orange", linestyle="--")
            ax.axhline(elev_upper, color="tab:orange", linestyle="--")

        ax.set_title(f"{receiver_names[receiver]} Elevation")
        ax.grid(alpha=0.2)
        ax.text(
            0.02,
            0.04,
            f"Retained: {np.count_nonzero(retained):,} / "
            f"{np.count_nonzero(available):,}",
            transform=ax.transAxes,
        )

    maximum_count = max(
        np.max(plot.get_array()) if len(plot.get_array()) else 0
        for plot in density_plots
    )
    density_norm = colors.Normalize(vmin=1, vmax=max(2, maximum_count))
    for density_plot in density_plots:
        density_plot.set_norm(density_norm)

    colorbar = fig.colorbar(density_plots[0], ax=axes.tolist())
    colorbar.set_label("Nearby retained observations")
    axes[0].legend(markerscale=4)
    axes[2].set_xlabel("Hours since January 15, 2023 00:00 UTC")
    axes[3].set_xlabel("Hours since January 15, 2023 00:00 UTC")
    axes[0].set_ylabel("Ellipsoidal height (m)")
    axes[2].set_ylabel("Ellipsoidal height (m)")
    mask_label = " on DOG crossover legs" if mask else ""
    fig.suptitle(f"Puerto Rico GNSS Receiver Elevations{mask_label}")

    if save:
        save_plot(fig, "PuertoRico_Filtered_Elevation")

    plt.show()
    return fig


def barycenter_elevation_plot(
    GNSS_start=0,
    GNSS_end=120,
    save=False,
    mask=True,
    gridsize=180,
):
    """Plot barycenter elevation colored by nearby-observation count."""
    if gridsize <= 0:
        raise ValueError(f"gridsize must be positive, received {gridsize}")

    GPS_Coordinates, GPS_data, _ = process_gnss(GNSS_start, GNSS_end)

    if mask:
        crossover_mask = np.logical_or.reduce(
            [
                crossover_leg_mask(GPS_data, leg_bounds)
                for leg_bounds in CROSSOVER_LEG_UTC_BOUNDS.values()
            ]
        )
        GPS_data = GPS_data[crossover_mask]
        GPS_Coordinates = GPS_Coordinates[crossover_mask]

    if len(GPS_data) == 0:
        raise ValueError("No GNSS epochs satisfy the requested time and crossover mask")

    GPS_barycenter = np.mean(GPS_Coordinates, axis=1)
    _, _, elevation = ECEF_Geodetic(GPS_barycenter)
    time_hours = GPS_data / 3600.0
    vertical_bins = max(1, round(gridsize / np.sqrt(3.0)))
    time_radius = np.ptp(time_hours) / gridsize
    elevation_radius = np.ptp(elevation) / vertical_bins
    if time_radius <= 0 or elevation_radius <= 0:
        raise ValueError("Density plotting requires varying times and elevations")

    scaled_points = np.column_stack(
        (time_hours / time_radius, elevation / elevation_radius)
    )
    nearby_counts = cKDTree(scaled_points).query_ball_point(
        scaled_points,
        r=1.0,
        return_length=True,
    )
    plot_order = np.argsort(nearby_counts)

    fig, ax = plt.subplots(figsize=(8, 6))
    density_plot = ax.scatter(
        time_hours[plot_order],
        elevation[plot_order],
        c=nearby_counts[plot_order],
        s=2,
        cmap="viridis",
        norm=colors.Normalize(vmin=1, vmax=max(2, np.max(nearby_counts))),
        edgecolors="none",
        rasterized=True,
    )
    colorbar = fig.colorbar(density_plot, ax=ax)
    colorbar.set_label("Nearby observations")
    mask_label = " on DOG crossover legs" if mask else ""
    ax.set_title(f"Barycenter Elevation Density{mask_label}")
    ax.set_xlabel("Hours since January 15, 2023 00:00 UTC")
    ax.set_ylabel("Ellipsoidal height (m)")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    if save:
        save_plot(fig, "PuertoRico_Barycenter_Elevation")

    plt.show()
    return fig


def trajectory_plot(GNSS_start=0, GNSS_end=120, save=False, mask=True):
    """Plot the vessel trajectory with points colored by elapsed time."""
    GPS_Coordinates, GPS_data, _ = process_gnss(GNSS_start, GNSS_end)

    if mask:
        crossover_mask = np.logical_or.reduce(
            [
                crossover_leg_mask(GPS_data, leg_bounds)
                for leg_bounds in CROSSOVER_LEG_UTC_BOUNDS.values()
            ]
        )
        GPS_data = GPS_data[crossover_mask]
        GPS_Coordinates = GPS_Coordinates[crossover_mask]

    if len(GPS_data) == 0:
        raise ValueError("No GNSS epochs satisfy the requested time and crossover mask")

    GPS_barycenter = np.mean(GPS_Coordinates, axis=1)
    latitude, longitude, _ = ECEF_Geodetic(GPS_barycenter)
    CDOG_latitude, CDOG_longitude, _ = ECEF_Geodetic(CDOG_guess[None, :])
    time_hours = GPS_data / 3600.0

    fig, ax = plt.subplots(figsize=(9, 7))
    trajectory = ax.scatter(
        longitude,
        latitude,
        c=time_hours,
        s=2,
        cmap="viridis",
        rasterized=True,
    )
    ax.scatter(
        longitude[0],
        latitude[0],
        s=90,
        color="limegreen",
        edgecolor="black",
        label="Start",
        zorder=3,
    )
    ax.scatter(
        CDOG_longitude[0],
        CDOG_latitude[0],
        s=140,
        color="tab:red",
        edgecolor="black",
        marker="*",
        label="C-DOG",
        zorder=4,
    )

    colorbar = fig.colorbar(trajectory, ax=ax)
    colorbar.set_label("Hours since January 15, 2023 00:00 UTC")
    mask_label = " on DOG crossover legs" if mask else ""
    ax.set_title(f"Puerto Rico Vessel Trajectory{mask_label}")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(latitude))))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    if save:
        save_plot(fig, "PuertoRico_Trajectory")

    plt.show()
    return fig


if __name__ == "__main__":
    GNSS_start = 45
    GNSS_end = 120
    elevation_limits = None
    save = True

    filtered_elevation_plots(
        GNSS_start,
        GNSS_end,
        elevation_limits=elevation_limits,
        save=save,
        mask=False,
    )
    trajectory_plot(GNSS_start, GNSS_end, save=save, mask=True)
    barycenter_elevation_plot(GNSS_start, GNSS_end, save=save, mask=True)
