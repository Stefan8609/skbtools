import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import combinations
from scipy.stats import norm

from jacksonville_data import gps_data_path


def load_and_process_data(path):
    """Load GNSS coordinates and position uncertainties."""
    data = sio.loadmat(path)

    days = data["days"].flatten() - 59011
    times = data["times"].flatten()
    time_GNSS = days * 86400 + times

    x = data["x"].flatten()
    y = data["y"].flatten()
    z = data["z"].flatten()
    elev = data["elev"].flatten()

    sig_n = data["sig_n"].flatten()
    sig_e = data["sig_e"].flatten()
    sig_u = data["sig_u"].flatten()
    nsv = data["nsv"].flatten()

    return (
        time_GNSS, x, y, z, elev,
        sig_n, sig_e, sig_u, nsv
    )


if __name__ == "__main__":
    # Plot settings
    USE_QUALITY_FILTER = False  # True = filtered, False = unfiltered

    # Position uncertainty thresholds (95%, meters)
    MAX_SIG_N = 0.05  # 5 cm
    MAX_SIG_E = 0.05  # 5 cm
    MAX_SIG_U = 0.10  # 10 cm
    MIN_SATELLITES = 10

    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.39341409, -4.22350344, 0.02941493],
            [-12.09568416, -0.94568462, 0.0043972],
            [-8.68674054, 5.16918806, 0.02499322],
        ]
    )


    paths = [
        gps_data_path("GPS_Data/JAX1.mat"),
        gps_data_path("GPS_Data/JAX2.mat"),
        gps_data_path("GPS_Data/JAX3.mat"),
        gps_data_path("GPS_Data/JAX4.mat"),
    ]

    # Load all four receivers
    all_data = [load_and_process_data(path) for path in paths]

    # Find timestamps shared by all receivers
    common_datetimes = set(all_data[0][0])
    for data in all_data[1:]:
        common_datetimes.intersection_update(data[0])
    common_datetimes = np.array(sorted(common_datetimes))

    if len(common_datetimes) == 0:
        raise ValueError("No timestamps are shared by all four GNSS receivers.")

    # Synchronize the arrays
    filtered_data = []
    for data in all_data:
        datetimes = data[0]

        time_indices = {time: idx for idx, time in enumerate(datetimes)}
        indices = np.array(
            [time_indices[t] for t in common_datetimes],
            dtype=int,
        )

        filtered_data.append(
            [np.asarray(values)[indices] for values in data]
        )

    filtered_data = np.asarray(filtered_data)

    # Coordinates: shape (epochs, receivers, xyz)
    GPS_Coordinates = np.stack(
        [
            filtered_data[:, 1, :],
            filtered_data[:, 2, :],
            filtered_data[:, 3, :],
        ],
        axis=-1,
    ).transpose(1, 0, 2)

    # Quality filtering: all four receivers must pass simultaneously
    sig_n = filtered_data[:, 5, :]
    sig_e = filtered_data[:, 6, :]
    sig_u = filtered_data[:, 7, :]
    nsv = filtered_data[:, 8, :]

    quality_mask = np.all(
        (sig_n < MAX_SIG_N)
        & (sig_e < MAX_SIG_E)
        & (sig_u < MAX_SIG_U)
        & (nsv >= MIN_SATELLITES),
        axis=0,
    )

    if USE_QUALITY_FILTER:
        GPS_Coordinates_plot = GPS_Coordinates[quality_mask]
        filter_label = "Filtered"
    else:
        GPS_Coordinates_plot = GPS_Coordinates
        filter_label = "Unfiltered"

    print(f"Common epochs: {len(common_datetimes):,}")
    print(f"Quality-accepted epochs: {quality_mask.sum():,}")
    print(f"Quality-accepted fraction: {quality_mask.mean() * 100:.2f}%")
    print(f"Plotting {filter_label.lower()} epochs: {len(GPS_Coordinates_plot):,}")

    if GPS_Coordinates_plot.shape[0] == 0:
        raise ValueError(
            "No epochs available to plot. "
            "Try relaxing the uncertainty thresholds or disabling filtering."
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
        distances = np.linalg.norm(
            GPS_Coordinates_plot[:, i, :]
            - GPS_Coordinates_plot[:, j, :],
            axis=1,
        )
        distances = distances[np.isfinite(distances)]

        if len(distances) < 2:
            raise ValueError(
                f"Insufficient finite distances for GNSS {i+1}–GNSS {j+1}."
            )

        mu, sigma = norm.fit(distances)
        grid_distance = np.linalg.norm(
            gps1_to_others[i] - gps1_to_others[j]
        )

        ax.hist(
            distances,
            bins=100,
            density=True,
            alpha=0.5,
            label="GNSS distances",
        )

        if sigma > 0:
            xx = np.linspace(distances.min(), distances.max(), 500)
            ax.plot(
                xx,
                norm.pdf(xx, mu, sigma),
                linewidth=2,
                label="Gaussian fit",
            )

        ax.axvline(
            grid_distance,
            linestyle="--",
            linewidth=2,
            label="Bermuda grid distance",
        )
        ax.text(
            0.97, 0.97,
            f"$\\sigma$ = {sigma * 100:.2f} cm\n"
            f"$\\mu$ = {mu:.2f} m",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.9),
        )

        ax.set_title(f"GNSS {i+1} – GNSS {j+1}")
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("density")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)

    # Use the same y-axis range on all six panels
    ylim = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(0, ylim)

    fig.suptitle(
        "Jacksonville GNSS Antenna Separations\n"
        f"{filter_label}: {len(GPS_Coordinates_plot):,} / "
        f"{len(common_datetimes):,} epochs",
        fontsize=16,
    )

    output_file = gps_data_path(
        f"Figs/Jacksonville_GNSS_Comparison_{filter_label}.pdf"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved {output_file}")
    plt.show()
