import numpy as np
from data import gps_data_path
import scipy.io as sio
import matplotlib.pyplot as plt


def load_and_process_data(path, GNSS_start, GNSS_end):
    """Load GNSS unit data within the specified time range."""
    data = sio.loadmat(path)
    days = data["days"].flatten() - 59958
    times = data["times"].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes / 3600 >= GNSS_start) & (datetimes / 3600 <= GNSS_end)

    datetimes = datetimes[condition_GNSS]
    time_GNSS = datetimes
    x, y, z, elev = (
        data["x"].flatten()[condition_GNSS],
        data["y"].flatten()[condition_GNSS],
        data["z"].flatten()[condition_GNSS],
        data["elev"].flatten()[condition_GNSS],
    )
    return time_GNSS, x, y, z, elev


def running_median(data, window=50):
    """Return the running median of ``data`` using the given window."""
    half = window // 2
    n = len(data)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.median(data[start:end])
    return result


def running_abs_dev(data, window=50):
    """Median absolute deviation computed over a moving window."""
    half = window // 2
    n = len(data)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_data = data[start:end]
        med = np.median(window_data)
        result[i] = np.median(np.abs(window_data - med))
    return result


def receiver_comparison(GNSS_start, GNSS_end):
    paths = [
        gps_data_path("GPS_Data/Puerto_Rico/1_PortFwd/combined/0007-combined.mat"),
        gps_data_path("GPS_Data/Puerto_Rico/2_StbdFwd/combined/0007-combined.mat"),
        gps_data_path("GPS_Data/Puerto_Rico/3_StbdAft/combined/0007-combined.mat"),
        gps_data_path("GPS_Data/Puerto_Rico/4_PortAft/combined/0007-combined.mat"),
    ]

    all_data = [load_and_process_data(path, GNSS_start, GNSS_end) for path in paths]
    common_datetimes = set(all_data[0][0])
    for data in all_data[1:]:
        common_datetimes.intersection_update(data[0])
    common_datetimes = sorted(common_datetimes)

    filtered_data = []
    for datetimes, x, y, z, elev in all_data:
        mask = np.isin(datetimes, common_datetimes)
        filtered_data.append(
            [
                np.array(datetimes)[mask],
                np.array(x)[mask],
                np.array(y)[mask],
                np.array(z)[mask],
                np.array(elev)[mask],
            ]
        )
    filtered_data = np.array(filtered_data)

    # Filter data based on elevation
    print("Filtering Data")
    window = 5000
    elev_upper = -52
    elev_lower = -66
    mask = np.array(
        [
            (filtered_data[0, 4, :] < elev_upper)
            & (filtered_data[0, 4, :] > elev_lower)
            & (filtered_data[1, 4, :] < elev_upper)
            & (filtered_data[1, 4, :] > elev_lower)
            & (filtered_data[2, 4, :] < elev_upper)
            & (filtered_data[2, 4, :] > elev_lower)
            & (filtered_data[3, 4, :] < elev_upper)
            & (filtered_data[3, 4, :] > elev_lower)
        ]
    )
    indices = np.where(mask[0])[0]
    filtered_data = filtered_data[:, :, indices]

    mask = np.ones(filtered_data.shape[2], dtype=bool)
    for i in range(4):
        elev = filtered_data[i, 4, :]
        median_elev = running_median(elev, window)
        abs_dev = running_abs_dev(elev, window)
        mask &= (elev >= median_elev - 2 * abs_dev) & (
            elev <= median_elev + 2 * abs_dev
        )
    indices = np.where(mask)[0]
    filtered_data = filtered_data[:, :, indices]

    print("End of Filtering Data")

    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    # Calculate Distances between GPS receivers
    Distances = np.zeros((len(filtered_data[0, 0]), 6))

    for i in range(len(filtered_data[0, 0])):
        coord = GPS_Coordinates[i, :, :]
        Distances[i, 0] = np.linalg.norm(coord[0, :] - coord[1, :])
        Distances[i, 1] = np.linalg.norm(coord[0, :] - coord[2, :])
        Distances[i, 2] = np.linalg.norm(coord[0, :] - coord[3, :])
        Distances[i, 3] = np.linalg.norm(coord[1, :] - coord[2, :])
        Distances[i, 4] = np.linalg.norm(coord[1, :] - coord[3, :])
        Distances[i, 5] = np.linalg.norm(coord[2, :] - coord[3, :])
    # Plotting the distances
    labels = [
        "PortFwd - StbdFwd",
        "PortFwd - StbdAft",
        "PortFwd - PortAft",
        "StbdFwd - StbdAft",
        "StbdFwd - PortAft",
        "StbdAft - PortAft",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i in range(6):
        axes[i].hist(Distances[:, i], bins=50, color="steelblue", edgecolor="black")
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("Distance (m)")
        axes[i].set_ylabel("Count")
        axes[i].grid(alpha=0.3)

    plt.suptitle("Baseline Distance Distributions Between GNSS Receivers", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    GNSS_start = 0  # in hours
    GNSS_end = 1000  # in hours
    receiver_comparison(GNSS_start, GNSS_end)
