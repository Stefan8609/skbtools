import numpy as np
import scipy.io as sio
from pymap3d import geodetic2ecef

"""Enable this for paper plots"""
# plt.rcParams.update({
#     "text.usetex":      True,
#     "font.family":      "serif",
#     "font.serif":       ["Computer Modern"],
#     "font.size": 14,
#     "mathtext.fontset": "cm",
#     "text.latex.preamble":
#         r"\usepackage[utf8]{inputenc}" "\n"
#         r"\usepackage{textcomp}",
# })


def initialize_bermuda(GNSS_start, GNSS_end, CDOG_augment, DOG_num=3, save=False):
    """Load DOG and GPS data for the Bermuda experiment.

    Parameters
    ----------
    GNSS_start, GNSS_end : float
        Start and end times (hours) for slicing the GNSS data.
    CDOG_augment : ndarray
        Offset applied to the base CDOG position.
    DOG_num : int, optional
        DOG data set number to load.
    save : bool, optional
        If ``True`` save processed arrays as ``.npz`` files.

    Returns
    -------
    tuple
        ``(GPS_Coordinates, GPS_data, CDOG_data, CDOG_guess, gps1_to_others)``.
    """
    print("Initializing Bermuda Data")

    # Load GNSS Data during the time of expedition (25 through 40.9) hours
    def load_and_process_data(path, GNSS_start, GNSS_end):
        """Load GNSS unit data within the specified time range."""
        data = sio.loadmat(path)
        days = data["days"].flatten() - 59015
        times = data["times"].flatten()
        datetimes = (days * 24 * 3600) + times
        condition_GNSS = (datetimes / 3600 >= GNSS_start) & (
            datetimes / 3600 <= GNSS_end
        )

        datetimes = datetimes[condition_GNSS]
        time_GNSS = datetimes
        x, y, z, elev = (
            data["x"].flatten()[condition_GNSS],
            data["y"].flatten()[condition_GNSS],
            data["z"].flatten()[condition_GNSS],
            data["elev"].flatten()[condition_GNSS],
        )
        return time_GNSS, x, y, z, elev

    from data import gps_data_path

    paths = [
        gps_data_path("Unit1-camp_bis.mat"),
        gps_data_path("Unit2-camp_bis.mat"),
        gps_data_path("Unit3-camp_bis.mat"),
        gps_data_path("Unit4-camp_bis.mat"),
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

    # Filtering Functions
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

    # Filter data based on elevation
    print("Filtering Data")
    window = 5000
    elev_upper = -35
    elev_lower = -38
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
    # Initialize Coordinates in form of Geiger's Method
    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    # Initialize time-tagged data for GPS and CDOG
    GPS_data = filtered_data[0, 0, :]
    CDOG_data = sio.loadmat(gps_data_path(f"DOG{DOG_num}-camp.mat"))["tags"].astype(
        float
    )

    lat = sio.loadmat(gps_data_path("Unit1-camp_bis.mat"))["lat"].flatten()
    lon = sio.loadmat(gps_data_path("Unit1-camp_bis.mat"))["lon"].flatten()

    """If elevation plotting is desired"""
    # alpha = {0: "A", 1: "B", 2: "C", 3: "D"}
    # if save:
    #     fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    #     mask = np.ones(filtered_data.shape[2], dtype=bool)
    #     for i in range(4):
    #         row, col = divmod(i, 2)
    #         elevation = filtered_data[i, 4, :]
    #         axs[row, col].scatter(
    #             GPS_data, elevation, s=1, color="blue", label=r"Elevation Data"
    #         )
    #         median_elev = running_median(elevation, window=window)
    #         abs_dev = running_abs_dev(elevation, window=window)
    #         mask &= (elevation >= median_elev - 2 * abs_dev) & (
    #             elevation <= median_elev + 2 * abs_dev
    #         )
    #
    #         upper_band = median_elev + 2 * abs_dev
    #         lower_band = median_elev - 2 * abs_dev
    #         axs[row, col].plot(
    #             GPS_data, median_elev, color="red",
    #             linewidth=2, label=r"Running Median"
    #         )
    #         axs[row, col].plot(
    #             GPS_data, upper_band, color="orange", label=r"2 Absolute Deviations"
    #         )
    #         axs[row, col].plot(GPS_data, lower_band, color="orange")
    #
    #         axs[row, col].set_title(f"GPS Unit {i + 1} Elevation")
    #         axs[row, col].set_xlabel(r"Time (s)")
    #         axs[row, col].set_ylabel(r"Elevation (m)")
    #         axs[row, col].set_ylim(-39, -34)
    #         axs[row, col].text(
    #             0.02,
    #             0.93,
    #             f"{alpha[i]}",
    #             transform=axs[row, col].transAxes,
    #         )
    #         if i == 0:
    #             axs[row, col].legend(loc="lower right")
    #     indices = np.where(mask)[0]
    #     axs[0, 1].text(
    #         0.95,
    #         0.05,
    #         f"Removed: {len(elevation) - len(indices)} / {len(elevation)} points \n"
    #         f"Window: {window} samples \n"
    #         f"Overlap: {window - 1} samples",
    #         transform=axs[0, 1].transAxes,
    #         ha="right",
    #         va="bottom",
    #         fontsize=14,
    #     )
    #     plt.tight_layout()
    #     plt.show()
    """end plotting"""

    CDOG_guess_geodetic = np.array(
        [np.mean(lat), np.mean(lon), np.mean(elev)]
    ) + np.array([0, 0, -5200])
    CDOG_guess_base = np.array(
        geodetic2ecef(
            CDOG_guess_geodetic[0], CDOG_guess_geodetic[1], CDOG_guess_geodetic[2]
        )
    )
    CDOG_guess = CDOG_guess_base + CDOG_augment

    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )

    # Scale GPS Clock slightly and scale CDOG clock to nanoseconds
    clock_adjustment = {1: 70000.0, 2: 70000.0, 3: 70000.0, 4: 70000.0}

    GPS_data = GPS_data - clock_adjustment[DOG_num]

    CDOG_data[:, 1] = CDOG_data[:, 1] / 1e9

    # Save the data if required
    if save:
        np.savez(
            gps_data_path(f"Processed_GPS_Receivers_DOG_{DOG_num}"),
            GPS_Coordinates=GPS_Coordinates,
            GPS_data=GPS_data,
            CDOG_data=CDOG_data,
            CDOG_guess=CDOG_guess,
            gps1_to_others=gps1_to_others,
        )

    print("Bermuda Data Initialized")
    return GPS_Coordinates, GPS_data, CDOG_data, CDOG_guess, gps1_to_others


if __name__ == "__main__":
    GNSS_start = 25
    # GNSS_end = 40.9
    GNSS_end = 39
    CDOG_augment = np.array([0, 0, 0])
    (
        GPS_Coordinates,
        GPS_data,
        CDOG_data,
        CDOG_guess,
        gps1_to_others,
    ) = initialize_bermuda(GNSS_start, GNSS_end, CDOG_augment, DOG_num=1, save=True)
