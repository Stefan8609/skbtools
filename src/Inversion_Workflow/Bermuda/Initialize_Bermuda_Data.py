import numpy as np
import scipy.io as sio
from bermuda_data import gps_data_path


def load_and_process_data(path, GNSS_start, GNSS_end):
    """Load GNSS unit data within the specified time range."""
    data = sio.loadmat(path)
    days = data["days"].flatten() - 59015
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


# Filtering Functions
def running_median(data, window=50):
    """Running median (Eq. A4) of ``data`` over ``window`` samples."""
    half = window // 2
    n = len(data)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.median(data[start:end])
    return result


def running_abs_dev(data, window=50):
    """Running median absolute deviation (Eq. A5) over ``window`` samples."""
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


def initialize_bermuda(GNSS_start, GNSS_end, DOG_num=3, save=False):
    """GPS_Coordinates (N, 4, 3), GPS_data (N,) and CDOG_data (M, 2) of one DOG.

    Keeps the GNSS epochs between ``GNSS_start`` and ``GNSS_end`` (hours since MJD
    59015) that all four receivers share and whose ellipsoidal heights all lie
    between -38 and -35 m. GPS times are shifted by 70000 s and the DOG fractional
    seconds converted from ns to s. ``save`` writes
    ``GPS_Data/Processed_GPS_Receivers_DOG_<n>.npz``.
    """
    print("Initializing Bermuda Data")

    paths = [
        gps_data_path("GPS_Data/PRIDE_final/Antenna1-camp.mat"),
        gps_data_path("GPS_Data/PRIDE_final/Antenna2-camp.mat"),
        gps_data_path("GPS_Data/PRIDE_final/Antenna3-camp.mat"),
        gps_data_path("GPS_Data/PRIDE_final/Antenna4-camp.mat"),
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

    # Hard cut on the ellipsoidal height (m) of all four receivers
    print("Filtering Data")
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

    # No running-MAD filter (Eq. A6 of the submitted paper, 2 MAD over 5000 epochs).
    # It removed antenna 4's height jumps in the submitted GNSS solutions; with the
    # PRIDE_final solutions every baseline scatters by 1-2 cm, and dropping the
    # filter (904 more epochs than a 4-MAD filter) moves the Gauss-Newton positions
    # of DOGs 1/3/4 by at most 1.6 cm.

    print("End of Filtering Data")
    # GPS_Coordinates (N, 4, 3)
    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    GPS_data = filtered_data[0, 0, :]
    CDOG_data = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{DOG_num}-camp.mat"))[
        "tags"
    ].astype(float)

    # Shift the GPS clock and convert the DOG fractional seconds from ns to s
    clock_adjustment = {1: 70000.0, 2: 70000.0, 3: 70000.0, 4: 70000.0}

    GPS_data = GPS_data - clock_adjustment[DOG_num]

    CDOG_data[:, 1] = CDOG_data[:, 1] / 1e9

    if save:
        np.savez(
            gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}"),
            GPS_Coordinates=GPS_Coordinates,
            GPS_data=GPS_data,
            CDOG_data=CDOG_data,
        )

    print("Bermuda Data Initialized")
    return GPS_Coordinates, GPS_data, CDOG_data


if __name__ == "__main__":
    GNSS_start = 25
    GNSS_end = 42  # 17 h of survey: the hourglass and the third diagonal leg
    for DOG_num in (1, 3, 4):
        initialize_bermuda(GNSS_start, GNSS_end, DOG_num=DOG_num, save=True)
