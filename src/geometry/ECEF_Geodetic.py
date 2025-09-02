"""
Function to convert from ECEF coordinates to Geodetic coordinates
    Uses the WGS84 ellipsoid model
    Adapted from https://gis.stackexchange.com/questions/265909/converting-from-ecef-to-geodetic-coordinates
"""

import numpy as np
from numba import njit
from pymap3d import ecef2geodetic
import scipy.io as sio


@njit(cache=True)
def ECEF_Geodetic(coords):
    """Convert ECEF coordinates to geodetic coordinates.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Array of ``(x, y, z)`` positions in metres.

    Returns
    -------
    tuple of ndarray
        ``(lat, lon, h)`` arrays in degrees/metres for each input point.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # --- WGS84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    # --- derived constants
    b = a - f * a
    e = np.sqrt(np.power(a, 2.0) - np.power(b, 2.0)) / a
    clambda = np.arctan2(y, x)
    p = np.sqrt(np.power(x, 2.0) + np.power(y, 2))
    theta = np.arctan2(z, p * (1.0 - np.power(e, 2.0)))
    cs = np.cos(theta)
    sn = np.sin(theta)
    N = np.power(a, 2.0) / np.sqrt(np.power(a * cs, 2.0) + np.power(b * sn, 2.0))
    h = p / cs - N
    k = 0
    while k < 2:
        theta = np.arctan2(z, p * (1.0 - np.power(e, 2.0) * N / (N + h)))
        cs = np.cos(theta)
        sn = np.sin(theta)
        N = np.power(a, 2.0) / np.sqrt(np.power(a * cs, 2.0) + np.power(b * sn, 2.0))
        h = p / cs - N
        k += 1
    clambda = np.degrees(clambda)
    theta = np.degrees(theta)
    return theta, clambda, h


if __name__ == "__main__":
    from Inversion_Workflow.Synthetic.Numba_Functions.Numba_Geiger import (
        findTransponder,
    )
    from data import gps_data_path
    import timeit

    def load_and_process_data(path):
        """Load unit data and slice to the GNSS window.

        Parameters
        ----------
        path : str
            Path to the ``.mat`` file containing receiver data.

        Returns
        -------
        tuple of ndarray
            ``(time, x, y, z)`` arrays filtered to the GNSS window.
        """
        data = sio.loadmat(path)
        days = data["days"].flatten() - 59015
        times = data["times"].flatten()
        datetimes = (days * 24 * 3600) + times
        condition_GNSS = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)
        # condition_GNSS = (datetimes/3600 >= 35.3) & (datetimes / 3600 <= 37.6)

        datetimes = datetimes[condition_GNSS]
        time_GNSS = datetimes
        x, y, z = (
            data["x"].flatten()[condition_GNSS],
            data["y"].flatten()[condition_GNSS],
            data["z"].flatten()[condition_GNSS],
        )
        # x,y,z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()

        return time_GNSS, x, y, z

    paths = [
        gps_data_path("GPS_Data/Unit1-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit2-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit3-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit4-camp_bis.mat"),
    ]

    all_data = [load_and_process_data(path) for path in paths]
    common_datetimes = set(all_data[0][0])
    for data in all_data[1:]:
        common_datetimes.intersection_update(data[0])
    common_datetimes = sorted(common_datetimes)

    filtered_data = []
    for datetimes, x, y, z in all_data:
        mask = np.isin(datetimes, common_datetimes)
        filtered_data.append(
            [
                np.array(datetimes)[mask],
                np.array(x)[mask],
                np.array(y)[mask],
                np.array(z)[mask],
            ]
        )
    filtered_data = np.array(filtered_data)

    # Initialize Coordinates in form of Geiger's Method
    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )
    initial_lever_guess = np.array([-12.48862757, 0.22622633, -15.89601934])

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, initial_lever_guess
    )

    theta, clambda, h = ECEF_Geodetic(transponder_coordinates)

    lat_arr = np.array(
        [
            ecef2geodetic(
                transponder_coordinates[i, 0],
                transponder_coordinates[i, 1],
                transponder_coordinates[i, 2],
            )[0]
            for i in range(len(transponder_coordinates))
        ]
    )
    lon_arr = np.array(
        [
            ecef2geodetic(
                transponder_coordinates[i, 0],
                transponder_coordinates[i, 1],
                transponder_coordinates[i, 2],
            )[1]
            for i in range(len(transponder_coordinates))
        ]
    )
    depth_arr = np.array(
        [
            ecef2geodetic(
                transponder_coordinates[i, 0],
                transponder_coordinates[i, 1],
                transponder_coordinates[i, 2],
            )[2]
            for i in range(len(transponder_coordinates))
        ]
    )
    print(np.max(np.abs(theta - lat_arr)))
    print(np.max(np.abs(clambda - lon_arr)))
    print(np.abs(h - depth_arr))

    start = timeit.default_timer()
    for _ in range(100):
        theta, clambda, h = ECEF_Geodetic(transponder_coordinates)
    stop = timeit.default_timer()
    print("Time: ", (stop - start) / 100)
