import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pymap3d import ecef2geodetic
from Inversion_Workflow.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from Inversion_Workflow.Synthetic.Numba_Functions.Numba_Geiger_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from data import gps_data_path


def bermuda_trajectory(
    time_noise, position_noise, dz_array, angle_array, esv_matrix, DOG_num=3
):
    """Generate synthetic Bermuda trajectory and travel times.

    Parameters
    ----------
    time_noise : float
        Standard deviation of timing noise added to the data.
    position_noise : float
        Standard deviation of position noise in metres.
    dz_array, angle_array, esv_matrix : ndarray
        Effective sound velocity lookup tables.
    DOG_num : int, optional
        DOG data set to load (1–4).

    Returns
    -------
    tuple of numpy.ndarray
        ``(CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates)``.
    """
    CDOG_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    CDOG_augment = np.array([974.12667502, -80.98121315, -805.07870249])
    CDOG = CDOG_base + CDOG_augment
    lever = np.array([-12.48862757, 0.22622633, -15.89601934])
    offset = 1991.01236648

    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    gps1_to_others = data["gps1_to_others"]

    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    synthetic_travel_times, esv = calculateTimesRayTracing_Bias_Real(
        CDOG, transponder_coordinates, 0.0, dz_array, angle_array, esv_matrix
    )

    CDOG_time = (
        GPS_data
        + synthetic_travel_times
        + np.random.normal(0, time_noise, len(GPS_data))
        + offset
    )
    CDOG_remain, CDOG_int = np.modf(CDOG_time)

    CDOG_data = np.stack((CDOG_int, CDOG_remain), axis=0)
    CDOG_data = CDOG_data.T

    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    return CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates


if __name__ == "__main__":
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    time_noise = 0
    position_noise = 0

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        transponder_coordinates,
    ) = bermuda_trajectory(
        time_noise, position_noise, dz_array, angle_array, esv_matrix
    )

    lat = sio.loadmat(gps_data_path("GPS_Data/Unit1-camp_bis.mat"))["lat"].flatten()
    lon = sio.loadmat(gps_data_path("GPS_Data/Unit1-camp_bis.mat"))["lon"].flatten()
    elev = sio.loadmat(gps_data_path("GPS_Data/Unit1-camp_bis.mat"))["elev"].flatten()
    times = sio.loadmat(gps_data_path("GPS_Data/Unit1-camp_bis.mat"))["times"].flatten()
    days = (
        sio.loadmat(gps_data_path("GPS_Data/Unit1-camp_bis.mat"))["days"].flatten()
        - 59015
    )

    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)

    datetimes = datetimes[condition_GNSS]
    lat = lat[condition_GNSS]
    lon = lon[condition_GNSS]
    elev = elev[condition_GNSS]

    points = len(lat)
    colors = plt.cm.viridis(np.linspace(0, 1, points))

    CDOG_lat, CDOG_lon, CDOG_depth = ecef2geodetic(CDOG[0], CDOG[1], CDOG[2])
    CDOG_lon += 360

    # Calculate time values in hours for proper colorbar range
    times_hours = datetimes / 3600  # Convert seconds to hours
    min_time = np.min(times_hours)
    max_time = np.max(times_hours)

    scatter = plt.scatter(
        lon, lat, s=1, c=times_hours, cmap="viridis", label="Surface Vessel"
    )
    plt.scatter(CDOG_lon, CDOG_lat, marker="x", s=5, label="CDOG")
    plt.colorbar(scatter, label="Elapsed Time (hours)")
    plt.clim(min_time, max_time)  # Set the colorbar to actual time range
    plt.title("Plot of Bermuda Trajectory and CDOG location")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()
    print(CDOG_data, GPS_data)
