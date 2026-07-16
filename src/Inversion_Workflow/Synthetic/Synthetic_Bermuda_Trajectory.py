import numpy as np
import scipy.io as sio
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from data import gps_data_path


def bermuda_trajectory(
    time_noise,
    position_noise,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    offset=1991.01236648,
    gps1_to_others=None,
    gps1_to_transponder=None,
    DOG_num=3,
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
    if gps1_to_others is None:
        gps1_to_others = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.39341409, -4.22350344, 0.02941493],
                [-12.09568416, -0.94568462, 0.0043972],
                [-8.68674054, 5.16918806, 0.02499322],
            ]
        )
    if gps1_to_transponder is None:
        gps1_to_transponder = np.array([-12.4659, 9.6021, -13.2993])

    CDOG_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    CDOG_augment = np.array([236.428385, -1307.98390221, -2189.21991698])
    CDOG = CDOG_base + CDOG_augment

    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )
    synthetic_travel_times, esv = calculateTimesRayTracing_Bias_Real(
        CDOG, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
    )

    synthetic_travel_times = synthetic_travel_times + time_bias

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
    from plotting.Plot_Modular import trajectory_plot
    from geometry.ECEF_Geodetic import ECEF_Geodetic

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
        time_noise, position_noise, 0, 0, dz_array, angle_array, esv_matrix
    )

    leg1 = (GPS_data / 3600 >= 9.0) & (GPS_data / 3600 <= 11)
    leg2 = (GPS_data / 3600 >= 12.4) & (GPS_data / 3600 <= 15)
    leg_mask = leg1 | leg2
    GPS_Coordinates = GPS_Coordinates[leg_mask]
    GPS_data = GPS_data[leg_mask]

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
    GPS_lat, GPS_lon, GPS_height = ECEF_Geodetic(GPS_Coordinates[:, 0, :])

    trajectory_plot(
        np.array([GPS_lon, GPS_lat, GPS_height]).T,
        GPS_data,
        np.array([CDOGs_lon, CDOGs_lat, CDOGs_height]).T,
    )
