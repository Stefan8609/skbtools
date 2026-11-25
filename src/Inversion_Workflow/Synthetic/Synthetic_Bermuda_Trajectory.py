import numpy as np
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from bermuda_data import CDOG_REFERENCE, GPS1_TO_OTHERS, gps_data_path


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
    drift_rate=0.0,
):
    """Synthetic arrivals on the real Bermuda track of one DOG.

    The GNSS grid is the rigid ``gps1_to_others`` (vessel frame, m) fitted to the
    real receivers at each epoch, so only GPS1 keeps its real noise. The DOG clock
    reads ``t + T + time_bias + offset`` plus ``time_noise`` (s) and
    ``drift_rate * (t - t_mid)`` (s/s, ``t_mid`` half-way through the GPS epochs).
    ``position_noise`` (m) is then added to every receiver. Returns ``(CDOG_data, CDOG,
    GPS_Coordinates, GPS_data, transponder_coordinates)``.
    """
    if gps1_to_others is None:
        gps1_to_others = GPS1_TO_OTHERS

    if gps1_to_transponder is None:
        gps1_to_transponder = np.array([-12.4659, 9.6021, -13.2993])

    CDOG_augments = {
        1: np.array([-396.80833750, 370.23261256, 774.14404697]),
        3: np.array([826.18512886, -111.85558415, -733.73615228]),
        4: np.array([235.77476616, -1305.38300727, -2190.89445398]),
    }

    CDOG = CDOG_REFERENCE + CDOG_augments[DOG_num]

    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
    GPS_Coordinates_real = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    # Rigid GNSS array at every epoch: GPS1 keeps its real position, the
    # orientation comes from the measured grid.
    GPS_Coordinates = np.empty_like(GPS_Coordinates_real)

    for i in range(len(GPS_Coordinates_real)):
        real_grid_relative = (GPS_Coordinates_real[i] - GPS_Coordinates_real[i, 0]).T

        rotation_matrix, _ = findRotationAndDisplacement(
            gps1_to_others.T,
            real_grid_relative,
        )

        GPS_Coordinates[i] = (
            GPS_Coordinates_real[i, 0] + (rotation_matrix @ gps1_to_others.T).T
        )

    transponder_coordinates = findTransponder(
        GPS_Coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )

    synthetic_travel_times, esv = calculateTimesRayTracing_Bias_Real(
        CDOG,
        transponder_coordinates,
        esv_bias,
        dz_array,
        angle_array,
        esv_matrix,
    )

    synthetic_travel_times = synthetic_travel_times + time_bias

    CDOG_time = (
        GPS_data
        + synthetic_travel_times
        + np.random.normal(0, time_noise, len(GPS_data))
        + offset
        + drift_rate * (GPS_data - 0.5 * (GPS_data[0] + GPS_data[-1]))
    )

    CDOG_remain, CDOG_int = np.modf(CDOG_time)

    CDOG_data = np.stack((CDOG_int, CDOG_remain), axis=0)
    CDOG_data = CDOG_data.T

    GPS_Coordinates += np.random.normal(
        0,
        position_noise,
        GPS_Coordinates.shape,
    )

    return (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        transponder_coordinates,
    )
