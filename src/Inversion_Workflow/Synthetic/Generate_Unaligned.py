import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Inversion_Workflow.Synthetic.Generate_Trajectories import (
    generateRealistic,
    generateRandomData,
    generateLine,
    generateCross,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)
from data import gps_data_path
from numba import njit


@njit
def generateUnaligned(
    n,
    time_noise,
    position_noise,
    offset,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    gps1_to_others=None,
    gps1_to_transponder=None,
    trajectory="realistic",
):
    """Generate noisy travel times and corrupted CDOG tags for a realistic trajectory.

    Parameters
    ----------
    n : int
        Number of points along the synthetic path.
    time_noise : float
        Standard deviation of arrival time noise.
    offset : float
        Base offset applied to DOG times.
    esv_bias, time_bias : float
        Bias values applied to ESV and times.
    dz_array, angle_array, esv_matrix : ndarray
        Lookup table for the ESV.
    gps1_to_others, gps1_to_transponder : ndarray, optional
        Geometry offsets. Defaults are used if not provided.

    Returns
    -------
    tuple
        (CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates)
        where CDOG_mat has shape (m, 2) with integer and fractional clock parts,
        after removal of several random contiguous segments.
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

    if trajectory == "random":
        print("Generating random trajectory")
        (
            CDOG,
            GPS_Coordinates,
            transponder_coordinates,
            gps1_to_others,
            gps1_to_transponder,
        ) = generateRandomData(n, gps1_to_others, gps1_to_transponder)
    elif trajectory == "line":
        print("Generating line trajectory")
        (
            CDOG,
            GPS_Coordinates,
            transponder_coordinates,
            gps1_to_others,
            gps1_to_transponder,
        ) = generateLine(n, gps1_to_others, gps1_to_transponder)
    elif trajectory == "cross":
        print("Generating cross trajectory")
        (
            CDOG,
            GPS_Coordinates,
            transponder_coordinates,
            gps1_to_others,
            gps1_to_transponder,
        ) = generateCross(n, gps1_to_others, gps1_to_transponder)
    else:
        print("Generating realistic trajectory")
        (
            CDOG,
            GPS_Coordinates,
            transponder_coordinates,
            gps1_to_others,
            gps1_to_transponder,
        ) = generateRealistic(n, gps1_to_others, gps1_to_transponder)

    # GPS time index
    GPS_time = np.arange(len(GPS_Coordinates))

    # True travel times (with ESV and time biases)
    true_travel_times, _true_esv = calculateTimesRayTracing_Bias(
        CDOG,
        transponder_coordinates,
        esv_bias,
        dz_array,
        angle_array,
        esv_matrix,
    )
    true_travel_times = true_travel_times + time_bias

    # Construct CDOG time and split into integer/fractional parts
    CDOG_time = (
        GPS_time
        + true_travel_times
        + np.random.normal(0.0, time_noise, len(GPS_time))
        + offset
    )

    CDOG_int = np.floor(CDOG_time)
    CDOG_remain = CDOG_time - CDOG_int

    CDOG_mat = np.stack((CDOG_int, CDOG_remain), axis=1)

    # Remove random contiguous segments from CDOG_mat
    # (indices are recomputed each iteration to respect the shrinking array)
    num_segments = 5
    for _ in range(num_segments):
        if len(CDOG_mat) == 0:
            break
        # Use conservative bounds when the array becomes short
        max_len = min(500, max(1, len(CDOG_mat)))
        min_len = min(200, max(1, len(CDOG_mat)))
        if max_len < min_len:
            seg_len = max_len
        else:
            seg_len = np.random.randint(min_len, max_len + 1)
        if len(CDOG_mat) - seg_len <= 0:
            start_index = 0
        else:
            start_index = np.random.randint(0, len(CDOG_mat) - seg_len + 1)
        mask = np.ones(len(CDOG_mat), dtype=np.bool_)
        mask[start_index : start_index + seg_len] = False
        CDOG_mat = CDOG_mat[mask]
        GPS_Coordinates += np.random.normal(
            0, position_noise, (len(GPS_Coordinates), 4, 3)
        )

    return CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates


if __name__ == "__main__":
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    esv_bias = 0.0
    time_bias = 0.0
    offset = 1200
    time_noise = 2 * 10**-5
    position_noise = 2 * 10**-2

    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates = (
        generateUnaligned(
            20000,
            time_noise,
            position_noise,
            offset,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            trajectory="realistic",
        )
    )

    CDOG_int = CDOG_mat[:, 0]
    CDOG_frac = np.unwrap(CDOG_mat[:, 1] * 2 * np.pi) / (2 * np.pi)
    print(CDOG_frac)

    plt.figure(figsize=(8, 5))
    plt.plot(CDOG_int, CDOG_frac, ".", markersize=2)
    plt.xlabel("Integer part of CDOG time")
    plt.ylabel("Unwrapped fractional part of CDOG time")
    plt.title("CDOG_mat Time Components")
    plt.grid(True)
    plt.show()
