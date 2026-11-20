import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Inversion_Workflow.Synthetic.Generate_Trajectories import generateRealistic
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)
from bermuda_data import gps_data_path
from numba import njit

# Fraction of DOG tags dropped in total (App. B.3: "10-20%").
MIN_DROP_FRACTION = 0.10
MAX_DROP_FRACTION = 0.20


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
    drift_rate=0.0,
):
    """Generate noisy travel times and corrupted CDOG tags for a realistic trajectory.

    Parameters
    ----------
    n : int
        Number of points along the synthetic path.
    time_noise : float
        Standard deviation of arrival time noise, s.
    position_noise : float
        Standard deviation of the noise added to each GNSS receiver coordinate, m.
    offset : float
        Base offset applied to DOG times.
    esv_bias, time_bias : float
        Bias values applied to ESV and times. The DOG clock reads
        ``t + T + offset + time_bias``, so the offset of Eq. 6 is
        ``offset + time_bias`` and the inversion recovers
        ``offset_est - time_bias_est = offset + time_bias``.
    dz_array, angle_array, esv_matrix : ndarray
        Lookup table for the ESV.
    gps1_to_others, gps1_to_transponder : ndarray, optional
        Geometry offsets. Defaults are used if not provided.
    drift_rate : float, optional
        Clock drift, s/s, Eq. 6 convention: the DOG clock also gains
        ``drift_rate * (t - t_mid)``, with ``t_mid`` half-way through the GPS
        epochs (the inversion's ``t_mid`` uses the paired epochs instead).

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
        + drift_rate * (GPS_time - 0.5 * (GPS_time[0] + GPS_time[-1]))
    )

    CDOG_int = np.floor(CDOG_time)
    CDOG_remain = CDOG_time - CDOG_int

    CDOG_mat = np.stack((CDOG_int, CDOG_remain), axis=1)

    # Dropped arrivals (App. B.3): 10-20 % of the tags in total, as isolated random
    # tags plus contiguous blocks of 200-500 tags (blocks may overlap).
    drop_fraction = np.random.uniform(MIN_DROP_FRACTION, MAX_DROP_FRACTION)
    keep = np.random.random(len(CDOG_mat)) >= drop_fraction / 2
    while np.sum(~keep) < drop_fraction * len(CDOG_mat):
        block_len = min(np.random.randint(200, 501), len(CDOG_mat))
        start_index = np.random.randint(0, len(CDOG_mat) - block_len + 1)
        keep[start_index : start_index + block_len] = False
    CDOG_mat = CDOG_mat[keep]

    # Independent noise on every receiver, applied once.
    GPS_Coordinates += np.random.normal(0, position_noise, GPS_Coordinates.shape)

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
