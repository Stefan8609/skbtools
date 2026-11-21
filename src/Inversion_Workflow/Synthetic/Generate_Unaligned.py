import numpy as np
from Inversion_Workflow.Synthetic.Generate_Trajectories import generateRealistic
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)
from numba import njit
from bermuda_data import GPS1_TO_OTHERS

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
    """Box-pattern synthetic arrivals (Eq. B35) with dropped tags (App. B.3).

    The DOG clock reads ``t + T + offset + time_bias`` plus ``time_noise`` (s) and
    ``drift_rate * (t - t_mid)`` (s/s, Eq. 6 convention; ``t_mid`` half-way through
    the GPS epochs, while the inversion's uses the paired epochs). So the offset of
    Eq. 6 is ``offset + time_bias`` and the inversion recovers ``offset_est -
    time_bias_est``. ``position_noise`` (m) is added to every receiver coordinate.
    Returns ``(CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates)``;
    ``CDOG_mat`` is (m, 2) integer and fractional seconds.
    """
    if gps1_to_others is None:
        gps1_to_others = GPS1_TO_OTHERS
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
