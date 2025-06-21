import numpy as np
from Numba_time_bias import find_esv


# @njit
def calculateTimesRayTracing_split(
    guess, transponder_coordinates, esv_biases, dz_array, angle_array, esv_matrix
):
    """Ray trace times with separate ESV biases for consecutive blocks.

    Parameters
    ----------
    guess : array-like of float, shape (3,)
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of transponder positions.
    esv_biases : ndarray
        Sequence of ESV biases applied in order along the track.
    dz_array : ndarray
        Depth grid used for the ESV table.
    angle_array : ndarray
        Angle grid used for the ESV table in degrees.
    esv_matrix : ndarray
        Effective sound velocity table.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and effective sound speeds.
    """
    times = np.zeros(len(transponder_coordinates))
    esv = np.zeros(len(transponder_coordinates))
    left, right = 0, 0
    for n in range(len(esv_biases)):
        # Split the transponder coordinates
        # into chunks based on the number of ESV biases
        left = right
        right = left + len(transponder_coordinates) // len(esv_biases)
        if n == len(esv_biases) - 1:
            right = len(transponder_coordinates)
        curr_transponder = transponder_coordinates[left:right]
        hori_dist = np.sqrt(
            (curr_transponder[:, 0] - guess[0]) ** 2
            + (curr_transponder[:, 1] - guess[1]) ** 2
        )
        abs_dist = np.sqrt(np.sum((curr_transponder - guess) ** 2, axis=1))
        beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
        dz = np.abs(guess[2] - curr_transponder[:, 2])
        esv[left:right] = (
            find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_biases[n]
        )
        times[left:right] = abs_dist / esv[left:right]
    return times, esv
