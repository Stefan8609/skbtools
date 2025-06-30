import numpy as np
from numba import njit
from GeigerMethod.Synthetic.Numba_Functions.Numba_time_bias import find_esv
from GeigerMethod.Synthetic.Numba_Functions.ECEF_Geodetic import ECEF_Geodetic


@njit
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
    N = transponder_coordinates.shape[0]
    B = esv_biases.shape[0]

    # Get depth of receiver guess
    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)

    # Output arrays
    times = np.zeros(N)
    esv = np.zeros(N)

    # Determine block sizes (equivalent to np.array_split)
    base = N // B
    remainder = N % B
    start = 0

    for n in range(B):
        size = base + 1 if n < remainder else base
        end = start + size

        blk = transponder_coordinates[start:end]
        depth_arr = ECEF_Geodetic(blk)[2]

        dz = depth_arr - depth
        abs_dist = np.sqrt(np.sum((blk - guess) ** 2, axis=1))
        beta = np.arcsin(dz / abs_dist) * 180 / np.pi

        block_esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix)
        esv[start:end] = block_esv + esv_biases[n]
        times[start:end] = abs_dist / esv[start:end]

        start = end

    return times, esv
