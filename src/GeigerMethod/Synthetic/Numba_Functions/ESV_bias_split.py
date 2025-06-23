import numpy as np
from GeigerMethod.Synthetic.Numba_Functions.Numba_time_bias import find_esv
from GeigerMethod.Synthetic.Numba_Functions.ECEF_Geodetic import ECEF_Geodetic


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
    N = len(transponder_coordinates)
    B = len(esv_biases)

    # Get depth of receiver guess
    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)

    # 1) Split coords into B blocks (some blocks may be 1 sample larger)
    blocks = np.array_split(transponder_coordinates, B, axis=0)
    # 2) Prepare output arrays
    times = np.zeros(N)
    esv = np.zeros(N)

    # 3) Compute cumulative indices so we know where each block lives in the full vector
    lengths = [blk.shape[0] for blk in blocks]
    cumidx = np.concatenate(([0], np.cumsum(lengths)))

    # 4) Loop each block/apply its bias
    for n in range(B):
        blk = blocks[n]
        left, right = cumidx[n], cumidx[n + 1]

        depth_arr = ECEF_Geodetic(blk)[2]

        # vertical & absolute distances on this block
        dz = depth_arr - depth
        abs_dist = np.sqrt(np.sum((blk - guess) ** 2, axis=1))
        beta = np.arcsin(dz / abs_dist) * 180 / np.pi

        # lookup & bias
        block_esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix)
        esv[left:right] = block_esv + esv_biases[n]
        times[left:right] = abs_dist / esv[left:right]
    return times, esv
