import numpy as np
from numba import njit
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import find_esv
from geometry.ECEF_Geodetic import ECEF_Geodetic


@njit(cache=True, fastmath=True)
def calculateTimesRayTracing_split(
    guess, transponder_coordinates, esv_biases, dz_array, angle_array, esv_matrix
):
    """As ``calculateTimesRayTracing_Bias_Real``, with an ESV bias per block of epochs.

    The epochs are split into ``len(esv_biases)`` consecutive blocks as
    ``np.array_split`` does. Returns ``(times, esv)``.
    """
    N = transponder_coordinates.shape[0]
    B = esv_biases.shape[0]

    # Geodetic height of the DOG
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


@njit(cache=True, fastmath=True)
def calculateTimesRayTracing_windows(
    guess,
    transponder_coordinates,
    window_biases,
    epoch_window,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Ray trace times with an ESV bias per time window.

    As ``calculateTimesRayTracing_Bias_Real``, but epoch i gets the bias
    ``window_biases[epoch_window[i]]`` (m/s); ``epoch_window`` is an (N,) integer
    array of window indices. Returns ``(times, esv)``.
    """
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    depth_arr = ECEF_Geodetic(transponder_coordinates)[2]

    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)
    dz = depth_arr - depth
    beta = np.arcsin(dz / abs_dist) * 180 / np.pi
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix)
    esv = esv + window_biases[epoch_window]
    times = abs_dist / esv
    return times, esv
