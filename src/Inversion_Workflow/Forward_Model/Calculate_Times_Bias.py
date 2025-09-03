import numpy as np
from numba import njit

from geometry.ECEF_Geodetic import ECEF_Geodetic


@njit
def find_esv(beta, dz, dz_array, angle_array, esv_matrix):
    """Interpolate effective sound velocities from a lookup table.

    Parameters
    ----------
    beta : ndarray
        Takeoff angles in degrees.
    dz : ndarray
        Vertical separation of receiver and source.
    dz_array, angle_array, esv_matrix : ndarray
        Discrete lookup grids defining the ESV table.

    Returns
    -------
    ndarray
        Interpolated effective sound velocities.
    """
    idx_closest_dz = np.empty_like(dz, dtype=np.int64)
    idx_closest_beta = np.empty_like(beta, dtype=np.int64)

    for i in range(len(dz)):
        idx_closest_dz[i] = np.searchsorted(dz_array, dz[i], side="left")
        if idx_closest_dz[i] < 0:
            idx_closest_dz[i] = 0
        elif idx_closest_dz[i] >= len(dz_array):
            idx_closest_dz[i] = len(dz_array) - 1

        idx_closest_beta[i] = np.searchsorted(angle_array, beta[i], side="left")
        if idx_closest_beta[i] < 0:
            idx_closest_beta[i] = 0
        elif idx_closest_beta[i] >= len(angle_array):
            idx_closest_beta[i] = len(angle_array) - 1

    closest_esv = np.empty_like(dz, dtype=np.float64)
    for i in range(len(dz)):
        closest_esv[i] = esv_matrix[idx_closest_dz[i], idx_closest_beta[i]]

    return closest_esv


@njit
def calculateTimesRayTracing_Bias(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Compute travel times with an optional ESV bias term.

    Parameters
    ----------
    guess : array-like, shape (3,)
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of transponder positions.
    esv_bias : float or ndarray
        Bias added to the effective sound velocity.
    dz_array, angle_array, esv_matrix : ndarray
        Lookup table for computing the ESV.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and sound velocities.
    """
    hori_dist = np.sqrt(
        (transponder_coordinates[:, 0] - guess[0]) ** 2
        + (transponder_coordinates[:, 1] - guess[1]) ** 2
    )
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv


@njit
def calculateTimesRayTracing_Bias_Real(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Calculate times for real data using geodetic depths."""
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    depth_arr = ECEF_Geodetic(transponder_coordinates)[2]

    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)
    dz = depth_arr - depth
    beta = np.arcsin(dz / abs_dist) * 180 / np.pi
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv
