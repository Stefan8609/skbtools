import numpy as np
from numba import njit


@njit
def find_esv(beta, dz, dz_array, angle_array, esv_matrix):
    """Look up effective sound velocities for given angles and depths.

    Parameters
    ----------
    beta : ndarray
        Ray takeoff angles in degrees.
    dz : ndarray
        Vertical distance between the receiver and the source.
    dz_array, angle_array, esv_matrix : ndarray
        Discrete lookup grids defining the ESV table.

    Returns
    -------
    ndarray
        Effective sound velocity for each input pair.
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
