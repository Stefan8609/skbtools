import numpy as np
from numba import njit

from Inversion_Workflow.Inversion.Numba_Geiger import dz_array, angle_array, esv_matrix
from geometry.ECEF_Geodetic import ECEF_Geodetic


@njit
def find_esv(beta, dz):
    """Look up effective sound velocities for given angles and depths.

    Parameters
    ----------
    beta : ndarray
        Ray takeoff angles in degrees.
    dz : ndarray
        Vertical distance between the receiver and the source.

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


@njit
def calculateTimesRayTracingReal(guess, transponder_coordinates):
    """Compute travel times using a real ESV table.

    Parameters
    ----------
    guess : ndarray
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of receiver positions.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and sound speeds.
    """
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    depth_arr = ECEF_Geodetic(transponder_coordinates)[2]

    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)
    dz = depth_arr - depth
    beta = np.arcsin(dz / abs_dist) * 180 / np.pi
    esv = find_esv(beta, dz)
    times = abs_dist / esv
    return times, esv


@njit
def calculateTimesRayTracing(guess, transponder_coordinates, ray=True):
    """Compute travel times using a synthetic or constant sound speed.

    Parameters
    ----------
    guess : ndarray
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of receiver positions.
    ray : bool, optional
        If ``False`` use a fixed 1515 m/s sound speed.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and sound speeds.
    """
    hori_dist = np.sqrt(
        (transponder_coordinates[:, 0] - guess[0]) ** 2
        + (transponder_coordinates[:, 1] - guess[1]) ** 2
    )
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz)
    times = abs_dist / esv
    if not ray:
        times = abs_dist / 1515.0
        esv = np.full(len(transponder_coordinates), 1515.0)
    return times, esv
