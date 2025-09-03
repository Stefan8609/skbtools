import numpy as np
from numba import njit

from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.Forward_Model.Find_ESV import find_esv

"""Need to add support for ESV in parameters

very easy fix"""


@njit
def calculateTimesRayTracingReal(
    guess, transponder_coordinates, dz_array, angle_array, esv_matrix
):
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
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix)
    times = abs_dist / esv
    return times, esv


@njit
def calculateTimesRayTracing(
    guess, transponder_coordinates, dz_array, angle_array, esv_matrix, ray=True
):
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
    if ray:
        esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix)
        times = abs_dist / esv
    if not ray:
        times = abs_dist / 1515.0
        esv = np.full(len(transponder_coordinates), 1515.0)
    return times, esv
