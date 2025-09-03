import random
import numpy as np
import scipy.io as sio
from numba import njit

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from data import gps_data_path
import timeit


@njit
def computeJacobianRayTracing(guess, transponder_coordinates, times, sound_speed):
    """Jacobian of travel times with respect to source position.

    Parameters
    ----------
    guess : ndarray
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of receiver positions.
    times : ndarray
        Travel times for the current guess.
    sound_speed : ndarray
        Sound speed used for each receiver.

    Returns
    -------
    ndarray
        Jacobian matrix of shape ``(N, 3)``.
    """
    # xyz coordinates and functions are the travel times
    diffs = transponder_coordinates - guess
    jacobian = -diffs / (times[:, np.newaxis] * (sound_speed[:, np.newaxis] ** 2))
    return jacobian


@njit
def geigersMethod(
    guess,
    CDog,
    transponder_coordinates_Actual,
    transponder_coordinates_Found,
    time_noise=0,
):
    """Iteratively refine a CDOG position using Geiger's algorithm.

    Parameters
    ----------
    guess : ndarray
        Initial estimate of the CDOG location.
    CDog : ndarray
        True CDOG location used to generate synthetic times.
    transponder_coordinates_Actual : ndarray
        Actual receiver coordinates for the data generation.
    transponder_coordinates_Found : ndarray
        Receiver coordinates used in the inversion.
    time_noise : float, optional
        Standard deviation of noise added to the travel times.

    Returns
    -------
    tuple of ndarray
        ``(estimate, times_known)`` final position and noisy arrival times.
    """
    from Inversion_Workflow.Forward_Model.Calculate_Times import (
        calculateTimesRayTracing,
    )

    # Define threshold
    epsilon = 10**-5
    times_known, esv = calculateTimesRayTracing(CDog, transponder_coordinates_Actual)

    # Apply noise to known times
    times_known += np.random.normal(0, time_noise, len(transponder_coordinates_Actual))

    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    # Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k < 100:
        times_guess, esv = calculateTimesRayTracing(
            guess, transponder_coordinates_Found
        )
        jacobian = computeJacobianRayTracing(
            guess, transponder_coordinates_Found, times_guess, esv
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (times_guess - times_known)
        )
        guess = guess + delta
        k += 1
    return guess, times_known


if __name__ == "__main__":
    from Inversion_Workflow.Synthetic.Generate_Trajectories import generateRealistic

    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    (
        CDog,
        GPS_Coordinates,
        transponder_coordinates_Actual,
        gps1_to_others,
        gps1_to_transponder,
    ) = generateRealistic(10000)

    time_noise = 2 * 10**-5
    position_noise = 2 * 10**-2

    # Apply noise to position
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    transponder_coordinates_Found = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )

    guess = np.array(
        [
            random.uniform(-5000, 5000),
            random.uniform(-5000, 5000),
            random.uniform(-5225, -5235),
        ]
    )

    start = timeit.default_timer()
    for _ in range(100):
        guess, times_known = geigersMethod(
            guess,
            CDog,
            transponder_coordinates_Actual,
            transponder_coordinates_Found,
            time_noise,
        )
    stop = timeit.default_timer()
    print("Time: ", (stop - start) / 100)
    print(times_known)
    print(guess - CDog)
