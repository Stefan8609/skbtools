import random
import numpy as np
import scipy.io as sio
from numba import njit
from Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from ECEF_Geodetic import ECEF_Geodetic


@njit(cache=True)
def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    # Given initial information relative GPS locations and transponder and GPS Coords at each timestep
    xs, ys, zs = gps1_to_others.T
    initial_transponder = gps1_to_transponder
    n = len(GPS_Coordinates)
    transponder_coordinates = np.zeros((n, 3))
    for i in range(n):
        new_xs, new_ys, new_zs = GPS_Coordinates[i].T
        xyzs_init = np.vstack((xs, ys, zs))
        xyzs_final = np.vstack((new_xs, new_ys, new_zs))

        R_mtrx, d = findRotationAndDisplacement(xyzs_init, xyzs_final)
        transponder_coordinates[i] = R_mtrx @ initial_transponder + d
    return transponder_coordinates


@njit
def find_esv(beta, dz):
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
    hori_dist = np.sqrt(
        (transponder_coordinates[:, 0] - guess[0]) ** 2
        + (transponder_coordinates[:, 1] - guess[1]) ** 2
    )
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz)
    times = abs_dist / esv
    if ray == False:
        times = abs_dist / 1515.0
        esv = np.full(len(transponder_coordinates), 1515.0)
    return times, esv


@njit
def computeJacobianRayTracing(guess, transponder_coordinates, times, sound_speed):
    # Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
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
    # Use Geiger's method to find the guess of CDOG location which minimizes sum of travel times squared
    # Define threshold
    epsilon = 10**-5
    times_known, esv = calculateTimesRayTracing(CDog, transponder_coordinates_Actual)

    # Apply noise to known times
    times_known += np.random.normal(0, time_noise, len(transponder_coordinates_Actual))

    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    # Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k < 100:
        # times_guess = calculateTimes(guess, transponder_coordinates_Found, sound_speed)
        times_guess, esv = calculateTimesRayTracing(
            guess, transponder_coordinates_Found
        )
        # jacobian = computeJacobian(guess, transponder_coordinates_Found, times_guess, sound_speed)
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


@njit
def generateRealistic(n):
    # Initialize CDog and GPS locations
    CDog = np.array(
        [
            random.uniform(-5000, 5000),
            random.uniform(-5000, 5000),
            random.uniform(-5225, -5235),
        ]
    )
    x_coords1 = np.sort((np.random.rand(n // 4) * 15000) - 7500)
    x_coords2 = -1 * np.sort(-1 * ((np.random.rand(n // 4) * 15000) - 7500))
    x_coords3 = np.sort((np.random.rand(n // 4) * 15000) - 7500)
    x_coords4 = -1 * np.sort(-1 * ((np.random.rand(n // 4) * 15000) - 7500))
    y_coords1 = x_coords1 + (np.random.rand(n // 4) * 50) - 25
    y_coords2 = 7500 + (np.random.rand(n // 4) * 50) - 25
    y_coords3 = -x_coords1 + (np.random.rand(n // 4) * 50) - 25
    y_coords4 = -7500 + (np.random.rand(n // 4) * 50) - 25
    x_coords = np.concatenate((x_coords1, x_coords2, x_coords3, x_coords4))
    y_coords = np.concatenate((y_coords1, y_coords2, y_coords3, y_coords4))
    z_coords = (np.random.rand(n // 4 * 4) * 5) - 10
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))

    GPS_Coordinates = np.zeros((n // 4 * 4, 4, 3))
    transponder_coordinates = np.zeros((n // 4 * 4, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    # Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n // 4 * 4):
        # Build rotation matrix at each time step
        xRot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rot[i, 0]), -np.sin(rot[i, 0])],
                [0.0, np.sin(rot[i, 0]), np.cos(rot[i, 0])],
            ]
        )
        yRot = np.array(
            [
                [np.cos(rot[i, 1]), 0.0, np.sin(rot[i, 1])],
                [0.0, 1.0, 0.0],
                [-np.sin(rot[i, 1]), 0.0, np.cos(rot[i, 1])],
            ]
        )
        zRot = np.array(
            [
                [np.cos(rot[i, 2]), -np.sin(rot[i, 2]), 0.0],
                [np.sin(rot[i, 2]), np.cos(rot[i, 2]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        totalRot = xRot @ yRot @ zRot
        for j in range(
            1, 4
        ):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i, 0] + totalRot @ gps1_to_others[j]
        # Initialize transponder location
        transponder_coordinates[i] = (
            GPS_Coordinates[i, 0] + totalRot @ gps1_to_transponder
        )
    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


if __name__ == "__main__":
    esv_table = sio.loadmat("../../../GPSData/global_table_esv.mat")
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

    import timeit

    start = timeit.default_timer()
    for i in range(100):
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
