import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from Numba_Geiger import find_esv, calculateTimesRayTracing, generateRealistic, findTransponder


esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

"""
Need an short algorithm to estimate the derivative of the ESV in x,y, and z
    According to the algorithm listed by bud (short and simple numerical method)
    Remove timing bias term from bud's algorithm (or investigate how it works as sub-integer offset)
    Implement Bud's algorithm for finding ESV bias
    
This algorithm finds a fixed bias (that is a constant in time and depth)
"""

@njit
def calculateTimesRayTracing_Bias(guess, transponder_coordinates, esv_bias, ray=True):
    """
    Ray Tracing calculation of times using ESV
        Capable of handling an ESV bias input term
        (whether a constant or an array with same length as transponder_coordinates)
    """
    hori_dist = np.sqrt((transponder_coordinates[:, 0] - guess[0])**2 + (transponder_coordinates[:, 1] - guess[1])**2)
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess)**2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz) + esv_bias
    times = abs_dist / esv
    if ray == False:
        times = abs_dist / 1515.0
        esv = np.full(len(transponder_coordinates), 1515.0)
    return times, esv

@njit(cache=True)
def compute_H_biased(transponder_coordinates, diffs, dist, esv, esv_bias):
    """Compute the Jacobian of the system with respect to the ESV bias term
    According to Bud's algorithm for Gauss-Newton Inversion"""

    H = np.zeros((len(transponder_coordinates), 5))

    #Scale factors
    scale = np.array([1e-3, 1e-3, 1e-3, 1e3, 1.0])  # position(m), time(s), esv(m/s)

    H[:, 0] = (diffs[:, 0] / dist) / scale[0]
    H[:, 1] = (diffs[:, 1] / dist) / scale[1]
    H[:, 2] = (diffs[:, 2] / dist) / scale[2]
    H[:, 3] = (esv_bias + esv[:]) / scale[3]
    H[:, 4] = (dist / (esv_bias + esv[:])) / scale[4]

    return H, scale

# @njit
def numba_bias_geiger(guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, esv_bias_input, time_noise=0):
    """Geiger method with estimation of ESV bias term"""
    epsilon = 10**-5
    lambda_damping = 1.0  # Levenberg-Marquardt damping factor

    #Calculate and apply noise for known times
    times_known, esv = calculateTimesRayTracing_Bias(CDog, transponder_coordinates_Actual, esv_bias_input)
    times_known+=np.random.normal(0, time_noise, len(transponder_coordinates_Actual))

    time_bias = 0.0
    esv_bias = 0.0

    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    k = 0
    delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    prev_residual = np.inf

    while np.linalg.norm(delta) > epsilon and k < 100:
        times_guess, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates_Found, esv_bias)
        diffs = transponder_coordinates_Found - guess
        dist = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Compute the residual
        z = np.zeros(len(transponder_coordinates_Found))
        z[:] = dist[:] - (esv[:] + esv_bias) * (times_known[:] + time_bias)

        # Compute H Matrix and Weighting Matrix
        H, scale = compute_H_biased(transponder_coordinates_Found, diffs, dist, esv, esv_bias)
        damping  = np.eye(5) * lambda_damping

        delta = np.linalg.inv((H.T @ H) + damping) @ H.T @ z
        delta = delta * scale

        # Update the estimate
        new_residual = np.sum(z**2)
        if new_residual < prev_residual:
            estimate = estimate + delta
            lambda_damping = max(lambda_damping/10, 1e-7)
        else:
            lambda_damping = min(lambda_damping*10, 1e7)
            k+=1
            continue

        prev_residual = new_residual
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        k += 1

        if k % 10 == 0:
            print(k, ": residual= ", new_residual, "lambda= ", lambda_damping)

    return estimate, times_known

if __name__ == "__main__":
    # Generate synthetic data
    CDOG, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(100)
    times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates_Actual)

    esv_bias = 0.0
    time_noise = 0.0
    position_noise = 0.0

    #Apply noise to position
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
    CDOG_guess = np.array([random.uniform(-5000, 5000), random.uniform(-5000, 5000), random.uniform(-5225, -5235)])

    # Run the Geiger method with ESV bias estimation
    estimate, times_known = numba_bias_geiger(CDOG_guess, CDOG, transponder_coordinates_Actual,
                                              transponder_coordinates_Found, esv_bias, time_noise)

    print(estimate)
    print(estimate - np.array([CDOG[0], CDOG[1], CDOG[2], 0.0, esv_bias]))