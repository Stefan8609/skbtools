import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from Numba_Geiger import computeJacobianRayTracing, findTransponder, calculateTimesRayTracing
from Numba_xAline import two_pointer_index, find_subint_offset

"""Need 3 xaline geigers

1) Finding the integer offset with a coarse guess
2) Finding the sub-int offset with a medium grain guess
3) Exact offset known (to high degree) and pinpointing the location of the receiver
    Exact offset should be bc it shifts the RMSE away from 0

"""

def initial_geiger(guess, CDOG_data, GPS_data, transponder_coordinates):
    """For use when looking for int offset"""
    epsilon = 10**-5
    k = 0
    delta = 1
    inversion_guess = guess
    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        times_guess, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates)
        offset = find_int_offset(CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)

        CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
            two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)
        )
        """Need to rewrite two-pointer-index with ability to accept non-exact offset"""

        abs_diff = np.abs(CDOG_full - GPS_full)
        indices = np.where(abs_diff >= 0.9)
        CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

        jacobian = computeJacobianRayTracing(inversion_guess, transponder_full, GPS_full, esv_full)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        estimate_arr = np.append(estimate_arr, inversion_guess, axis=0)
        k += 1

        if np.linalg.norm(inversion_guess - guess) > 1000:
            print("ERROR: Inversion too far from starting value")
            estimate_arr = np.reshape(estimate_arr, (-1, 3))
            return inversion_guess, estimate_arr, offset

    return inversion_guess, offset

def transition_geiger()
    """ For when looking for sub-int offset"""

def final_geiger():
    """ For use when offset is known to a high degree"""