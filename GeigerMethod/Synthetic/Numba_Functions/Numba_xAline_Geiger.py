import numpy as np
from numba import njit
from Numba_Geiger import computeJacobianRayTracing, findTransponder, calculateTimesRayTracing
from Numba_xAline import two_pointer_index, find_subint_offset, find_int_offset
from Generate_Unaligned_Realistic import generateUnalignedRealistic

"""Need 3 xaline geigers

1) Finding the integer offset with a coarse guess
2) Finding the sub-int offset with a medium grain guess
3) Exact offset known (to high degree) and pinpointing the location of the receiver
    Exact offset should be bc it shifts the RMSE away from 0

Seems like initial guess has to be within a couple 100 meters of CDOG for it to converge
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
        jacobian = computeJacobianRayTracing(inversion_guess, transponder_coordinates_full, GPS_full, esv_full)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        k += 1

        if np.linalg.norm(inversion_guess - guess) > 1000:
            print("ERROR: Inversion too far from starting value")
            return inversion_guess, offset
    return inversion_guess, offset

def transition_geiger(guess, CDOG_data, GPS_data, transponder_coordinates, offset):
    """ For when looking for sub-int offset"""
    epsilon = 10 ** -5
    k = 0
    delta = 1
    inversion_guess = guess

    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        times_guess, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates)
        offset = find_subint_offset(offset, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)

        CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
            two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True)
        )
        jacobian = computeJacobianRayTracing(inversion_guess, transponder_coordinates_full, GPS_full, esv_full)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        k += 1

        if np.linalg.norm(inversion_guess - guess) > 1000:
            print("ERROR: Inversion too far from starting value")
            return inversion_guess, offset

    return inversion_guess, offset


def final_geiger(guess, CDOG_data, GPS_data, transponder_coordinates, offset):
    """ For use when offset is known to a high degree"""
    epsilon = 10 ** -5
    k = 0
    delta = 1
    inversion_guess = guess

    times_guess, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates)
    CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
        two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True)
    )

    while np.linalg.norm(delta) > epsilon and k < 10:
        GPS_full, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates_full)

        jacobian = computeJacobianRayTracing(inversion_guess, transponder_coordinates_full, GPS_full, esv_full)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        k += 1

    times_guess, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates)
    CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
        two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True)
    )

    return inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock


if __name__ == "__main__":
    true_offset = np.random.rand() * 9000 + 1000
    print(true_offset)

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(
        20000, time_noise, true_offset
    )
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

    guess = CDOG + [200, 200, 400]

    inversion_result, offset = initial_geiger(guess, CDOG_data, GPS_data, transponder_coordinates)
    print("INT Offset: ", offset, "DIFF:", offset - true_offset)
    print("CDOG:", CDOG)
    print("Inversion:", inversion_result)
    print("Distance:", np.linalg.norm(inversion_result - CDOG) * 100, 'cm')
    print("\n")

    inversion_result, offset = transition_geiger(inversion_result, CDOG_data, GPS_data, transponder_coordinates, offset)

    print("SUB-INT Offset: ", offset, "DIFF", offset - true_offset)
    print("CDOG:", CDOG)
    print("Inversion:", inversion_result)
    print("Distance:", np.linalg.norm(inversion_result - CDOG) * 100, 'cm')
    print("\n")

    times_guess, esv = calculateTimesRayTracing(inversion_result, transponder_coordinates)
    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(
        inversion_result, CDOG_data, GPS_data, transponder_coordinates, offset
    )
    print("CDOG:", CDOG)
    print("Inversion:", inversion_result)
    print("Distance:", np.linalg.norm(inversion_result - CDOG) * 100, 'cm')