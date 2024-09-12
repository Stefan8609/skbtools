import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import calculateTimesRayTracing, computeJacobianRayTracing, findTransponder
from Generate_Unaligned_Realistic import generateUnalignedRealistic
from xAline import index_data, find_int_offset, find_subint_offset

"""Have a far annealing where only integer offset matters - Approximating right offset


Then after running that have a close offset where sub-integer offset matters too
"""

def xAline_Geiger(guess, CDOG_data, GPS_data, transponder_coordinates):
    #Threshold
    epsilon = 10**-5

    k=0
    delta = 1
    inversion_guess = guess
    while np.linalg.norm(delta) > epsilon and k<100:
        times_guess, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates)

        offset = find_int_offset(CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)
        full_times, CDOG_full, GPS_full, transponder_full, esv_full = index_data(offset, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)

        abs_diff = np.abs(CDOG_full - GPS_full)
        indices = np.where(abs_diff >= 0.9)
        CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

        ##What goes in for times in this??
        jacobian = computeJacobianRayTracing(inversion_guess, transponder_full, GPS_full, esv_full)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (GPS_full-CDOG_full)
        inversion_guess = inversion_guess + delta
        k+=1

        print(inversion_guess, offset, k)

    return inversion_guess


if __name__ == "__main__":
    true_offset = int(np.random.rand() * 10000)
    print(true_offset)

    position_noise = 2*10**-2
    # Generate the arrival time series for the generated offset (aswell as GPS Coordinates)
    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(20000,
                                                                                                          true_offset)
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

    guess = CDOG + [100, 200 , -200]

    result = xAline_Geiger(guess, CDOG_data, GPS_data, transponder_coordinates)
    print("CDOG:", CDOG)
    print("Inversion:", result)
