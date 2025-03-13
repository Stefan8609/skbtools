import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline_bias import calculateTimesRayTracing_Bias_Real
from Generate_Unaligned_Realistic import generateUnalignedRealistic
from Numba_xAline import two_pointer_index, find_int_offset
from Numba_time_bias import calculateTimesRayTracing_Bias, find_esv, compute_Jacobian_biased
from Numba_Geiger import findTransponder
from Numba_xAline_bias import initial_bias_geiger, transition_bias_geiger, final_bias_geiger
from numba import njit

"""
Incorporate simulated annealing to find the transducer location in addition to the bias terms.
"""

def simulated_annealing_bias(iter, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others, initial_guess, initial_lever,
                             dz_array, angle_array, esv_matrix, initial_offset=0, real_data = False):
    """Algorithm to determine the best lever arm, offset, and seafloor receiver position"""
    # Initialize variables
    status = "int"
    old_offset = initial_offset

    inversion_guess = initial_guess
    time_bias = 0.0
    esv_bias = 0.0
    inversion_estimate = np.array([initial_guess[0], initial_guess[1], initial_guess[2], time_bias, esv_bias])

    transponder_coordinates_found = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever)
    if real_data == False:
        times_guess, esv = calculateTimesRayTracing_Bias(initial_guess, transponder_coordinates_found, esv_bias, dz_array, angle_array, esv_matrix)
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(initial_guess, transponder_coordinates_found, esv_bias, dz_array, angle_array, esv_matrix)
    CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
        two_pointer_index(initial_offset, 0.5, CDOG_data, GPS_data, times_guess, transponder_coordinates_found, esv)
    )

    best_rmse = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
    best_lever = initial_lever
    k=0

    while k < 300:
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / (iter)))
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = best_lever + displacement

        transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)

        if status == "int":
            inversion_estimate, offset = initial_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, dz_array, angle_array, esv_matrix, real_data)
            if offset == old_offset:
                status = "subint"
        elif status == "subint":
            inversion_estimate, offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, offset, esv_bias, time_bias, dz_array, angle_array, esv_matrix, real_data)
            status = "constant"
        else:
            if k == 100 or k == 200:
                transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, best_lever)
                inversion_estimate, offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, offset, esv_bias, time_bias, dz_array, angle_array, esv_matrix, real_data)
            else:
                inversion_estimate, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, offset, esv_bias, time_bias, dz_array, angle_array, esv_matrix, real_data)

        inversion_guess = inversion_estimate[:3]
        time_bias = inversion_estimate[3]
        esv_bias = inversion_estimate[4]

        if real_data == False:
            times_guess, esv = calculateTimesRayTracing_Bias(inversion_guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
        else:
            times_guess, esv = calculateTimesRayTracing_Bias_Real(inversion_guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
            two_pointer_index(offset, 0.5, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)
        )

        RMSE = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever

        if k % 10 == 0:
            print(k, RMSE * 100 * 1515, offset, lever)
        old_offset = offset
        k += 1

    return lever, offset, inversion_estimate



if __name__ == "__main__":
    esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
    dz_array = esv_table['distance'].flatten()
    angle_array = esv_table['angle'].flatten()
    esv_matrix = esv_table['matrice']

    true_offset = np.random.rand() * 9000 + 1000
    print(true_offset)
    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(
        10000, time_noise, true_offset
    )
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    initial_guess = CDOG + np.array([100, -100, -50], dtype=np.float64)
    initial_lever = np.array([-5.0, 7.0, -10.0], dtype=np.float64)

    lever, offset, inversion_result = simulated_annealing_bias(300, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others,
                                                              initial_guess, initial_lever, dz_array, angle_array, esv_matrix)


    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion:", np.around(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
