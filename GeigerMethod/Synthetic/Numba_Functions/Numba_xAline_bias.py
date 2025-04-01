import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from Generate_Unaligned_Realistic import generateUnalignedRealistic
from Bermuda_Trajectory import bermuda_trajectory
from Numba_xAline import two_pointer_index, find_int_offset
from Numba_time_bias import calculateTimesRayTracing_Bias, calculateTimesRayTracing_Bias_Real, find_esv, compute_Jacobian_biased
from Numba_Geiger import findTransponder
from numba import njit
from ECEF_Geodetic import ECEF_Geodetic
from Plot_Modular import time_series_plot

"""
Improve Geiger's method such that the time-bias become included in the offset for transition geiger... 
    (Necessary for properly aligning the data)
    
Prediction ability not changing between the methods because the int-offset is giving the correct alignment between
    each data set
"""

def initial_bias_geiger(guess, CDOG_data, GPS_data, transponder_coordinates, dz_array,
                        angle_array, esv_matrix, real_data=False):
    """For use when looking for int offset"""
    epsilon = 10**-5
    k = 0
    delta = 1
    time_bias = 0.0
    esv_bias = 0.0
    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if real_data == False:
            times_guess, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
        else:
            times_guess, esv = calculateTimesRayTracing_Bias_Real(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
        offset = find_int_offset(CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
            two_pointer_index(offset, .6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv)
        )
        J = compute_Jacobian_biased(guess, transponder_coordinates_full, GPS_full, esv_full, esv_bias)
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ ((GPS_full - time_bias) - CDOG_full)
        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        k += 1
    return estimate, offset

@njit
def transition_bias_geiger(guess, CDOG_data, GPS_data, transponder_coordinates, offset, esv_bias, time_bias, dz_array,
                        angle_array, esv_matrix, real_data=False):
    """ For when looking for sub-int offset using time bias"""
    epsilon = 10 ** -5
    delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    k=0
    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if real_data == False:
            times_guess, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
        else:
            times_guess, esv = calculateTimesRayTracing_Bias_Real(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
            two_pointer_index(offset, .6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True)
        )
        J = compute_Jacobian_biased(guess, transponder_coordinates_full, GPS_full, esv_full, esv_bias)
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ ((GPS_full - time_bias) - CDOG_full)
        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        offset -= time_bias
        k += 1
    return estimate, offset

@njit
def final_bias_geiger(guess, CDOG_data, GPS_data, transponder_coordinates, offset, esv_bias, time_bias, dz_array,
                 angle_array, esv_matrix, real_data=False):
    """ For when looking for sub-int offset"""
    epsilon = 10 ** -5
    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])

    if real_data == False:
        times_guess, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)

    CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
        two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True)
    )

    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if real_data == False:
            GPS_full, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates_full, esv_bias, dz_array, angle_array, esv_matrix)
        else:
            GPS_full, esv = calculateTimesRayTracing_Bias_Real(guess, transponder_coordinates_full, esv_bias, dz_array, angle_array, esv_matrix)

        J = compute_Jacobian_biased(guess, transponder_coordinates_full, GPS_full, esv_full, esv_bias)
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ ((GPS_full - time_bias) - CDOG_full)
        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        k += 1

    if real_data == False:
        times_guess, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix)
    CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
        two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True)
    )
    return estimate, CDOG_full, GPS_full, CDOG_clock, GPS_clock

if __name__ == '__main__':
    # Table to generate synthetic times
    esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
    dz_array = esv_table['distance'].flatten()
    angle_array = esv_table['angle'].flatten()
    esv_matrix = esv_table['matrice']

    position_noise = 2 * 10 ** -2
    time_noise = 2 * 10 ** -5

    esv_bias = 0
    time_bias = 0
    """Either generate a realistic or use bermuda trajectory"""

    # true_offset = np.random.rand() * 9000 + 1000
    # print(true_offset)
    # CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(
    #     20000, time_noise, true_offset, esv_bias, time_bias, dz_array, angle_array, esv_matrix
    # )
    # GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    # gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    # gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = bermuda_trajectory(time_noise, position_noise,
                                                                                                    dz_array, angle_array, esv_matrix)
    true_offset = 1991.01236648
    gps1_to_others = np.array([[0.0, 0.0, 0.0], [-2.4054, -4.20905, 0.060621], [-12.1105, -0.956145, 0.00877],
                               [-8.70446831, 5.165195, 0.04880436]])
    gps1_to_transponder = np.array([-12.48862757, 0.22622633, -15.89601934])

    """After Generating run through the analysis"""

    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

    guess = CDOG + [100, 100, 200]

    inversion_result, offset = initial_bias_geiger(guess, CDOG_data, GPS_data, transponder_coordinates, dz_array,
                        angle_array, esv_matrix)
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("INT Offset: {:.4f}".format(offset), "DIFF: {:.4f}".format(offset - true_offset))
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion:", np.around(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
    print("\n")

    inversion_result, offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, offset,
                                                      esv_bias, time_bias, dz_array, angle_array, esv_matrix)
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("SUB-INT Offset: {:.4f}".format(offset), "DIFF: {:.4f}".format(offset - true_offset))
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion:", np.around(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
    print("\n")

    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,
                                                                                     transponder_coordinates, offset, esv_bias, time_bias,
                                                                                     dz_array, angle_array, esv_matrix)
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion:", np.around(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))

    # Plot the results
    time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise)

