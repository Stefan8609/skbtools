import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from Generate_Unaligned_Realistic import generateUnalignedRealistic
from Bermuda_Trajectory import bermuda_trajectory
from Numba_Geiger import findTransponder
from Numba_xAline import two_pointer_index
from Numba_xAline_bias import initial_bias_geiger, transition_bias_geiger, final_bias_geiger
from Numba_xAline_Annealing_bias import simulated_annealing_bias
from Plot_Modular import time_series_plot, trajectory_plot

"""
File to allow for easy changing of parameters when running synthetic

Write a code to check the indexing of the Bermuda data
"""

def modular_synthetic(time_noise, position_noise, esv1 = "global_table_esv", esv2="global_table_esv_perturbed", generate_type = 0, inversion_type = 0):
    np.set_printoptions(suppress=True)
    # Choose ESV table for generation and to run synthetic
    #   Perhaps make the file link a parameter of the function
    esv_table_generate = sio.loadmat(f'../../../GPSData/{esv1}.mat')
    dz_array_generate = esv_table_generate['distance'].flatten()
    angle_array_generate = esv_table_generate['angle'].flatten()
    esv_matrix_generate = esv_table_generate['matrice']

    esv_table_inversion = sio.loadmat(f'../../../GPSData/{esv2}.mat')
    dz_array_inversion = esv_table_inversion['distance'].flatten()
    angle_array_inversion = esv_table_inversion['angle'].flatten()
    esv_matrix_inversion = esv_table_inversion['matrice']

    # Choose Generate type:
    #   0: Generate Unaligned Realistic Data
    #   1: Use Bermuda Dataset
    if generate_type == 0:
        # Generate Unaligned Realistic Data
        true_offset = np.random.rand() * 9000 + 1000
        print(true_offset)
        CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(20000, time_noise, true_offset,
            dz_array_generate, angle_array_generate, esv_matrix_generate)
        GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
        gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    else:
        # Use Bermuda Dataset
        CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = bermuda_trajectory(time_noise, position_noise,
            dz_array_generate, angle_array_generate, esv_matrix_generate)
        true_offset = 1991.01236648
        gps1_to_others = np.array([[0.0, 0.0, 0.0], [-2.4054, -4.20905, 0.060621], [-12.1105, -0.956145, 0.00877],
                                   [-8.70446831, 5.165195, 0.04880436]])

    # Choose Inversion Type
    #   0: Just xAline Geiger
    #   1: xAline Geiger with Simulated Annealing
    initial_guess = CDOG + [100, 100, 200]
    if inversion_type == 0:
        # Just xAline Geiger
        lever = np.array([-10.0, 3.0, -15.0]) if generate_type == 0 else np.array([-12.48862757, 0.22622633, -15.89601934])
        # lever = np.array([-10.0, 3.0, -15.0]) if generate_type == 0 else np.array([-12.48862757, 2.22622633, -14.89601934])


        transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        inversion_result, offset = initial_bias_geiger(initial_guess, CDOG_data, GPS_data, transponder_coordinates,
                                                                   dz_array_inversion, angle_array_inversion, esv_matrix_inversion)
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]

        print("INT Offset: {:.4f}".format(offset), "DIFF: {:.4f}".format(offset - true_offset))
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion:", np.round(inversion_result, 3))
        print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
        print("\n")

        inversion_result, offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates,
                                                          offset, esv_bias, time_bias, dz_array_inversion, angle_array_inversion, esv_matrix_inversion)
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print("SUB-INT Offset: {:.4f}".format(offset), "DIFF: {:.4f}".format(offset - true_offset))
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion:", np.round(inversion_result, 3))
        print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
        print("\n")

        inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data,GPS_data, transponder_coordinates,
                                                                                         offset, esv_bias, time_bias, dz_array_inversion,
                                                                                         angle_array_inversion, esv_matrix_inversion)
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion:", np.round(inversion_result, 3))
        print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))

    else:
        real_lever = np.array([-10.0, 3.0, -15.0]) if generate_type == 0 else np.array([-12.48862757, 0.22622633, -15.89601934])
        initial_lever = np.array([-13.0, 0.0, -14.0])
        """True levers: Realistic Generate [-10, 3, -15], Bermuda Generate: [-12.48862757, 0.22622633, -15.89601934]"""
        lever, offset, inversion_result = simulated_annealing_bias(300, CDOG_data, GPS_data, GPS_Coordinates,gps1_to_others,
                                                                   initial_guess, initial_lever, dz_array_inversion, angle_array_inversion,
                                                                   esv_matrix_inversion)
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion:", np.round(inversion_result, 3))
        print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
        print(f"Lever Error: {np.round(np.linalg.norm(lever - real_lever), 2) * 100} cm")

        transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,transponder_coordinates,
                                                                                         offset, esv_bias, time_bias, dz_array_inversion, angle_array_inversion, esv_matrix_inversion)

    time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise)
    trajectory_plot(transponder_coordinates, GPS_data, CDOG)


if __name__ == "__main__":
    modular_synthetic(2 * 10**-5, 2 * 10**-2,"global_table_esv","global_table_esv_realistic_perturbed", generate_type = 0, inversion_type=0)
    # modular_synthetic(2 * 10**-5, 2 * 10**-2,"global_table_esv","global_table_esv", generate_type = 1, inversion_type=1)

