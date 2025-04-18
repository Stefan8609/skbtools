from idlelib.pyparse import trans

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ECEF_Geodetic import ECEF_Geodetic
from pymap3d import geodetic2ecef, ecef2geodetic

from Numba_Geiger import findTransponder
from Numba_xAline_bias import initial_bias_geiger, transition_bias_geiger, final_bias_geiger
from Plot_Modular import time_series_plot
from Numba_xAline_Annealing_bias import simulated_annealing_bias
from Initialize_Bermuda_Data import initialize_bermuda

"""    
Look at the Durbin watson and Q test for determining if the small section is normal

Add density plot to the GPS elevation distribution 
    Find a better way to delete points (median over time with 2 absolute deivations above and below

"""

esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

# CDOG_guess_augment = np.array([ 974.12667502,  -80.98121315, -805.07870249])
CDOG_guess_augment = np.array([825.182985, -111.05670221, -734.10011698])
# initial_lever_guess = np.array([-12.48862757, 0.22622633, -15.89601934])
# initial_lever_guess = np.array([ -8.1379,   2.6067, -17.7846 ])
# initial_lever_guess = np.array([-8.74068827,  7.78977386, -7.27690523])
# initial_lever_guess = np.array([-12.7632,   9.4474, -12.0])
# initial_lever_guess = np.array([-10.9211, 8.3947, -6.0])
# initial_lever_guess = np.array([-10.7368,  8.9474, -16.5789])
initial_lever_guess = np.array([-12.4659, 9.6021, -13.2993])
"""[ 826.74003 -113.4907  -732.66118]"""
offset = 2001.0
# offset = 1991.01236648
# offset = 2076.0242

GNSS_start, GNSS_end = 25, 40.9
# GNSS_start, GNSS_end = 31.9, 34.75
# GNSS_start, GNSS_end = 35.3, 37.6
GPS_Coordinates, GPS_data, CDOG_data, CDOG_guess, gps1_to_others = initialize_bermuda(GNSS_start, GNSS_end, CDOG_guess_augment)

transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)
"""No Simulated Annealing"""
inversion_result, best_offset = initial_bias_geiger(CDOG_guess, CDOG_data, GPS_data, transponder_coordinates, dz_array,
                        angle_array, esv_matrix, real_data=True)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
print("Initial Complete:", best_offset)

inversion_result, best_offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset,
                                                      esv_bias, time_bias, dz_array, angle_array, esv_matrix, real_data=True)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
print("Transition Complete:", best_offset)

# inversion_result = CDOG_guess
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]

print("offsets: ", best_offset, offset)

"""If we don't want offset found by our method"""
best_offset = 2001.0

inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,
                                                                                     transponder_coordinates, best_offset, esv_bias, time_bias,
                                                                                     dz_array, angle_array, esv_matrix, real_data=True)
print("Final Complete")
best_lever = initial_lever_guess
"""End No simulated annealing"""

"""Simulated Annealing"""
# best_lever, best_offset, inversion_result = simulated_annealing_bias(300, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others,
#                                                                 CDOG_guess, initial_lever_guess,dz_array, angle_array, esv_matrix, offset, True, True)
#
# inversion_guess = inversion_result[:3]
# time_bias = inversion_result[3]
# esv_bias = inversion_result[4]
#
# transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, best_lever)
# inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,
#                                                                                      transponder_coordinates, best_offset, esv_bias, time_bias,
#                                                                                      dz_array, angle_array, esv_matrix, real_data=True)
# inversion_guess = inversion_result[:3]
# time_bias = inversion_result[3]
# esv_bias = inversion_result[4]
# GPS_full = GPS_full - time_bias
"""End Simulated Annealing"""

CDOG_guess_base = np.array([1976671.618715,  -5069622.53769779,  3306330.69611698])
best_offset = best_offset - inversion_result[3]
print(f"Estimate: {np.round(inversion_result, 4)}")
print(f"Best Lever: {np.round(best_lever,3)}, Offset: {np.round(best_offset,4)}, Inversion Guess: {np.round(inversion_guess-CDOG_guess_base, 5)}")
diff_data = (CDOG_full - GPS_full) * 1000
RMSE = np.sqrt(np.nanmean(diff_data**2))/1000 * 1515 * 100
print("RMSE:", np.round(RMSE,3), "cm")

time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full)