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
Bud's Algorithm gives another drastic drop in RMSE and definitely has room for improvement
    Down to 41 cm RMSE for later stretch of data
    
    
Run synthetic with the real data location and see what happens

Write modular codes
Run the inversion on all the dogs and see if we get similar parameters

Run the numba time bias with multople different severities of esv offset to see how the noise propogates

Make the algorithms easier to run with options (turning on and off certain parts of the algorithm)
    Sound speed
    Time_noise
    Lever_arm
    trajectory
    etc...
    
How wrong can sound speed be for us to still recover it (build out various variations)
    Especially the uppermost part of the ocean
"""

esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

CDOG_guess_augment = np.array([ 974.12667502,  -80.98121315, -805.07870249])
# initial_lever_guess = np.array([-12.48862757, 0.22622633, -15.89601934])
# initial_lever_guess = np.array([ -8.1379,   2.6067, -17.7846 ])
# initial_lever_guess = np.array([-8.74068827,  7.78977386, -7.27690523])
# initial_lever_guess = np.array([-12.39728684,  9.58745143, -7.13177909])
initial_lever_guess = np.array([-10.9211, 8.3947, -6.0])
"""[ 826.74003 -113.4907  -732.66118]"""
offset = 2002
# offset = 1991.01236648
# offset = 2076.0242

GNSS_start, GNSS_end = 25, 40.9
# GNSS_start, GNSS_end = 31.9, 34.75
# GNSS_start, GNSS_end = 35.3, 37.6
GPS_Coordinates, GPS_data, CDOG_data, CDOG_guess, gps1_to_others = initialize_bermuda(GNSS_start, GNSS_end, CDOG_guess_augment)

transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)

"""No Simulated Annealing"""


# inversion_result, best_offset = initial_bias_geiger(CDOG_guess, CDOG_data, GPS_data, transponder_coordinates, dz_array,
#                         angle_array, esv_matrix, real_data=True)
# inversion_guess = inversion_result[:3]
# time_bias = inversion_result[3]
# esv_bias = inversion_result[4]
# print("Initial Complete:", best_offset)
#
# inversion_result, best_offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset,
#                                                       esv_bias, time_bias, dz_array, angle_array, esv_matrix, real_data=True)
# inversion_guess = inversion_result[:3]
# time_bias = inversion_result[3]
# esv_bias = inversion_result[4]
# print("Transition Complete:", best_offset)
#
# # inversion_result = CDOG_guess
# inversion_guess = inversion_result[:3]
# time_bias = inversion_result[3]
# esv_bias = inversion_result[4]
#
# print("offsets: ", best_offset, offset)
#
# """If we don't want offset found by our method"""
# best_offset = 2002
#
# inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,
#                                                                                      transponder_coordinates, best_offset, esv_bias, time_bias,
#                                                                                      dz_array, angle_array, esv_matrix, real_data=True)
# print("Final Complete")
# best_lever = initial_lever_guess
"""End No simulated annealing"""

"""Simulated Annealing"""
best_lever, best_offset, inversion_result = simulated_annealing_bias(300, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others,
                                                                CDOG_guess, initial_lever_guess,dz_array, angle_array, esv_matrix, offset, True, True)

inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]

transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, best_lever)
inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,
                                                                                     transponder_coordinates, best_offset, esv_bias, time_bias,
                                                                                     dz_array, angle_array, esv_matrix, real_data=True)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
GPS_full = GPS_full - time_bias
"""End Simulated Annealing"""

CDOG_guess_base = np.array([1976671.618715,  -5069622.53769779,  3306330.69611698])
best_offset = best_offset - inversion_result[3]
print(f"Estimate: {np.round(inversion_result, 4)}")
print(f"Best Lever: {np.round(best_lever,3)}, Offset: {np.round(best_offset,4)}, Inversion Guess: {np.round(inversion_guess-CDOG_guess_base, 5)}")
diff_data = (CDOG_full - GPS_full) * 1000
RMSE = np.sqrt(np.nanmean(diff_data**2))/1000 * 1515 * 100
print("RMSE:", np.round(RMSE,3), "cm")

time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full)

"""
Later Stretch with offset 1991.01236648:
    Estimate: [ 1.97756330e+06 -5.06971291e+06  3.30558847e+06 -6.15213141e-03
      3.45269699e-01]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 1991.01236648, Inversion Guess: [ 891.68438411  -90.36853403 -742.22868432]
    RMSE: 41.76412114430489 cm

Early stetch with offset 1991.01236648:
    Estimate: [ 1.97749469e+06 -5.06975002e+06  3.30557765e+06 -4.69472577e-03
     -8.25014171e-03]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 1991.01236648, Inversion Guess: [ 823.07273856 -127.4784012  -753.04546293]
    RMSE: 34.92721519014112 cm

Full data with found offset:
    Estimate: [ 1.97749415e+06 -5.06972687e+06  3.30558970e+06 -5.71262003e-03
     -1.42138328e+00]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 2002.0024736817998, Inversion Guess: [ 822.53498628 -104.33636845 -740.99398228]
    RMSE: 346.6423130226219 cm
    
Full data with 2003 offset:
    Estimate: [ 1.97749420e+06 -5.06972741e+06  3.30558995e+06 -8.76245851e-03
     -1.33482911e+00]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 2003, Inversion Guess: [ 822.58142272 -104.87236322 -740.7419129 ]
    RMSE: 302.33173949545716 cm

Note that the ESV bias vary greatly between the two stretches of data
"""