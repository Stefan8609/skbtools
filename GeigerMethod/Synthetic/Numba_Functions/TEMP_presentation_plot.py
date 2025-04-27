import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from Bermuda_Trajectory import bermuda_trajectory
from Numba_time_bias import calculateTimesRayTracing_Bias_Real
from Numba_xAline import two_pointer_index
from Plot_Modular import time_series_plot
from Numba_xAline_bias import final_bias_geiger

time_noise = 2 * 10**-5
position_noise = 2 * 10**-2
esv_table = sio.loadmat('../../../GPSData/global_table_esv_normal.mat')
dz_array_generate = esv_table['distance'].flatten()
angle_array_generate = esv_table['angle'].flatten()
esv_matrix_generate = esv_table['matrice']
DOG_num = 3

esv_table = sio.loadmat('../../../GPSData/global_table_esv_realistic_perturbed.mat')
dz_array_inversion = esv_table['distance'].flatten()
angle_array_inversion = esv_table['angle'].flatten()
esv_matrix_inversion = esv_table['matrice']

CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = bermuda_trajectory(time_noise,
                                                                                              position_noise,
                                                                                              dz_array_generate,
                                                                                              angle_array_generate,
                                                                                              esv_matrix_generate,
                                                                                              DOG_num)
true_offset = 1991.01236648
gps1_to_others = np.array([[0.0, 0.0, 0.0], [-2.4054, -4.20905, 0.060621], [-12.1105, -0.956145, 0.00877],
                           [-8.70446831, 5.165195, 0.04880436]])
lever = np.array([-12.48862757, 0.22622633, -15.89601934])

esv_bias = 0
time_bias = 0

times, esv = calculateTimesRayTracing_Bias_Real(CDOG, true_transponder_coordinates, esv_bias,
                                                dz_array_inversion, angle_array_inversion, esv_matrix_inversion)

CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
    two_pointer_index(true_offset, .6, CDOG_data, GPS_data, times, true_transponder_coordinates, esv, True)
)

# Plotting
time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise)

inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(CDOG, CDOG_data, GPS_data,
                                                                                 true_transponder_coordinates,
                                                                                 true_offset, esv_bias, time_bias,
                                                                                 dz_array_inversion,
                                                                                 angle_array_inversion,
                                                                                 esv_matrix_inversion,
                                                                                 real_data=True)

time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise)
