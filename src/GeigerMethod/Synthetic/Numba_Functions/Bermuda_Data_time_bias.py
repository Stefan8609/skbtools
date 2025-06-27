import numpy as np
import scipy.io as sio

from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline_bias import (
    initial_bias_geiger,
    transition_bias_geiger,
    final_bias_geiger,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_time_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import two_pointer_index
from GeigerMethod.Synthetic.Numba_Functions.Plot_Modular import (
    time_series_plot,
    range_residual,
)


# esv_table = sio.loadmat('../../../Data/global_table_esv.mat')
from data import gps_data_path

esv_table = sio.loadmat(gps_data_path("global_table_esv_normal.mat"))

dz_array = esv_table["distance"].flatten()
angle_array = esv_table["angle"].flatten()
esv_matrix = esv_table["matrice"]

DOG_num = 4

data = np.load(gps_data_path(f"Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
GPS_Coordinates = data["GPS_Coordinates"]
GPS_data = data["GPS_data"]
CDOG_data = data["CDOG_data"]
# CDOG_guess = data['CDOG_guess']
CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
# gps1_to_others = data['gps1_to_others']

gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)

"""Set up the initial estimations"""
initial_lever_guess = np.array([-12.4659, 9.6021, -13.2993])

# initial_lever_guess = np.array([-12.5841,  9.6593, -11.9162])
# gps1_to_others =  np.array([[ -0.2763,   0.2039,  -0.2825],
#  [ -1.718,   -4.9097,   0.0369],
#  [-12.464,    0.2169,  -0.0625],
#  [ -8.523,    4.007,    0.0896]])
if DOG_num == 1:
    CDOG_guess_augment = np.array([-398.16, 371.90, 773.02])
    offset = 1866.0
if DOG_num == 3:
    CDOG_guess_augment = np.array([825.182985, -111.05670221, -734.10011698])
    offset = 3175.0
if DOG_num == 4:
    CDOG_guess_augment = np.array([236.428385, -1307.98390221, -2189.21991698])
    offset = 1939.0

CDOG_guess = CDOG_guess_base + CDOG_guess_augment

transponder_coordinates = findTransponder(
    GPS_Coordinates, gps1_to_others, initial_lever_guess
)
"""No Simulated Annealing"""
inversion_result, best_offset = initial_bias_geiger(
    CDOG_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=True,
)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
print("Initial Complete:", best_offset)

inversion_result, best_offset = transition_bias_geiger(
    inversion_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    best_offset,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=True,
)
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
best_offset = offset

inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
    inversion_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    best_offset,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=True,
)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]

print("Final Complete")
best_lever = initial_lever_guess
"""End No simulated annealing"""

print(inversion_result[3])
best_offset = best_offset - inversion_result[3]
print(f"Estimate: {np.round(inversion_result, 4)}")
print(
    f"Best Lever: {np.round(best_lever, 3)}, "
    f"Offset: {np.round(best_offset, 4)}, "
    f"Inversion Guess: {np.round(inversion_guess - CDOG_guess_base, 5)}"
)
diff_data = (CDOG_full - GPS_full) * 1000
RMSE = np.sqrt(np.nanmean(diff_data**2)) / 1000 * 1515 * 100
print("RMSE:", np.round(RMSE, 3), "cm")

time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full)

"""Plot range residuals"""
times_guess, esv = calculateTimesRayTracing_Bias_Real(
    inversion_guess,
    transponder_coordinates,
    esv_bias,
    dz_array,
    angle_array,
    esv_matrix,
)

_, _, GPS_clock, _, transponder_coordinates_full, esv_full = two_pointer_index(
    offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv, True
)


range_residual(
    transponder_coordinates_full,
    esv_full,
    inversion_guess,
    CDOG_full,
    GPS_full,
    GPS_clock,
)

print(best_offset, inversion_result[3])
