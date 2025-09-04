import numpy as np
import scipy.io as sio

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger import (
    initial_geiger,
    transition_geiger,
    final_geiger,
)

import matplotlib.pyplot as plt

from data import gps_data_path

esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
dz_array = esv_table["distance"].flatten()
angle_array = esv_table["angle"].flatten()
esv_matrix = esv_table["matrice"]

"""
Process:
    Load the data in the form of CDOG_data, GPS_data, GPS_coordinates
    Need to figure out how to find the depth and angle for ray-tracing

    Need to configure what the coordinate systems are for each different type of data
        GET absolute distance using ECEF, and vertical distance from geodetic

    Need to get elevation of transducer (how?) Convert ecef to geodetic for transducer?
"""

CDOG_guess_augment = np.array([974.12667502, -80.98121315, -805.07870249])
# initial_lever_guess = np.array([-30.22391079,  -0.22850613, -21.97254162])
initial_lever_guess = np.array([-12.48862757, 0.22622633, -15.89601934])
offset = 1991.01236648
# offset = 2076.0242

DOG_num = 3

data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
GPS_Coordinates = data["GPS_Coordinates"]
GPS_data = data["GPS_data"]
CDOG_data = data["CDOG_data"]
# CDOG_guess = data['CDOG_guess']
CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
# gps1_to_others = data['gps1_to_others']

initial_lever_guess = np.array([-12.4659, 9.6021, -13.2993])


gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)

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

"""Running Geiger"""
print("Starting Geiger Method")
transponder_coordinates = findTransponder(
    GPS_Coordinates, gps1_to_others, initial_lever_guess
)
inversion_guess, best_offset = initial_geiger(
    CDOG_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=True,
)
print("Initial Complete:", best_offset)
inversion_guess, best_offset = transition_geiger(
    inversion_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    best_offset,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=True,
)
print("Transition Complete:", best_offset)

inversion_guess = CDOG_guess
best_offset = offset
inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(
    inversion_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    best_offset,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=True,
)
print("Final Complete")
best_lever = initial_lever_guess

CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
print(
    f"Best Lever: {best_lever}, Offset: {best_offset}, "
    f"Inversion Guess: {inversion_guess - CDOG_guess_base}"
)
diff_data = (CDOG_full - GPS_full) * 1000
RMSE = np.sqrt(np.nanmean(diff_data**2)) / 1000 * 1515 * 100
print("RMSE:", RMSE, "cm")

# Figure with 2 plots arranged vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
# axes[0].scatter(CDOG_clock, CDOG_full, s=10, marker="x", label="CDOG")
axes[0].scatter(
    CDOG_clock, CDOG_clock - (GPS_clock - GPS_full), s=10, marker="x", label="CDOG"
)
axes[0].scatter(GPS_clock, GPS_full, s=1, marker="x", label="GPS")
axes[0].set_title("Travel Times")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Travel Time (s)")
axes[0].legend()

axes[1].scatter(GPS_clock, diff_data, s=1)
axes[1].set_title("Difference between CDOG and GPS")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Residual (ms)")
plt.tight_layout()
plt.show()

"""
Adjust sound speed with a constant
Make some minimal inversion

Run on three hours of the data that is continuous in the residual

Isolate a geometric effect from the part that nearly fits
    Adjust the sound speed by a constant because its not constant
        Need to rewrite Thalia's code to do this

    If no way I can make this better than it has to be the sound speed


Make Synthetic with the incorrect sound speed profile to see the variation
"""
