import numpy as np
import scipy.io as sio

from Numba_Geiger import (
    findTransponder,
)
from Numba_xAline_Geiger import initial_geiger, transition_geiger, final_geiger
from Initialize_Bermuda_Data import initialize_bermuda

import matplotlib.pyplot as plt

esv_table = sio.loadmat("../../../GPSData/global_table_esv.mat")

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

GNSS_start, GNSS_end = 25, 40.9
# GNSS_start, GNSS_end = 31.9, 34.75
# GNSS_start, GNSS_end = 35.3, 37.6
GPS_Coordinates, GPS_data, CDOG_data, CDOG_guess, gps1_to_others = initialize_bermuda(
    GNSS_start, GNSS_end, CDOG_guess_augment
)

"""Running Geiger"""
transponder_coordinates = findTransponder(
    GPS_Coordinates, gps1_to_others, initial_lever_guess
)
inversion_guess, best_offset = initial_geiger(
    CDOG_guess, CDOG_data, GPS_data, transponder_coordinates, real_data=True
)
print("Initial Complete:", best_offset)
inversion_guess, best_offset = transition_geiger(
    inversion_guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    best_offset,
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
