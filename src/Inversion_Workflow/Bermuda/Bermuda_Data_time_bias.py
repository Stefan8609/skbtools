import numpy as np
import scipy.io as sio

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    initial_bias_geiger,
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from plotting.Plot_Modular import (
    time_series_plot,
    range_residual,
)
from geometry.ECEF_Geodetic import ECEF_Geodetic
from data import gps_data_path

# -------------------------
# Config
# -------------------------
DOG_num = 4
simulated_annealing = True
save = True

# -------------------------
# Known data
# -------------------------

esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))
dz_array = esv_table["distance"].flatten()
angle_array = esv_table["angle"].flatten()
esv_matrix = esv_table["matrice"]

data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
GPS_Coordinates = data["GPS_Coordinates"]
GPS_data = data["GPS_data"]
CDOG_data = data["CDOG_data"]

CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)

initial_lever_guess = np.array([-12.69796149, 9.51739301, -15.0743129])

# -------------------------
# Initial guesses by DOG
# -------------------------
if DOG_num == 1:
    CDOG_guess_augment = np.array([-398.16, 371.90, 773.02])
    offset_hint = 1866.0
elif DOG_num == 3:
    CDOG_guess_augment = np.array([825.182985, -111.05670221, -734.10011698])
    offset_hint = 3175.0
elif DOG_num == 4:
    CDOG_guess_augment = np.array([236.428385, -1307.98390221, -2189.21991698])
    offset_hint = 1939.0
else:
    CDOG_guess_augment = np.zeros(3)
    offset_hint = 0.0

CDOG_guess = CDOG_guess_base + CDOG_guess_augment

if simulated_annealing:
    print("\n ---------Using Simulated Annealing--------- \n")
    best_lever, best_offset, inversion_result = simulated_annealing_bias(
        300,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others,  # keep consistent with how data were generated
        CDOG_guess,
        initial_lever_guess,
        dz_array,
        angle_array,
        esv_matrix,
        initial_offset=offset_hint,
        real_data=True,
        z_sample=True,
    )

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, best_lever
    )

    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
        inversion_result[:3],
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        best_offset,
        inversion_result[4],
        inversion_result[3],
        dz_array,
        angle_array,
        esv_matrix,
        real_data=True,
    )

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
else:
    # -------------------------
    # Inversion: Initial → Final
    # -------------------------
    print("\n ---------Using Geiger Inversion--------- \n")

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, initial_lever_guess
    )
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
    print("\nInitial complete. Offset =", best_offset)

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]

    print("Starting Final Geiger (fixed offset)...")
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
    print("\nFinal complete.")

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    best_lever = initial_lever_guess


# -------------------------
# Summary + diagnostics
# -------------------------
# Keep your original convention (fold time_bias into offset for reporting)
best_offset_eff = best_offset - time_bias

diff_data_ms = (CDOG_full - GPS_full) * 1000.0
RMSE_cm = (np.sqrt(np.nanmean(diff_data_ms**2)) / 1000.0) * 1515.0 * 100.0

print(f"\nEstimate: {np.round(inversion_result, 4)}")
print(
    f"Best Lever: {np.round(best_lever, 3)}, "
    f"Offset: {np.round(best_offset_eff, 4)}, "
    f"Inversion Guess (from base): {np.round(inversion_guess - CDOG_guess_base, 5)}"
)
print("RMSE:", np.round(RMSE_cm, 3), "cm")


# -------------------------
# Plots
# -------------------------
DOG_name = {1: "DOG1", 3: "DOG3", 4: "DOG4"}
save_path = "Figs/Inversion_Workflow"
file_tag = (
    DOG_name[DOG_num] + "_SimAnn"
    if simulated_annealing
    else DOG_name[DOG_num] + "_Geiger"
)

time_series_plot(
    CDOG_clock,
    CDOG_full,
    GPS_clock,
    GPS_full,
    save=save,
    path=save_path,
    chain_name=file_tag,
    zoom_start=40000,
)

# Range residual plot inputs: recompute travel times at final estimate
times_guess, esv = calculateTimesRayTracing_Bias_Real(
    inversion_guess,
    transponder_coordinates,
    esv_bias,
    dz_array,
    angle_array,
    esv_matrix,
)

CDOG_clock_ex, CDOG_full_ex, GPS_clock_ex, GPS_full_ex, trans_coords_full, esv_full = (
    two_pointer_index(
        best_offset_eff,
        0.6,
        CDOG_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )
)

range_residual(
    trans_coords_full,
    esv_full,
    inversion_guess,
    CDOG_full_ex,
    GPS_full_ex,
    GPS_clock_ex,
    save=save,
    path=save_path,
    chain_name=file_tag,
)

# Elevation-angle residuals
depth_arr = ECEF_Geodetic(trans_coords_full)[2]
inv_xyz = inversion_guess[np.newaxis, :]
lat, lon, depth0 = ECEF_Geodetic(inv_xyz)

dz = depth_arr - depth0
abs_dist = np.sqrt(np.sum((trans_coords_full - inv_xyz) ** 2, axis=1))
beta = np.arcsin(dz / abs_dist) * 180.0 / np.pi

# elevation_angle_residual(beta, CDOG_full_ex, GPS_full_ex, save=save,
# path= save_path, chain_name=file_tag)

# plot_integer_pick_metrics_dog(
#     best_offset,
#     CDOG_data,
#     GPS_data,
#     times_guess,
#     transponder_coordinates,
#     esv,
#     half_window=50,
#     step=0.1,
# )
