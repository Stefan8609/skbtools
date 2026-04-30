import numpy as np
import scipy.io as sio

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    initial_bias_geiger,
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_xAline_Joint_Annealing_bias import (
    simulated_annealing_bias_joint,
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
from numba.typed import List

# -------------------------
# Config
# -------------------------
DOG_nums = [1, 3, 4]
downsample = 10

# -------------------------
# Known data
# -------------------------
esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))
dz_array = esv_table["distance"].flatten()
angle_array = esv_table["angle"].flatten()
esv_matrix = esv_table["matrice"]

CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)

initial_lever_guess = np.array([-12.69796149, 9.51739301, -12.0743129])


# -------------------------
# Initial guesses by DOG
# -------------------------
initial_guess_augments = {
    1: np.array([-396.16, 369.90, 773.02]),
    3: np.array([825.182985, -111.05670221, -734.10011698]),
    4: np.array([236.428385, -1307.98390221, -2189.21991698]),
}

offset_hints = {
    1: 1866.0,
    3: 3175.0,
    4: 1939.0,
}


# -------------------------
# Load all DOG datasets
# -------------------------

CDOG_data_list = List()
GPS_data_list = List()

initial_guess_list = []
initial_offset_list = []

GPS_Coordinates_ref = None

for DOG_num in DOG_nums:
    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))

    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    CDOG_data = data["CDOG_data"]

    if GPS_Coordinates_ref is None:
        GPS_Coordinates_ref = GPS_Coordinates
    else:
        if GPS_Coordinates.shape != GPS_Coordinates_ref.shape:
            raise ValueError(
                f"GPS_Coordinates shape mismatch for DOG {DOG_num}: "
                f"{GPS_Coordinates.shape} != {GPS_Coordinates_ref.shape}"
            )

    CDOG_guess = CDOG_guess_base + initial_guess_augments.get(DOG_num, np.zeros(3))
    offset_hint = offset_hints.get(DOG_num, 0.0)

    CDOG_data_list.append(CDOG_data[::downsample])
    GPS_data_list.append(GPS_data)
    initial_guess_list.append(CDOG_guess)
    initial_offset_list.append(offset_hint)

initial_guess_list = np.asarray(initial_guess_list)
initial_offset_list = np.asarray(initial_offset_list)
GPS_Coordinates = GPS_Coordinates_ref


# -------------------------
# Joint inversion
# -------------------------

print("\n ---------Using Joint Simulated Annealing--------- \n")

best_lever, best_offset_list, inversion_result_list, rmse_list = (
    simulated_annealing_bias_joint(
        300,
        CDOG_data_list,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others,
        initial_guess_list,
        initial_lever_guess,
        dz_array,
        angle_array,
        esv_matrix,
        initial_offset_list=initial_offset_list,
        real_data=True,
        z_sample=True,
    )
)



# -------------------------
# Summary + diagnostics
# -------------------------
print("\n---------------- Joint Results ----------------")
print("Best Lever:", np.round(best_lever, 4))

joint_rmse_cm = 0.0

for i, DOG_num in enumerate(DOG_nums):
    inversion_result = inversion_result_list[i]
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    best_offset = best_offset_list[i]

    best_offset_eff = best_offset - time_bias
    joint_rmse_cm += rmse_list[i]

    print(f"\nDOG {DOG_num}")
    print(f"Estimate: {np.round(inversion_result, 4)}")
    print(
        f"Offset: {np.round(best_offset_eff, 4)}, "
        f"Inversion Guess (from base): {np.round(inversion_guess - CDOG_guess_base, 5)}"
    )
    print("RMSE:", np.round(rmse_list[i] * 100 * 1515, 3), "cm")

print("\nJoint objective (sum of DOG RMSE, cm):", np.round(joint_rmse_cm * 100 * 1515, 3))