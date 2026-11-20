import numpy as np

from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Inversion.Numba_xAline_Joint_Annealing_bias import (
    simulated_annealing_bias_joint,
)
from bermuda_data import (
    CDOG_GUESS_AUGMENTS,
    CDOG_REFERENCE,
    GPS1_TO_OTHERS,
    OFFSET_HINTS_S,
    gps_data_path,
)
from numba.typed import List

# Config
DOG_nums = [1, 3, 4]
downsample = 10
# Linear clock drift per DOG as config["clock_drift"] in the MCMC.
clock_drift = False

# Known data
dz_array, angle_array, esv_matrix = load_esv_table(
    gps_data_path("ESV_Tables/global_table_esv_extended.mat")
)

initial_lever_guess = np.array([-12.69796149, 9.51739301, -12.0743129])


# Initial guesses by DOG
# DOG 1 starts 2 m from the guess in Bermuda_Data_time_bias.py (+2 m ECEF x, -2 m y).
initial_guess_augments = {
    1: np.array([-396.16, 369.90, 773.02]),
    3: CDOG_GUESS_AUGMENTS[3],
    4: CDOG_GUESS_AUGMENTS[4],
}


# Load all DOG datasets

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

    CDOG_guess = CDOG_REFERENCE + initial_guess_augments[DOG_num]
    offset_hint = OFFSET_HINTS_S[DOG_num]

    CDOG_data_list.append(CDOG_data[::downsample])
    GPS_data_list.append(GPS_data)
    initial_guess_list.append(CDOG_guess)
    initial_offset_list.append(offset_hint)

initial_guess_list = np.asarray(initial_guess_list)
initial_offset_list = np.asarray(initial_offset_list)
GPS_Coordinates = GPS_Coordinates_ref


# Joint inversion

print("\n ---------Using Joint Simulated Annealing--------- \n")

best_lever, best_offset_list, inversion_result_list, rmse_list = (
    simulated_annealing_bias_joint(
        300,
        CDOG_data_list,
        GPS_data,
        GPS_Coordinates,
        GPS1_TO_OTHERS,
        initial_guess_list,
        initial_lever_guess,
        dz_array,
        angle_array,
        esv_matrix,
        initial_offset_list=initial_offset_list,
        real_data=True,
        z_sample=True,
        clock_drift=clock_drift,
    )
)


# Summary + diagnostics
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
    if clock_drift:
        print(f"Drift rate: {inversion_result[5]:.4e} s/s (Eq. 6 convention)")
    print(
        f"Offset: {np.round(best_offset_eff, 4)}, "
        f"Inversion Guess (from base): {np.round(inversion_guess - CDOG_REFERENCE, 5)}"
    )
    print("RMSE:", np.round(rmse_list[i] * 100 * NOMINAL_SOUND_SPEED_MPS, 3), "cm")

print(
    "\nJoint objective (sum of DOG RMSE, cm):",
    np.round(joint_rmse_cm * 100 * NOMINAL_SOUND_SPEED_MPS, 3),
)
