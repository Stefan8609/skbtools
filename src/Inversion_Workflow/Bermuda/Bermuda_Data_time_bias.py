import numpy as np
import os
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    gauss_newton_offset_search,
    initial_bias_geiger,
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from bermuda_data import (
    CDOG_GUESS_AUGMENTS,
    CDOG_REFERENCE,
    GPS1_TO_OTHERS,
    OFFSET_HINTS_S,
    gps_data_path,
    gps_output_path,
    leg_mask,
)
from plotting.Plot_Modular import time_series_plot, range_residual


# -------------------------
# Config
# -------------------------
DOG_num = 4
# Lever arm and clock offset:
#   "gauss_newton": integer second and lever arm by the Gauss-Newton offset search
#       (no hint used; the found integer must match the hint below).
#   "annealing": simulated annealing at the hinted integer, as in the submitted
#       paper; kept as a validation of "gauss_newton".
#   "fixed": lever arm fixed at the initial guess, Geiger inversion only.
lever_method = "gauss_newton"
# Paper I (Table 1, Figs. 5, 6, C12-C15) uses the full trajectory. The leg mask
# (9-11 h and 12.4-15 h) is a sensitivity test: it lowers c*sigma to ~8 cm but
# moves the DOG depths by metres.
use_leg_mask = False
save = True
downsample = 1
# Linear clock drift (theme T5), as config["clock_drift"] in the MCMC. The estimate
# then gets drift_rate (s/s, Eq. 6 convention) as a sixth entry, and the reported
# offset is dt0, the offset at t_mid (half-way between the first and last paired
# GPS epochs).
clock_drift = False

# -------------------------
# Known data
# -------------------------

dz_array, angle_array, esv_matrix = load_esv_table(
    gps_data_path("ESV_Tables/global_table_esv_extended.mat")
)

data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
GPS_Coordinates = data["GPS_Coordinates"]
GPS_data = data["GPS_data"]
CDOG_data = data["CDOG_data"][::downsample]

initial_lever_guess = np.array([-12.69796149, 9.51739301, -15.0743129])

CDOG_guess = CDOG_REFERENCE + CDOG_GUESS_AUGMENTS[DOG_num]
offset_hint = OFFSET_HINTS_S[DOG_num]

if use_leg_mask:
    # Same legs for every DOG. Earlier alternative for DOGs 3 and 4:
    # 8.7-11 h and 15.75-18.15 h (not a true leg for DOG 4, as the GPS data
    # were corrupted overlying DOG 4).
    keep = leg_mask(GPS_data)
    GPS_Coordinates = GPS_Coordinates[keep]
    GPS_data = GPS_data[keep]

if lever_method == "gauss_newton":
    print("\n ---------Using Gauss-Newton Offset Search--------- \n")
    best_lever, best_offset, inversion_result, offset_diagnostics = (
        gauss_newton_offset_search(
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            GPS1_TO_OTHERS,
            CDOG_guess,
            initial_lever_guess,
            dz_array,
            angle_array,
            esv_matrix,
            real_data=True,
            clock_drift=clock_drift,
        )
    )
    print(
        f"Offset {best_offset:.5f} s, lever {np.round(best_lever, 3)}, "
        f"delta chi^2 to second best {offset_diagnostics['delta_chi2']:.0f}"
    )
    if np.round(best_offset) != offset_hint:
        raise ValueError(
            f"Gauss-Newton offset search found integer second "
            f"{np.round(best_offset):.0f} s for DOG {DOG_num}, expected the hint "
            f"{offset_hint:.0f} s"
        )

    transponder_coordinates = findTransponder(
        GPS_Coordinates, GPS1_TO_OTHERS, best_lever
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
        clock_drift=clock_drift,
        drift_rate=inversion_result[5] if clock_drift else 0.0,
    )

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
elif lever_method == "annealing":
    print("\n ---------Using Simulated Annealing--------- \n")
    best_lever, best_offset, inversion_result = simulated_annealing_bias(
        300,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        GPS1_TO_OTHERS,  # keep consistent with how data were generated
        CDOG_guess,
        initial_lever_guess,
        dz_array,
        angle_array,
        esv_matrix,
        initial_offset=offset_hint,
        real_data=True,
        z_sample=True,
        clock_drift=clock_drift,
    )

    transponder_coordinates = findTransponder(
        GPS_Coordinates, GPS1_TO_OTHERS, best_lever
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
        clock_drift=clock_drift,
        drift_rate=inversion_result[5] if clock_drift else 0.0,
    )

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
elif lever_method == "fixed":
    # -------------------------
    # Inversion: Initial → Final
    # -------------------------
    print("\n ---------Using Geiger Inversion--------- \n")

    transponder_coordinates = findTransponder(
        GPS_Coordinates, GPS1_TO_OTHERS, initial_lever_guess
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
        clock_drift=clock_drift,
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
        clock_drift=clock_drift,
        drift_rate=inversion_result[5] if clock_drift else 0.0,
    )
    print("\nFinal complete.")

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    best_lever = initial_lever_guess
else:
    raise ValueError(
        "lever_method must be 'gauss_newton', 'annealing' or 'fixed', "
        f"got {lever_method!r}"
    )


# -------------------------
# Summary + diagnostics
# -------------------------
# Keep your original convention (fold time_bias into offset for reporting)
best_offset_eff = best_offset - time_bias
drift_rate = inversion_result[5] if clock_drift else 0.0

diff_data_ms = (CDOG_full - GPS_full) * 1000.0
RMSE_cm = (
    (np.sqrt(np.nanmean(diff_data_ms**2)) / 1000.0) * NOMINAL_SOUND_SPEED_MPS * 100.0
)

print(f"\nEstimate: {np.round(inversion_result, 4)}")
print(
    f"Best Lever: {np.round(best_lever, 3)}, "
    f"Offset: {np.round(best_offset_eff, 4)}, "
    f"Inversion Guess (from base): {np.round(inversion_guess - CDOG_REFERENCE, 5)}"
)
print("RMSE:", np.round(RMSE_cm, 3), "cm")
if clock_drift:
    print(f"Drift rate: {drift_rate:.4e} s/s (Eq. 6 convention)")


# -------------------------
# Plots
# -------------------------
DOG_name = {1: "1", 3: "3", 4: "4"}
save_path = "Figs/Inversion_Workflow"
file_tag = (
    DOG_name[DOG_num]
    + {"gauss_newton": "_GN", "annealing": "_SimAnn", "fixed": "_Geiger"}[lever_method]
)

time_series_plot(
    CDOG_clock,
    CDOG_full,
    GPS_clock,
    GPS_full,
    save=save,
    path=save_path,
    DOG_num=file_tag,
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
    DOG_num=file_tag,
)

# Save Data for Easy replotting

# -------------------------
# Save plotting inputs for later font-size experiments
# -------------------------
plot_data_dir = gps_output_path("Plot_Data")
plot_data_path = f"{plot_data_dir}/integer_pick_metrics_DOG{DOG_num}.npz"

if save:
    os.makedirs(plot_data_dir, exist_ok=True)

    np.savez_compressed(
        plot_data_path,
        CDOG_clock=CDOG_clock,
        CDOG_full=CDOG_full,
        GPS_clock=GPS_clock,
        GPS_full=GPS_full,
        best_offset=best_offset,
        best_lever=best_lever,
        inversion_result=inversion_result,
        esv_bias=esv_bias,
        time_bias=time_bias,
        clock_drift=clock_drift,
        drift_rate_s_per_s=drift_rate,
        CDOG_data=CDOG_data,
        GPS_data=GPS_data,
        times_guess=times_guess,
        esv=esv,
        transponder_coordinates=transponder_coordinates,
        DOG_num=DOG_num,
    )

    print(f"Saved integer-pick metric plot data to: {plot_data_path}")

# -------------------------
# Save small repo sample
# -------------------------
save_repo_sample = True

if save_repo_sample:
    repo_sample_dir = gps_output_path("Paper_Data")
    os.makedirs(repo_sample_dir, exist_ok=True)

    n_repo_samples = 1000

    # Use the first 1000 samples, preserving the same imported-file structure
    GPS_Coordinates_sample = GPS_Coordinates[:n_repo_samples]
    GPS_data_sample = GPS_data[:n_repo_samples]
    CDOG_data_sample = CDOG_data[:n_repo_samples]

    repo_sample_path = os.path.join(
        repo_sample_dir,
        f"Processed_GPS_Receivers_DOG_{DOG_num}_sample.npz",
    )

    np.savez_compressed(
        repo_sample_path,
        GPS_Coordinates=GPS_Coordinates_sample,
        GPS_data=GPS_data_sample,
        CDOG_data=CDOG_data_sample,
    )

    print(f"Saved repo sample data to: {repo_sample_path}")
