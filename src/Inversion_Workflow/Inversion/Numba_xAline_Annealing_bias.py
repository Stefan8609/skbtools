import numpy as np
import scipy.io as sio

from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from Inversion_Workflow.Synthetic.Generate_Unaligned import (
    generateUnaligned,
)

from Inversion_Workflow.Inversion.Numba_xAline import (
    two_pointer_index,
    find_subint_offset,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    initial_bias_geiger,
    final_bias_geiger,
)
from plotting.Plot_Modular import time_series_plot
from data import gps_data_path
from numba import njit


@njit(cache=True)
def simulated_annealing_bias(
    iter,
    CDOG_data,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    initial_guess,
    initial_lever,
    dz_array,
    angle_array,
    esv_matrix,
    initial_offset=0,
    real_data=False,
    z_sample=True,
):
    """Estimate lever arm and biases using simulated annealing."""
    # Working state
    inversion_guess = initial_guess
    time_bias = 0.0
    esv_bias = 0.0
    offset = initial_offset

    best_lever = initial_lever
    best_rmse = np.inf

    # ---- initial solve at initial lever ----
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, best_lever
    )

    inversion_estimate, offset = initial_bias_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
    )

    inversion_guess = inversion_estimate[:3]
    time_bias = inversion_estimate[3]
    esv_bias = inversion_estimate[4]

    # Evaluate RMSE at this solution
    if not real_data:
        times_guess, esv = calculateTimesRayTracing_Bias(
            inversion_guess,
            transponder_coordinates,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            inversion_guess,
            transponder_coordinates,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )

    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset, 0.5, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
    )[1:4]
    best_rmse = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))

    # ---- annealing loop ----
    k = 0
    while k < 300:
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / (iter)))
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = best_lever + displacement

        transponder_coordinates = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )

        (
            inversion_estimate_k,
            CDOG_full_k,
            GPS_full_k,
            CDOG_clock_k,
            GPS_clock_k,
        ) = final_bias_geiger(
            inversion_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            offset,  # fixed offset during SA
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
        )

        RMSE = np.sqrt(np.nanmean((GPS_full_k - CDOG_full_k) ** 2))

        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever

            # update "best" state to this candidate
            inversion_estimate = inversion_estimate_k
            inversion_guess = inversion_estimate_k[:3]
            time_bias = inversion_estimate_k[3]
            esv_bias = inversion_estimate_k[4]

        if k % 10 == 0:
            print(
                k,
                np.round(RMSE * 100 * 1515, 2),
                np.round(offset, 5),
                np.round(lever, 3),
            )
        if k % 50 == 0 and k > 0:
            if not real_data:
                times_guess, esv = calculateTimesRayTracing_Bias(
                    inversion_guess,
                    transponder_coordinates,
                    esv_bias,
                    dz_array,
                    angle_array,
                    esv_matrix,
                )
            else:
                times_guess, esv = calculateTimesRayTracing_Bias_Real(
                    inversion_guess,
                    transponder_coordinates,
                    esv_bias,
                    dz_array,
                    angle_array,
                    esv_matrix,
                )
            offset = find_subint_offset(
                offset,
                CDOG_data,
                GPS_data,
                times_guess,
                transponder_coordinates,
                esv,
            )
            print("    Updated offset:", np.round(offset, 5))

        k += 1

    # ---- optional z scan around best lever ----
    if z_sample:
        print("\nStarting Z-scan around best lever...\n")
        best_lever_new = best_lever
        for dz in np.arange(-5, 5, 0.1):
            lever = best_lever + np.array([0.0, 0.0, dz])
            transponder_coordinates = findTransponder(
                GPS_Coordinates, gps1_to_others, lever
            )

            (
                inversion_estimate_k,
                CDOG_full_k,
                GPS_full_k,
                CDOG_clock_k,
                GPS_clock_k,
            ) = final_bias_geiger(
                inversion_guess,
                CDOG_data,
                GPS_data,
                transponder_coordinates,
                offset,
                esv_bias,
                time_bias,
                dz_array,
                angle_array,
                esv_matrix,
                real_data,
            )

            RMSE = np.sqrt(np.nanmean((GPS_full_k - CDOG_full_k) ** 2))

            if RMSE < best_rmse:
                best_rmse = RMSE
                best_lever_new = lever

                inversion_estimate = inversion_estimate_k
                inversion_guess = inversion_estimate_k[:3]
                time_bias = inversion_estimate_k[3]
                esv_bias = inversion_estimate_k[4]

        best_lever = best_lever_new

    # Final solve at best lever (keep your original final_bias_geiger usage)
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, best_lever
    )
    inversion_estimate, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        esv_bias,
        time_bias,
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
    )

    return best_lever, offset, inversion_estimate


if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio

    # --- Load ESV lookup table ---
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    # --- Synthetic settings ---
    position_noise = 2e-2
    time_noise = 2e-5

    true_esv_bias = 3.0
    true_time_bias = 0.0
    true_offset = np.random.rand() * 10000

    gps1_to_others_true = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )
    true_lever = np.array([-12.48862757, 0.22622633, -15.89601934])

    print("True Offset:", true_offset)
    print("True Lever :", np.round(true_lever, 4))

    # --- Generate data (same style as your GN test) ---
    type = "bermuda"  # or "unaligned"

    if type == "bermuda":
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = bermuda_trajectory(
            time_noise,
            position_noise,
            true_esv_bias,
            true_time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            offset=true_offset,
            gps1_to_others=gps1_to_others_true,
            gps1_to_transponder=true_lever,
        )
        real = True

    if type == "unaligned":
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = generateUnaligned(
            20000,
            time_noise,
            position_noise,
            true_offset,
            true_esv_bias,
            true_time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            gps1_to_others=gps1_to_others_true,
            gps1_to_transponder=true_lever,
        )
        real = False

    # --- Starting guesses for annealing ---
    initial_lever = true_lever + (np.random.rand(3) * 6.0 - 3.0)  # +/-3 m perturbation
    CDOG_position_adjustment = (np.random.rand(3) * 400.0) - 200.0
    initial_guess = CDOG + CDOG_position_adjustment

    print("\nInitial lever guess:", np.round(initial_lever, 4))
    print("CDOG position adjustment (m):", np.round(CDOG_position_adjustment, 2))
    print("Initial position guess:", np.round(initial_guess, 2))

    # --- Annealing params ---
    SA_iter = 300  # temperature schedule control (your "iter" argument)
    initial_offset = 0.0  # let initial_bias_geiger find it inside annealing
    z_sample = True

    print("\n------ Starting Simulated Annealing ------\n")
    lever_est, offset_est, inversion_est = simulated_annealing_bias(
        SA_iter,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others_true,  # keep consistent with how data were generated
        initial_guess,
        initial_lever,
        dz_array,
        angle_array,
        esv_matrix,
        initial_offset=initial_offset,
        real_data=real,
        z_sample=z_sample,
    )

    est_xyz = inversion_est[:3]
    est_time_bias = inversion_est[3]
    est_esv_bias = inversion_est[4]

    print("\n------ Annealing Results ------")
    print("Estimated lever:", np.round(lever_est, 4))
    print("True lever     :", np.round(true_lever, 4))
    print("Lever error (m):", np.round(lever_est - true_lever, 4))

    print("\nEstimated offset:", float(offset_est))
    print("True offset     :", float(true_offset))
    print("Offset error (s):", float(offset_est - true_offset))

    print("\nTrue CDOG (m):", np.round(CDOG, 2))
    print("Estimated xyz:", np.round(est_xyz, 2))
    print("XYZ error (cm):", np.linalg.norm(est_xyz - CDOG) * 100.0)

    print("\nEstimated time_bias:", float(est_time_bias))
    print("Estimated esv_bias :", float(est_esv_bias))

    # --- Final evaluation / plot (same pattern as GN test) ---
    transponder_coordinates_est = findTransponder(
        GPS_Coordinates, gps1_to_others_true, lever_est
    )

    inversion_final, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
        est_xyz,
        CDOG_data,
        GPS_data,
        transponder_coordinates_est,
        offset_est,
        est_esv_bias,
        est_time_bias,
        dz_array,
        angle_array,
        esv_matrix,
        real_data=real,
    )

    rmse = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2)) * 1515 * 100
    print("\nFinal RMSE (cm):", float(rmse))

    time_series_plot(
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise
    )
