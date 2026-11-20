import numpy as np


from Inversion_Workflow.Inversion.Numba_xAline import (
    filter_close_receptions,
    find_subint_offset,
    two_pointer_index,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    modelled_times,
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    initial_bias_geiger,
    final_bias_geiger,
    misfit_rms,
    INT_OFFSET_SHIFT_SECONDS,
)
from numba import njit


@njit()
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
    clock_drift=False,
):
    """Lever arm, offset and estimate by simulated annealing (Eqs. A28-A29).

    300 lever-arm moves U(-Theta, Theta) m with Theta = exp(-7 k / iter) around the
    best lever arm; a move is kept only if it lowers the RMSE, so A30 (accepting a
    worse move) is not implemented. The fraction of the offset is refit every 50
    moves; ``z_sample`` then scans the lever-arm z over +-5 m in 0.1 m steps.
    Returns ``(lever, offset, estimate)``. With ``clock_drift`` the estimate carries
    ``drift_rate`` (s/s, Eq. 6 convention) as a sixth entry, as in
    ``final_bias_geiger``.
    """
    inversion_guess = initial_guess
    time_bias = 0.0
    esv_bias = 0.0
    drift_rate = 0.0

    best_lever = initial_lever
    best_rmse = np.inf

    # Alignment and fit at the initial lever arm
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
        initial_offset,
        clock_drift,
    )

    inversion_guess = inversion_estimate[:3]
    time_bias = inversion_estimate[3]
    esv_bias = inversion_estimate[4]
    if clock_drift:
        drift_rate = inversion_estimate[5]

    times_guess, esv = modelled_times(
        inversion_guess,
        transponder_coordinates,
        esv_bias,
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
    )

    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset, 0.5, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
    )[1:4]
    best_rmse = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))

    # Annealing
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
            offset,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
            clock_drift,
            drift_rate,
        )

        RMSE = np.sqrt(np.nanmean((GPS_full_k - CDOG_full_k) ** 2))

        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever

            inversion_estimate = inversion_estimate_k
            inversion_guess = inversion_estimate_k[:3]
            time_bias = inversion_estimate_k[3]
            esv_bias = inversion_estimate_k[4]
            if clock_drift:
                drift_rate = inversion_estimate_k[5]

        if k % 10 == 0:
            print(
                k,
                np.round(RMSE * 100 * NOMINAL_SOUND_SPEED_MPS, 2),
                np.round(offset, 5),
                np.round(lever, 3),
            )
        if k % 50 == 0 and k > 0:
            best_transponder_coordinates = findTransponder(
                GPS_Coordinates, gps1_to_others, best_lever
            )

            times_guess, esv = modelled_times(
                inversion_guess,
                best_transponder_coordinates,
                esv_bias,
                dz_array,
                angle_array,
                esv_matrix,
                real_data,
            )
            offset = find_subint_offset(
                offset,
                filter_close_receptions(CDOG_data),
                GPS_data,
                times_guess,
                best_transponder_coordinates,
                esv,
            )
            print("    Updated offset:", np.round(offset, 5))

        k += 1

    # Vertical scan of the lever arm
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
                clock_drift,
                drift_rate,
            )

            RMSE = np.sqrt(np.nanmean((GPS_full_k - CDOG_full_k) ** 2))

            if RMSE < best_rmse:
                best_rmse = RMSE
                best_lever_new = lever

                inversion_estimate = inversion_estimate_k
                inversion_guess = inversion_estimate_k[:3]
                time_bias = inversion_estimate_k[3]
                esv_bias = inversion_estimate_k[4]
                if clock_drift:
                    drift_rate = inversion_estimate_k[5]

        best_lever = best_lever_new

    # Final fit at the best lever arm
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
        clock_drift,
        drift_rate,
    )

    return best_lever, offset, inversion_estimate


@njit()
def anneal_with_offset_search(
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
    real_data=False,
    z_sample=True,
    clock_drift=False,
):
    """Anneal when the DOG clock offset is unknown.

    A wrong integer second is partly absorbed by the lever arm during annealing, so
    the fit cannot tell it from the right one until the lever arm is good. The data
    are aligned at the initial lever arm, every integer second within
    ``INT_OFFSET_SHIFT_SECONDS`` of that offset is annealed in full, and the lowest
    final RMS wins. Returns ``(lever, offset, estimate)`` like
    ``simulated_annealing_bias``.

    The workflow uses ``gauss_newton_offset_search`` instead (it solves the lever
    arm exactly at every integer second); this is kept to validate it.
    """
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, initial_lever
    )
    aligned_estimate, aligned_offset = initial_bias_geiger(
        initial_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
        clock_drift=clock_drift,
    )

    best_rmse = np.inf
    best_lever = initial_lever
    best_offset = aligned_offset
    best_estimate = aligned_estimate
    for shift in range(-INT_OFFSET_SHIFT_SECONDS, INT_OFFSET_SHIFT_SECONDS + 1):
        lever, offset, estimate = simulated_annealing_bias(
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
            initial_offset=aligned_offset + shift,
            real_data=real_data,
            z_sample=z_sample,
            clock_drift=clock_drift,
        )
        rmse = misfit_rms(
            estimate,
            offset,
            CDOG_data,
            GPS_data,
            findTransponder(GPS_Coordinates, gps1_to_others, lever),
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
            clock_drift,
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_lever = lever
            best_offset = offset
            best_estimate = estimate
    return best_lever, best_offset, best_estimate
