import numpy as np
from numba import njit

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    modelled_times,
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import (
    transponder_from_rotations,
    vessel_rotations,
)
from Inversion_Workflow.Inversion.Numba_xAline import (
    find_subint_offset,
    two_pointer_index,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    initial_bias_geiger,
)


@njit(cache=True)
def _evaluate_single_dog(
    inversion_guess,
    time_bias,
    esv_bias,
    offset,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data,
    clock_drift=False,
    drift_rate=0.0,
):
    """Final fit of one DOG at a fixed offset.

    Returns ``(estimate, rmse, CDOG_full, GPS_full, CDOG_clock, GPS_clock)``, RMSE
    in s.
    """
    (
        inversion_estimate,
        CDOG_full,
        GPS_full,
        CDOG_clock,
        GPS_clock,
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

    rmse = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
    return inversion_estimate, rmse, CDOG_full, GPS_full, CDOG_clock, GPS_clock


@njit(cache=True)
def simulated_annealing_bias_joint(
    iter,
    CDOG_data_array,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    initial_guess_list,
    initial_lever,
    dz_array,
    angle_array,
    esv_matrix,
    initial_offset_list,
    real_data=False,
    z_sample=True,
    clock_drift=False,
):
    """Simulated annealing of one lever arm shared by all DOGs.

    Each DOG keeps its own position, time bias, ESV bias and offset, and with
    ``clock_drift`` a ``drift_rate`` (s/s) in the sixth column of the estimates. The
    objective is the sum of the DOG RMSEs; the moves are those of
    ``simulated_annealing_bias``. ``CDOG_data_array`` is a typed List of (N_j, 2)
    arrival arrays; ``GPS_data`` and ``GPS_Coordinates`` are shared;
    ``initial_guess_list`` is (n_dogs, 3) and ``initial_offset_list`` (n_dogs,).
    Returns ``(lever, offsets, estimates, rmses)``.
    """
    n_dogs = len(CDOG_data_array)
    n_params = 6 if clock_drift else 5

    # Shared lever arm
    best_lever = initial_lever.copy()
    best_total_rmse = np.inf

    # Per-DOG state
    inversion_guesses = np.empty((n_dogs, 3))
    inversion_estimates = np.empty((n_dogs, n_params))
    time_biases = np.zeros(n_dogs)
    esv_biases = np.zeros(n_dogs)
    drift_rates = np.zeros(n_dogs)
    offsets = initial_offset_list.copy()

    # The vessel attitude does not change with the lever arm: fit it once
    rotations, displacements = vessel_rotations(GPS_Coordinates, gps1_to_others)

    # Alignment and fit of every DOG at the initial lever arm
    transponder_coordinates = transponder_from_rotations(
        rotations, displacements, best_lever
    )

    total_rmse = 0.0
    for j in range(n_dogs):
        inversion_estimate_j, offset_j = initial_bias_geiger(
            initial_guess_list[j],
            CDOG_data_array[j],
            GPS_data,
            transponder_coordinates,
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
            clock_drift=clock_drift,
        )

        inversion_estimates[j] = inversion_estimate_j
        inversion_guesses[j] = inversion_estimate_j[:3]
        time_biases[j] = inversion_estimate_j[3]
        esv_biases[j] = inversion_estimate_j[4]
        if clock_drift:
            drift_rates[j] = inversion_estimate_j[5]
        offsets[j] = offset_j

        times_guess, esv = modelled_times(
            inversion_guesses[j],
            transponder_coordinates,
            esv_biases[j],
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
        )

        CDOG_full, GPS_clock, GPS_full = two_pointer_index(
            offsets[j],
            0.5,
            CDOG_data_array[j],
            GPS_data,
            times_guess,
            transponder_coordinates,
            esv,
        )[1:4]

        rmse_j = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
        total_rmse += rmse_j

        print(
            "Initial DOG",
            j,
            "RMSE (cm):",
            np.round(rmse_j * 100.0 * NOMINAL_SOUND_SPEED_MPS, 2),
            "Offset:",
            np.round(offsets[j], 5),
        )

    best_total_rmse = total_rmse

    print(
        "Initial joint RMSE (cm):",
        np.round(best_total_rmse * 100.0 * NOMINAL_SOUND_SPEED_MPS, 2),
    )
    print("Initial lever:", np.round(best_lever, 3))

    # Save best per-DOG state
    best_inversion_guesses = inversion_guesses.copy()
    best_time_biases = time_biases.copy()
    best_esv_biases = esv_biases.copy()
    best_drift_rates = drift_rates.copy()
    best_offsets = offsets.copy()

    # Annealing
    k = 0
    while k < 300:
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / iter))
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = best_lever + displacement

        transponder_coordinates = transponder_from_rotations(
            rotations, displacements, lever
        )

        candidate_total_rmse = 0.0
        candidate_inversion_guesses = np.empty((n_dogs, 3))
        candidate_inversion_estimates = np.empty((n_dogs, n_params))
        candidate_rmse_list = np.empty(n_dogs)

        for j in range(n_dogs):
            (
                inversion_estimate_j,
                rmse_j,
                _CDOG_full_j,
                _GPS_full_j,
                _CDOG_clock_j,
                _GPS_clock_j,
            ) = _evaluate_single_dog(
                best_inversion_guesses[j],
                best_time_biases[j],
                best_esv_biases[j],
                best_offsets[j],
                CDOG_data_array[j],
                GPS_data,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
                real_data,
                clock_drift,
                best_drift_rates[j],
            )

            candidate_inversion_estimates[j] = inversion_estimate_j
            candidate_inversion_guesses[j] = inversion_estimate_j[:3]
            candidate_rmse_list[j] = rmse_j
            candidate_total_rmse += rmse_j

        if candidate_total_rmse < best_total_rmse:
            best_total_rmse = candidate_total_rmse
            best_lever = lever.copy()

            best_inversion_guesses = candidate_inversion_guesses.copy()

            for j in range(n_dogs):
                best_time_biases[j] = candidate_inversion_estimates[j, 3]
                best_esv_biases[j] = candidate_inversion_estimates[j, 4]
                if clock_drift:
                    best_drift_rates[j] = candidate_inversion_estimates[j, 5]

        if k % 10 == 0:
            print("\nIteration", k)
            print("Current Lever:", np.round(lever, 3))
            print("Best Lever:", np.round(best_lever, 3))
            print(
                "Current Total RMSE (cm):",
                np.round(candidate_total_rmse * 100.0 * NOMINAL_SOUND_SPEED_MPS, 2),
            )
            print(
                "Best Total RMSE (cm):",
                np.round(best_total_rmse * 100.0 * NOMINAL_SOUND_SPEED_MPS, 2),
            )
            print("Best Offsets:", np.round(best_offsets, 3))

        if k % 50 == 0 and k > 0:
            best_transponder_coordinates = transponder_from_rotations(
                rotations, displacements, best_lever
            )

            for j in range(n_dogs):
                times_guess, esv = modelled_times(
                    best_inversion_guesses[j],
                    best_transponder_coordinates,
                    best_esv_biases[j],
                    dz_array,
                    angle_array,
                    esv_matrix,
                    real_data,
                )

                best_offsets[j] = find_subint_offset(
                    best_offsets[j],
                    CDOG_data_array[j],
                    GPS_data,
                    times_guess,
                    best_transponder_coordinates,
                    esv,
                )
        k += 1

    # Vertical scan of the lever arm
    if z_sample:
        print("\nStarting Z-scan around best lever...\n")
        best_lever_new = best_lever.copy()

        for dz in np.arange(-5, 5, 0.1):
            lever = best_lever + np.array([0.0, 0.0, dz])
            transponder_coordinates = transponder_from_rotations(
                rotations, displacements, lever
            )

            candidate_total_rmse = 0.0
            candidate_inversion_guesses = np.empty((n_dogs, 3))
            candidate_inversion_estimates = np.empty((n_dogs, n_params))

            for j in range(n_dogs):
                (
                    inversion_estimate_j,
                    rmse_j,
                    _CDOG_full_j,
                    _GPS_full_j,
                    _CDOG_clock_j,
                    _GPS_clock_j,
                ) = _evaluate_single_dog(
                    best_inversion_guesses[j],
                    best_time_biases[j],
                    best_esv_biases[j],
                    best_offsets[j],
                    CDOG_data_array[j],
                    GPS_data,
                    transponder_coordinates,
                    dz_array,
                    angle_array,
                    esv_matrix,
                    real_data,
                    clock_drift,
                    best_drift_rates[j],
                )

                candidate_inversion_estimates[j] = inversion_estimate_j
                candidate_inversion_guesses[j] = inversion_estimate_j[:3]
                candidate_total_rmse += rmse_j

            if candidate_total_rmse < best_total_rmse:
                best_total_rmse = candidate_total_rmse
                best_lever_new = lever.copy()
                best_inversion_guesses = candidate_inversion_guesses.copy()

                for j in range(n_dogs):
                    best_time_biases[j] = candidate_inversion_estimates[j, 3]
                    best_esv_biases[j] = candidate_inversion_estimates[j, 4]
                    if clock_drift:
                        best_drift_rates[j] = candidate_inversion_estimates[j, 5]

        best_lever = best_lever_new

    # Final fit of every DOG at the best lever arm
    best_transponder_coordinates = transponder_from_rotations(
        rotations, displacements, best_lever
    )

    final_inversion_estimates = np.empty((n_dogs, n_params))
    final_rmses = np.empty(n_dogs)

    for j in range(n_dogs):
        (
            inversion_estimate_j,
            rmse_j,
            _CDOG_full_j,
            _GPS_full_j,
            _CDOG_clock_j,
            _GPS_clock_j,
        ) = _evaluate_single_dog(
            best_inversion_guesses[j],
            best_time_biases[j],
            best_esv_biases[j],
            best_offsets[j],
            CDOG_data_array[j],
            GPS_data,
            best_transponder_coordinates,
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
            clock_drift,
            best_drift_rates[j],
        )

        final_inversion_estimates[j] = inversion_estimate_j
        final_rmses[j] = rmse_j
    return best_lever, best_offsets, final_inversion_estimates, final_rmses
