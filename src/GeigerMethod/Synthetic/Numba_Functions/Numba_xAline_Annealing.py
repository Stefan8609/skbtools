import numpy as np

from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import (
    two_pointer_index,
    find_subint_offset,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline_Geiger import (
    initial_geiger,
    transition_geiger,
    final_geiger,
    calculateTimesRayTracing,
    findTransponder,
    calculateTimesRayTracingReal,
)
from GeigerMethod.Synthetic.Numba_Functions.Generate_Unaligned_Realistic import (
    generateUnalignedRealistic,
)

"""
Maybe try to implement the more complicated schematic
"""


def simulated_annealing(
    iter,
    CDOG_data,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    initial_guess,
    initial_lever,
    initial_offset=0,
    real_data=False,
):
    """Estimate lever arm and receiver location via simulated annealing."""

    # Initialize variables
    status = "int"
    old_offset = initial_offset
    inversion_guess = initial_guess

    transponder_coordinates_found = findTransponder(
        GPS_Coordinates, gps1_to_others, initial_lever
    )
    if not real_data:
        times_guess, esv = calculateTimesRayTracing(
            inversion_guess, transponder_coordinates_found
        )
    else:
        times_guess, esv = calculateTimesRayTracingReal(
            inversion_guess, transponder_coordinates_found
        )
    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        initial_offset,
        0.5,
        CDOG_data,
        GPS_data,
        times_guess,
        transponder_coordinates_found,
        esv,
    )

    best_rmse = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
    best_lever = initial_lever
    k = 0
    while k < 300:
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / (iter)))
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = best_lever + displacement

        transponder_coordinates_found = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )

        if status == "int":
            inversion_guess, offset = initial_geiger(
                inversion_guess,
                CDOG_data,
                GPS_data,
                transponder_coordinates_found,
                real_data,
            )
            if offset == old_offset:
                status = "subint"
        elif status == "subint":
            inversion_guess, offset = transition_geiger(
                inversion_guess,
                CDOG_data,
                GPS_data,
                transponder_coordinates_found,
                offset,
                real_data,
            )
            status = "constant"
        else:
            if k == 100 or k == 200:
                transponder_coordinates_found = findTransponder(
                    GPS_Coordinates, gps1_to_others, best_lever
                )
                inversion_guess, offset = transition_geiger(
                    inversion_guess,
                    CDOG_data,
                    GPS_data,
                    transponder_coordinates_found,
                    offset,
                    real_data,
                )
            else:
                (
                    inversion_guess,
                    CDOG_full,
                    GPS_full,
                    CDOG_clock,
                    GPS_clock,
                ) = final_geiger(
                    inversion_guess,
                    CDOG_data,
                    GPS_data,
                    transponder_coordinates_found,
                    offset,
                    real_data,
                )

        if not real_data:
            times_guess, esv = calculateTimesRayTracing(
                inversion_guess, transponder_coordinates_found
            )
        else:
            times_guess, esv = calculateTimesRayTracingReal(
                inversion_guess, transponder_coordinates_found
            )
        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset,
            0.5,
            CDOG_data,
            GPS_data,
            times_guess,
            transponder_coordinates_found,
            esv,
        )

        RMSE = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever

        if k % 10 == 0:
            print(k, RMSE * 100 * 1515, offset, lever)
        old_offset = offset
        k += 1

    return best_lever, offset, inversion_guess


if __name__ == "__main__":
    true_offset = np.random.rand() * 9000 + 1000
    print(true_offset)
    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = generateUnalignedRealistic(10000, time_noise, true_offset)
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    initial_guess = CDOG + np.array([100, -100, -50], dtype=np.float64)
    initial_lever = np.array([-5.0, 7.0, -10.0], dtype=np.float64)

    lever, offset, inversion_guess = simulated_annealing(
        300,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others,
        initial_guess,
        initial_lever,
    )

    lever, offset, inversion_guess = simulated_annealing(
        300,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others,
        inversion_guess,
        lever,
        offset,
    )

    transponder_coordinates_found = findTransponder(
        GPS_Coordinates, gps1_to_others, lever
    )
    times_found, esv = calculateTimesRayTracing(
        inversion_guess, transponder_coordinates_found
    )

    offset = find_subint_offset(
        offset, CDOG_data, GPS_data, times_found, transponder_coordinates_found, esv
    )

    inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(
        inversion_guess, CDOG_data, GPS_data, transponder_coordinates_found, offset
    )

    print(np.linalg.norm(inversion_guess - CDOG) * 100, "cm")
    print("Found Lever", lever, "Found Offset", offset)
    print("Actual Lever", gps1_to_transponder, "Actual Offset", true_offset)
