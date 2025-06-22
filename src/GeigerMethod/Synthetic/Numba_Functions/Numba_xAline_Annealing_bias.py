import numpy as np
import scipy.io as sio

from Bermuda_Trajectory import bermuda_trajectory
from Numba_xAline import two_pointer_index
from Numba_time_bias import calculateTimesRayTracing_Bias
from Numba_Geiger import findTransponder
from Numba_xAline_bias import (
    initial_bias_geiger,
    transition_bias_geiger,
    final_bias_geiger,
    calculateTimesRayTracing_Bias_Real,
)
from Plot_Modular import time_series_plot

"""
Incorporate simulated annealing to find
the transducer location in addition to the bias terms.
"""


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
    enforce_offset=False,
    z_sample=False,
):
    """Estimate lever arm and biases using simulated annealing."""
    # Initialize variables
    status = "int"
    if enforce_offset:
        status = "subint"
        offset = initial_offset
    old_offset = initial_offset

    inversion_guess = initial_guess
    time_bias = 0.0
    esv_bias = 0.0
    inversion_estimate = np.array(
        [initial_guess[0], initial_guess[1], initial_guess[2], time_bias, esv_bias]
    )

    transponder_coordinates_found = findTransponder(
        GPS_Coordinates, gps1_to_others, initial_lever
    )
    if not real_data:
        times_guess, esv = calculateTimesRayTracing_Bias(
            initial_guess,
            transponder_coordinates_found,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            initial_guess,
            transponder_coordinates_found,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
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

        transponder_coordinates = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        if status == "int":
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
            if offset == old_offset:
                status = "subint"
        elif status == "subint":
            inversion_estimate, offset = transition_bias_geiger(
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
            status = "constant"
        else:
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
            )

        inversion_guess = inversion_estimate[:3]
        time_bias = inversion_estimate[3]
        esv_bias = inversion_estimate[4]

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
        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset, 0.5, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )

        RMSE = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever

        if k % 10 == 0:
            print(
                k,
                np.round(RMSE * 100 * 1515, 2),
                np.round(offset, 5),
                np.round(lever, 3),
            )
        old_offset = offset
        k += 1

    # Sample z values in case where it is poorly constrained
    if z_sample:
        best_lever_new = best_lever
        for dz in np.arange(-5, 5, 0.1):
            lever = best_lever + np.array([0.0, 0.0, dz])
            transponder_coordinates = findTransponder(
                GPS_Coordinates, gps1_to_others, lever
            )
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
            )
            inversion_guess = inversion_estimate[:3]
            time_bias = inversion_estimate[3]
            esv_bias = inversion_estimate[4]

            RMSE = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
            if RMSE < best_rmse:
                best_rmse = RMSE
                best_lever_new = lever
        best_lever = best_lever_new

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
    from data import gps_data_path

    esv_table = sio.loadmat(gps_data_path("global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    esv_bias = 0
    time_bias = 0

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = bermuda_trajectory(
        time_noise, position_noise, dz_array, angle_array, esv_matrix
    )
    GPS_Coordinates = GPS_Coordinates[::20]
    GPS_data = GPS_data[::20]

    true_offset = 1991.01236648
    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )

    """After Generating run through the analysis"""

    # initial_lever = np.array([-10.62639549,  -0.47739287, -11.5380207 ])
    initial_lever = np.array([-13.0, 0.0, -14.0])
    initial_guess = CDOG + [100, 100, 200]

    lever, offset, inversion_result = simulated_annealing_bias(
        300,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others,
        initial_guess,
        initial_lever,
        dz_array,
        angle_array,
        esv_matrix,
        real_data=True,
    )

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion:", np.round(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))

    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
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
    )

    time_series_plot(
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise
    )
