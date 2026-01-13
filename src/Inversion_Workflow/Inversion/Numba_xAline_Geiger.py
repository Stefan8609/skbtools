import numpy as np
import scipy.io as sio

from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from Inversion_Workflow.Inversion.Numba_xAline import (
    two_pointer_index,
    find_int_offset,
    refine_offset,
)
from Inversion_Workflow.Inversion.Numba_Geiger import (
    computeJacobianRayTracing,
)
from Inversion_Workflow.Forward_Model.Calculate_Times import (
    calculateTimesRayTracingReal,
    calculateTimesRayTracing,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from numba import njit
from plotting.Plot_Modular import time_series_plot
from data import gps_data_path


@njit(cache=True, fastmath=True)
def initial_geiger(
    guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=False,
):
    """Initial Geiger solver used with integer offset search."""
    epsilon = 10**-5
    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    inversion_guess = guess
    while np.linalg.norm(delta) > epsilon and k < 10:
        # Find the best offset
        if not real_data:
            times_guess, esv = calculateTimesRayTracing(
                inversion_guess,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times_guess, esv = calculateTimesRayTracingReal(
                inversion_guess,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
            )
        # offset = find_int_offset(
        #     CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        # )
        offset = find_int_offset(CDOG_data, GPS_data, times_guess)
        print(offset, k)
        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset,
            0.6,
            CDOG_data,
            GPS_data,
            times_guess,
            transponder_coordinates,
            esv,
        )
        J = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        k += 1
    """Refine offset in local region"""
    print("Offset before sub-int:", offset)
    offset = refine_offset(
        offset, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
    )
    return inversion_guess, offset


@njit(cache=True, fastmath=True)
def transition_geiger(
    guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    offset,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=False,
):
    """Refine position using sub-integer offset adjustments."""
    epsilon = 10**-5
    delta = np.array([1.0, 1.0, 1.0])
    inversion_guess = guess
    k = 0
    while np.linalg.norm(delta) > epsilon and k < 10:
        # Find the best offset
        if not real_data:
            times_guess, esv = calculateTimesRayTracing(
                inversion_guess,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times_guess, esv = calculateTimesRayTracingReal(
                inversion_guess,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
            )
        offset = refine_offset(
            offset, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )
        print(offset, k)
        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset,
            0.6,
            CDOG_data,
            GPS_data,
            times_guess,
            transponder_coordinates,
            esv,
        )
        J = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        k += 1
    return inversion_guess, offset


@njit(cache=True, fastmath=True)
def final_geiger(
    guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    offset,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=False,
):
    """Final refinement assuming the offset is accurately known."""
    epsilon = 10**-5
    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    inversion_guess = guess

    if not real_data:
        times_guess, esv = calculateTimesRayTracing(
            inversion_guess,
            transponder_coordinates,
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracingReal(
            inversion_guess,
            transponder_coordinates,
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
        offset,
        0.6,
        CDOG_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )

    while np.linalg.norm(delta) > epsilon and k < 10:
        if not real_data:
            GPS_full, esv = calculateTimesRayTracing(
                inversion_guess,
                transponder_coordinates_full,
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            GPS_full, esv = calculateTimesRayTracingReal(
                inversion_guess,
                transponder_coordinates_full,
                dz_array,
                angle_array,
                esv_matrix,
            )

        J = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ (GPS_full - CDOG_full)
        inversion_guess += delta
        k += 1

    if not real_data:
        times_guess, esv = calculateTimesRayTracing(
            inversion_guess,
            transponder_coordinates,
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracingReal(
            inversion_guess,
            transponder_coordinates,
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
        offset,
        0.6,
        CDOG_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )

    return inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock


if __name__ == "__main__":
    # Table to generate synthetic times
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    esv_bias = 0
    time_bias = 0
    """Either generate a realistic or use bermuda trajectory"""

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = bermuda_trajectory(
        time_noise, position_noise, 0.0, 0.0, dz_array, angle_array, esv_matrix
    )
    true_offset = 1991.51236648
    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )
    gps1_to_transponder = np.array([-12.48862757, 0.22622633, -15.89601934])

    print("True Offset: ", true_offset)

    """After Generating run through the analysis"""

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )

    guess = CDOG + [100, 100, 200]

    print("Starting Initial Geiger")
    inversion_guess, offset = initial_geiger(
        guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
    )
    print(
        "INT Offset: {:.4f}".format(offset), "DIFF: {:.4f}".format(offset - true_offset)
    )
    print("CDOG:", np.around(CDOG, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
    print("\n")

    inversion_guess, offset = transition_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        dz_array,
        angle_array,
        esv_matrix,
    )
    print(
        "SUB-INT Offset: {:.4f}".format(offset),
        "DIFF: {:.4f}".format(offset - true_offset),
    )
    print("CDOG:", np.around(CDOG, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
    print("\n")

    inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        dz_array,
        angle_array,
        esv_matrix,
    )

    print("CDOG:", np.around(CDOG, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))

    # Plot the results
    time_series_plot(
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise
    )
