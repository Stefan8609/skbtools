import numpy as np
from numba import njit
from Numba_Geiger import (
    computeJacobianRayTracing,
    findTransponder,
    calculateTimesRayTracing,
    calculateTimesRayTracingReal,
)
from Numba_xAline import two_pointer_index, find_subint_offset, find_int_offset
from Generate_Unaligned_Realistic import generateUnalignedRealistic


def initial_geiger(
    guess, CDOG_data, GPS_data, transponder_coordinates, real_data=False
):
    """Initial Geiger solver used with integer offset search."""
    epsilon = 10**-5
    k = 0
    delta = 1
    inversion_guess = guess
    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if not real_data:
            times_guess, esv = calculateTimesRayTracing(
                inversion_guess, transponder_coordinates
            )
        else:
            times_guess, esv = calculateTimesRayTracingReal(
                inversion_guess, transponder_coordinates
            )
        offset = find_int_offset(
            CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )

        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )
        jacobian = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (GPS_full - CDOG_full)
        )
        inversion_guess += delta
        k += 1
        if np.linalg.norm(inversion_guess - guess) > 1000:
            print("ERROR: Inversion too far from starting value")
            return inversion_guess, offset

    return inversion_guess, offset


@njit
def transition_geiger(
    guess, CDOG_data, GPS_data, transponder_coordinates, offset, real_data=False
):
    """Refine position using sub-integer offset adjustments."""
    epsilon = 10**-5
    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    inversion_guess = guess

    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if not real_data:
            times_guess, esv = calculateTimesRayTracing(
                inversion_guess, transponder_coordinates
            )
        else:
            times_guess, esv = calculateTimesRayTracingReal(
                inversion_guess, transponder_coordinates
            )
        offset = find_subint_offset(
            offset, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
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
            True,
        )
        jacobian = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (GPS_full - CDOG_full)
        )
        inversion_guess += delta
        k += 1

        if np.linalg.norm(inversion_guess - guess) > 1000:
            print("ERROR: Inversion too far from starting value")
            return inversion_guess, offset

    return inversion_guess, offset


@njit
def final_geiger(
    guess, CDOG_data, GPS_data, transponder_coordinates, offset, real_data=False
):
    """Final refinement assuming the offset is accurately known."""
    epsilon = 10**-5
    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    inversion_guess = guess

    if not real_data:
        times_guess, esv = calculateTimesRayTracing(
            inversion_guess, transponder_coordinates
        )
    else:
        times_guess, esv = calculateTimesRayTracingReal(
            inversion_guess, transponder_coordinates
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
        True,
    )

    while np.linalg.norm(delta) > epsilon and k < 10:
        if not real_data:
            GPS_full, esv = calculateTimesRayTracing(
                inversion_guess, transponder_coordinates_full
            )
        else:
            GPS_full, esv = calculateTimesRayTracingReal(
                inversion_guess, transponder_coordinates_full
            )

        jacobian = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (GPS_full - CDOG_full)
        )
        inversion_guess += delta
        k += 1

    if not real_data:
        times_guess, esv = calculateTimesRayTracing(
            inversion_guess, transponder_coordinates
        )
    else:
        times_guess, esv = calculateTimesRayTracingReal(
            inversion_guess, transponder_coordinates
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
        True,
    )

    return inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock


if __name__ == "__main__":
    import scipy.io as sio
    from data import gps_data_path

    true_offset = np.random.rand() * 9000 + 1000
    print(true_offset)

    esv_table = sio.loadmat(gps_data_path("global_table_esv_normal.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5
    esv_bias = 1.5
    time_bias = 0.43

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = generateUnalignedRealistic(
        20000,
        time_noise,
        true_offset,
        esv_bias,
        time_bias,
        dz_array,
        angle_array,
        esv_matrix,
        main=False,
    )
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )

    guess = CDOG + [200, 200, 400]

    inversion_result, offset = initial_geiger(
        guess, CDOG_data, GPS_data, transponder_coordinates
    )
    print("INT Offset: ", offset, "DIFF:", offset - true_offset)
    print("CDOG:", CDOG)
    print("Inversion:", inversion_result)
    print("Distance:", np.linalg.norm(inversion_result - CDOG) * 100, "cm")
    print("\n")

    inversion_result, offset = transition_geiger(
        inversion_result, CDOG_data, GPS_data, transponder_coordinates, offset
    )

    print("SUB-INT Offset: ", offset, "DIFF", offset - true_offset)
    print("CDOG:", CDOG)
    print("Inversion:", inversion_result)
    print("Distance:", np.linalg.norm(inversion_result - CDOG) * 100, "cm")
    print("\n")

    times_guess, esv = calculateTimesRayTracing(
        inversion_result, transponder_coordinates
    )
    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(
        inversion_result, CDOG_data, GPS_data, transponder_coordinates, offset
    )
    print("CDOG:", CDOG)
    print("Inversion:", inversion_result)
    print("Distance:", np.linalg.norm(inversion_result - CDOG) * 100, "cm")
