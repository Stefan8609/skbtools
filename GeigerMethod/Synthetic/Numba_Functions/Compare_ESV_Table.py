import numpy as np
import scipy.io as sio
import random
from Numba_time_bias import numba_bias_geiger, find_esv
from Numba_Geiger import generateRealistic, findTransponder

"""
Plot the distribution of esv differences
"""

"""
Investigating how the ESV bias compares to the difference in the sections of the ESV table that are sampled
"""


def compare_tables(esv_table1, esv_table2):
    """
    Compare the relevant angles and depths within ESV table to see if their average difference is the ESV-bias

    esv1 used to generate the data
    esv2 is used in inversion
    """
    dz_array_gen = esv_table1["distance"].flatten()
    angle_array_gen = esv_table1["angle"].flatten()
    esv_matrix_gen = esv_table1["matrice"]

    dz_array_inv = esv_table2["distance"].flatten()
    angle_array_inv = esv_table2["angle"].flatten()
    esv_matrix_inv = esv_table2["matrice"]

    (
        CDOG,
        GPS_Coordinates,
        transponder_coordinates_Actual,
        gps1_to_others,
        gps1_to_transponder,
    ) = generateRealistic(20000)

    esv_bias = 0.0
    time_bias = 0.0

    # time_noise = 0
    # position_noise = 0
    time_noise = 2.0 * 10**-5
    position_noise = 2.0 * 10**-2

    # Apply noise to position
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    transponder_coordinates_Found = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )
    CDOG_guess = np.array(
        [
            random.uniform(-5000, 5000),
            random.uniform(-5000, 5000),
            random.uniform(-5225, -5235),
        ]
    )

    # Run the Geiger method with ESV bias estimation
    estimate, times_known = numba_bias_geiger(
        CDOG_guess,
        CDOG,
        transponder_coordinates_Actual,
        transponder_coordinates_Found,
        esv_bias,
        time_bias,
        dz_array_inv,
        angle_array_inv,
        esv_matrix_inv,
        dz_array_gen,
        angle_array_gen,
        esv_matrix_gen,
        time_noise,
    )

    print(f"Input: {[CDOG[0], CDOG[1], CDOG[2], time_bias, esv_bias]}")
    print(f"Output: {estimate}")
    print(
        f"Diff: {estimate - np.array([CDOG[0], CDOG[1], CDOG[2], -time_bias, esv_bias])}"
    )
    print(
        f"Distance: {np.sqrt(np.sum((estimate[:3] - np.array([CDOG[0], CDOG[1], CDOG[2]]))) ** 2)}"
    )

    hori_dist = np.sqrt(
        (transponder_coordinates_Found[:, 0] - estimate[0]) ** 2
        + (transponder_coordinates_Found[:, 1] - estimate[1]) ** 2
    )
    abs_dist = np.sqrt(
        np.sum((transponder_coordinates_Found - estimate[:3]) ** 2, axis=1)
    )
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(transponder_coordinates_Found[:, 2] - CDOG[2])

    print(f"max dz: {np.max(dz)}, min dz: {np.min(dz)}")
    print(f"max beta: {np.max(beta)}, min beta: {np.min(beta)}")

    # Somehow average over difference at angles and depths
    gen_esv = find_esv(beta, dz, dz_array_gen, angle_array_gen, esv_matrix_gen)
    inv_esv = find_esv(beta, dz, dz_array_inv, angle_array_inv, esv_matrix_inv)

    tot = 0
    for i in range(len(gen_esv)):
        tot += gen_esv[i] - inv_esv[i]
    print(
        "Average Difference in ESV table at Relevant Angles and Depths: ",
        tot / len(gen_esv),
        "\nESV Bias: ",
        estimate[4],
    )


if __name__ == "__main__":
    esv_table1 = sio.loadmat("../../../GPSData/global_table_esv_perturbed.mat")

    esv_table2 = sio.loadmat("../../../GPSData/global_table_esv.mat")

    compare_tables(esv_table1, esv_table2)

    # plt.figure(figsize=(10, 6))
    # # plt.contourf(angle_array1, dz_array1, esv_matrix1 - esv_matrix2, levels=10)
    # angle_idx = np.array([np.abs(angle_array1 - a).argmin() for a in angle_array2])
    # depth_idx = np.array([np.abs(dz_array1 - d).argmin() for d in dz_array2])
    # plt.contourf(angle_array1[angle_idx], dz_array1[depth_idx], esv_matrix1[np.ix_(depth_idx, angle_idx)] - esv_matrix2,
    #              levels=10)
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.xlabel("Elevation Angle (degrees)")
    # plt.ylabel("Depth (m)")
    # plt.title("Effective Sound Velocity (m/s)")
    # plt.show()
