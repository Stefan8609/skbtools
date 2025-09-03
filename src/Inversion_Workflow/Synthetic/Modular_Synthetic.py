import numpy as np
import scipy.io as sio

from src.Inversion_Workflow.Synthetic.Generate_Unaligned import (
    generateUnalignedRealistic,
)
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    initial_bias_geiger,
    transition_bias_geiger,
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from plotting.Plot_Modular import time_series_plot
from data import gps_data_path

"""
File to allow for easy changing of parameters when running synthetic

Write a code to check the indexing of the Bermuda data vs exact indexing
"""


def modular_synthetic(
    time_noise,
    position_noise,
    in_esv_bias,
    in_time_bias,
    esv1="global_table_esv",
    esv2="global_table_esv_perturbed",
    generate_type=0,
    inversion_type=0,
    plot=True,
    DOG_num=3,
):
    np.set_printoptions(suppress=True)
    # Choose ESV table for generation and to run synthetic
    #   Perhaps make the file link a parameter of the function

    esv_table_generate = sio.loadmat(gps_data_path(f"ESV_Tables/{esv1}.mat"))
    dz_array_generate = esv_table_generate["distance"].flatten()
    angle_array_generate = esv_table_generate["angle"].flatten()
    esv_matrix_generate = esv_table_generate["matrice"]

    esv_table_inversion = sio.loadmat(gps_data_path(f"ESV_Tables/{esv2}.mat"))
    dz_array_inversion = esv_table_inversion["distance"].flatten()
    angle_array_inversion = esv_table_inversion["angle"].flatten()
    esv_matrix_inversion = esv_table_inversion["matrice"]

    # Choose Generate type:
    #   0: Generate Unaligned Realistic Data
    #   1: Use Bermuda Dataset
    if generate_type == 0:
        # Generate Unaligned Realistic Data
        true_offset = np.random.rand() * 9000 + 1000
        print(true_offset)
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
            in_esv_bias,
            in_time_bias,
            dz_array_generate,
            angle_array_generate,
            esv_matrix_generate,
        )
        GPS_Coordinates += np.random.normal(
            0, position_noise, (len(GPS_Coordinates), 4, 3)
        )
        gps1_to_others = np.array(
            [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
        )
        z_sample = False
    else:
        # Use Bermuda Dataset
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = bermuda_trajectory(
            time_noise,
            position_noise,
            dz_array_generate,
            angle_array_generate,
            esv_matrix_generate,
            DOG_num,
        )
        true_offset = 1991.01236648
        gps1_to_others = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.4054, -4.20905, 0.060621],
                [-12.1105, -0.956145, 0.00877],
                [-8.70446831, 5.165195, 0.04880436],
            ]
        )

        downsample = 1
        GPS_Coordinates = GPS_Coordinates[::downsample]
        GPS_data = GPS_data[::downsample]
        true_transponder_coordinates = true_transponder_coordinates[::downsample]
        z_sample = True

    # Choose Inversion_Workflow Type
    #   0: Just xAline Geiger
    #   1: xAline Geiger with Simulated Annealing
    real_data = True if generate_type == 1 else False
    initial_guess = CDOG + [100, 100, 200]
    if inversion_type == 0:
        # Just xAline Geiger
        lever = (
            np.array([-10.0, 3.0, -15.0])
            if generate_type == 0
            else np.array([-12.48862757, 0.22622633, -15.89601934])
        )

        # Add some randomness to the lever
        lever += np.random.normal(0, 3, 3)

        transponder_coordinates = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        inversion_result, offset = initial_bias_geiger(
            initial_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
        )

        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]

        print(
            "INT Offset: {:.4f}".format(offset),
            "DIFF: {:.4f}".format(offset - true_offset),
        )
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )
        print("\n")
        inversion_result, offset = transition_bias_geiger(
            inversion_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            offset,
            esv_bias,
            time_bias,
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
        )
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print(
            "SUB-INT Offset: {:.4f}".format(offset),
            "DIFF: {:.4f}".format(offset - true_offset),
        )
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )
        print("\n")

        (
            inversion_result,
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
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
        )
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )

    else:
        real_lever = (
            np.array([-10.0, 3.0, -15.0])
            if generate_type == 0
            else np.array([-12.48862757, 0.22622633, -15.89601934])
        )
        initial_lever = np.array([-12.478, 0.667, -14.292])

        """True levers: Realistic Generate [-10, 3, -15],
        Bermuda Generate: [-12.48862757, 0.22622633, -15.89601934]"""

        lever, offset, inversion_result = simulated_annealing_bias(
            300,
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            gps1_to_others,
            initial_guess,
            initial_lever,
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
            enforce_offset=True,
            initial_offset=1991,
            z_sample=z_sample,
        )
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )
        print(
            f"Lever Error: {np.round(np.linalg.norm(lever - real_lever) * 100, 2)} cm"
        )

        transponder_coordinates = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        (
            inversion_result,
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
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
        )

    if plot:
        time_series_plot(
            CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise
        )
        # trajectory_plot(transponder_coordinates, GPS_data, CDOG)

    return (
        inversion_result,
        CDOG_data,
        CDOG_full,
        GPS_data,
        GPS_full,
        CDOG_clock,
        GPS_clock,
        CDOG,
        transponder_coordinates,
        GPS_Coordinates,
        offset,
        lever,
    )


if __name__ == "__main__":
    esv_bias = 0
    time_bias = 0

    modular_synthetic(
        2 * 10**-5,
        2 * 10**-2,
        0,
        0,
        "global_table_esv",
        "global_table_esv_realistic_perturbed",
        generate_type=1,
        inversion_type=1,
        DOG_num=3,
    )
