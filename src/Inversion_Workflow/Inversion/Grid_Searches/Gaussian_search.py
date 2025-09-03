import numpy as np
import scipy.io as sio

from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_Geiger import findTransponder
from data import gps_data_path, gps_output_path


def gaussian_search_individual(
    dog_idx,
    num_points,
    output_file=None,
    initial_lever_base=None,
    initial_gps_grid=None,
    sigma_lever=None,
    sigma_gps_grid=None,
    downsample=50,
):
    """Sample lever and grid parameters for a single DOG.

    Parameters
    ----------
    dog_idx : int
        Index of the DOG (0–2).
    num_points : int
        Number of samples to generate.
    output_file : str, optional
        Destination file for the RMSE results.
    initial_lever_base : ndarray, optional
        Base estimate of the lever arm.
    initial_gps_grid : ndarray, optional
        Estimated GPS grid for DOG1.
    sigma_lever : ndarray, optional
        Standard deviation of lever perturbations.
    sigma_gps_grid : ndarray, optional
        Standard deviation of GPS grid perturbations.
    downsample : int, optional
        Downsample factor for the raw data.

    Returns
    -------
    None
        Results are written to ``output_file``.
    """
    if output_file is None:
        output_file = gps_output_path("gaussian_individual_output.txt")
    if initial_lever_base is None:
        initial_lever_base = np.array([-12.4659, 9.6021, -13.2993])
    if initial_gps_grid is None:
        initial_gps_grid = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.39341409, -4.22350344, 0.02941493],
                [-12.09568416, -0.94568462, 0.0043972],
                [-8.68674054, 5.16918806, -0.02499322],
            ]
        )
    if sigma_lever is None:
        sigma_lever = np.array([0.5, 0.5, 2.0])
    if sigma_gps_grid is None:
        sigma_gps_grid = np.array([0.5, 0.5, 0.1])

    np.set_printoptions(suppress=True)

    # Load the external ESV data
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    # Load the external GPS data
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"][::downsample]
    GPS_data = data["GPS_data"][::downsample]
    CDOG_guess = data["CDOG_guess"]

    # Load the DOG data for each DOG (skipping i==2 as before)
    CDOG_all_data = []
    for i in range(1, 5):
        if i == 2:
            continue
        CDOG_mat = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{i}-camp.mat"))[
            "tags"
        ].astype(float)
        CDOG_mat[:, 1] /= 1e9
        CDOG_all_data.append(CDOG_mat)

    # Initialize the best-known augments and offsets
    CDOG_augments = np.array(
        [
            [-398.16, 371.90, 773.02],
            [825.182985, -111.05670221, -734.10011698],
            [236.428385, -1307.98390221, -2189.21991698],
        ]
    )
    offsets = np.array([1866.0, 3175.0, 1939.0])
    esv_bias = 0.0
    time_bias = 0.0

    # Open and write header
    with open(output_file, "w") as file:
        file.write("Lever, GPS_Grid, Offset, RMSE_individual\n")

        for i in range(num_points):
            # perturb lever and GPS grid
            lever_guess = initial_lever_base + np.random.normal(0, sigma_lever, 3)
            gps1_grid_guess = initial_gps_grid + np.random.normal(
                0, sigma_gps_grid, (4, 3)
            )

            # compute transponder coordinates
            transponder_coordinates = findTransponder(
                GPS_Coordinates, gps1_grid_guess, lever_guess
            )

            # for this DOG, find best offset and RMSE
            best_offset_rmse = np.inf
            best_offset = None

            for off_adjust in (-1, 0, 1):
                offset = offsets[dog_idx] + off_adjust
                inversion_guess = CDOG_guess + CDOG_augments[dog_idx]
                CDOG_data = CDOG_all_data[dog_idx]

                try:
                    _, CDOG_full, GPS_full, _, _ = final_bias_geiger(
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
                        real_data=True,
                    )
                    RMSE = np.sqrt(np.mean((CDOG_full - GPS_full) ** 2))
                except Exception:
                    RMSE = np.inf

                if RMSE < best_offset_rmse:
                    best_offset_rmse = RMSE
                    best_offset = offset

            # Write results
            file.write(
                f"[{lever_guess.round(4).tolist()}], "
                f"[{gps1_grid_guess.round(4).tolist()}], "
                f"{best_offset}, "
                f"{best_offset_rmse:.6f}\n"
            )
            file.flush()

            print(
                f"Sample {i + 1}/{num_points} – DOG{dog_idx + 1}: "
                f"Lever={lever_guess}, Offset={best_offset}, "
                f"RMSE={best_offset_rmse * 100 * 1515:.6f}"
            )

    print(f"Done: RMSE samples for DOG{dog_idx + 1} saved to {output_file}")


def gaussian_search(
    num_points,
    output_file=None,
    initial_lever_base=None,
    initial_gps_grid=None,
    sigma_lever=None,
    sigma_gps_grid=None,
    downsample=50,
):
    """Perform a Gaussian search over lever and grid parameters.

    Parameters
    ----------
    num_points : int
        Number of samples to generate.
    output_file : str, optional
        File where the combined RMSE values are stored.
    initial_lever_base : ndarray, optional
        Base estimate of the lever arm.
    initial_gps_grid : ndarray, optional
        Estimated GPS grid for DOG1.
    sigma_lever : ndarray, optional
        Standard deviation of lever perturbations.
    sigma_gps_grid : ndarray, optional
        Standard deviation of GPS grid perturbations.
    downsample : int, optional
        Downsample factor for the raw data.

    Returns
    -------
    None
        The RMSE values are written to ``output_file``.
    """
    if output_file is None:
        output_file = gps_output_path("gaussian_output.txt")
    if initial_lever_base is None:
        initial_lever_base = np.array([-12.4659, 9.6021, -13.2993])
    if initial_gps_grid is None:
        initial_gps_grid = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.39341409, -4.22350344, 0.02941493],
                [-12.09568416, -0.94568462, 0.0043972],
                [-8.68674054, 5.16918806, -0.02499322],
            ]
        )
    if sigma_lever is None:
        sigma_lever = np.array([0.5, 0.5, 2.0])
    if sigma_gps_grid is None:
        sigma_gps_grid = np.array([0.5, 0.5, 0.1])

    np.set_printoptions(suppress=True)

    # Load the external ESV data
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    # Load the external GPS data
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    CDOG_guess = data["CDOG_guess"]

    # Load the DOG data for each DOG
    CDOG_all_data = []
    for i in range(1, 5):
        if i == 2:
            continue
        CDOG_data = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{i}-camp.mat"))[
            "tags"
        ].astype(float)
        CDOG_data[:, 1] = CDOG_data[:, 1] / 1e9
        CDOG_all_data.append(CDOG_data)

    # Initialize the best known augments and offsets
    CDOG_augments = np.array(
        [
            [-398.16, 371.90, 773.02],
            [825.182985, -111.05670221, -734.10011698],
            [236.428385, -1307.98390221, -2189.21991698],
        ]
    )
    offsets = np.array([1866.0, 3175.0, 1939.0])

    esv_bias = 0.0
    time_bias = 0.0

    # Downsample Data
    GPS_Coordinates = GPS_Coordinates[::downsample]
    GPS_data = GPS_data[::downsample]

    # Initialize the output file
    with open(output_file, "w") as file:
        # Write a header line
        file.write("Grid Search Results\n")
        file.write("[Lever], [[GPS Grid Estimate]], Offset Adjustment, RMSE combined\n")
        for i in range(num_points):
            lever_guess = initial_lever_base + np.random.normal(0, sigma_lever, 3)
            gps1_grid_guess = initial_gps_grid + np.random.normal(
                0, sigma_gps_grid, (4, 3)
            )

            # Calculate the transponder coordinates
            transponder_coordinates = findTransponder(
                GPS_Coordinates, gps1_grid_guess, lever_guess
            )

            # Run final_geiger for each DOG
            best_offset_rmse = np.full(3, np.inf)
            best_offset = np.zeros(3)
            for off_adjust in range(-1, 2, 1):
                RMSE_sum = 0
                for j in range(3):
                    offset = offsets[j] + off_adjust
                    inversion_guess = CDOG_guess + CDOG_augments[j]
                    CDOG_data = CDOG_all_data[j]
                    try:
                        inversion_result, CDOG_full, GPS_full, _, _ = final_bias_geiger(
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
                            real_data=True,
                        )
                        RMSE = np.sqrt(np.mean((CDOG_full - GPS_full) ** 2))
                    except Exception as error:
                        print(error)
                        print(
                            f"Error in final_bias_geiger, skipping this iteration {j}"
                        )
                        RMSE_sum = np.inf
                        break
                    if RMSE < best_offset_rmse[j]:
                        best_offset_rmse[j] = RMSE
                        best_offset[j] = offset
                        """Extend on this idea post meeting"""
                RMSE_sum = np.sum(best_offset_rmse)
                lever_str = np.array2string(
                    lever_guess, precision=4, separator=", ", max_line_width=1000
                )[1:-1]
                gps1_str = str(gps1_grid_guess.round(4)).replace("\n", " ")[1:-1]
                offset_str = np.array2string(
                    best_offset, precision=4, separator=", ", max_line_width=1000
                )[1:-1]
                rmse_val = RMSE_sum * 100 * 1515

                file.write(
                    f"[{lever_str}], [{gps1_str}], [{offset_str}], {rmse_val:.4f}\n"
                )
                file.flush()  # ensures immediate write

                lever_str = np.array2string(
                    lever_guess, precision=4, separator=", ", max_line_width=1000
                )
                gps1_str = str(gps1_grid_guess.round(4)).replace("\n", " ")
                offset_str = np.array2string(
                    best_offset, precision=4, separator=", ", max_line_width=1000
                )[1:-1]
                rmse_offset_str = np.array2string(
                    best_offset_rmse * 100 * 1515,
                    precision=4,
                    separator=", ",
                    max_line_width=1000,
                )[1:-1]
                rmse_val = RMSE_sum * 100 * 1515

                print(
                    f"Iteration {i + 1}/{num_points}:\n"
                    f"Lever: {lever_str},\n"
                    f"gps1_grid_guess: {gps1_str},\n"
                    f"Best Offsets: {offset_str}\n"
                    f"Best RMSE: {rmse_offset_str},\n"
                    f"RMSE: {rmse_val:.4f}\n"
                )


if __name__ == "__main__":
    # gaussian_search(
    #     num_points=100,
    #     output_file='gaussian_output.txt',
    #     initial_lever_base=np.array([-12.4659, 9.6021, -13.2993]),
    #     initial_gps_grid=np.array([[0.0, 0.0, 0.0],
    #                                [-2.39341409, -4.22350344, 0.02941493],
    #                                [-12.09568416, -0.94568462, 0.0043972],
    #                                [-8.68674054, 5.16918806, -0.02499322]]),
    #     sigma_lever=np.array([0.2, 0.2, 0.2]),
    #     sigma_gps_grid=np.array([0.01, 0.01, 0.01]),
    #     downsample=100,
    # )

    if __name__ == "__main__":
        gaussian_search_individual(
            dog_idx=1,
            num_points=200,
            output_file=gps_output_path("gaussian_output.txt"),
            initial_lever_base=np.array([-12.4659, 9.6021, -13.2993]),
            initial_gps_grid=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-2.3934, -4.2235, 0.0294],
                    [-12.0957, -0.9457, 0.0044],
                    [-8.6867, 5.1692, -0.0250],
                ]
            ),
            sigma_lever=np.array([0.5, 0.5, 0.5]),
            sigma_gps_grid=np.array([0.1, 0.1, 0.1]),
            downsample=50,
        )
