import numpy as np
import scipy.io as sio
import itertools

from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline_bias import (
    final_bias_geiger,
    initial_bias_geiger,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from GeigerMethod.Synthetic.Numba_Functions.Real_Annealing import (
    simulated_annealing_real,
)
from data import gps_data_path, gps_output_path

"""Plot all of the Seafloor guesses from the grid search and find the correlation

Add GPS grid search to the plot (that'll add a lot of complexity)
"""


def grid_search_annealing(
    xl,
    xh,
    yl,
    yh,
    zl,
    zh,
    num_points,
    output_file=None,
    CDOG_guess_augment=None,
    initial_lever_base=None,
    downsample=50,
    sa_iterations=300,
    offset_range=4,
    current_offset=0,
    z_range=4,
    DOG_num=3,
):
    """Run a grid search over lever arm and offset using annealing.

    Parameters
    ----------
    xl, xh, yl, yh, zl, zh : float
        Bounds for the lever-arm search grid.
    num_points : int
        Number of samples along each axis.
    output_file : str, optional
        Path for the text output of the search.
    CDOG_guess_augment : ndarray, optional
        Base adjustment to the DOG initial guess.
    initial_lever_base : ndarray, optional
        Initial lever arm used for the search origin.
    downsample : int
        Downsampling factor for data.
    sa_iterations : int
        Number of annealing steps per grid point.
    offset_range : int
        Magnitude of offset adjustments to test.
    current_offset : float
        Starting offset for annealing.
    z_range : int
        Number of z-axis points to evaluate.
    DOG_num : int
        DOG data set identifier.

    Returns
    -------
    None
        Results are appended to ``output_file``.
    """
    if output_file is None:
        output_file = gps_output_path("output.txt")
    if CDOG_guess_augment is None:
        CDOG_guess_augment = np.array([0.0, 0.0, 0.0])
    if initial_lever_base is None:
        initial_lever_base = np.array([-16.0, 0.0, -15.0])

    # Load the external ESV data
    esv_table = sio.loadmat(gps_data_path("global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    data = np.load(gps_data_path(f"Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    CDOG_data = data["CDOG_data"]
    CDOG_guess = data["CDOG_guess"]
    GPS1_to_others = data["gps1_to_others"]

    # For a quick initial offset guess (if needed)
    # you could call initial_bias_geiger once per (x, y, z) if relevant:
    _, current_offset = initial_bias_geiger(
        CDOG_guess,
        CDOG_data,
        GPS_data,
        findTransponder(GPS_Coordinates, GPS1_to_others, initial_lever_base),
        dz_array,
        angle_array,
        esv_matrix,
        real_data=True,
    )

    # Downsample data
    GPS_Coordinates = GPS_Coordinates[::downsample]
    GPS_data = GPS_data[::downsample]

    # Create search grids
    x_grid = np.linspace(xl, xh, num_points)
    y_grid = np.linspace(yl, yh, num_points)
    z_grid = np.linspace(zl, zh, num_points)
    offset_adjusts = np.linspace(-offset_range, offset_range, offset_range * 2 + 1)

    # Calculate total iterations for progress tracking
    total_iterations = len(x_grid) * len(y_grid) * len(z_grid) * len(offset_adjusts)
    iteration_count = 0

    print("Starting Grid Search...")
    with open(output_file, "w") as file:
        # Write a header line
        file.write("Grid Search Results\n")
        file.write(
            "[Lever], [CDOG Estimate], Input Offset, Time Bias, ESV Bias, RMSE\n"
        )

        for x, y, z in itertools.product(x_grid, y_grid, z_grid):
            # Compute the lever-arm guess for this (x, y, z)
            lever_guess = initial_lever_base + np.array([x, y, z])

            for off_adjust in offset_adjusts:
                iteration_count += 1

                # Run simulated annealing
                inversion_result, best_lever, RMSE = simulated_annealing_real(
                    sa_iterations,
                    CDOG_data,
                    GPS_data,
                    GPS_Coordinates,
                    GPS1_to_others,
                    CDOG_guess,
                    lever_guess,
                    dz_array,
                    angle_array,
                    esv_matrix,
                    current_offset + off_adjust,
                )

                inversion_guess = inversion_result[:3]
                time_bias = inversion_result[3]
                esv_bias = inversion_result[4]

                """Search the z-range to find the lowest RMSE
                while evaluating the poorly resolved z-axis"""
                for z in np.linspace(-z_range, z_range, z_range * 2 + 1):
                    offset_perturbed = current_offset + off_adjust
                    best_lever_perturbed = best_lever + np.array([0, 0, z])
                    transponder_perturbed = findTransponder(
                        GPS_Coordinates, GPS1_to_others, best_lever_perturbed
                    )
                    (
                        inversion_result_pert,
                        CDOG_full,
                        GPS_full,
                        _,
                        _,
                    ) = final_bias_geiger(
                        inversion_guess,
                        CDOG_data,
                        GPS_data,
                        transponder_perturbed,
                        offset_perturbed,
                        esv_bias,
                        time_bias,
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=True,
                    )
                    RMSE_pert = np.sqrt(np.mean((CDOG_full - GPS_full) ** 2))
                    if RMSE_pert < RMSE:
                        RMSE = RMSE_pert
                        best_lever = best_lever_perturbed
                        inversion_result = inversion_result_pert

                inversion_guess = inversion_result[:3]
                time_bias = inversion_result[3]
                esv_bias = inversion_result[4]

                # Write line to file
                file.write(
                    f"[{
                        np.array2string(best_lever, precision=4, separator=', ')[1:-1]
                    }], "
                    f"[{
                        np.array2string(inversion_guess, precision=4, separator=', ')[
                            1:-1
                        ]
                    }], "
                    f"{current_offset + off_adjust:.4f}, "
                    f"{time_bias:.4e}, {esv_bias:.4f}, {RMSE * 100 * 1515:.4f}\n"
                )
                file.flush()  # ensures immediate write

                print(
                    f"Iteration {iteration_count}/{total_iterations}: "
                    f"Lever: {best_lever}, "
                    f"Offset: {current_offset + off_adjust:.4f}"
                    f", RMSE: {RMSE * 100 * 1515:.4f}"
                )

    print("Grid Search Completed. Results saved to", output_file)


def grid_search_discrete(
    xl,
    xh,
    yl,
    yh,
    zl,
    zh,
    num_points,
    output_file=None,
    CDOG_guess_augment=None,
    initial_lever_base=None,
    downsample=50,
    offset_range=4,
    current_offset=0,
    DOG_num=1,
):
    """
    Performs a grid search over lever-arm values (x, y, z) and small offset adjustments,
    running final geiger and getting RMSE at each iteration. Saves intermediate
    and final results to a text file (and prints progress to console).

    Parameters
    ----------
    xl, xh : float
        Lower and upper bounds for the x coordinate of the lever-arm search range.
    yl, yh : float
        Lower and upper bounds for the y coordinate of the lever-arm search range.
    zl, zh : float
        Lower and upper bounds for the z coordinate of the lever-arm search range.
    num_points : int
        Number of points to sample along each axis (x, y, z).
    output_file : str, optional
        Path to the output text file where results are appended.
    CDOG_guess_augment : np.ndarray, optional
        Base vector for initial CDOG guess augmentation.
    initial_lever_base : np.ndarray, optional
        Base vector for the lever-arm guess.
        Start and end times for GNSS data slicing.
    downsample : int
        Downsampling step used for the GPS data.
    offset_range : tuple (start, stop, num)
        Defines the np.linspace() range for searching offset around offset_initial.
    current_offset : float
        Initial offset used as a starting point for finer searches.
    DOG_num : int
        Number of the DOG that is analyzed
    """
    if output_file is None:
        output_file = gps_output_path("output.txt")
    if CDOG_guess_augment is None:
        CDOG_guess_augment = np.array([0.0, 0.0, 0.0])
    if initial_lever_base is None:
        initial_lever_base = np.array([-16.0, 0.0, -15.0])

    # Load the external ESV data
    esv_table = sio.loadmat(gps_data_path("global_table_esv_normal.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    data = np.load(gps_data_path(f"Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    CDOG_data = data["CDOG_data"]
    CDOG_guess_base = data["CDOG_guess"]
    GPS1_to_others = data["gps1_to_others"]

    CDOG_guess = CDOG_guess_base + CDOG_guess_augment

    # For a quick initial offset guess (if needed)
    # you could call initial_bias_geiger once per (x, y, z) if relevant:
    if current_offset == 0:
        _, current_offset = initial_bias_geiger(
            CDOG_guess,
            CDOG_data,
            GPS_data,
            findTransponder(GPS_Coordinates, GPS1_to_others, initial_lever_base),
            dz_array,
            angle_array,
            esv_matrix,
            real_data=True,
        )

    # Downsample data
    GPS_Coordinates = GPS_Coordinates[::downsample]
    GPS_data = GPS_data[::downsample]

    inversion_guess = CDOG_guess
    esv_bias = 0
    time_bias = 0

    # Create search grids
    x_grid = np.linspace(xl, xh, num_points)
    y_grid = np.linspace(yl, yh, num_points)
    z_grid = np.linspace(zl, zh, num_points)
    offset_adjusts = np.linspace(-offset_range, offset_range, offset_range * 2 + 1)

    # Calculate total iterations for progress tracking
    total_iterations = len(x_grid) * len(y_grid) * len(z_grid) * len(offset_adjusts)
    iteration_count = 0

    print("Starting Grid Search...")
    with open(output_file, "w") as file:
        # Write a header line
        file.write("Grid Search Results\n")
        file.write(
            "[Lever], [CDOG Estimate], Input Offset, Time Bias, ESV Bias, RMSE\n"
        )

        for x, y, z in itertools.product(x_grid, y_grid, z_grid):
            # Compute the lever-arm guess for this (x, y, z)
            lever_guess = initial_lever_base + np.array([x, y, z])
            transponder_coordinates = findTransponder(
                GPS_Coordinates, GPS1_to_others, lever_guess
            )
            for off_adjust in offset_adjusts:
                iteration_count += 1
                # Run final_geiger
                try:
                    """Reset inversion_guess each time"""
                    # inversion_guess = CDOG_guess
                    inversion_result, CDOG_full, GPS_full, _, _ = final_bias_geiger(
                        inversion_guess,
                        CDOG_data,
                        GPS_data,
                        transponder_coordinates,
                        current_offset + off_adjust,
                        esv_bias,
                        time_bias,
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=True,
                    )
                    inversion_guess = inversion_result[:3]
                    time_bias = inversion_result[3]
                    esv_bias = inversion_result[4]
                except Exception:
                    print(
                        f"Error in final_bias_geiger, "
                        f"skipping this iteration. offset = "
                        f"{current_offset + off_adjust}"
                    )
                    continue

                diff_data = (CDOG_full - GPS_full) * 1000
                RMSE = np.sqrt(np.nanmean(diff_data**2)) / 1000 * 1515 * 100
                # Write line to file
                file.write(
                    f"[{
                        np.array2string(lever_guess, precision=4, separator=', ')[1:-1]
                    }], "
                    f"[{
                        np.array2string(inversion_guess, precision=4, separator=', ')[
                            1:-1
                        ]
                    }], "
                    f"{current_offset + off_adjust:.4f}, "
                    f"{time_bias:.4e}, {esv_bias:.4f}, {RMSE:.4f}\n"
                )
                file.flush()  # ensures immediate write

                print(
                    f"Iteration {iteration_count}/{total_iterations}: "
                    f"Lever: {lever_guess}, Offset: {current_offset + off_adjust:.4f},"
                    f" RMSE: {RMSE:.4f} \n"
                    f"Inversion Guess: "
                    f"{np.round(inversion_guess - CDOG_guess_base, 5)}, "
                    f"time_bias: {time_bias:.4f}, esv_bias: {esv_bias:.4f} \n"
                )

    print("Grid Search Completed. Results saved to", output_file)


if __name__ == "__main__":
    # Example usage
    # grid_search_annealing(
    #     xl=-5, xh=5,
    #     yl=-5, yh=5,
    #     zl=-5, zh=5,
    #     num_points=2,
    #     output_file='output.txt',
    #     downsample=100,
    #     offset_range = 4,
    #     z_range = 10
    # )

    grid_search_discrete(
        xl=-0.05,
        xh=0.05,
        yl=-0.05,
        yh=0.05,
        zl=-1,
        zh=1,
        num_points=20,
        output_file=gps_output_path("output.txt"),
        CDOG_guess_augment=np.array([-398.16, 371.90, 773.02]),
        downsample=50,
        offset_range=1,
        current_offset=1886.0,
        initial_lever_base=np.array([-12.4659, 9.6021, -13.2993]),
        DOG_num=1,
    )
