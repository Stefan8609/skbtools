import numpy as np
import scipy.io as sio

from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_Geiger import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from Inversion_Workflow.Bermuda.Initialize_Bermuda_Data import (
    initialize_bermuda,
)
from data import gps_data_path, gps_output_path


def grid_search_annealing(xl, xh, yl, yh, zl, zh, iter):
    """Brute force lever-arm search using simulated annealing.

    Parameters
    ----------
    xl, xh, yl, yh, zl, zh : float
        Bounds for the lever-arm search grid.
    iter : int
        Number of grid points along each axis.

    Returns
    -------
    None
        Results are written to ``output.txt``.
    """
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    CDOG_guess_augment = np.array([974.12667502, -80.98121315, -805.07870249])
    # initial_lever_guess = np.array([-30.22391079,  -0.22850613, -21.97254162])
    initial_lever_guess = np.array([-12.5, 0.5, -16])
    offset = 1991.01236648
    # offset = 2076.0242

    GNSS_start, GNSS_end = 25, 40.9
    # GNSS_start, GNSS_end = 31.9, 34.75
    # GNSS_start, GNSS_end = 35.3, 37.6
    (
        GPS_Coordinates,
        GPS_data,
        CDOG_data,
        CDOG_guess,
        gps1_to_others,
    ) = initialize_bermuda(GNSS_start, GNSS_end, CDOG_guess_augment)

    GPS_Coordinates = GPS_Coordinates[::25]
    GPS_data = GPS_data[::25]
    x_grid = np.linspace(xl, xh, iter)
    y_grid = np.linspace(yl, yh, iter)
    z_grid = np.linspace(zl, zh, iter)

    # Open a text file in write mode
    with open(gps_output_path("output.txt"), "w") as file:
        # Write some text to the file
        file.write("Grid Search Results\n")
        file.write("[Lever], [CDOG Estimate], Offset, Time Bias, ESV Bias, RMSE \n")

        iteration = 0
        print("Starting Grid Search")
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    initial_lever = np.array([x, y, z]) + initial_lever_guess
                    (
                        best_lever,
                        best_offset,
                        inversion_result,
                    ) = simulated_annealing_bias(
                        300,
                        CDOG_data,
                        GPS_data,
                        GPS_Coordinates,
                        gps1_to_others,
                        CDOG_guess,
                        initial_lever,
                        dz_array,
                        angle_array,
                        esv_matrix,
                        offset,
                        True,
                    )
                    inversion_guess = inversion_result[:3]
                    time_bias = inversion_result[3]
                    esv_bias = inversion_result[4]

                    transponder_coordinates = findTransponder(
                        GPS_Coordinates, gps1_to_others, best_lever
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
                        best_offset,
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

                    diff_data = (CDOG_full - GPS_full) * 1000
                    RMSE = np.sqrt(np.nanmean(diff_data**2)) / 1000 * 1515 * 100
                    file.write(
                        "[{} ], [{} ], {:.4f}, {:.4e}, {:.4f}, {:.4f}\n".format(
                            np.array2string(np.round(best_lever, 4), separator=", ")[
                                1:-1
                            ],
                            np.array2string(
                                np.round(inversion_guess, 4), separator=", "
                            )[1:-1],
                            best_offset,
                            time_bias,
                            esv_bias,
                            RMSE,
                        )
                    )
                    iteration += 1
                    print(
                        f"Iteration {iteration}/{iter**3}: Lever: "
                        f"{best_lever}, Offset: {best_offset}, RMSE: {RMSE}"
                    )
    print("Grid Search Completed. Results saved to", gps_output_path("output.txt"))


if __name__ == "__main__":
    # Define the grid limits
    xl, xh = -2, 2
    yl, yh = -2, 2
    zl, zh = -2, 2

    iter = 2

    # Call the grid search function
    grid_search_annealing(xl, xh, yl, yh, zl, zh, iter)
