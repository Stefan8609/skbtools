import numpy as np
import scipy.io as sio
from numba import njit

from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    initial_bias_geiger,
)
from GeigerMethod.Synthetic.Numba_Functions.Bermuda_Trajectory import bermuda_trajectory
from data import gps_data_path


@njit
def _evaluate(
    CDOG_data,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    inversion_in,
    lever,
    dz_array,
    angle_array,
    esv_matrix,
    offset,
):
    """Evaluate RMSE for a given lever and offset.

    Parameters
    ----------
    CDOG_data, GPS_data : ndarray
        Raw arrival data for the DOG and GPS receivers.
    GPS_Coordinates : ndarray
        ``(N, 4, 3)`` array of GPS positions.
    gps1_to_others : ndarray
        Relative positions of the other GPS receivers.
    inversion_in : ndarray
        Current estimate of ``(x, y, z, time_bias, esv_bias)``.
    lever : ndarray
        Lever arm vector from GPS1 to the transponder.
    dz_array, angle_array, esv_matrix : ndarray
        Effective sound velocity lookup tables.
    offset : float
        Initial timing offset between DOG and GPS data.

    Returns
    -------
    tuple
        ``(inversion_result, RMSE)`` best estimate and associated error.
    """
    inversion_guess = inversion_in[:3]
    time_bias = inversion_in[3]
    esv_bias = inversion_in[4]

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
        True,
    )

    # Calculate the RMSE
    RMSE = np.sqrt(np.mean((CDOG_full - GPS_full) ** 2))
    return inversion_result, RMSE


@njit
def simulated_annealing_real(
    max_iterations,
    CDOG_data,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    initial_guess,
    initial_lever,
    dz_array,
    angle_array,
    esv_matrix,
    offset=2000.0,
):
    """Run simulated annealing to estimate lever arm and biases.

    Parameters
    ----------
    max_iterations : int
        Number of annealing steps to perform.
    CDOG_data, GPS_data : ndarray
        Arrival data for the DOG and GPS receivers.
    GPS_Coordinates : ndarray
        ``(N, 4, 3)`` array of GPS positions.
    gps1_to_others : ndarray
        Relative offsets of the other GPS receivers.
    initial_guess : ndarray
        Starting estimate of the CDOG position.
    initial_lever : ndarray
        Initial guess for the lever arm.
    dz_array, angle_array, esv_matrix : ndarray
        Effective sound velocity lookup tables.
    offset : float, optional
        Initial timing offset between DOG and GPS data.

    Returns
    -------
    tuple
        ``(inversion_estimate, best_lever, RMSE)`` final state and error.
    """
    inversion_estimate = np.array(
        [initial_guess[0], initial_guess[1], initial_guess[2], 0.0, 0.0]
    )
    best_rmse = np.inf
    best_lever = initial_lever

    k = 0
    while k < max_iterations:
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / (max_iterations)))
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = best_lever + displacement

        # Calculate the RMSE
        inversion_estimate, RMSE = _evaluate(
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            gps1_to_others,
            inversion_estimate,
            lever,
            dz_array,
            angle_array,
            esv_matrix,
            offset,
        )
        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever.copy()
            offset = offset - inversion_estimate[3]

        if k % 20 == 0:
            print(
                "Iteration: ",
                k,
                "RMSE: ",
                np.round(RMSE * 100 * 1515, 3),
                "Offset: ",
                np.round(offset, 4),
                "Lever: ",
                np.round(best_lever, 3),
            )
        k += 1
    return inversion_estimate, best_lever, RMSE


if __name__ == "__main__":
    esv_table_generate = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array_generate = esv_table_generate["distance"].flatten()
    angle_array_generate = esv_table_generate["angle"].flatten()
    esv_matrix_generate = esv_table_generate["matrice"]

    esv_table = sio.loadmat(
        gps_data_path("ESV_Tables/global_table_esv_realistic_perturbed.mat")
    )
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = bermuda_trajectory(
        time_noise, position_noise, dz_array, angle_array, esv_matrix
    )

    # true_offset = 1991.01236648
    true_offset = 2003.0
    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )
    # initial_lever = np.array([-13.0, 0.0, -13])
    initial_lever = np.array([-8.74068827, 7.78977386, -7.27690523])
    #    lever = np.array([-12.48862757, 0.22622633, -15.89601934])
    initial_guess = CDOG + np.array([100.0, 100.0, 200.0])

    initial_transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, initial_lever
    )
    estimate, offset = initial_bias_geiger(
        initial_guess,
        CDOG_data,
        GPS_data,
        initial_transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
    )

    # GPS_Coordinates = GPS_Coordinates[::100]
    # GPS_data = GPS_data[::100]

    for off_adj in np.linspace(-3, 3, 7):
        inversion_result, best_lever, RMSE = simulated_annealing_real(
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
            offset=offset + off_adj,
        )

        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]

        print("Input Offset: ", np.round(offset + off_adj, 4))
        print("Lever :", np.round(best_lever, 3))
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion:", np.round(inversion_result, 2))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100),
            "RMSE: ",
            np.round(RMSE * 100 * 1515, 3),
        )
        print("\n")
