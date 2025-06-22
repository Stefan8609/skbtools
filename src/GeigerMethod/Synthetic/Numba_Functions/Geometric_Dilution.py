import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from Numba_time_bias import compute_Jacobian_biased, calculateTimesRayTracing_Bias
from Modular_Synthetic import modular_synthetic
from data import gps_data_path


def geometric_dilution(interval):
    """Plot the geometric dilution of precision for a given interval.

    Parameters
    ----------
    interval : int
        Step between successive transponder points used in the GDOP
        calculation.

    Returns
    -------
    None
        A figure showing GDOP versus time is displayed.
    """

    table_str = "global_table_esv_realistic_perturbed"

    (
        inversion_result,
        CDOG_data,
        CDOG_full,
        GPS_data,
        GPS_full,
        CDOG_clock,
        GPS_clock,
        transponder_coordinates,
        GPS_Coordinates,
        offset,
    ) = modular_synthetic(
        2 * 10**-5,
        2 * 10**-2,
        0,
        0,
        "global_table_esv",
        table_str,
        generate_type=1,
        inversion_type=0,
        plot=False,
    )

    esv_table = sio.loadmat(gps_data_path(f"{table_str}.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    CDOG_guess = inversion_result[:3]
    esv_bias = inversion_result[4]
    times, esv = calculateTimesRayTracing_Bias(
        CDOG_guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
    )

    length = np.size(times[::interval])
    GDOP_array = np.zeros(length)
    idx = 0
    for i in range(0, len(transponder_coordinates) - interval, interval):
        coords_inter = transponder_coordinates[i : i + interval]
        times_inter = times[i : i + interval]
        esv_inter = esv[i : i + interval]
        J = compute_Jacobian_biased(
            CDOG_guess, coords_inter, times_inter, esv_inter, esv_bias
        )
        Q = np.linalg.inv(J.T @ J)
        GDOP_array[idx] = np.sqrt(np.trace(Q))
        idx += 1

    std = np.std(GDOP_array)
    plt.scatter(GPS_data[::interval] / 3600, GDOP_array, s=1)
    plt.ylim(0, 3 * std)
    plt.title(f"Geometric Dilution of Precision for intervals of {interval} points")
    plt.ylabel("GDOP")
    plt.xlabel("Time (hours)")
    plt.show()


if __name__ == "__main__":
    geometric_dilution(50)
