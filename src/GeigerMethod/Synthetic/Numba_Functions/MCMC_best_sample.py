import numpy as np
from data import gps_output_path, gps_data_path
import scipy.io as sio

from plotting.Plot_Modular import (
    time_series_plot,
    range_residual,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_time_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import two_pointer_index
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from GeigerMethod.Synthetic.Numba_Functions.ESV_bias_split import (
    calculateTimesRayTracing_split,
)


def load_min_logpost_params(npz_path):
    """
    Load an MCMC chain saved with np.savez and return the parameter set
    corresponding to the minimum logpost.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file (e.g. gps_output_path("mcmc_chain_good.npz")).

    Returns
    -------
    dict
        A dictionary with keys 'lever', 'gps1_grid', 'CDOG_aug',
        'esv_bias', 'time_bias', 'logpost', each holding the value
        (or array) at the minimum-logpost index.
    """
    data = np.load(npz_path)
    logpost = data["logpost"]
    idx_min = np.argmax(logpost)

    return {
        "lever": data["lever"][idx_min],
        "gps1_grid": data["gps1_grid"][idx_min],
        "CDOG_aug": data["CDOG_aug"][idx_min],
        "esv_bias": data["esv_bias"][idx_min],
        "time_bias": data["time_bias"][idx_min],
        "logpost": logpost[idx_min],
    }


def plot_best_sample(
    npz_path,
    GPS_Coordinates,
    GPS_data,
    CDOG_guess,
    CDOG_all_data,
    offsets,
    dz_array,
    angle_array,
    esv_matrix,
    CDOG_num=3,
    timestamp=None,
):
    """
    Plot the best sample from an MCMC chain based on minimum logpost.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing the MCMC chain.
    downsample : int, optional
        Step used when plotting chain values (default is 1).
    """
    CDOG_to_index = {1: 0, 3: 1, 4: 2}
    CDOG_index = CDOG_to_index[CDOG_num]

    best = load_min_logpost_params(npz_path)

    lever_guess = best["lever"]
    gps1_grid_guess = best["gps1_grid"]
    CDOG_augments = best["CDOG_aug"]
    esv_bias = best["esv_bias"]
    time_bias = best["time_bias"]
    logpost = best["logpost"]

    print("Best parameters:")
    print("Lever guess:", lever_guess)
    print("GPS1 grid guess:", gps1_grid_guess)
    print("CDOG augment:", CDOG_augments[CDOG_index])
    print("ESV bias:", esv_bias)
    print("Time bias:", time_bias)
    print("Log posterior:", logpost)

    split_esv = False
    if esv_bias.ndim == 2:
        split_esv = True
    trans_coords = findTransponder(GPS_Coordinates, gps1_grid_guess, lever_guess)

    inv_guess = CDOG_guess + CDOG_augments[CDOG_index]
    CDOG_data = CDOG_all_data[CDOG_index]

    if split_esv:
        times_guess, esv = calculateTimesRayTracing_split(
            inv_guess,
            trans_coords,
            esv_bias[CDOG_index],
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            inv_guess,
            trans_coords,
            esv_bias[CDOG_index],
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
        offsets[CDOG_index],
        0.6,
        CDOG_data,
        GPS_data + time_bias[CDOG_index],
        times_guess,
        trans_coords,
        esv,
        True,
    )

    # Plotting
    time_series_plot(
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        position_noise=0,
        time_noise=0,
        block=True,
        save=True,
        path="Figs/MCMC",
        timestamp=timestamp,
    )
    range_residual(
        transponder_coordinates_full,
        esv_full,
        inv_guess,
        CDOG_full,
        GPS_full,
        GPS_clock,
        save=True,
        path="Figs/MCMC",
        timestamp=timestamp,
    )


# Example usage:
if __name__ == "__main__":
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_normal.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 1
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"][::downsample]
    GPS_data = data["GPS_data"][::downsample]
    CDOG_guess = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    CDOG_all_data = []
    for i in (1, 3, 4):
        tmp = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{i}-camp.mat"))["tags"].astype(
            float
        )
        tmp[:, 1] /= 1e9
        CDOG_all_data.append(tmp)

    offsets = np.array([1866.0, 3175.0, 1939.0])
    plot_best_sample(
        gps_output_path("mcmc_chain_adroit_5_test_xy_lever.npz"),
        GPS_Coordinates,
        GPS_data,
        CDOG_guess,
        CDOG_all_data,
        offsets,
        dz_array,
        angle_array,
        esv_matrix,
        CDOG_num=4,
        timestamp=timestamp,
    )
