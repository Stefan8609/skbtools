import numpy as np
import scipy.io as sio
from data import gps_output_path, gps_data_path
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import two_pointer_index
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from plotting.MCMC_best_sample import (
    load_min_logpost_params,
)
from plotting.Plot_Modular import time_series_plot


def plot_combined_segments(n_splits, path, DOG_num=0):
    """
    Plot the combined segments from multiple MCMC runs.

    Parameters:
    path (str): Path to the directory containing the split data files.
    """
    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_normal.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 1
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"][::downsample]
    GPS_data = data["GPS_data"][::downsample]
    CDOG_reference = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    CDOG_all_data = []
    for i in (1, 3, 4):
        tmp = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{i}-camp.mat"))["tags"].astype(
            float
        )
        tmp[:, 1] /= 1e9
        CDOG_all_data.append(tmp)

    offsets = np.array([1866.0, 3175.0, 1939.0])

    split_GPS_Coordinates = np.array_split(GPS_Coordinates, n_splits)
    split_GPS_data = np.array_split(GPS_data, n_splits)

    CDOG_clock_all = np.array([])
    CDOG_full_all = np.array([])
    GPS_clock_all = np.array([])
    GPS_full_all = np.array([])
    for i in range(n_splits):
        current_GPS_Coordinates = split_GPS_Coordinates[i]
        current_GPS_data = split_GPS_data[i]
        segment_best = load_min_logpost_params(f"{path}/split_{i}.npz")

        lever = segment_best["lever"]
        gps_grid = segment_best["gps1_grid"]
        CDOG_aug = segment_best["CDOG_aug"]
        esv_bias = segment_best["esv_bias"]
        time_bias = segment_best["time_bias"]

        trans_coords = findTransponder(current_GPS_Coordinates, gps_grid, lever)

        inv_guess = CDOG_reference + CDOG_aug[DOG_num]
        CDOG_data = CDOG_all_data[DOG_num]

        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            inv_guess,
            trans_coords,
            esv_bias[DOG_num],
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
            offsets[DOG_num],
            0.6,
            CDOG_data,
            current_GPS_data + time_bias[DOG_num],
            times_guess,
            trans_coords,
            esv,
            True,
        )
        CDOG_clock_all = np.append(CDOG_clock_all, CDOG_clock)
        CDOG_full_all = np.append(CDOG_full_all, CDOG_full)
        GPS_clock_all = np.append(GPS_clock_all, GPS_clock)
        GPS_full_all = np.append(GPS_full_all, GPS_full)

    time_series_plot(
        CDOG_clock_all,
        CDOG_full_all,
        GPS_clock_all,
        GPS_full_all,
        position_noise=0,
        time_noise=0,
        block=True,
        save=True,
        path="Figs",
        chain_name="7_individual_splits_esv_20250806_143356",
        segments=n_splits,
    )


if __name__ == "__main__":
    path = gps_output_path("7_individual_splits_esv_20250806_143356")
    plot_combined_segments(7, path)
