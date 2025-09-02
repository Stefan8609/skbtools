import numpy as np
import scipy.io as sio
from data import gps_data_path
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import two_pointer_index
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from plotting.MCMC_plots import get_init_params_and_prior


def print_GPS_points_per_segment(
    n_splits,
):
    initial_params, _ = get_init_params_and_prior({})

    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_normal.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 5
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

    gps1_grid_guess = initial_params["gps_grid"]
    lever_guess = initial_params["lever"]
    CDOG_augments = initial_params["CDOG_aug"]
    esv_bias = initial_params["esv_bias"]
    time_bias = initial_params["time_bias"]

    num_points = np.zeros((3, n_splits), dtype=int)
    for i in range(n_splits):
        current_GPS_Coordinates = split_GPS_Coordinates[i]
        current_GPS_data = split_GPS_data[i]

        trans_coords = findTransponder(
            current_GPS_Coordinates, gps1_grid_guess, lever_guess
        )
        # loop each DOG
        for j in range(3):
            inv_guess = CDOG_reference + CDOG_augments[j]
            CDOG_data = CDOG_all_data[j]

            times_guess, esv = calculateTimesRayTracing_Bias_Real(
                inv_guess,
                trans_coords,
                esv_bias[j],
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
                offsets[j],
                0.6,
                CDOG_data,
                current_GPS_data + time_bias[j],
                times_guess,
                trans_coords,
                esv,
                True,
            )

            num_points[j, i] = len(GPS_clock)

    print("Number of GPS points per segment:")
    for i in range(n_splits):
        print(f"Segment {i + 1}: {num_points[:, i]} points (DOG1, DOG2, DOG3)")
    return num_points


if __name__ == "__main__":
    n_splits = 10  # Example number of splits
    print_GPS_points_per_segment(n_splits)
