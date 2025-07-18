import numpy as np
import scipy.io as sio
from datetime import datetime
import os
from numba.typed import List
from MCMC_sampler import mcmc_sampler
from data import gps_data_path, gps_output_path


"""Function that will take only the GPS points in the
    individual splits and calculate a reasonable ESV bias for each split.

    Will use MCMC with reasonable priors on levers and DOG position"""


def individual_splits_esv(
    n_splits,
    n_iters,
    initial_params,
    dz_array,
    angle_array,
    esv_matrix,
    GPS_Coordinates,
    GPS_data,
    CDOG_reference,
    CDOG_all_data,
    offsets,
):
    # Get split GPS coordinates
    split_GPS_Coordinates = np.array_split(GPS_Coordinates, n_splits)
    split_GPS_data = np.array_split(GPS_data, n_splits)

    # Current time for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = gps_output_path(f"individual_splits_esv_{timestamp}")
    os.makedirs(path, exist_ok=True)

    init_lever = initial_params["lever"]
    init_gps_grid = initial_params["gps_grid"]
    init_aug = initial_params["CDOG_aug"]
    init_ebias = initial_params["esv_bias"]
    init_tbias = initial_params["time_bias"]

    # Reshape esv bias to ensure it is a 2D array
    init_ebias = np.tile(init_ebias.reshape(-1, 1), (1, 1))

    # Run MCMC sampler for each split
    for i in range(n_splits):
        print(f"Processing split {i + 1}/{n_splits}")

        # Extract the current split of GPS coordinates
        current_GPS_coordinates = split_GPS_Coordinates[i]
        current_GPS_data = split_GPS_data[i]

        # Run MCMC sampler
        (
            lever_chain,
            gps_chain,
            cdog_aug_chain,
            ebias_chain,
            tbias_chain,
            loglike_chain,
            logpost_chain,
        ) = mcmc_sampler(
            n_iters=n_iters,
            initial_lever_base=init_lever,
            initial_gps_grid=init_gps_grid,
            initial_CDOG_augments=init_aug,
            initial_esv_bias=init_ebias,
            initial_time_bias=init_tbias,
            dz_array=dz_array,
            angle_array=angle_array,
            esv_matrix=esv_matrix,
            GPS_Coordinates=current_GPS_coordinates,
            GPS_data=current_GPS_data,
            CDOG_reference=CDOG_reference,
            CDOG_all_data=CDOG_all_data,
            offsets=offsets,
            proposal_lever=np.array([0.005, 0.005, 0.01]),
            proposal_gps_grid=0.0,
            proposal_CDOG_aug=0.01,
            proposal_esv_bias=0.05,
            proposal_time_bias=0.000005,
        )

        # Save results for the current split
        chain = {
            "lever": lever_chain,
            "gps1_grid": gps_chain,
            "CDOG_aug": cdog_aug_chain,
            "esv_bias": ebias_chain,
            "time_bias": tbias_chain,
            "loglike": loglike_chain,
            "logpost": logpost_chain,
        }
        np.savez(f"{path}/split_{i}", **chain)
    return


if __name__ == "__main__":
    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_normal.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 50
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

    typed_CDOG_all_data = List()
    for arr in CDOG_all_data:
        typed_CDOG_all_data.append(arr)

    offsets = np.array([1866.0, 3175.0, 1939.0])

    # initial parameters
    init_lever = np.array([-12.4659, 9.6021, -16.2993])
    init_gps_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.393414, -4.223503, 0.029415],
            [-12.095685, -0.945685, 0.004397],
            [-8.686741, 5.169188, -0.024993],
        ]
    )
    init_aug = np.array(
        [
            [-397.63809, 371.47355, 773.26347],
            [825.31541, -110.93683, -734.15039],
            [236.27742, -1307.44426, -2189.59746],
        ]
    )
    init_ebias = np.array([-0.4775, -0.3199, 0.1122])
    init_tbias = np.array([0.01518602, 0.015779, 0.018898])

    initial_params = {
        "lever": init_lever,
        "gps_grid": init_gps_grid,
        "CDOG_aug": init_aug,
        "esv_bias": init_ebias,
        "time_bias": init_tbias,
    }

    individual_splits_esv(
        4,
        50000,
        initial_params,
        dz_array,
        angle_array,
        esv_matrix,
        GPS_Coordinates,
        GPS_data,
        CDOG_reference,
        typed_CDOG_all_data,
        offsets,
    )
