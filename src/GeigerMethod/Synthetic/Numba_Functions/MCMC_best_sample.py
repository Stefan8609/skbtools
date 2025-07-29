import numpy as np
from data import gps_output_path
import matplotlib.pyplot as plt

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


def split_samples(path, n_splits):
    """
    Split the MCMC chain into multiple parts for parallel processing.

    Parameters
    ----------
    path : str
        Path to the .npz file containing the MCMC chain.
    n_splits : int, optional
        Number of splits to create (default is 5).
    """
    dir = gps_output_path(path)
    esv_biases = np.zeros((n_splits, 3))
    for i in range(n_splits):
        # Load the MCMC chain
        data = np.load(f"{dir}/split_{i}.npz")
        logpost = data["logpost"]
        idx_min = np.argmax(logpost)

        # print(f"Best parameters for split {i} \n",
        #       f"Lever guess: {data['lever'][idx_min]}\n",
        #       f"GPS1 grid guess: {data['gps1_grid'][idx_min]}\n",
        #       f"CDOG augment: {data['CDOG_aug'][idx_min]}\n",
        #       f"ESV bias: {data['esv_bias'][idx_min]}\n",
        #       f"Time bias: {data['time_bias'][idx_min]}\n",)
        esv_biases[i, :] = data["esv_bias"][idx_min].T

        print(f"ESV bias for split {i}: {data['esv_bias'][idx_min].T}")
    print(esv_biases[:, 0], esv_biases[:, 1], esv_biases[:, 2])
    print(esv_biases)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    splits = np.arange(n_splits)
    for dog in range(3):
        ax = axes[dog]
        ax.plot(splits, esv_biases[:, dog], marker="o", linestyle="-")
        ax.set_title(f"DOG {dog + 1}")
        ax.set_xlabel("Split index")
        if dog == 0:
            ax.set_ylabel("ESV bias (m)")
        ax.set_xticks(splits)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("ESV Bias Across Splits for Each DOG")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


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
    split_samples("individual_splits_esv_20250717_115727", 10)

    # from datetime import datetime
    #
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #
    # esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_normal.mat"))
    # dz_array = esv["distance"].flatten()
    # angle_array = esv["angle"].flatten()
    # esv_matrix = esv["matrice"]
    #
    # downsample = 1
    # data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    # GPS_Coordinates = data["GPS_Coordinates"][::downsample]
    # GPS_data = data["GPS_data"][::downsample]
    # CDOG_guess = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    #
    # CDOG_all_data = []
    # for i in (1, 3, 4):
    #     tmp = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{i}-camp.mat"))["tags"].astype(
    #         float
    #     )
    #     tmp[:, 1] /= 1e9
    #     CDOG_all_data.append(tmp)
    #
    # offsets = np.array([1866.0, 3175.0, 1939.0])
    # plot_best_sample(
    #     gps_output_path("mcmc_chain_adroit_6_test_xy_lever.npz"),
    #     GPS_Coordinates,
    #     GPS_data,
    #     CDOG_guess,
    #     CDOG_all_data,
    #     offsets,
    #     dz_array,
    #     angle_array,
    #     esv_matrix,
    #     CDOG_num=4,
    #     timestamp=timestamp,
    # )
