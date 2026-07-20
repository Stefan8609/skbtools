"""Plot the best sample from the fixed-geometry ESV and timing-bias sampler."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from data import gps_data_path, gps_output_path
from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import find_esv
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from plotting.Plot_Modular import (
    range_residual,
    time_series_plot,
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
CHAIN_FILE = "mcmc_esv_time_chain.npz"

# Use 1, 3, or 4.
# Set to None to plot every DOG present in the chain.
DOG_NUM = 1

# Choose "logpost" or "loglike".
BEST_BY = "loglike"

# Use the same threshold as the sampler.
PAIR_THRESHOLD = 0.4

SAVE = True
SHOW = True


# ------------------------------------------------------------
# Calculate times using the optimized split edges
# ------------------------------------------------------------
def calculate_times_split_edges(
    guess,
    transponder_coordinates,
    esv_biases,
    split_edges,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Calculate travel times using explicit ESV split boundaries."""

    guess = np.asarray(
        guess,
        dtype=float,
    )

    dog_depth = ECEF_Geodetic(guess[np.newaxis, :])[2][0]

    transponder_depth = ECEF_Geodetic(transponder_coordinates)[2]

    difference = transponder_coordinates - guess

    distance = np.sqrt(
        np.sum(
            difference**2,
            axis=1,
        )
    )

    dz = transponder_depth - dog_depth

    beta = np.degrees(
        np.arcsin(
            np.clip(
                dz / distance,
                -1.0,
                1.0,
            )
        )
    )

    esv = find_esv(
        beta,
        dz,
        dz_array,
        angle_array,
        esv_matrix,
    ).copy()

    for split in range(esv_biases.shape[0]):
        start = split_edges[split]
        end = split_edges[split + 1]

        esv[start:end] += esv_biases[split]

    times = distance / esv

    return times, esv


# ------------------------------------------------------------
# Plot one DOG
# ------------------------------------------------------------
def plot_best_sample_for_dog(
    dog_num,
    best_index,
    chain,
    dz_array,
    angle_array,
    esv_matrix,
    output_path,
):
    DOG_order = chain["DOG_order"].astype(int)

    if dog_num not in DOG_order:
        raise ValueError(
            f"DOG {dog_num} is not present in this chain. "
            f"Available DOGs are {DOG_order.tolist()}."
        )

    # The sampler saves only the selected DOGs, so this is
    # the index within the selected-DOG arrays.
    dog_index = np.where(DOG_order == dog_num)[0][0]

    # --------------------------------------------------------
    # Load best-sample parameters
    # --------------------------------------------------------
    esv_biases = chain["esv_bias"][
        best_index,
        dog_index,
    ]

    time_bias = chain["time_bias"][
        best_index,
        dog_index,
    ]

    fixed_CDOG_aug = chain["fixed_CDOG_aug"]

    offsets = chain["offsets"]

    split_edges = chain["split_edges"].astype(int)

    GPS_data = chain["GPS_data"]

    transponder_coordinates = chain["transponder_coordinates"]

    # --------------------------------------------------------
    # Fixed DOG position
    # --------------------------------------------------------
    CDOG_reference = np.array(
        [
            1976671.618715,
            -5069622.53769779,
            3306330.69611698,
        ]
    )

    inv_guess = CDOG_reference + fixed_CDOG_aug[dog_index]

    # --------------------------------------------------------
    # Load DOG acoustic observations
    # --------------------------------------------------------
    CDOG_data = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{dog_num}-camp.mat"))[
        "tags"
    ].astype(float)

    CDOG_data[:, 1] /= 1e9

    # --------------------------------------------------------
    # Best-sample predicted travel times
    # --------------------------------------------------------
    times_guess, esv = calculate_times_split_edges(
        inv_guess,
        transponder_coordinates,
        esv_biases,
        split_edges,
        dz_array,
        angle_array,
        esv_matrix,
    )

    # --------------------------------------------------------
    # Pair the acoustic and GPS observations
    # --------------------------------------------------------
    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        offsets[dog_index],
        PAIR_THRESHOLD,
        CDOG_data,
        GPS_data - time_bias,
        times_guess,
        transponder_coordinates,
        esv,
    )

    # --------------------------------------------------------
    # Print best-sample information
    # --------------------------------------------------------
    print()
    print(f"Best sample for DOG {dog_num}")
    print("======================")
    print("Sample index:", best_index)
    print("DOG augment:", fixed_CDOG_aug[dog_index])
    print("DOG coordinates:", inv_guess)
    print("ESV biases (m/s):", esv_biases)
    print("Timing bias (s):", time_bias)
    print("Timing bias (ms):", time_bias * 1000.0)
    print("Matched pairs:", GPS_clock.shape[0])

    for split in range(4):
        start = split_edges[split]
        end = split_edges[split + 1]

        print(
            f"Split {split + 1}:",
            f"indices {start} to {end - 1},",
            f"ESV bias = {esv_biases[split]:.6f} m/s",
        )

    if GPS_clock.shape[0] == 0:
        raise ValueError(f"No matched observations were found for DOG {dog_num}.")

    # --------------------------------------------------------
    # Time-series comparison
    # --------------------------------------------------------
    time_series_plot(
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        position_noise=0,
        time_noise=0,
        block=False,
        save=SAVE,
        path=str(output_path),
        DOG_num=dog_num,
    )

    # --------------------------------------------------------
    # Range residual
    # --------------------------------------------------------
    range_residual(
        transponder_coordinates_full,
        esv_full,
        inv_guess,
        CDOG_full,
        GPS_full,
        GPS_clock,
        save=SAVE,
        path=str(output_path),
        DOG_num=dog_num,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    if BEST_BY not in (
        "logpost",
        "loglike",
    ):
        raise ValueError('BEST_BY must be either "logpost" or "loglike".')

    chain_path = gps_output_path(CHAIN_FILE)

    chain = np.load(chain_path)

    print("Saved chain keys:")
    print(chain.files)
    print()

    best_index = int(np.argmax(chain[BEST_BY]))

    print("Chain:", chain_path)
    print("Best sample selected by:", BEST_BY)
    print("Best sample index:", best_index)
    print(
        "Log posterior:",
        chain["logpost"][best_index],
    )
    print(
        "Log likelihood:",
        chain["loglike"][best_index],
    )

    # --------------------------------------------------------
    # Load ESV lookup table
    # --------------------------------------------------------
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))

    dz_array = esv_table["distance"].flatten()

    angle_array = esv_table["angle"].flatten()

    esv_matrix = esv_table["matrice"]

    DOG_order = chain["DOG_order"].astype(int)

    if DOG_NUM is None:
        dogs_to_plot = DOG_order

    else:
        dogs_to_plot = np.array(
            [DOG_NUM],
            dtype=int,
        )

    output_path = Path("Figs") / "MCMC" / f"best_{BEST_BY}_{Path(CHAIN_FILE).stem}"

    for dog_num in dogs_to_plot:
        plot_best_sample_for_dog(
            int(dog_num),
            best_index,
            chain,
            dz_array,
            angle_array,
            esv_matrix,
            output_path,
        )

    chain.close()

    if SHOW:
        plt.show()
