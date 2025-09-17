import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from data import gps_data_path, gps_output_path
from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.MCMC_Functions.MCMC_best_sample import load_min_logpost_params
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_split,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from plotting.save import save_plot


def tide_residual_correlation(chain, CDOG_num=3):
    best = load_min_logpost_params(chain)

    lever_guess = best["lever"]
    gps1_grid_guess = best["gps1_grid"]
    CDOG_augments = best["CDOG_aug"]
    esv_bias = best["esv_bias"]
    time_bias = best["time_bias"]

    offsets = np.array([1866.0, 3175.0, 1939.0])

    data = np.load(
        gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{CDOG_num}.npz")
    )
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    split_esv = False
    if esv_bias.ndim == 2:
        split_esv = True
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_grid_guess, lever_guess
    )

    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    CDOG_to_index = {1: 0, 3: 1, 4: 2}
    CDOG_index = CDOG_to_index[CDOG_num]
    CDOG_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    inv_guess = CDOG_base + CDOG_augments[CDOG_index]

    CDOG_data = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{CDOG_num}-camp.mat"))[
        "tags"
    ].astype(float)
    CDOG_data[:, 1] /= 1e9

    if split_esv:
        times_guess, esv = calculateTimesRayTracing_split(
            inv_guess,
            transponder_coordinates,
            esv_bias[CDOG_index],
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            inv_guess,
            transponder_coordinates,
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
        transponder_coordinates,
        esv,
        True,
    )

    residuals = GPS_full - CDOG_full
    _, _, elev = ECEF_Geodetic(transponder_coordinates_full)

    # Center data
    residuals -= np.mean(residuals)
    elev = elev - np.mean(elev)

    # Normalize
    residuals /= np.std(residuals)
    elev /= np.std(elev)

    # Compute correlation
    correlation = np.correlate(residuals, elev, mode="full")

    lags = np.arange(-len(residuals) + 1, len(residuals))
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlation)
    plt.title(f"Tidal Residual Correlation (DOG {CDOG_num})")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid()
    plt.xlim(-500, 500)
    plt.axvline(0, color="k", linestyle="--", label="Zero Lag")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(np.corrcoef(residuals, elev)[0, 1])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(CDOG_clock, residuals, alpha=0.4)
    ax.scatter(CDOG_clock, elev, alpha=0.05)
    ax.set_title(f"Residuals and Elevation vs Time (DOG {CDOG_num})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Values")
    ax.legend(["Residuals", "Elevation"], loc="lower right")
    ax.grid()
    ax.text(
        0.05,
        0.95,
        f"Correlation: {np.corrcoef(residuals, elev)[0, 1]:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    fig.tight_layout()
    plt.show()

    func_name = f"tidal_residual_correlation_DOG_{CDOG_num}"
    save_plot(fig, func_name)

    return


if __name__ == "__main__":
    file = "mcmc_chain_9_16_new_table"
    chain = gps_output_path(f"{file}.npz")

    tide_residual_correlation(chain, CDOG_num=3)
