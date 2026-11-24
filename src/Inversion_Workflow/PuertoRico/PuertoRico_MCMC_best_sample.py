from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline import pair_indices
from Inversion_Workflow.PuertoRico.Crossover_Mask import crossover_leg_mask
from Inversion_Workflow.PuertoRico.PuertoRico_MCMC_sampler import (
    PAIR_THRESHOLD_SECONDS,
    calculate_travel_times,
    filter_pair_outliers,
)
from plotting.Plot_Modular import range_residual, time_series_plot
from puertorico_data import gps_data_path, gps_output_path


def load_best_sample(npz_path, loglike=False):
    """Load the maximum-posterior or maximum-likelihood chain sample."""
    with np.load(npz_path) as chain:
        score = chain["loglike"] if loglike else chain["logpost"]
        best_index = np.argmax(score)
        return {
            "lever": chain["lever"][best_index],
            "CDOG_aug": chain["CDOG_aug"][best_index],
            "esv_bias": chain["esv_bias"][best_index],
            "time_bias": chain["time_bias"][best_index],
            "loglike": chain["loglike"][best_index],
            "logpost": chain["logpost"][best_index],
            "init_lever": chain["init_lever"],
            "init_CDOG_aug": chain["init_CDOG_aug"],
            "init_esv_bias": chain["init_esv_bias"],
            "init_time_bias": chain["init_time_bias"],
        }


def plot_best_sample(
    npz_path,
    clock_segment="before",
    trajectory_mode="cross",
    downsample=1,
    loglike=False,
    save=True,
):
    """Plot the best joint-chain sample for one Puerto Rico survey."""
    survey_indices = {"before": 0, "after": 1}
    if clock_segment not in survey_indices:
        raise ValueError("clock_segment must be 'before' or 'after'")
    if trajectory_mode not in ("cross", "full"):
        raise ValueError("trajectory_mode must be 'cross' or 'full'")

    survey = survey_indices[clock_segment]
    best = load_best_sample(npz_path, loglike=loglike)
    CDOG_augment = np.asarray(best["CDOG_aug"])
    if CDOG_augment.ndim == 2:
        CDOG_augment = CDOG_augment[0]
    survey_esv_bias = np.atleast_1d(best["esv_bias"][survey])
    survey_time_bias = best["time_bias"][survey]
    initial_CDOG_augment = np.asarray(best["init_CDOG_aug"])
    if initial_CDOG_augment.ndim == 2:
        initial_CDOG_augment = initial_CDOG_augment[0]
    initial_esv_bias = np.atleast_1d(best["init_esv_bias"][survey])
    initial_time_bias = best["init_time_bias"][survey]

    data = np.load(
        gps_data_path(
            f"GPS_Data/Processed_GPS_Receivers_DOG_1_{clock_segment}_clock_loss.npz"
        )
    )
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    GPS_time_bounds = data["GPS_time_bounds"]
    crossover_leg_GPS_bounds = data["crossover_leg_GPS_bounds"]
    CDOG_data = data["CDOG_data"]
    CDOG_guess = data["CDOG_guess"]
    gps1_to_others = data["gps1_to_others"]

    trajectory_mask = (GPS_data >= GPS_time_bounds[0]) & (
        GPS_data <= GPS_time_bounds[1]
    )
    if trajectory_mode == "cross":
        trajectory_mask &= crossover_leg_mask(GPS_data, crossover_leg_GPS_bounds)

    GPS_Coordinates = GPS_Coordinates[trajectory_mask][::downsample]
    GPS_data = GPS_data[trajectory_mask][::downsample]

    dz_array, angle_array, esv_matrix = load_esv_table(
        gps_data_path("ESV_Tables/PuertoRico_table_transducer_60_DOG_2950_3050.mat")
    )

    transponder_coordinates = findTransponder(
        GPS_Coordinates,
        gps1_to_others,
        best["lever"],
    )
    inversion_guess = CDOG_guess + CDOG_augment
    survey_bounds = np.array([[0, len(GPS_data)]])
    times_guess, esv = calculate_travel_times(
        transponder_coordinates,
        CDOG_augment,
        survey_esv_bias[None, :],
        CDOG_guess,
        survey_bounds,
        dz_array,
        angle_array,
        esv_matrix,
    )

    initial_transponder_coordinates = findTransponder(
        GPS_Coordinates,
        gps1_to_others,
        best["init_lever"],
    )
    initial_times, _ = calculate_travel_times(
        initial_transponder_coordinates,
        initial_CDOG_augment,
        initial_esv_bias[None, :],
        CDOG_guess,
        survey_bounds,
        dz_array,
        angle_array,
        esv_matrix,
    )

    offsets = np.array([-19.32504165, -7458.62104948])
    cdog_indices, gps_indices = pair_indices(
        offsets[survey],
        PAIR_THRESHOLD_SECONDS,
        CDOG_data,
        GPS_data - initial_time_bias,
        initial_times,
    )
    cdog_indices, gps_indices = filter_pair_outliers(
        offsets[survey],
        CDOG_data,
        GPS_data - initial_time_bias,
        initial_times,
        cdog_indices,
        gps_indices,
    )
    CDOG_clock = (
        CDOG_data[cdog_indices, 0] + CDOG_data[cdog_indices, 1] - offsets[survey]
    )
    GPS_full = times_guess[gps_indices]
    GPS_clock = GPS_data[gps_indices] - survey_time_bias + GPS_full
    CDOG_full = CDOG_clock - (GPS_clock - GPS_full)
    transponder_coordinates_full = transponder_coordinates[gps_indices]
    esv_full = esv[gps_indices]

    print(f"Best {'loglikelihood' if loglike else 'log-posterior'} sample")
    print("Survey:", clock_segment)
    print("Lever:", best["lever"])
    print("CDOG augment:", CDOG_augment)
    print("ESV bias:", survey_esv_bias)
    print("Time bias:", survey_time_bias)
    print("Log posterior:", best["logpost"])
    print("Log likelihood:", best["loglike"])

    time_series_figure, _ = time_series_plot(
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        block=False,
        save=False,
        zoom_start=CDOG_clock[0],
        DOG_num=f"PuertoRico_{clock_segment}",
    )
    residual_figure, _ = range_residual(
        transponder_coordinates_full,
        esv_full,
        inversion_guess,
        CDOG_full,
        GPS_full,
        GPS_clock,
        block=False,
        save=False,
        DOG_num=f"PuertoRico_{clock_segment}",
    )

    if save:
        output_directory = gps_data_path("Figs/MCMC")
        output_directory.mkdir(parents=True, exist_ok=True)
        chain_name = Path(npz_path).stem
        file_stem = f"{chain_name}_{clock_segment}"
        time_series_figure.savefig(
            output_directory / f"{file_stem}_time_series.pdf",
            bbox_inches="tight",
        )
        residual_figure.savefig(
            output_directory / f"{file_stem}_range_residual.pdf",
            bbox_inches="tight",
        )

    return time_series_figure, residual_figure


if __name__ == "__main__":
    clock_segment = "after"  # "before" or "after"
    trajectory_mode = "cross"  # "cross" or "full"
    downsample = 1
    loglike = False
    save = True

    plot_best_sample(
        gps_output_path(f"mcmc_chain_puertorico_{trajectory_mode}.npz"),
        clock_segment=clock_segment,
        trajectory_mode=trajectory_mode,
        downsample=downsample,
        loglike=loglike,
        save=save,
    )
    plt.show()
