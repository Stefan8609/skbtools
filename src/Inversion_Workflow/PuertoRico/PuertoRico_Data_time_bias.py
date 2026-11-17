import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    simulated_annealing_bias,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    initial_bias_geiger,
    residual_inliers,
)
from Inversion_Workflow.PuertoRico.Crossover_Mask import crossover_leg_mask
from plotting.Plot_Modular import range_residual, time_series_plot
from puertorico_data import gps_data_path, gps_output_path


# Config
clock_segment = "after"  # Change to "before" for the first clock segment.

if clock_segment == "before":
    initial_offset = -19.32
elif clock_segment == "after":
    initial_offset = -7458.6

simulated_annealing = True
annealing_iterations = 300
annealing_z_sample = True
save = True
downsample = 1
mask_crossover_legs = True

# Known data
esv_table = sio.loadmat(
    gps_data_path("ESV_Tables/PuertoRico_table_transducer_60_DOG_2950_3050.mat")
)
dz_array = esv_table["distance"].flatten()
angle_array = esv_table["angle"].flatten()
esv_matrix = esv_table["matrice"]


def run_inversion():
    """Run the Puerto Rico inversion for one clock segment."""
    if clock_segment not in ("before", "after"):
        raise ValueError("clock_segment must be 'before' or 'after'")

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
    initial_lever_guess = data["initial_lever_guess"]
    GPS_source_indices = np.arange(len(GPS_data))

    time_mask = (GPS_data >= GPS_time_bounds[0]) & (
        GPS_data <= GPS_time_bounds[1]
    )
    GPS_Coordinates = GPS_Coordinates[time_mask]
    GPS_data = GPS_data[time_mask]
    GPS_source_indices = GPS_source_indices[time_mask]

    transponder_coordinates = findTransponder(
        GPS_Coordinates,
        gps1_to_others,
        initial_lever_guess,
    )
    ranges = np.linalg.norm(transponder_coordinates - CDOG_guess, axis=1)
    transducer_height = ECEF_Geodetic(transponder_coordinates)[2]
    CDOG_height = ECEF_Geodetic(CDOG_guess[None, :])[2][0]
    vertical_distance = transducer_height - CDOG_height
    elevation_angle = np.degrees(np.arcsin(vertical_distance / ranges))
    esv_mask = (
        (vertical_distance >= dz_array[0])
        & (vertical_distance <= dz_array[-1])
        & (elevation_angle >= angle_array[0])
        & (elevation_angle <= angle_array[-1])
    )

    crossover_mask = crossover_leg_mask(GPS_data, crossover_leg_GPS_bounds)
    inversion_mask = esv_mask.copy()
    if mask_crossover_legs:
        inversion_mask &= crossover_mask
    esv_epoch_count = np.count_nonzero(esv_mask)
    inversion_epoch_count = np.count_nonzero(inversion_mask)

    GPS_Coordinates = GPS_Coordinates[inversion_mask][::downsample]
    GPS_data = GPS_data[inversion_mask][::downsample]
    GPS_source_indices = GPS_source_indices[inversion_mask][::downsample]
    transponder_coordinates = transponder_coordinates[inversion_mask][::downsample]
    CDOG_data = CDOG_data[::downsample]

    if len(GPS_data) == 0:
        raise ValueError("No GNSS epochs satisfy the ESV and trajectory masks")

    print(f"Clock segment: {clock_segment}")
    print(f"GNSS epochs inside ESV table: {esv_epoch_count:,}")
    if mask_crossover_legs:
        print(f"GNSS epochs on crossover legs: {inversion_epoch_count:,}")
    else:
        print("Crossover-leg mask disabled")
    print(f"Acoustic tags: {len(CDOG_data):,}")

    preliminary_times, _ = calculateTimesRayTracing_Bias_Real(
        CDOG_guess,
        transponder_coordinates,
        0.0,
        dz_array,
        angle_array,
        esv_matrix,
    )
    offset_center = initial_offset
    print(f"Offset search center: {offset_center:.6f} s")

    if simulated_annealing:
        print("\n ---------Using Simulated Annealing--------- \n")
        best_lever, best_offset, inversion_result = simulated_annealing_bias(
            annealing_iterations,
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            gps1_to_others,
            CDOG_guess,
            initial_lever_guess,
            dz_array,
            angle_array,
            esv_matrix,
            initial_offset=offset_center,
            real_data=True,
            z_sample=annealing_z_sample,
        )
        transponder_coordinates = findTransponder(
            GPS_Coordinates,
            gps1_to_others,
            best_lever,
        )
        (
            inversion_result,
            CDOG_full,
            GPS_full,
            CDOG_clock,
            GPS_clock,
        ) = final_bias_geiger(
            inversion_result[:3],
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            best_offset,
            inversion_result[4],
            inversion_result[3],
            dz_array,
            angle_array,
            esv_matrix,
            real_data=True,
        )
    else:
        print("\n ---------Using Geiger Inversion--------- \n")
        best_lever = initial_lever_guess
        transponder_coordinates = findTransponder(
            GPS_Coordinates,
            gps1_to_others,
            best_lever,
        )
        inversion_result, best_offset = initial_bias_geiger(
            CDOG_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            dz_array,
            angle_array,
            esv_matrix,
            real_data=True,
            initial_offset=offset_center,
        )
        print("\nInitial complete. Offset =", best_offset)

        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]

        print("Starting Final Geiger (fixed offset)...")
        (
            inversion_result,
            CDOG_full,
            GPS_full,
            CDOG_clock,
            GPS_clock,
        ) = final_bias_geiger(
            inversion_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            best_offset,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            real_data=True,
        )
        print("\nFinal complete.")

    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    best_offset_eff = best_offset - time_bias
    diff_data_ms = (CDOG_full - GPS_full) * 1000.0
    RMSE_cm = np.sqrt(np.nanmean(diff_data_ms**2)) / 1000.0 * 1515.0 * 100.0

    print(f"\nEstimate: {np.round(inversion_result, 4)}")
    print(
        f"Lever: {np.round(best_lever, 3)}, "
        f"Offset: {best_offset_eff:.6f}, "
        f"DOG adjustment: {np.round(inversion_guess - CDOG_guess, 3)}"
    )
    print("RMSE:", np.round(RMSE_cm, 3), "cm")

    method_tag = "_SimAnn" if simulated_annealing else ""
    mask_tag = "_XMask" if mask_crossover_legs else ""
    file_tag = f"PuertoRico_{clock_segment}{method_tag}{mask_tag}"
    time_series_figure, _ = time_series_plot(
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        block=False,
        save=False,
        DOG_num=file_tag,
    )
    if save:
        time_series_path = gps_data_path(
            "Figs/Inversion_Workflow",
            f"time_series_plot_{file_tag}.pdf",
        )
        time_series_path.parent.mkdir(parents=True, exist_ok=True)
        time_series_figure.savefig(time_series_path, bbox_inches="tight")
        print(f"Saved {time_series_path}")
    plt.close(time_series_figure)

    times_guess, esv = calculateTimesRayTracing_Bias_Real(
        inversion_guess,
        transponder_coordinates,
        esv_bias,
        dz_array,
        angle_array,
        esv_matrix,
    )
    (
        CDOG_clock_ex,
        CDOG_full_ex,
        GPS_clock_ex,
        GPS_full_ex,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        best_offset,
        0.05,
        CDOG_data,
        GPS_data - time_bias,
        times_guess,
        transponder_coordinates,
        esv,
    )
    range_inliers = residual_inliers(GPS_full_ex - CDOG_full_ex)
    CDOG_full_ex = CDOG_full_ex[range_inliers]
    GPS_full_ex = GPS_full_ex[range_inliers]
    GPS_clock_ex = GPS_clock_ex[range_inliers]
    transponder_coordinates_full = transponder_coordinates_full[range_inliers]
    esv_full = esv_full[range_inliers]

    residual_figure, _ = range_residual(
        transponder_coordinates_full,
        esv_full,
        inversion_guess,
        CDOG_full_ex,
        GPS_full_ex,
        GPS_clock_ex,
        save=False,
        block=False,
    )
    if save:
        residual_path = gps_data_path(
            "Figs/Inversion_Workflow",
            f"range_residual_{file_tag}.pdf",
        )
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        residual_figure.savefig(residual_path, bbox_inches="tight")
        print(f"Saved {residual_path}")
    plt.close(residual_figure)

    if save:
        output_file = gps_output_path(
            "Plot_Data",
            f"{file_tag}_inversion.npz",
        )
        np.savez_compressed(
            output_file,
            CDOG_clock=CDOG_clock,
            CDOG_full=CDOG_full,
            GPS_clock=GPS_clock,
            GPS_full=GPS_full,
            best_offset=best_offset,
            lever=best_lever,
            inversion_result=inversion_result,
            esv_bias=esv_bias,
            time_bias=time_bias,
            CDOG_data=CDOG_data,
            GPS_data=GPS_data,
            GPS_source_indices=GPS_source_indices,
            times_guess=times_guess,
            esv=esv,
            transponder_coordinates=transponder_coordinates,
            clock_segment=clock_segment,
            crossover_leg_mask_enabled=mask_crossover_legs,
            crossover_leg_GPS_bounds=crossover_leg_GPS_bounds,
            simulated_annealing=simulated_annealing,
            annealing_iterations=annealing_iterations,
            offset_search_center=offset_center,
        )
        print(f"Saved inversion plot data to: {output_file}")

    return inversion_result, best_lever, best_offset_eff, RMSE_cm


if __name__ == "__main__":
    run_inversion()
