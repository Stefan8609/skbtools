import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Inversion.Numba_xAline import (
    OFFSET_SEARCH_THRESHOLD_SECONDS,
    two_pointer_index,
)


@njit(cache=True)
def _rmse_cm_for_offset(
    offset,
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    thresh=OFFSET_SEARCH_THRESHOLD_SECONDS,
):
    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset,
        thresh,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )[1:4]

    N = len(CDOG_full)
    if N == 0:
        return np.nan, 0

    # integer-cycle correction (same as elsewhere)
    for i in range(N):
        diff = GPS_full[i] - CDOG_full[i]
        if abs(diff) >= 0.9:
            CDOG_full[i] += np.round(diff)

    rmse_cm = (
        np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2))
        * NOMINAL_SOUND_SPEED_MPS
        * 100.0
    )
    return rmse_cm, N


@njit(cache=True)
def _coherence_obj_for_offset(
    offset,
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    thresh=OFFSET_SEARCH_THRESHOLD_SECONDS,
):
    """
    Phase-coherence objective:
      r = CDOG_full - GPS_full
      w = wrap(r) into [-0.5, 0.5)
      R = |mean(exp(i 2π w))|
      obj = (1 - R) / sqrt(N)   (smaller is better)
    """
    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset,
        thresh,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )[1:4]

    N = len(CDOG_full)
    if N < 5:
        return np.nan, N, np.nan

    r = CDOG_full - GPS_full
    w = (r + 0.5) % 1.0 - 0.5  # wrap to [-0.5, 0.5)
    # Numba supports complex128; this is fine
    R = np.abs(np.mean(np.exp(1j * 2.0 * np.pi * w)))

    obj = (1.0 - R) / np.sqrt(N)
    return obj, N, R


def plot_integer_pick_metrics_dog(
    center_offset,
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    half_window=100.0,
    step=0.1,
    thresh=OFFSET_SEARCH_THRESHOLD_SECONDS,
    vlines=None,
    title_suffix="",
    filtered_CDOG_data=None,
):
    """
    vlines: optional dict like {"best_offset": value, "offset_hint": value}
    """
    offsets = np.round(
        np.arange(
            center_offset - half_window, center_offset + half_window + step, step
        ),
        8,
    )

    rmse = np.empty(offsets.shape[0], dtype=np.float64)
    nmatch = np.empty(offsets.shape[0], dtype=np.float64)
    coh_obj = np.empty(offsets.shape[0], dtype=np.float64)
    coh_R = np.empty(offsets.shape[0], dtype=np.float64)
    filtered_coh_obj = None
    filtered_nmatch = None
    if filtered_CDOG_data is not None:
        filtered_coh_obj = np.empty(offsets.shape[0], dtype=np.float64)
        filtered_nmatch = np.empty(offsets.shape[0], dtype=np.float64)

    for i in range(offsets.shape[0]):
        off = offsets[i]
        rmse[i], nmatch[i] = _rmse_cm_for_offset(
            off,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            thresh=thresh,
        )
        coh_obj[i], _, coh_R[i] = _coherence_obj_for_offset(
            off,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            thresh=thresh,
        )
        if filtered_coh_obj is not None:
            filtered_coh_obj[i], filtered_nmatch[i], _ = _coherence_obj_for_offset(
                off,
                filtered_CDOG_data,
                GPS_data,
                travel_times,
                transponder_coordinates,
                esv,
                thresh=thresh,
            )

    finite = np.isfinite(coh_obj)
    best_off = offsets[np.nanargmin(coh_obj)] if finite.any() else np.nan

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    axes[0].plot(offsets, rmse)
    axes[0].set_ylabel(r"$\mathcal{R}$ (cm)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].grid(True)

    axes[1].plot(offsets, coh_obj, label="All receptions")
    if filtered_coh_obj is not None:
        axes[1].plot(
            offsets,
            filtered_coh_obj,
            label="Intervals at least 0.95 s",
        )
        filtered_best = offsets[np.nanargmin(filtered_coh_obj)]
        axes[1].axvline(
            filtered_best,
            color="tab:orange",
            linestyle="--",
            label="Filtered optimum",
        )
    axes[1].set_ylabel(r"$v_\circ$")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].grid(True)

    axes[2].plot(offsets, nmatch, label="All receptions")
    if filtered_nmatch is not None:
        axes[2].plot(offsets, filtered_nmatch, label="Intervals at least 0.95 s")
        axes[2].legend()
    axes[2].set_ylabel("Pairings")
    axes[2].set_xlabel(r"$\delta t$ (s)")
    axes[2].grid(True)

    # Annotate best by Circular Variance
    if np.isfinite(best_off):
        axes[0].axvline(best_off, color="tab:blue", linestyle="--")
        axes[1].axvline(
            best_off,
            color="tab:blue",
            linestyle="--",
            label="All-reception optimum",
        )
    if filtered_coh_obj is not None:
        axes[1].legend()

    # Additional vlines (hint / solved offset etc.)
    if vlines is not None:
        for _, val in vlines.items():
            if val is None:
                continue
            for ax in axes:
                ax.axvline(val, linestyle=":")
        # Don't overwrite the title if coherence already set it
        if not np.isfinite(best_off):
            axes[0].set_title(f"Integer-pick metrics{title_suffix}")

    plt.tight_layout()
    plt.show()

    return offsets, rmse, coh_obj, nmatch, coh_R, best_off


def plot_puertorico_metrics():
    """Plot after-segment metrics using the schematic transducer lever."""

    from geometry.ECEF_Geodetic import ECEF_Geodetic
    from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
        calculateTimesRayTracing_Bias_Real,
    )
    from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
    from Inversion_Workflow.Inversion.Numba_xAline import filter_close_receptions
    from puertorico_data import gps_data_path, gps_output_path

    clock_segment = "after"
    center_offset = -7470
    processed_path = gps_data_path(
        f"GPS_Data/Processed_GPS_Receivers_DOG_1_{clock_segment}_clock_loss.npz"
    )
    before_inversion_path = gps_output_path(
        "Plot_Data", "PuertoRico_before_inversion.npz"
    )
    dz_array, angle_array, esv_matrix = load_esv_table(
        gps_data_path("ESV_Tables/PuertoRico_table_transducer_60_DOG_2950_3050.mat")
    )

    with np.load(before_inversion_path) as before_inversion:
        CDOG_guess = before_inversion["inversion_result"][:3].copy()

    with np.load(processed_path) as processed:
        GPS_Coordinates = processed["GPS_Coordinates"]
        GPS_data = processed["GPS_data"]
        GPS_time_bounds = processed["GPS_time_bounds"]
        CDOG_data = processed["CDOG_data"]
        schematic_lever = processed["initial_lever_guess"]

        time_mask = (GPS_data >= GPS_time_bounds[0]) & (GPS_data <= GPS_time_bounds[1])
        GPS_Coordinates = GPS_Coordinates[time_mask]
        GPS_data = GPS_data[time_mask]

        transponder_coordinates = findTransponder(
            GPS_Coordinates,
            processed["gps1_to_others"],
            schematic_lever,
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

        GPS_data = GPS_data[esv_mask]
        transponder_coordinates = transponder_coordinates[esv_mask]
        travel_times, esv = calculateTimesRayTracing_Bias_Real(
            CDOG_guess,
            transponder_coordinates,
            0.0,
            dz_array,
            angle_array,
            esv_matrix,
        )
        filtered_cdog = filter_close_receptions(
            CDOG_data,
            minimum_interval=0.95,
        )
        removed_receptions = len(CDOG_data) - len(filtered_cdog)
        results = plot_integer_pick_metrics_dog(
            center_offset,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            half_window=100,
            step=0.1,
            vlines={"UTC center": center_offset},
            title_suffix=f": PuertoRico_{clock_segment}",
            filtered_CDOG_data=filtered_cdog,
        )

    offsets, rmse, _, _, _, circular_offset = results
    rmse_offset = offsets[np.nanargmin(rmse)]
    print(f"Schematic lever from STPF: {np.round(schematic_lever, 3)} m")
    print(f"Before-section DOG estimate: {np.round(CDOG_guess, 3)} m ECEF")
    print(f"Offset center: {center_offset:.6f} s")
    print(f"Filtered receptions: {removed_receptions:,} removed")
    print(f"RMSE-optimal offset: {rmse_offset:.6f} s")
    print(f"Circular-variance-optimal offset: {circular_offset:.6f} s")
    print(f"Minimum RMSE {np.min(rmse)}")
    return results


if __name__ == "__main__":
    plot_puertorico_metrics()
