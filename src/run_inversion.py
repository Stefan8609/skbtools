"""Invert DOG arrival times for the DOG position, clock offset and lever arm.

Edit the settings below and run from src/: ``python run_inversion.py``. With the
defaults it repeats Inversion_Workflow/Bermuda/Bermuda_Data_time_bias.py for every
DOG in DOGS (same estimates, figures and Output/Plot_Data files).
"""

import numpy as np
from numba import njit
from numba.typed import List

from acoustics.synthetic_esv_tables import ESV_GENERATE, ESV_MODERATE
from bermuda_data import (
    CDOG_GUESS_AUGMENTS,
    CDOG_REFERENCE,
    GPS1_TO_OTHERS,
    OFFSET_HINTS_S,
    leg_mask,
)
import bermuda_data
import puertorico_data
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    check_esv_coverage,
    modelled_times,
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    anneal_with_offset_search,
    simulated_annealing_bias,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    gauss_newton_offset_search,
    initial_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_xAline_Joint_Annealing_bias import (
    simulated_annealing_bias_joint,
)
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from plotting.Plot_Modular import range_residual, time_series_plot

# Data
DATA = "bermuda"  # "bermuda", "puertorico" or "synthetic"
DOGS = [1, 3, 4]  # each DOG inverted on its own (synthetic: the DOG whose track)
CLOCK_SEGMENT = "after"  # Puerto Rico: "before" or "after" the clock loss
ESV_TABLE = None  # inversion ESV table name; None: the default of DATA below
USE_LEG_MASK = False  # Bermuda: 9-11 h and 12.4-15 h only (sensitivity test)
MASK_CROSSOVER_LEGS = True  # Puerto Rico: crossover legs only
DOWNSAMPLE = 1  # keep every n-th DOG arrival (Puerto Rico: and GNSS epoch)

# Synthetic data (DATA = "synthetic"); errors against the truth are printed
TRAJECTORY = "bermuda"  # "bermuda" (real ship track) or "unaligned" (box pattern)
ESV_TABLE_GENERATE = ESV_GENERATE  # table that makes the travel times
TIME_NOISE_S = 2e-5  # sd of the arrival-time noise
POSITION_NOISE_M = 2e-2  # sd of the noise on every GNSS coordinate
TIME_BIAS_S = 0.0  # true time bias; the Eq. 6 offset is TRUE_OFFSET_S + TIME_BIAS_S
ESV_BIAS_MPS = 0.0  # true dc
DRIFT_RATE = 0.0  # true clock drift, s/s (Eq. 6 convention)
TRUE_OFFSET_S = None  # None: drawn from U(1000, 10000) s
GPS_DOWNSAMPLE = 50  # keep every n-th GNSS epoch of the Bermuda track
SEED = 0  # NumPy and Numba random state (synthetics, annealing)

# Inversion
METHOD = "gauss_newton"  # "gauss_newton", "annealing", "joint_annealing" or "fixed"
# Bermuda starting values. Puerto Rico takes its own from the processed file; the
# synthetics start 50 m (DOG) and 3 m (lever arm) sd from the truth.
INITIAL_LEVER_M = [-12.69796149, 9.51739301, -15.0743129]  # GPS1 to transducer, body
INITIAL_GUESS_M = CDOG_GUESS_AUGMENTS  # {DOG: ECEF offset (m) from CDOG_REFERENCE}
# Integer second per DOG: Gauss-Newton must find it, annealing and "fixed" start at it
OFFSET_HINT_S = OFFSET_HINTS_S
ANNEALING_ITERATIONS = 300  # cooling scale of the lever-arm moves (Eq. A29)
CLOCK_DRIFT = False  # fit a linear clock drift (sixth entry of the estimate)

# Output
OUTPUT_NAME = "integer_pick_metrics"  # Output/Plot_Data/<name>_<run>.npz
SAVE = True
PLOT = True  # Bermuda-style figures; PuertoRico_Data_time_bias.py makes Puerto Rico's
SAVE_DATA_SAMPLE = True  # Bermuda: first 1000 epochs to Output/Paper_Data

ESV_TABLES = {
    "bermuda": "global_table_esv_extended",
    "puertorico": "PuertoRico_table_transducer_60_DOG_2950_3050",
    "synthetic": ESV_MODERATE,
}
METHOD_TAGS = {
    "gauss_newton": "_GN",
    "annealing": "_SimAnn",
    "joint_annealing": "_Joint",
    "fixed": "_Geiger",
}


@njit
def numba_seed(seed):
    np.random.seed(seed)


def load_bermuda(DOG_num):
    """Inputs of one Bermuda DOG, as in Bermuda_Data_time_bias.py."""
    data = np.load(
        bermuda_data.gps_data_path(
            f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"
        )
    )
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    if USE_LEG_MASK:
        keep = leg_mask(GPS_data)
        GPS_Coordinates = GPS_Coordinates[keep]
        GPS_data = GPS_data[keep]
    return {
        "name": f"DOG{DOG_num}",
        "DOG_num": DOG_num,
        "GPS_Coordinates": GPS_Coordinates,
        "GPS_data": GPS_data,
        "CDOG_data": data["CDOG_data"][::DOWNSAMPLE],
        "gps1_to_others": GPS1_TO_OTHERS,
        "CDOG_guess": CDOG_REFERENCE + INITIAL_GUESS_M[DOG_num],
        "initial_lever": np.array(INITIAL_LEVER_M),
        "offset_hint": OFFSET_HINT_S[DOG_num],
        "offset_center": None,
        "reference": CDOG_REFERENCE,
        "real_data": True,
        "paths": bermuda_data,
    }


def load_puertorico(dz_array, angle_array):
    """Inputs of the Puerto Rico DOG for CLOCK_SEGMENT, as its driver masks them."""
    # Imported here: the driver loads its ESV table from PuertoRico_Data at import
    from Inversion_Workflow.PuertoRico.PuertoRico_Data_time_bias import (
        INITIAL_OFFSETS_S,
        load_inversion_inputs,
    )

    inputs = load_inversion_inputs(
        CLOCK_SEGMENT, dz_array, angle_array, MASK_CROSSOVER_LEGS, DOWNSAMPLE
    )
    return {
        "name": f"PuertoRico_{CLOCK_SEGMENT}",
        "DOG_num": 1,
        "GPS_Coordinates": inputs["GPS_Coordinates"],
        "GPS_data": inputs["GPS_data"],
        "CDOG_data": inputs["CDOG_data"],
        "gps1_to_others": inputs["gps1_to_others"],
        "CDOG_guess": inputs["CDOG_guess"],
        "initial_lever": inputs["initial_lever_guess"],
        "offset_hint": None,
        "offset_center": INITIAL_OFFSETS_S[CLOCK_SEGMENT],
        "reference": inputs["CDOG_guess"],
        "real_data": True,
        "paths": puertorico_data,
    }


def make_synthetic(DOG_num, dz_array, angle_array):
    """Synthetic arrivals on the track of one Bermuda DOG or a box pattern (App. B).

    Generated as in ``modular_synthetic``. The truth is kept under "truth".
    """
    # Imported here: Modular_Synthetic sets its own figure style at import
    from Inversion_Workflow.Synthetic.Modular_Synthetic import (
        BERMUDA_GPS1_TO_OTHERS,
        TRUE_LEVER,
    )

    dz_generate, angle_generate, esv_generate = load_esv_table(
        bermuda_data.gps_data_path(f"ESV_Tables/{ESV_TABLE_GENERATE}.mat")
    )
    true_offset = TRUE_OFFSET_S
    if true_offset is None:
        true_offset = np.random.rand() * 9000 + 1000
    if TRAJECTORY == "unaligned":
        gps1_to_others = np.array(
            [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
        )
        true_lever = TRUE_LEVER[0]
        CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates = (
            generateUnaligned(
                20000,
                TIME_NOISE_S,
                POSITION_NOISE_M,
                true_offset,
                ESV_BIAS_MPS,
                TIME_BIAS_S,
                dz_generate,
                angle_generate,
                esv_generate,
                gps1_to_others=gps1_to_others,
                gps1_to_transponder=true_lever,
                drift_rate=DRIFT_RATE,
            )
        )
        real_data = False
    elif TRAJECTORY == "bermuda":
        gps1_to_others = BERMUDA_GPS1_TO_OTHERS
        true_lever = TRUE_LEVER[1]
        CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates = (
            bermuda_trajectory(
                TIME_NOISE_S,
                POSITION_NOISE_M,
                ESV_BIAS_MPS,
                TIME_BIAS_S,
                dz_generate,
                angle_generate,
                esv_generate,
                offset=true_offset,
                DOG_num=DOG_num,
                gps1_to_others=gps1_to_others,
                gps1_to_transponder=true_lever,
                drift_rate=DRIFT_RATE,
            )
        )
        keep = leg_mask(GPS_data) if USE_LEG_MASK else slice(None, None, GPS_DOWNSAMPLE)
        GPS_Coordinates = GPS_Coordinates[keep]
        GPS_data = GPS_data[keep]
        transponder_coordinates = transponder_coordinates[keep]
        real_data = True
    else:
        raise ValueError(
            f"TRAJECTORY must be 'bermuda' or 'unaligned', got {TRAJECTORY!r}"
        )

    for table, dz, angle in (
        (ESV_TABLE_GENERATE, dz_generate, angle_generate),
        ("inversion table", dz_array, angle_array),
    ):
        check_esv_coverage(
            CDOG, transponder_coordinates, dz, angle, real_data, table, "warn"
        )
    return {
        "name": f"synthetic_{TRAJECTORY}_DOG{DOG_num}",
        "DOG_num": DOG_num,
        "GPS_Coordinates": GPS_Coordinates,
        "GPS_data": GPS_data,
        "CDOG_data": CDOG_data,
        "gps1_to_others": gps1_to_others,
        "CDOG_guess": CDOG + np.random.normal(0, 50, 3),
        "initial_lever": true_lever - np.random.normal(0, 3, 3),
        "offset_hint": None,
        "offset_center": None,
        "reference": CDOG,
        "real_data": real_data,
        "paths": bermuda_data,
        # The inversion recovers the Eq. 6 offset true_offset + TIME_BIAS_S.
        "truth": {"CDOG": CDOG, "lever": true_lever, "dt": true_offset + TIME_BIAS_S},
    }


def invert(inputs, dz_array, angle_array, esv_matrix):
    """Lever arm and clock offset by METHOD, then the final fit at that lever arm.

    Returns ``(lever, offset, estimate, transponder_coordinates, CDOG_full,
    GPS_full, CDOG_clock, GPS_clock)``; the estimate is that of
    ``final_bias_geiger``.
    """
    CDOG_data = inputs["CDOG_data"]
    GPS_data = inputs["GPS_data"]
    GPS_Coordinates = inputs["GPS_Coordinates"]
    gps1_to_others = inputs["gps1_to_others"]
    real_data = inputs["real_data"]
    # Known integer second (Bermuda hint) or search center (Puerto Rico), if any
    known_offset = inputs["offset_center"]
    if known_offset is None:
        known_offset = inputs["offset_hint"]

    if METHOD == "gauss_newton":
        print("\n ---------Using Gauss-Newton Offset Search--------- \n")
        best_lever, best_offset, inversion_result, offset_diagnostics = (
            gauss_newton_offset_search(
                CDOG_data,
                GPS_data,
                GPS_Coordinates,
                gps1_to_others,
                inputs["CDOG_guess"],
                inputs["initial_lever"],
                dz_array,
                angle_array,
                esv_matrix,
                initial_offset=inputs["offset_center"],
                real_data=real_data,
                clock_drift=CLOCK_DRIFT,
            )
        )
        print(
            f"Offset {best_offset:.5f} s, lever {np.round(best_lever, 3)}, "
            f"delta chi^2 to second best {offset_diagnostics['delta_chi2']:.0f}"
        )
        hint = inputs["offset_hint"]
        if hint is not None and np.round(best_offset) != hint:
            raise ValueError(
                f"Gauss-Newton offset search found integer second "
                f"{np.round(best_offset):.0f} s for {inputs['name']}, expected the "
                f"hint {hint:.0f} s"
            )
    elif METHOD == "annealing" and known_offset is not None:
        print("\n ---------Using Simulated Annealing--------- \n")
        best_lever, best_offset, inversion_result = simulated_annealing_bias(
            ANNEALING_ITERATIONS,
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            gps1_to_others,
            inputs["CDOG_guess"],
            inputs["initial_lever"],
            dz_array,
            angle_array,
            esv_matrix,
            initial_offset=known_offset,
            real_data=real_data,
            z_sample=True,
            clock_drift=CLOCK_DRIFT,
        )
    elif METHOD == "annealing":
        print("\n ---------Using Simulated Annealing over integer seconds--------- \n")
        best_lever, best_offset, inversion_result = anneal_with_offset_search(
            ANNEALING_ITERATIONS,
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            gps1_to_others,
            inputs["CDOG_guess"],
            inputs["initial_lever"],
            dz_array,
            angle_array,
            esv_matrix,
            real_data=real_data,
            z_sample=real_data,
            clock_drift=CLOCK_DRIFT,
        )
    elif METHOD == "fixed":
        print("\n ---------Using Geiger Inversion--------- \n")
        best_lever = inputs["initial_lever"]
        inversion_result, best_offset = initial_bias_geiger(
            inputs["CDOG_guess"],
            CDOG_data,
            GPS_data,
            findTransponder(GPS_Coordinates, gps1_to_others, best_lever),
            dz_array,
            angle_array,
            esv_matrix,
            real_data=real_data,
            initial_offset=inputs["offset_center"],
            clock_drift=CLOCK_DRIFT,
        )
        print("\nInitial complete. Offset =", best_offset)
    else:
        raise ValueError(
            "METHOD must be 'gauss_newton', 'annealing', 'joint_annealing' or "
            f"'fixed', got {METHOD!r}"
        )
    return final_fit(
        inputs,
        best_lever,
        best_offset,
        inversion_result,
        dz_array,
        angle_array,
        esv_matrix,
    )


def final_fit(
    inputs, lever, offset, inversion_result, dz_array, angle_array, esv_matrix
):
    """``final_bias_geiger`` at the lever arm and offset found by the method."""
    transponder_coordinates = findTransponder(
        inputs["GPS_Coordinates"], inputs["gps1_to_others"], lever
    )
    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
        inversion_result[:3],
        inputs["CDOG_data"],
        inputs["GPS_data"],
        transponder_coordinates,
        offset,
        inversion_result[4],
        inversion_result[3],
        dz_array,
        angle_array,
        esv_matrix,
        real_data=inputs["real_data"],
        clock_drift=CLOCK_DRIFT,
        drift_rate=inversion_result[5] if CLOCK_DRIFT else 0.0,
    )
    return (
        lever,
        offset,
        inversion_result,
        transponder_coordinates,
        CDOG_full,
        GPS_full,
        CDOG_clock,
        GPS_clock,
    )


def invert_joint(inputs_list, dz_array, angle_array, esv_matrix):
    """One lever arm for all DOGs by joint annealing, then each DOG's final fit.

    All DOGs share the GNSS epochs (the processed Bermuda files are identical).
    """
    print("\n ---------Using Joint Simulated Annealing--------- \n")
    first = inputs_list[0]
    for inputs in inputs_list[1:]:
        if not np.array_equal(inputs["GPS_data"], first["GPS_data"]):
            raise ValueError("Joint annealing needs the same GNSS epochs for all DOGs")
    CDOG_data_list = List()
    for inputs in inputs_list:
        CDOG_data_list.append(inputs["CDOG_data"])
    best_lever, best_offsets, inversion_results, _ = simulated_annealing_bias_joint(
        ANNEALING_ITERATIONS,
        CDOG_data_list,
        first["GPS_data"],
        first["GPS_Coordinates"],
        first["gps1_to_others"],
        np.array([inputs["CDOG_guess"] for inputs in inputs_list]),
        first["initial_lever"],
        dz_array,
        angle_array,
        esv_matrix,
        initial_offset_list=np.array(
            [inputs["offset_hint"] for inputs in inputs_list], dtype=float
        ),
        real_data=first["real_data"],
        z_sample=True,
        clock_drift=CLOCK_DRIFT,
    )
    return [
        final_fit(
            inputs,
            best_lever,
            best_offsets[j],
            inversion_results[j],
            dz_array,
            angle_array,
            esv_matrix,
        )
        for j, inputs in enumerate(inputs_list)
    ]


def report(inputs, result, dz_array, angle_array, esv_matrix):
    """Print, plot and save one inversion as Bermuda_Data_time_bias.py does."""
    (
        best_lever,
        best_offset,
        inversion_result,
        transponder_coordinates,
        CDOG_full,
        GPS_full,
        CDOG_clock,
        GPS_clock,
    ) = result
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    drift_rate = inversion_result[5] if CLOCK_DRIFT else 0.0
    # Eq. 6 clock offset: dt = offset - time_bias
    best_offset_eff = best_offset - time_bias

    diff_data_ms = (CDOG_full - GPS_full) * 1000.0
    RMSE_cm = (
        (np.sqrt(np.nanmean(diff_data_ms**2)) / 1000.0)
        * NOMINAL_SOUND_SPEED_MPS
        * 100.0
    )
    print(f"\n{inputs['name']}")
    print(f"Estimate: {np.round(inversion_result, 4)}")
    print(
        f"Best Lever: {np.round(best_lever, 3)}, "
        f"Offset: {np.round(best_offset_eff, 4)}, "
        f"Inversion Guess (from base): "
        f"{np.round(inversion_guess - inputs['reference'], 5)}"
    )
    print("RMSE:", np.round(RMSE_cm, 3), "cm")
    if CLOCK_DRIFT:
        print(f"Drift rate: {drift_rate:.4e} s/s (Eq. 6 convention)")
    if "truth" in inputs:
        truth = inputs["truth"]
        position_error_cm = np.linalg.norm(inversion_guess - truth["CDOG"]) * 100
        lever_error_cm = np.linalg.norm(best_lever - truth["lever"]) * 100
        print(
            f"Position error: {position_error_cm:.2f} cm, lever error: "
            f"{lever_error_cm:.2f} cm, offset error: "
            f"{best_offset_eff - truth['dt']:.6f} s"
        )
        if CLOCK_DRIFT:
            print(f"Drift rate error: {drift_rate - DRIFT_RATE:.3e} s/s")

    # Travel times at the final estimate, paired with a wide gate for the residuals
    times_guess, esv = modelled_times(
        inversion_guess,
        transponder_coordinates,
        esv_bias,
        dz_array,
        angle_array,
        esv_matrix,
        inputs["real_data"],
    )
    if PLOT:
        run_tag = str(inputs["DOG_num"]) if DATA == "bermuda" else inputs["name"]
        file_tag = run_tag + METHOD_TAGS[METHOD]
        save_path = "Figs/Inversion_Workflow"
        if DATA == "puertorico":
            # save_plot writes inside Bermuda_Data unless the path is absolute
            save_path = str(puertorico_data.gps_data_path(save_path))
        time_series_plot(
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            save=SAVE,
            path=save_path,
            DOG_num=file_tag,
        )
        CDOG_full_ex, GPS_clock_ex, GPS_full_ex, trans_coords_full, esv_full = (
            two_pointer_index(
                best_offset_eff,
                0.6,
                inputs["CDOG_data"],
                inputs["GPS_data"],
                times_guess,
                transponder_coordinates,
                esv,
            )[1:]
        )
        range_residual(
            trans_coords_full,
            esv_full,
            inversion_guess,
            CDOG_full_ex,
            GPS_full_ex,
            GPS_clock_ex,
            save=SAVE,
            path=save_path,
            DOG_num=file_tag,
        )

    if not SAVE:
        return
    plot_data_path = inputs["paths"].gps_output_path(
        "Plot_Data", f"{OUTPUT_NAME}_{inputs['name']}.npz"
    )
    np.savez_compressed(
        plot_data_path,
        CDOG_clock=CDOG_clock,
        CDOG_full=CDOG_full,
        GPS_clock=GPS_clock,
        GPS_full=GPS_full,
        best_offset=best_offset,
        best_lever=best_lever,
        inversion_result=inversion_result,
        esv_bias=esv_bias,
        time_bias=time_bias,
        clock_drift=CLOCK_DRIFT,
        drift_rate_s_per_s=drift_rate,
        CDOG_data=inputs["CDOG_data"],
        GPS_data=inputs["GPS_data"],
        times_guess=times_guess,
        esv=esv,
        transponder_coordinates=transponder_coordinates,
        DOG_num=inputs["DOG_num"],
    )
    print(f"Saved integer-pick metric plot data to: {plot_data_path}")

    if DATA == "bermuda" and SAVE_DATA_SAMPLE:
        # First 1000 epochs, same keys as the processed file
        sample_path = bermuda_data.gps_output_path(
            "Paper_Data",
            f"Processed_GPS_Receivers_DOG_{inputs['DOG_num']}_sample.npz",
        )
        np.savez_compressed(
            sample_path,
            GPS_Coordinates=inputs["GPS_Coordinates"][:1000],
            GPS_data=inputs["GPS_data"][:1000],
            CDOG_data=inputs["CDOG_data"][:1000],
        )
        print(f"Saved repo sample data to: {sample_path}")


if __name__ == "__main__":
    if DATA not in ESV_TABLES:
        raise ValueError(f"DATA must be one of {list(ESV_TABLES)}, got {DATA!r}")
    if METHOD == "joint_annealing" and DATA != "bermuda":
        raise ValueError("joint_annealing is set up for the Bermuda DOGs only")
    np.random.seed(SEED)
    numba_seed(SEED)

    table_paths = puertorico_data if DATA == "puertorico" else bermuda_data
    dz_array, angle_array, esv_matrix = load_esv_table(
        table_paths.gps_data_path(f"ESV_Tables/{ESV_TABLE or ESV_TABLES[DATA]}.mat")
    )
    tables = (dz_array, angle_array, esv_matrix)

    if DATA == "bermuda":
        inputs_list = [load_bermuda(DOG_num) for DOG_num in DOGS]
    elif DATA == "puertorico":
        inputs_list = [load_puertorico(dz_array, angle_array)]
    else:
        inputs_list = [
            make_synthetic(DOG_num, dz_array, angle_array) for DOG_num in DOGS
        ]

    if METHOD == "joint_annealing":
        results = invert_joint(inputs_list, *tables)
    else:
        results = [invert(inputs, *tables) for inputs in inputs_list]
    for inputs, result in zip(inputs_list, results, strict=True):
        report(inputs, result, *tables)
