"""High-level, top-down driver for inversion workflows (no argparse).

Edit the constants in the **USER EDITABLE CONSTANTS** section below.
The script then executes top-to-bottom: configure → load/generate data →
select solvers → run pipeline → report quality → optionally save results.

This version preserves the functionality of the CLI-driven version but removes
all argument parsing and centralizes inputs as global constants.
"""

from __future__ import annotations

import numpy as np
import scipy.io as sio
from typing import Optional
from dataclasses import dataclass
import logging

from data import gps_data_path, gps_output_path
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Calculate_Times import (
    calculateTimesRayTracing,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from plotting.Plot_Modular import time_series_plot

logger = logging.getLogger(__name__)

# =============================================================================
# USER EDITABLE CONSTANTS
# =============================================================================
# Data/trajectory
DATA_TYPE: str = "synthetic"  # "synthetic" or "real"
TRAJECTORY: str = "realistic"  # "realistic", "random", "line", "cross", "bermuda"
N_SAMPLES: int = 5000  # synthetic only
DOG_NUM: int = 3  # real data set selection (and used by bermuda)

# Noise used for *generation* (synthetic only)
TIME_NOISE: float = 0.0
POSITION_NOISE: float = 0.0

# --- Generation knobs (synthetic only) ---
# If set to None, defaults (below) are used
GEN_LEVER: Optional[np.ndarray] = None  # (3,)
GEN_GPS_GRID: Optional[np.ndarray] = None  # (4,3)
GEN_OFFSET: float = 151.25  # scalar
GEN_ESV_BIAS: float = 0.0  # m/s
GEN_TIME_BIAS: float = 0.0  # s
GEN_ESV_TABLE: str = "global_table_esv_extended"

# --- Solving knobs ---
# If set to None, sensible defaults (below) are used
SOLVE_LEVER: Optional[np.ndarray] = None  # (3,)
SOLVE_GPS_GRID: Optional[np.ndarray] = None  # (4,3)
SOLVE_OFFSET: float = 0.0  # scalar
SOLVE_ESV_BIAS: float = 0.0  # used only in biased GN
SOLVE_TIME_BIAS: float = 0.0  # used only in biased GN
SOLVE_ESV_TABLE: str = "global_table_esv_extended"

# Inversion choices
GAUSS_NEWTON: str = "unbiased"  # "none", "biased", "unbiased"
ALIGNMENT: bool = True
ANNEALING: bool = False
ANNEAL_ITER: int = 300

# Output
SAVE_OUTPUT: bool = False
OUTPUT_NAME: str = "inversion_result"

# Plotting
MAKE_PLOTS: bool = False
PLOT_BLOCK: bool = False
PLOT_SAVE: bool = False
PLOT_PATH: str = "Figs"
PLOT_CHAIN_NAME: Optional[str] = None
PLOT_SEGMENTS: int = 0

# Logging
LOG_LEVEL = logging.INFO

# =============================================================================
# DEFAULTS / CONSTANTS (do not usually edit)
# =============================================================================
DEFAULT_GPS_GRID = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)

# Lever arm defaults (transponder_coordinates relative to GPS1)
DEFAULT_SYNTHETIC_LEVER = np.array([-12.0, 1.0, -16.0])
DEFAULT_REAL_LEVER = np.array([-12.48862757, 0.22622633, -15.89601934])

# Default offset applied to the CDOG initial location guess (absolute)
DEFAULT_INITIAL_DOG_OFFSET = np.array([100.0, 100.0, 200.0])

REAL_DATA_FLAG = DATA_TYPE == "real" or TRAJECTORY == "bermuda"

# -----------------------------------------------------------------------------
# Default to values if None (after checks)
# -----------------------------------------------------------------------------
if GEN_LEVER is None:
    GEN_LEVER = DEFAULT_SYNTHETIC_LEVER
if SOLVE_LEVER is None:
    SOLVE_LEVER = DEFAULT_REAL_LEVER if DATA_TYPE == "real" else DEFAULT_SYNTHETIC_LEVER
if GEN_GPS_GRID is None:
    GEN_GPS_GRID = DEFAULT_GPS_GRID
if SOLVE_GPS_GRID is None:
    SOLVE_GPS_GRID = DEFAULT_GPS_GRID

# =============================================================================
# TOP-LEVEL RUN SEQUENCE
# =============================================================================


def run():
    """Execute the configured inversion workflow."""
    _configure_logging()
    _log_configuration_summary()

    # Prepare data (synthetic vs real)
    if DATA_TYPE == "synthetic":
        dz_gen, angle_gen, esv_gen = load_esv_table(GEN_ESV_TABLE)
        (
            CDOG_data,
            CDOG_location,
            GPS_Coordinates,
            GPS_data,
            transponder_coordinates_true,
        ) = generate_synthetic(dz_gen, angle_gen, esv_gen, GEN_GPS_GRID, GEN_LEVER)
        logger.info("Synthetic data generated.")
    elif DATA_TYPE == "real":
        CDOG_data, CDOG_location, GPS_Coordinates, GPS_data = load_real_data()
        logger.info("Real data loaded.")
    else:
        raise ValueError("DATA_TYPE must be 'synthetic' or 'real'")

    # Run inversion
    dz_inv, angle_inv, esv_inv = load_esv_table(SOLVE_ESV_TABLE)
    inversion_result = run_inversion(
        CDOG_data,
        CDOG_location,
        GPS_Coordinates,
        GPS_data,
        SOLVE_GPS_GRID,
        SOLVE_LEVER,
        dz_inv,
        angle_inv,
        esv_inv,
    )

    # Report quality (if synthetic)
    if DATA_TYPE == "synthetic":
        _report_quality_synthetic(
            CDOG_data,
            GPS_data,
            CDOG_location,
            GPS_Coordinates,
            transponder_coordinates_true,
            SOLVE_GPS_GRID,
            inversion_result,
            GEN_LEVER,
        )

    # ADD PLOTTING AND REPORTING ON REAL DATA

    # Save
    if SAVE_OUTPUT:
        out_path = gps_output_path(f"{OUTPUT_NAME}.npz")
        np.savez(
            out_path,
            CDOG_guess=inversion_result["CDOG_guess"],
            lever=inversion_result["lever"],
            offset=inversion_result["offset"],
            time_bias=inversion_result["time_bias"]
            if inversion_result["time_bias"] is not None
            else np.nan,
            esv_bias=inversion_result["esv_bias"]
            if inversion_result["esv_bias"] is not None
            else np.nan,
        )
        print("Saved output to", out_path)

    return inversion_result


# =============================================================================
# UTILITIES & HELPERS
# =============================================================================


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
    else:
        logging.getLogger().setLevel(LOG_LEVEL)


def _arrfmt(a) -> str:
    """Compact array format for logs."""
    return np.array2string(
        np.asarray(a), precision=4, suppress_small=True, separator=", "
    )


def _log_configuration_summary() -> None:
    print("\n")
    print("====== Program start ======")
    logger.info(
        "Config: DATA_TYPE=%s  TRAJECTORY=%s  N_SAMPLES=%d  DOG_NUM=%d",
        DATA_TYPE,
        TRAJECTORY,
        N_SAMPLES,
        DOG_NUM,
    )
    logger.info(
        "Inversion: GN=%s  alignment=%s  annealing=%s(iter=%d)",
        GAUSS_NEWTON,
        ALIGNMENT,
        ANNEALING,
        ANNEAL_ITER,
    )


# ----------------------------- Data loading ------------------------------- #


def load_esv_table(name: str):
    """Load an effective sound speed (ESV) lookup table.

    Returns
    -------
    tuple(ndarray, ndarray, ndarray)
        (dz_array, angle_array, esv_matrix)
    """
    table = sio.loadmat(gps_data_path(f"ESV_Tables/{name}.mat"))
    dz_array = table["distance"].flatten()
    angle_array = table["angle"].flatten()
    esv_matrix = table["matrice"]
    return dz_array, angle_array, esv_matrix


def generate_synthetic(dz, angle, esv, gps_grid: np.ndarray, lever: np.ndarray):
    """
    Generate a synthetic trajectory according to configured options.
    """
    print("\n")
    print("====== Generating Synthetic ======")
    logger.info(
        "trajectory = %s  n_samples = %d  time_noise = %.6g  position_noise = %.6g",
        TRAJECTORY,
        N_SAMPLES,
        TIME_NOISE,
        POSITION_NOISE,
    )
    logger.info(
        "gen_offset = %.6g  gen_esv_bias = %.6g  gen_time_bias = %.6g",
        GEN_OFFSET,
        GEN_ESV_BIAS,
        GEN_TIME_BIAS,
    )
    logger.info("gen_esv_table = %s", GEN_ESV_TABLE)
    logger.info("gen_gps_grid = %s", _arrfmt(gps_grid))
    logger.info("gen_lever = %s", _arrfmt(lever))

    if TRAJECTORY == "bermuda":
        (
            CDOG_data,
            CDOG_location,
            GPS_Coordinates,
            GPS_data,
            transponder_coordinates_true,
        ) = bermuda_trajectory(
            TIME_NOISE,
            POSITION_NOISE,
            GEN_ESV_BIAS,
            GEN_TIME_BIAS,
            dz,
            angle,
            esv,
            offset=GEN_OFFSET,
            gps1_to_others=gps_grid,
            gps1_to_transponder=lever,
            DOG_num=DOG_NUM,
        )
    else:
        (
            CDOG_data,
            CDOG_location,
            GPS_Coordinates,
            GPS_data,
            transponder_coordinates_true,
        ) = generateUnaligned(
            N_SAMPLES,
            TIME_NOISE,
            POSITION_NOISE,
            GEN_OFFSET,
            GEN_ESV_BIAS,
            GEN_TIME_BIAS,
            dz,
            angle,
            esv,
            gps1_to_others=gps_grid,
            gps1_to_transponder=lever,
            trajectory=TRAJECTORY,
        )

    logger.info(
        "Generated synthetic dataset with sizes: CDOG_clock = %s GPS_time = %s",
        np.asarray(CDOG_data).shape,
        np.asarray(GPS_data).shape,
    )
    logger.info("Generated CDOG location = %s", _arrfmt(CDOG_location))
    return (
        CDOG_data,
        CDOG_location,
        GPS_Coordinates,
        GPS_data,
        transponder_coordinates_true,
    )


def load_real_data():
    """
    Load processed real GPS/DOG data from disk.
    """
    logger.info("Loading real data for DOG %d", DOG_NUM)
    path = gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_NUM}.npz")

    data = np.load(path)
    GPS_Coordinates = data["GPS_Coordinates"].astype(float)
    GPS_data = data["GPS_data"].astype(float)
    CDOG_data = data["CDOG_data"].astype(float)
    CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    logger.info("Loaded real data from %s", path)
    logger.info(
        "GPS_Coordinates shape = %s  GPS_time shape = %s",
        GPS_Coordinates.shape,
        GPS_data.shape,
    )
    logger.info(
        "CDOG_clock shape = %s  CDOG_guess = %s",
        CDOG_data.shape,
        _arrfmt(CDOG_guess_base),
    )
    return CDOG_data, CDOG_guess_base, GPS_Coordinates, GPS_data


# ------------------------------ Solver choice ----------------------------- #


def choose_inversion_functions():
    """Select inversion callables based on configured options."""
    funcs: dict[str, object] = {}
    logger.info(
        "Choosing inversion functions: gauss_newton=%s, alignment=%s, annealing=%s",
        GAUSS_NEWTON,
        bool(ALIGNMENT),
        bool(ANNEALING),
    )

    if GAUSS_NEWTON == "biased":
        from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
            initial_bias_geiger,
            transition_bias_geiger,
            final_bias_geiger,
        )

        funcs["gauss_newton"] = (
            initial_bias_geiger,
            transition_bias_geiger,
            final_bias_geiger,
        )
        if ANNEALING:
            from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
                simulated_annealing_bias,
            )

            funcs["annealing"] = simulated_annealing_bias
        logger.info(
            "Selected Gauss–Newton (biased) pipeline with optional annealing=%s",
            "annealing" in funcs,
        )

    elif GAUSS_NEWTON == "unbiased":
        from Inversion_Workflow.Inversion.Numba_xAline_Geiger import (
            initial_geiger,
            transition_geiger,
            final_geiger,
        )

        funcs["gauss_newton"] = (initial_geiger, transition_geiger, final_geiger)
        if ANNEALING:
            from Inversion_Workflow.Inversion.Numba_xAline_Annealing import (
                simulated_annealing,
            )

            funcs["annealing"] = simulated_annealing
        logger.info(
            "Selected Gauss–Newton (unbiased) pipeline with optional annealing=%s",
            "annealing" in funcs,
        )

    elif ANNEALING:
        # Simulated annealing without Gauss–Newton
        from Inversion_Workflow.Inversion.Numba_xAline_Annealing import (
            simulated_annealing,
        )

        funcs["annealing"] = simulated_annealing
        logger.info("Selected annealing-only pipeline (no Gauss–Newton)")

    return funcs


def solve_pipeline(steps, state):
    """Run a list of step(state)->state callables sequentially."""
    logger.info("Running solver pipeline with %d step(s)...", len(steps))
    for step in steps:
        state = step(state)
    return state


def _build_solver_steps(
    funcs: dict,
    CDOG_data: np.ndarray,
    GPS_data: np.ndarray,
    GPS_Coordinates: np.ndarray,
    GPS_grid: np.ndarray,
    transponder_coordinates_est: np.ndarray,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Create a list of step functions that operate on a shared state dict."""
    steps = []
    logger.info("Building solver steps...")

    if "annealing" in funcs:
        anneal = funcs["annealing"]

        def step_anneal(state):
            logger.info(
                "[STEP] Annealing: start (iter=%d, realistic_ray_tracing=%s)",
                ANNEAL_ITER,
                REAL_DATA_FLAG,
            )
            lever_guess, offset, result = anneal(
                ANNEAL_ITER,
                CDOG_data,
                GPS_data,
                GPS_Coordinates,
                GPS_grid,
                state["CDOG_guess"],
                state["lever"],
                dz_array,
                angle_array,
                esv_matrix,
                initial_offset=state["offset"],
                real_data=REAL_DATA_FLAG,
            )
            state["lever"] = np.asarray(lever_guess, dtype=float)
            state["offset"] = float(offset)
            result_arr = np.asarray(result, dtype=float)
            if result_arr.size >= 3:
                state["CDOG_guess"] = result_arr[:3]
            if result_arr.size >= 5:
                state["time_bias"] = float(result_arr[3])
                state["esv_bias"] = float(result_arr[4])

            logger.info(
                "[STEP] Annealing: updated lever=%s  offset=%.6g  CDOG_guess=%s",
                _arrfmt(state["lever"]),
                state["offset"],
                _arrfmt(state["CDOG_guess"]),
            )
            if state.get("time_bias") is not None and state.get("esv_bias") is not None:
                logger.info(
                    "[STEP] Annealing: biases time=%.6g  esv=%.6g",
                    state["time_bias"],
                    state["esv_bias"],
                )
            return state

        steps.append(step_anneal)

    if "gauss_newton" in funcs:
        initial_fn, transition_fn, final_fn = funcs["gauss_newton"]

        def step_initial(state):
            logger.info(
                "[STEP] Gauss–Newton initial: start (realistic_ray_tracing=%s)",
                REAL_DATA_FLAG,
            )
            result, off = initial_fn(
                state["CDOG_guess"],
                CDOG_data,
                GPS_data,
                transponder_coordinates_est,
                dz_array,
                angle_array,
                esv_matrix,
                real_data=REAL_DATA_FLAG,
            )
            result_arr = np.asarray(result, dtype=float)
            state["CDOG_guess"] = result_arr[:3]
            state["offset"] = float(off)
            if result.size >= 5:
                state["time_bias"] = float(result_arr[3])
                state["esv_bias"] = float(result_arr[4])
            logger.info(
                "[STEP] Gauss–Newton initial: CDOG_guess=%s  "
                "offset=%.6g  time_bias=%s  esv_bias=%s",
                _arrfmt(state["CDOG_guess"]),
                state["offset"],
                "None" if state["time_bias"] is None else f"{state['time_bias']:.6g}",
                "None" if state["esv_bias"] is None else f"{state['esv_bias']:.6g}",
            )
            return state

        steps.append(step_initial)

        if ALIGNMENT:

            def step_transition(state):
                logger.info(
                    "[STEP] Gauss–Newton transition: start (biased=%s)",
                    state.get("time_bias") is not None
                    and state.get("esv_bias") is not None,
                )
                if (
                    state.get("time_bias") is not None
                    and state.get("esv_bias") is not None
                ):
                    result, off = transition_fn(
                        state["CDOG_guess"],
                        CDOG_data,
                        GPS_data,
                        transponder_coordinates_est,
                        state["offset"],
                        state["esv_bias"],
                        state["time_bias"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=REAL_DATA_FLAG,
                    )
                    result_arr = np.asarray(result, dtype=float)
                    state["CDOG_guess"] = result_arr[:3]
                    state["offset"] = float(off)
                    state["time_bias"] = float(result_arr[3])
                    state["esv_bias"] = float(result_arr[4])
                else:
                    result, off = transition_fn(
                        state["CDOG_guess"],
                        CDOG_data,
                        GPS_data,
                        transponder_coordinates_est,
                        state["offset"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=REAL_DATA_FLAG,
                    )
                    state["CDOG_guess"] = np.asarray(result, dtype=float)
                    state["offset"] = float(off)
                logger.info(
                    "[STEP] Gauss–Newton transition: CDOG_guess=%s  offset=%.6g",
                    _arrfmt(state["CDOG_guess"]),
                    state["offset"],
                )
                return state

            steps.append(step_transition)

            def step_final(state):
                logger.info(
                    "[STEP] Gauss–Newton final: start (biased=%s)",
                    state.get("time_bias") is not None
                    and state.get("esv_bias") is not None,
                )
                if (
                    state.get("time_bias") is not None
                    and state.get("esv_bias") is not None
                ):
                    result, *_ = final_fn(
                        state["CDOG_guess"],
                        CDOG_data,
                        GPS_data,
                        transponder_coordinates_est,
                        state["offset"],
                        state["esv_bias"],
                        state["time_bias"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=REAL_DATA_FLAG,
                    )
                    result_arr = np.asarray(result, dtype=float)
                    state["CDOG_guess"] = result_arr[:3]
                    state["time_bias"] = float(result_arr[3])
                    state["esv_bias"] = float(result_arr[4])
                else:
                    result, *_ = final_fn(
                        state["CDOG_guess"],
                        CDOG_data,
                        GPS_data,
                        transponder_coordinates_est,
                        state["offset"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=REAL_DATA_FLAG,
                    )
                    state["CDOG_guess"] = np.asarray(result, dtype=float)
                logger.info(
                    "[STEP] Gauss–Newton final: CDOG_guess=%s",
                    _arrfmt(state["CDOG_guess"]),
                )
                return state

            steps.append(step_final)

    return steps


def run_inversion(
    CDOG_data,
    CDOG_guess_base,
    GPS_Coordinates,
    GPS_data,
    GPS_grid: np.ndarray,
    lever_est: np.ndarray,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Execute the selected inversion workflow."""

    print("\n")
    print("====== Inversion Setup ======")
    logger.info("realistic_ray_tracing = %s", REAL_DATA_FLAG)
    logger.info("inversion_esv_table = %s", SOLVE_ESV_TABLE)
    logger.info("inversion_gps_grid = %s", _arrfmt(GPS_grid))
    logger.info("inversion_CDOG_offset = %s", _arrfmt(DEFAULT_INITIAL_DOG_OFFSET))

    print("\n")
    print("====== Initial Values ======")

    if DATA_TYPE == "real":
        if DOG_NUM == 1:
            CDOG_guess_augment = np.array([-398.16, 371.90, 773.02])
        if DOG_NUM == 3:
            CDOG_guess_augment = np.array([825.182985, -111.05670221, -734.10011698])
        if DOG_NUM == 4:
            CDOG_guess_augment = np.array([236.428385, -1307.98390221, -2189.21991698])
    else:
        CDOG_guess_augment = DEFAULT_INITIAL_DOG_OFFSET

    CDOG_initial_guess = CDOG_guess_base + CDOG_guess_augment
    logger.info(
        "initial_offset = %.6g  initial_esv_bias = %.6g  initial_time_bias = %.6g",
        SOLVE_OFFSET,
        SOLVE_ESV_BIAS,
        SOLVE_TIME_BIAS,
    )
    logger.info("initial_CDOG_guess = %s", _arrfmt(CDOG_initial_guess))
    logger.info("initial_lever = %s", _arrfmt(lever_est))

    offset = float(SOLVE_OFFSET)

    transponder_coordinates_est = findTransponder(
        GPS_Coordinates,
        GPS_grid,
        lever_est,
    )

    time_bias_guess = float(SOLVE_TIME_BIAS) if GAUSS_NEWTON == "biased" else None
    esv_bias_guess = float(SOLVE_ESV_BIAS) if GAUSS_NEWTON == "biased" else None

    print("\n")
    print("====== Inversion Start ======")

    funcs = choose_inversion_functions()

    state = {
        "CDOG_guess": CDOG_initial_guess,
        "offset": offset,
        "time_bias": time_bias_guess,
        "esv_bias": esv_bias_guess,
        "lever": lever_est,
    }

    steps = _build_solver_steps(
        funcs,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        GPS_grid,
        transponder_coordinates_est,
        dz_array,
        angle_array,
        esv_matrix,
    )

    state = solve_pipeline(steps, state)

    print("\n")
    print("====== Inversion complete ======")
    logger.info(
        "Final CDOG_guess=%s  lever=%s  offset=%.6g",
        _arrfmt(state["CDOG_guess"]),
        _arrfmt(state["lever"]),
        state["offset"],
    )
    logger.info(
        "Final biases: time=%s  esv=%s",
        "None" if state["time_bias"] is None else f"{state['time_bias']:.6g}",
        "None" if state["esv_bias"] is None else f"{state['esv_bias']:.6g}",
    )

    return state


# ---------------------------- Quality reporting --------------------------- #


def _report_quality_synthetic(
    CDOG_data,
    GPS_data,
    CDOG_location,
    GPS_Coordinates,
    transponder_coordinates_true,
    solve_gps_grid: np.ndarray,
    inversion_result,
    gen_lever_used: np.ndarray,
) -> None:
    pos_err_vec = (
        np.asarray(inversion_result["CDOG_guess"], dtype=float) - CDOG_location
    )
    pos_err_norm = float(np.linalg.norm(pos_err_vec))

    lever_err_vec = np.asarray(inversion_result["lever"], dtype=float) - np.asarray(
        gen_lever_used, dtype=float
    )
    lever_err_norm = float(np.linalg.norm(lever_err_vec))

    transponder_coordinates_est = findTransponder(
        GPS_Coordinates,
        solve_gps_grid,
        inversion_result["lever"],
    )
    transp_err_vec = np.asarray(transponder_coordinates_est, dtype=float) - np.asarray(
        transponder_coordinates_true, dtype=float
    )
    transp_err_norm = float(np.linalg.norm(transp_err_vec))

    time_bias_err = (
        None
        if inversion_result["time_bias"] is None
        else float(inversion_result["time_bias"] - float(GEN_TIME_BIAS))
    )
    esv_bias_err = (
        None
        if inversion_result["esv_bias"] is None
        else float(inversion_result["esv_bias"] - float(GEN_ESV_BIAS))
    )

    offset_err = float(inversion_result["offset"] - float(GEN_OFFSET))

    print("\n")
    print("====== Quality (synthetic) ======")
    logger.info(
        "Position error: vec=%s  norm=%.6g m", _arrfmt(pos_err_vec), pos_err_norm
    )
    logger.info(
        "Lever error:    vec=%s  norm=%.6g m", _arrfmt(lever_err_vec), lever_err_norm
    )
    logger.info(
        "Transponder error: vec=%s  norm=%.6g m",
        _arrfmt(transp_err_vec),
        transp_err_norm,
    )
    logger.info("Offset error: %.6g", offset_err)
    logger.info(
        "Bias errors: time=%s  esv=%s",
        "None" if time_bias_err is None else f"{time_bias_err:.6g} s",
        "None" if esv_bias_err is None else f"{esv_bias_err:.6g} m/s",
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
