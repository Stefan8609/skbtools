"""High-level configurable driver for inversion workflows.

This module provides a command line interface that exposes the various
building blocks found in :mod:`Inversion_Workflow`.  The intent is to
offer a single entry point where different data sources (real or
synthetic), trajectory generators, inversion algorithms and noise/bias
parameters can be selected without modifying the underlying library
code.

The script is purposely lightweight – it wires together the available
components and leaves the heavy numerical work to the specialised
modules.  It is therefore well suited as a starting point for
experimentation or for building higher level applications.
"""

from __future__ import annotations

import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import logging

from data import gps_data_path, gps_output_path
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder

logger = logging.getLogger(__name__)


def _arrfmt(a) -> str:
    """Compact array format for logs."""
    return np.array2string(
        np.asarray(a), precision=4, suppress_small=True, separator=", "
    )


def validate_args(args: argparse.Namespace) -> None:
    """Emit warnings for incompatible or ignored combinations.

    This does *not* modify args; it only surfaces likely user mistakes.
    """
    # 1) Data-type vs. generation/trajectory knobs
    if args.data_type == "real":
        logging.warning("Using real data: ignoring --trajectory (synthetic-only).")
        # All --gen-* and --gen-esv-table are ignored when not generating
        ignored_gen = []
        if args.gen_lever is not None:
            ignored_gen.append("--gen-lever")
        if args.gen_gps_grid is not None:
            ignored_gen.append("--gen-gps-grid")
        if args.gen_cdog is not None:
            ignored_gen.append("--gen-cdog")
        if float(args.gen_offset) != 0.0:
            ignored_gen.append("--gen-offset")
        if float(args.gen_esv_bias) != 0.0:
            ignored_gen.append("--gen-esv-bias")
        if float(args.gen_time_bias) != 0.0:
            ignored_gen.append("--gen-time-bias")
        if args.gen_esv_table != "global_table_esv_extended":
            ignored_gen.append("--gen-esv-table")
        if ignored_gen:
            logging.warning(
                "Real-data mode: the following generation options are ignored: %s",
                ", ".join(ignored_gen),
            )

    # 2) Solver pipeline consistency
    if args.gauss_newton == "none" and args.alignment:
        logging.warning("--alignment has no effect because --gauss-newton is 'none'.")

    # ESV/time bias hints are only used in the *biased* GN path
    if args.gauss_newton != "biased":
        bias_flags = []
        if float(args.solve_esv_bias) != 0.0:
            bias_flags.append("--solve-esv-bias")
        if float(args.solve_time_bias) != 0.0:
            bias_flags.append("--solve-time-bias")
        if bias_flags:
            logging.warning(
                "Ignoring %s unless you use --gauss-newton biased.",
                ", ".join(bias_flags),
            )

    # 3) Annealing
    if not args.annealing and int(args.anneal_iter) != 1000:
        logging.warning("--anneal-iter has no effect unless --annealing is enabled.")

    # 4) Output ergonomics
    if not args.save_output and args.output_name != "inversion_result":
        logging.warning(
            "--output-name will not be written because --save-output is not set."
        )


# ---------------------------------------------------------------------------
# Defaults / module-level constants
DEFAULT_GPS_GRID = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)

# Initial lever guess (transponder relative to GPS1)
DEFAULT_LEVER_GUESS = np.array([-10.0, 3.0, -15.0])

# Default offset applied to the CDOG initial location guess (absolute)
DEFAULT_INITIAL_DOG_OFFSET = np.array([100.0, 100.0, 200.0])

# Placeholder for unknown transponder coordinates in real-data mode
DEFAULT_TRANSPONDER_UNKNOWN = np.zeros(3)

# ---------------------------------------------------------------------------
# Result container


@dataclass
class InversionResult:
    """Normalized output from the solver pipeline.

    position : np.ndarray of shape (3,)
    time_bias : Optional[float]  # seconds; None if not estimated
    esv_bias : Optional[float]   # m/s; None if not estimated
    offset : float               # scalar offset returned by solvers
    """

    position: np.ndarray
    lever: np.ndarray
    time_bias: Optional[float]
    esv_bias: Optional[float]
    offset: float


# ---------------------------------------------------------------------------
# Utilities


def _parse_vector3(val: Optional[str]) -> Optional[np.ndarray]:
    """Parse a 3-vector from 'x,y,z' or a .npy path. Returns None if val is falsy."""
    if not val:
        return None
    p = Path(val)
    if p.suffix == ".npy" and p.exists():
        arr = np.load(p)
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size != 3:
            raise ValueError(
                f"Expected 3 elements in vector file '{val}', got {arr.size}"
            )
        return arr.astype(float)
    parts = [s for s in val.split(",") if s.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"Expected 'x,y,z' with 3 numbers, got {val!r}")
    return np.array(list(map(float, parts)), dtype=float)


def _parse_grid4x3(val: Optional[str]) -> Optional[np.ndarray]:
    """Parse a (4,3) GPS grid from 12 comma-separated numbers (row-major) or a .
    npy path."""
    if not val:
        return None
    p = Path(val)
    if p.suffix == ".npy" and p.exists():
        arr = np.load(p).astype(float)
        arr = np.asarray(arr)
        if arr.shape != (4, 3):
            raise ValueError(
                f"Expected grid of shape (4,3) in '{val}', got {arr.shape}"
            )
        return arr
    parts = [s for s in val.split(",") if s.strip() != ""]
    if len(parts) != 12:
        raise ValueError(
            f"Expected 12 numbers for a 4x3 grid, got {len(parts)}: {val!r}"
        )
    return np.array(list(map(float, parts)), dtype=float).reshape(4, 3)


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


def generate_synthetic(args: argparse.Namespace, dz, angle, esv):
    """Generate a synthetic trajectory according to command line options."""
    logger.info("Generating synthetic data with:")
    logger.info(
        "  trajectory=%s  n_samples=%d  time_noise=%.6g  position_noise=%.6g",
        args.trajectory,
        args.n_samples,
        args.time_noise,
        args.position_noise,
    )
    logger.info(
        "  gen_offset=%.6g  gen_esv_bias=%.6g  gen_time_bias=%.6g",
        args.gen_offset,
        args.gen_esv_bias,
        args.gen_time_bias,
    )
    logger.info("  gen_esv_table=%s", getattr(args, "gen_esv_table", "N/A"))
    logger.info(
        "  gen_gps_grid=%s",
        "None" if args.gen_gps_grid is None else _arrfmt(args.gen_gps_grid),
    )
    logger.info(
        "  gen_lever=%s", "None" if args.gen_lever is None else _arrfmt(args.gen_lever)
    )
    logger.info(
        "  gen_cdog=%s", "None" if args.gen_cdog is None else _arrfmt(args.gen_cdog)
    )
    logger.info(
        "  ESV table arrays: dz[%d], angle[%d], esv%s",
        np.size(dz),
        np.size(angle),
        " (matrix)" if hasattr(esv, "shape") else "",
    )
    return generateUnaligned(
        args.n_samples,
        args.time_noise,
        args.position_noise,
        args.gen_offset,
        args.gen_esv_bias,
        args.gen_time_bias,
        dz,
        angle,
        esv,
        gps1_to_others=args.gen_gps_grid,
        gps1_to_transponder=args.gen_lever,
        trajectory=args.trajectory,
    )


def load_real_data(args: argparse.Namespace):
    """Load processed real GPS/DOG data from disk.

    Returns
    -------
    tuple
        (CDOG_data, CDOG_guess, GPS_Coordinates, GPS_data, transponder)
    """
    logger.info("Loading real data for DOG %d", args.dog_num)
    path = gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{args.dog_num}.npz")
    data = np.load(path)
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    CDOG_data = data["CDOG_data"]
    CDOG_guess = data["CDOG_guess"]

    # True transponder coordinates are unknown for real data; provide zeros
    transponder = DEFAULT_TRANSPONDER_UNKNOWN
    logger.info("Loaded real data from %s", path)
    logger.info(
        "  GPS_Coordinates shape=%s  GPS_data shape=%s",
        GPS_Coordinates.shape,
        GPS_data.shape,
    )
    logger.info(
        "  CDOG_data shape=%s  CDOG_guess=%s", CDOG_data.shape, _arrfmt(CDOG_guess)
    )
    return CDOG_data, CDOG_guess, GPS_Coordinates, GPS_data, transponder


def choose_inversion_functions(args: argparse.Namespace):
    """Select inversion callables based on user options."""
    funcs: dict[str, object] = {}
    logger.info(
        "Choosing inversion functions: gauss_newton=%s, alignment=%s, annealing=%s",
        args.gauss_newton,
        bool(args.alignment),
        bool(args.annealing),
    )

    # Alignment + Gauss–Newton (unbiased or biased)
    if args.gauss_newton == "biased":
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
        if args.annealing:
            from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
                simulated_annealing_bias,
            )

            funcs["annealing"] = simulated_annealing_bias
        logger.info(
            "Selected Gauss–Newton (biased) pipeline with optional annealing=%s",
            "annealing" in funcs,
        )

    elif args.gauss_newton == "unbiased":
        from Inversion_Workflow.Inversion.Numba_xAline_Geiger import (
            initial_geiger,
            transition_geiger,
            final_geiger,
        )

        funcs["gauss_newton"] = (initial_geiger, transition_geiger, final_geiger)
        if args.annealing:
            from Inversion_Workflow.Inversion.Numba_xAline_Annealing import (
                simulated_annealing,
            )

            funcs["annealing"] = simulated_annealing
        logger.info(
            "Selected Gauss–Newton (unbiased) pipeline with optional annealing=%s",
            "annealing" in funcs,
        )

    elif args.annealing:
        # Simulated annealing without Gauss–Newton (rare but allowed)
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
    args: argparse.Namespace,
    funcs: dict,
    CDOG_data,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Create a list of step functions that operate on a shared state dict.

    The state dict keys used/updated are: 'initial_guess', 'position', 'offset',
    'time_bias', 'esv_bias', 'lever', 'transponder_est'.
    """
    steps = []

    logger.info("Building solver steps...")
    logger.info("  GPS grid (gps1_to_others): %s", _arrfmt(gps1_to_others))
    logger.info(
        "  dz size=%d  angle size=%d  esv_matrix shape=%s",
        np.size(dz_array),
        np.size(angle_array),
        getattr(esv_matrix, "shape", None),
    )

    # Optional annealing step (works for both biased/unbiased variants)
    if "annealing" in funcs:
        anneal = funcs["annealing"]

        def step_anneal(state):
            logger.info(
                "[STEP] Annealing: start (iter=%d, real_data=%s)",
                args.anneal_iter,
                args.data_type == "real",
            )
            lever_guess, offset, result = anneal(
                args.anneal_iter,
                CDOG_data,
                GPS_data,
                GPS_Coordinates,
                gps1_to_others,
                state["initial_guess"],
                state["lever"],
                dz_array,
                angle_array,
                esv_matrix,
                initial_offset=state["offset"],
                real_data=args.data_type == "real",
            )
            state["lever"] = lever_guess
            state["offset"] = float(offset)
            # Update guess from annealed result
            state["initial_guess"] = (
                result[:3] if np.asarray(result).size > 3 else result
            )
            # Recompute transponder estimate with annealed lever
            state["transponder_est"] = findTransponder(
                GPS_Coordinates, gps1_to_others, state["lever"]
            )
            logger.info(
                "[STEP] Annealing: updated lever=%s  offset=%.6g  new initial_guess=%s",
                _arrfmt(state["lever"]),
                state["offset"],
                _arrfmt(state["initial_guess"]),
            )
            logger.info(
                "[STEP] Annealing: recomputed transponder_est=%s",
                _arrfmt(state["transponder_est"]),
            )
            return state

        steps.append(step_anneal)

    if "gauss_newton" in funcs:
        initial_fn, transition_fn, final_fn = funcs["gauss_newton"]

        def step_initial(state):
            logger.info(
                "[STEP] Gauss–Newton initial: start (real_data=%s)",
                args.data_type == "real",
            )
            result, off = initial_fn(
                state["initial_guess"],
                CDOG_data,
                GPS_data,
                state["transponder_est"],
                dz_array,
                angle_array,
                esv_matrix,
                real_data=args.data_type == "real",
            )
            arr = np.asarray(result)
            state["position"] = arr[:3]
            state["offset"] = float(off)
            # If the solver returns biases, capture them
            if arr.size >= 5:
                state["time_bias"] = float(arr[3])
                state["esv_bias"] = float(arr[4])
            logger.info(
                "[STEP] Gauss–Newton initial: position=%s  "
                "offset=%.6g  time_bias=%s  esv_bias=%s",
                _arrfmt(state["position"]),
                state["offset"],
                "None" if state["time_bias"] is None else f"{state['time_bias']:.6g}",
                "None" if state["esv_bias"] is None else f"{state['esv_bias']:.6g}",
            )
            return state

        steps.append(step_initial)

        if args.alignment:
            # Transition step (signature differs between biased/unbiased)
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
                    # Biased path
                    result, off = transition_fn(
                        state["position"],
                        CDOG_data,
                        GPS_data,
                        state["transponder_est"],
                        state["offset"],
                        state["esv_bias"],
                        state["time_bias"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    state["position"] = np.asarray(result)[:3]
                    state["offset"] = float(off)
                else:
                    # Unbiased path
                    result, off = transition_fn(
                        state["position"],
                        CDOG_data,
                        GPS_data,
                        state["transponder_est"],
                        state["offset"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    state["position"] = np.asarray(result)
                    state["offset"] = float(off)
                logger.info(
                    "[STEP] Gauss–Newton transition: position=%s  offset=%.6g",
                    _arrfmt(state["position"]),
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
                        state["position"],
                        CDOG_data,
                        GPS_data,
                        state["transponder_est"],
                        state["offset"],
                        state["esv_bias"],
                        state["time_bias"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    state["position"] = np.asarray(result)[:3]
                else:
                    result, *_ = final_fn(
                        state["position"],
                        CDOG_data,
                        GPS_data,
                        state["transponder_est"],
                        state["offset"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    state["position"] = np.asarray(result)
                logger.info(
                    "[STEP] Gauss–Newton final: position=%s", _arrfmt(state["position"])
                )
                return state

            steps.append(step_final)

    return steps


def run_inversion(
    args: argparse.Namespace,
    data,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Execute the selected inversion workflow."""
    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = data

    logger.info("==== Inversion setup ====")
    logger.info("data_type=%s", args.data_type)
    logger.info("Initial CDOG guess from data: %s", _arrfmt(CDOG))
    logger.info("Default/initial DOG offset: %s", _arrfmt(DEFAULT_INITIAL_DOG_OFFSET))

    # -------------------- SOLVING-SIDE CONFIG --------------------
    # GPS antenna geometry (relative to antenna 1)
    gps1_to_others = (
        args.solve_gps_grid if args.solve_gps_grid is not None else DEFAULT_GPS_GRID
    )

    # Initial lever guess (transponder relative to GPS1)
    lever_guess = (
        args.solve_lever if args.solve_lever is not None else DEFAULT_LEVER_GUESS
    )

    # Optional override of CDOG initial location (absolute)
    if args.solve_cdog_init is not None:
        initial_guess = args.solve_cdog_init.copy()
    else:
        initial_guess = CDOG + DEFAULT_INITIAL_DOG_OFFSET

    # Scalar offset for solver stages
    offset = float(args.solve_offset)

    # -------------------------------------------------------------

    logger.info("Using GPS grid (solve): %s", _arrfmt(gps1_to_others))
    logger.info("Using lever guess (solve): %s", _arrfmt(lever_guess))
    logger.info("Using initial DOG position guess: %s", _arrfmt(initial_guess))
    logger.info("Initial scalar offset: %.6g", offset)

    # Transponder estimate from the current lever guess
    transponder_est = findTransponder(GPS_Coordinates, gps1_to_others, lever_guess)
    logger.info(
        "Transponder estimate from lever & GPS grid: %s", _arrfmt(transponder_est)
    )

    funcs = choose_inversion_functions(args)

    # Initialize state for the pipeline
    state = {
        "initial_guess": initial_guess,
        "position": None,
        "offset": offset,
        "time_bias": None,
        "esv_bias": None,
        "lever": lever_guess,
        "transponder_est": transponder_est,
    }

    steps = _build_solver_steps(
        args,
        funcs,
        CDOG_data,
        GPS_data,
        GPS_Coordinates,
        gps1_to_others,
        dz_array,
        angle_array,
        esv_matrix,
    )

    state = solve_pipeline(steps, state)

    # Fallback in case no GN ran: use initial guess as position
    if state["position"] is None:
        state["position"] = np.asarray(state["initial_guess"])[:3]

    logger.info("==== Inversion complete ====")
    logger.info(
        "Final position=%s  lever=%s  offset=%.6g",
        _arrfmt(state["position"]),
        _arrfmt(state["lever"]),
        state["offset"],
    )
    logger.info(
        "Final biases: time=%s  esv=%s",
        "None" if state["time_bias"] is None else f"{state['time_bias']:.6g}",
        "None" if state["esv_bias"] is None else f"{state['esv_bias']:.6g}",
    )

    return InversionResult(
        position=np.asarray(state["position"]).astype(float),
        lever=np.asarray(state["lever"]).astype(float),
        time_bias=None if state["time_bias"] is None else float(state["time_bias"]),
        esv_bias=None if state["esv_bias"] is None else float(state["esv_bias"]),
        offset=float(state["offset"]),
    )


# ---------------------------------------------------------------------------
# Command line interface


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data/trajectory options
    parser.add_argument(
        "--data-type",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Use synthetic or real data",
    )
    parser.add_argument(
        "--trajectory",
        default="realistic",
        choices=["realistic", "random", "line", "cross", "bermuda"],
        help="Synthetic trajectory type",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20000,
        help="Number of samples when generating synthetic data",
    )
    parser.add_argument(
        "--dog-num",
        type=int,
        default=3,
        help="DOG data set number when using real data",
    )

    # Noise (generation)
    parser.add_argument("--time-noise", type=float, default=0.0)
    parser.add_argument("--position-noise", type=float, default=0.0)

    # -------------------- NEW: GENERATION ARGUMENTS --------------------
    parser.add_argument(
        "--gen-lever",
        type=str,
        default=None,
        help="Transponder lever (x,y,z) RELATIVE TO GPS1 used in data generation, "
        "or path to .npy with shape (3,). Currently not passed to generator.",
    )
    parser.add_argument(
        "--gen-gps-grid",
        type=str,
        default=None,
        help="Generation GPS grid (4x3) as 12 comma-separated numbers or path to .npy "
        "with shape (4,3). Currently not passed to generator.",
    )
    parser.add_argument(
        "--gen-offset",
        type=float,
        default=0.0,
        help="Scalar offset used when generating synthetic data.",
    )
    parser.add_argument(
        "--gen-cdog",
        type=str,
        default=None,
        help="CDOG base location (x,y,z) for generation, or .npy path. Currently not "
        "plumbed into the generator.",
    )
    parser.add_argument(
        "--gen-esv-bias",
        type=float,
        default=0.0,
        help="ESV bias used in synthetic data generation.",
    )
    parser.add_argument(
        "--gen-time-bias",
        type=float,
        default=0.0,
        help="Time bias used in synthetic data generation.",
    )
    # -------------------------------------------------------------------

    # -------------------- NEW: SOLVING ARGUMENTS -----------------------
    parser.add_argument(
        "--solve-lever",
        type=str,
        default=None,
        help="Initial lever guess (x,y,z) RELATIVE TO GPS1 for solving, or .npy path.",
    )
    parser.add_argument(
        "--solve-gps-grid",
        type=str,
        default=None,
        help="Solving GPS grid (4x3) as 12 comma-separated numbers or path to .npy "
        "with shape (4,3). Overrides the default hardcoded grid.",
    )
    parser.add_argument(
        "--solve-offset",
        type=float,
        default=0.0,
        help="Initial scalar offset for solver (annealing/GN).",
    )
    parser.add_argument(
        "--solve-cdog-init",
        type=str,
        default=None,
        help="Initial DOG location guess (x,y,z) or .npy path. If omitted, "
        "uses CDOG + [100,100,200].",
    )
    parser.add_argument(
        "--solve-esv-bias",
        type=float,
        default=0.0,
        help="Initial ESV bias guess for solving (used only if the solver "
        "path supports it).",
    )
    parser.add_argument(
        "--solve-time-bias",
        type=float,
        default=0.0,
        help="Initial time bias guess for solving (used only if the solver "
        "path supports it).",
    )
    # -------------------------------------------------------------------

    # Inversion choices
    parser.add_argument(
        "--gauss-newton",
        choices=["none", "biased", "unbiased"],
        default="unbiased",
        help="Type of Gauss–Newton solver to apply",
    )
    parser.add_argument(
        "--alignment", action="store_true", help="Apply alignment steps"
    )
    parser.add_argument(
        "--annealing", action="store_true", help="Use simulated annealing"
    )
    parser.add_argument(
        "--anneal-iter", type=int, default=1000, help="Iterations for annealing"
    )

    # ESV table names
    parser.add_argument(
        "--gen-esv-table",
        default="global_table_esv_extended",
        help="ESV table used for synthetic generation",
    )
    parser.add_argument(
        "--inv-esv-table",
        default="global_table_esv_extended",
        help="ESV table used during inversion",
    )

    # Misc
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save inversion result as a NumPy .npz file",
    )
    parser.add_argument(
        "--output-name", default="inversion_result", help="Base name for saved output"
    )

    return parser


def _postprocess_parsed_args(args: argparse.Namespace) -> None:
    """Convert string CLI values into arrays where appropriate."""
    # Generation (stored for symmetry; generator may not use yet)
    args.gen_lever = (
        _parse_vector3(args.gen_lever) if isinstance(args.gen_lever, str) else None
    )
    args.gen_gps_grid = (
        _parse_grid4x3(args.gen_gps_grid)
        if isinstance(args.gen_gps_grid, str)
        else None
    )
    args.gen_cdog = (
        _parse_vector3(args.gen_cdog) if isinstance(args.gen_cdog, str) else None
    )

    # Solving
    args.solve_lever = (
        _parse_vector3(args.solve_lever) if isinstance(args.solve_lever, str) else None
    )
    args.solve_gps_grid = (
        _parse_grid4x3(args.solve_gps_grid)
        if isinstance(args.solve_gps_grid, str)
        else None
    )
    args.solve_cdog_init = (
        _parse_vector3(args.solve_cdog_init)
        if isinstance(args.solve_cdog_init, str)
        else None
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _postprocess_parsed_args(args)
    validate_args(args)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    else:
        logging.getLogger().setLevel(logging.INFO)
    logger.info("==== Program start ====")
    logger.info("Parsed CLI arguments: %s", vars(args))
    # Show resolved array-like arguments explicitly
    logger.info("Resolved arguments:")
    logger.info(
        "  solve_lever=%s",
        "None" if args.solve_lever is None else _arrfmt(args.solve_lever),
    )
    logger.info(
        "  solve_gps_grid=%s",
        "None" if args.solve_gps_grid is None else _arrfmt(args.solve_gps_grid),
    )
    logger.info(
        "  solve_cdog_init=%s",
        "None" if args.solve_cdog_init is None else _arrfmt(args.solve_cdog_init),
    )
    logger.info(
        "  gen_lever=%s", "None" if args.gen_lever is None else _arrfmt(args.gen_lever)
    )
    logger.info(
        "  gen_gps_grid=%s",
        "None" if args.gen_gps_grid is None else _arrfmt(args.gen_gps_grid),
    )
    logger.info(
        "  gen_cdog=%s", "None" if args.gen_cdog is None else _arrfmt(args.gen_cdog)
    )

    dz_inv, angle_inv, esv_inv = load_esv_table(args.inv_esv_table)
    logger.info(
        "Loaded inversion ESV table '%s': dz[%d], angle[%d], esv shape=%s",
        args.inv_esv_table,
        np.size(dz_inv),
        np.size(angle_inv),
        getattr(esv_inv, "shape", None),
    )

    if args.data_type == "synthetic":
        dz_gen, angle_gen, esv_gen = load_esv_table(args.gen_esv_table)
        data = generate_synthetic(args, dz_gen, angle_gen, esv_gen)
        # Unpack for metrics
        CDOG_data_m, CDOG_true_m, GPS_Coordinates_m, GPS_data_m, transponder_true_m = (
            data
        )
        logger.info("Synthetic data generated.")
    else:
        data = load_real_data(args)
        # Unpack for metrics
        CDOG_data_m, CDOG_true_m, GPS_Coordinates_m, GPS_data_m, transponder_true_m = (
            data
        )
        logger.info("Real data loaded.")

    res = run_inversion(args, data, dz_inv, angle_inv, esv_inv)

    # ---------------- Quality metrics (synthetic-only) ----------------
    if args.data_type == "synthetic":
        # True DOG position (from generator) and estimated position
        pos_err_vec = (
            np.asarray(res.position, dtype=float)
            - np.asarray(CDOG_true_m, dtype=float)[:3]
        )
        pos_err_norm = float(np.linalg.norm(pos_err_vec))

        # Lever error if a true lever was provided
        lever_err_vec = None
        lever_err_norm = None
        if args.gen_lever is not None:
            lever_err_vec = np.asarray(res.lever, dtype=float) - np.asarray(
                args.gen_lever, dtype=float
            )
            lever_err_norm = float(np.linalg.norm(lever_err_vec))

        # Bias errors if supported/estimated
        time_bias_err = (
            None
            if res.time_bias is None
            else float(res.time_bias - float(args.gen_time_bias))
        )
        esv_bias_err = (
            None
            if res.esv_bias is None
            else float(res.esv_bias - float(args.gen_esv_bias))
        )

        # Transponder error from final lever & solve grid
        gps1_to_others_used = (
            args.solve_gps_grid if args.solve_gps_grid is not None else DEFAULT_GPS_GRID
        )
        transponder_est_m = findTransponder(
            GPS_Coordinates_m, gps1_to_others_used, res.lever
        )
        transp_err_vec = np.asarray(transponder_est_m, dtype=float) - np.asarray(
            transponder_true_m, dtype=float
        )
        transp_err_norm = float(np.linalg.norm(transp_err_vec))

        logger.info("==== Quality (synthetic) ====")
        logger.info(
            "Position error: vec=%s  norm=%.6g m", _arrfmt(pos_err_vec), pos_err_norm
        )
        if lever_err_vec is not None:
            logger.info(
                "Lever error:    vec=%s  norm=%.6g m",
                _arrfmt(lever_err_vec),
                lever_err_norm,
            )
        else:
            logger.info("Lever error:    (no ground truth lever provided)")
        logger.info(
            "Transponder error: vec=%s  norm=%.6g m",
            _arrfmt(transp_err_vec),
            transp_err_norm,
        )
        logger.info(
            "Bias errors: time=%s  esv=%s",
            "None" if time_bias_err is None else f"{time_bias_err:.6g} s",
            "None" if esv_bias_err is None else f"{esv_bias_err:.6g} m/s",
        )
    # ------------------------------------------------------------------

    print("Inversion estimate:", np.round(res.position, 3))
    print("Estimated lever:", np.round(res.lever, 3))
    print("Estimated offset:", round(float(res.offset), 6))
    if res.time_bias is not None or res.esv_bias is not None:
        print("Time bias:", None if res.time_bias is None else round(res.time_bias, 6))
        print("ESV bias:", None if res.esv_bias is None else round(res.esv_bias, 6))
    if args.data_type == "synthetic":
        print(
            "Position error (m):",
            np.round(pos_err_vec, 4),
            " | ||e|| =",
            round(pos_err_norm, 6),
        )
        if args.gen_lever is not None:
            print(
                "Lever error (m):   ",
                np.round(lever_err_vec, 4),
                " | ||e|| =",
                round(lever_err_norm, 6),
            )
        print(
            "Transponder error (m):",
            np.round(transp_err_vec, 4),
            " | ||e|| =",
            round(transp_err_norm, 6),
        )
        if res.time_bias is not None:
            print("Time bias error (s):", round(time_bias_err, 9))
        if res.esv_bias is not None:
            print("ESV bias error (m/s):", round(esv_bias_err, 9))

    logger.info(
        "Summary: position=%s  lever=%s  offset=%.6g  time_bias=%s  esv_bias=%s",
        _arrfmt(res.position),
        _arrfmt(res.lever),
        float(res.offset),
        "None" if res.time_bias is None else f"{res.time_bias:.6g}",
        "None" if res.esv_bias is None else f"{res.esv_bias:.6g}",
    )

    if args.save_output:
        out_path = gps_output_path(f"{args.output_name}.npz")
        np.savez(
            out_path,
            estimate=res.position,
            lever=res.lever,
            offset=res.offset,
            time_bias=res.time_bias if res.time_bias is not None else np.nan,
            esv_bias=res.esv_bias if res.esv_bias is not None else np.nan,
        )
        print("Saved output to", out_path)


if __name__ == "__main__":
    main()
