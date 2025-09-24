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
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
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

# Lever arm defaults (transponder relative to GPS1)
DEFAULT_SYNTHETIC_LEVER = np.array([-12.4659, 9.6021, -13.2993])
DEFAULT_REAL_LEVER = np.array([-12.48862757, 0.22622633, -15.89601934])

# Default offset applied to the CDOG initial location guess (absolute)
DEFAULT_INITIAL_DOG_OFFSET = np.array([100.0, 100.0, 200.0])

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
# Structured containers used throughout the workflow


@dataclass
class Geometry:
    """Geometric relationship between GPS receivers and the transponder."""

    gps1_to_others: np.ndarray
    gps1_to_transponder: np.ndarray


@dataclass
class SolverSetup:
    """Configuration for the inversion stage."""

    initial_position: np.ndarray
    offset: float
    geometry: Geometry
    time_bias: Optional[float] = None
    esv_bias: Optional[float] = None


@dataclass
class WorkflowData:
    """Bundle of DOG/GPS observations passed through the workflow."""

    cdog_clock: np.ndarray
    cdog_reference: np.ndarray
    gps_coordinates: np.ndarray
    gps_time: np.ndarray
    transponder_coordinates: np.ndarray

    @classmethod
    def from_tuple(cls, data: "WorkflowData | tuple") -> "WorkflowData":
        """Coerce tuples returned by legacy helpers into ``WorkflowData``."""

        if isinstance(data, cls):
            return data
        cdog_clock, cdog_reference, gps_coordinates, gps_time, transponder = data
        return cls(
            cdog_clock=np.asarray(cdog_clock, dtype=float),
            cdog_reference=np.asarray(cdog_reference, dtype=float),
            gps_coordinates=np.asarray(gps_coordinates, dtype=float),
            gps_time=np.asarray(gps_time, dtype=float),
            transponder_coordinates=np.asarray(transponder, dtype=float),
        )


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


def _resolve_vector3(value: Optional[np.ndarray], default: np.ndarray) -> np.ndarray:
    """Return a copy of ``value`` if provided, otherwise the default vector."""

    if value is None:
        return np.array(default, dtype=float, copy=True)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"Expected vector of length 3, received shape {np.asarray(value).shape}"
        )
    return np.array(arr, dtype=float, copy=True)


def _resolve_grid(value: Optional[np.ndarray], default: np.ndarray) -> np.ndarray:
    """Return a copy of ``value`` (shape ``(4,3)``) or the provided default."""

    if value is None:
        arr = np.array(default, dtype=float, copy=True)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 3):
        raise ValueError(f"Expected grid of shape (4,3), received {arr.shape}")
    return np.array(arr, dtype=float, copy=True)


def resolve_generation_geometry(args: argparse.Namespace) -> Geometry:
    """Resolve geometry used when synthesising data."""

    gps_grid = _resolve_grid(args.gen_gps_grid, DEFAULT_GPS_GRID)
    lever = _resolve_vector3(args.gen_lever, DEFAULT_SYNTHETIC_LEVER)
    return Geometry(gps_grid, lever)


def resolve_solver_geometry(args: argparse.Namespace) -> Geometry:
    """Resolve geometry used during the inversion stage."""

    default_lever = (
        DEFAULT_REAL_LEVER if args.data_type == "real" else DEFAULT_SYNTHETIC_LEVER
    )
    gps_grid = _resolve_grid(args.solve_gps_grid, DEFAULT_GPS_GRID)
    lever = _resolve_vector3(args.solve_lever, default_lever)
    return Geometry(gps_grid, lever)


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


def generate_synthetic(
    args: argparse.Namespace, dz, angle, esv, geometry: Geometry
) -> WorkflowData:
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
    logger.info("  gps geometry=%s", _arrfmt(geometry.gps1_to_others))
    logger.info("  lever used=%s", _arrfmt(geometry.gps1_to_transponder))
    logger.info(
        "  ESV table arrays: dz[%d], angle[%d], esv%s",
        np.size(dz),
        np.size(angle),
        " (matrix)" if hasattr(esv, "shape") else "",
    )

    if args.trajectory == "bermuda":
        dataset = bermuda_trajectory(
            args.time_noise,
            args.position_noise,
            args.gen_esv_bias,
            args.gen_time_bias,
            dz,
            angle,
            esv,
            offset=args.gen_offset,
            gps1_to_others=geometry.gps1_to_others,
            gps1_to_transponder=geometry.gps1_to_transponder,
            DOG_num=args.dog_num,
        )
    else:
        dataset = generateUnaligned(
            args.n_samples,
            args.time_noise,
            args.position_noise,
            args.gen_offset,
            args.gen_esv_bias,
            args.gen_time_bias,
            dz,
            angle,
            esv,
            gps1_to_others=geometry.gps1_to_others,
            gps1_to_transponder=geometry.gps1_to_transponder,
            trajectory=args.trajectory,
        )

    workflow = WorkflowData.from_tuple(dataset)
    logger.info(
        "Generated synthetic dataset: CDOG_clock=%s GPS_time=%s",
        workflow.cdog_clock.shape,
        workflow.gps_time.shape,
    )
    return workflow


def load_real_data(args: argparse.Namespace) -> WorkflowData:
    """Load processed real GPS/DOG data from disk."""

    logger.info("Loading real data for DOG %d", args.dog_num)
    path = gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{args.dog_num}.npz")
    data = np.load(path)
    gps_coordinates = data["GPS_Coordinates"].astype(float)
    gps_time = data["GPS_data"].astype(float)
    cdog_clock = data["CDOG_data"].astype(float)
    cdog_guess = data["CDOG_guess"].astype(float)

    # True transponder coordinates are unknown for real data; provide zeros
    transponder = np.zeros_like(gps_coordinates[:, 0, :])
    logger.info("Loaded real data from %s", path)
    logger.info(
        "  GPS_Coordinates shape=%s  GPS_time shape=%s",
        gps_coordinates.shape,
        gps_time.shape,
    )
    logger.info(
        "  CDOG_clock shape=%s  CDOG_guess=%s", cdog_clock.shape, _arrfmt(cdog_guess)
    )
    return WorkflowData(
        cdog_clock=cdog_clock,
        cdog_reference=cdog_guess,
        gps_coordinates=gps_coordinates,
        gps_time=gps_time,
        transponder_coordinates=transponder,
    )


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
    data: WorkflowData,
    setup: SolverSetup,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Create a list of step functions that operate on a shared state dict.

    The state dict keys used/updated are: 'initial_guess', 'position', 'offset',
    'time_bias', 'esv_bias', 'lever', 'transponder_est'.
    """
    steps = []

    cdog_clock = data.cdog_clock
    gps_time = data.gps_time
    gps_coordinates = data.gps_coordinates
    gps1_to_others = setup.geometry.gps1_to_others

    logger.info("Building solver steps...")
    logger.info("  GPS grid (gps1_to_others): %s", _arrfmt(gps1_to_others))
    logger.info(
        "  dz size=%d  angle size=%d  esv_matrix shape=%s",
        np.size(dz_array),
        np.size(angle_array),
        getattr(esv_matrix, "shape", None),
    )

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
                cdog_clock,
                gps_time,
                gps_coordinates,
                gps1_to_others,
                state["initial_guess"],
                state["lever"],
                dz_array,
                angle_array,
                esv_matrix,
                initial_offset=state["offset"],
                real_data=args.data_type == "real",
            )
            state["lever"] = np.asarray(lever_guess, dtype=float)
            state["offset"] = float(offset)
            result_arr = np.asarray(result, dtype=float)
            if result_arr.size >= 3:
                state["initial_guess"] = result_arr[:3]
            if result_arr.size >= 5:
                state["time_bias"] = float(result_arr[3])
                state["esv_bias"] = float(result_arr[4])
            state["transponder_est"] = findTransponder(
                gps_coordinates, gps1_to_others, state["lever"]
            )
            logger.info(
                "[STEP] Annealing: updated lever=%s  offset=%.6g  new initial_guess=%s",
                _arrfmt(state["lever"]),
                state["offset"],
                _arrfmt(state["initial_guess"]),
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
                "[STEP] Gauss–Newton initial: start (real_data=%s)",
                args.data_type == "real",
            )
            result, off = initial_fn(
                state["initial_guess"],
                cdog_clock,
                gps_time,
                state["transponder_est"],
                dz_array,
                angle_array,
                esv_matrix,
                real_data=args.data_type == "real",
            )
            arr = np.asarray(result, dtype=float)
            state["position"] = arr[:3]
            state["offset"] = float(off)
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
                        state["position"],
                        cdog_clock,
                        gps_time,
                        state["transponder_est"],
                        state["offset"],
                        state["esv_bias"],
                        state["time_bias"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    arr = np.asarray(result, dtype=float)
                    state["position"] = arr[:3]
                    state["offset"] = float(off)
                    state["time_bias"] = float(arr[3])
                    state["esv_bias"] = float(arr[4])
                else:
                    result, off = transition_fn(
                        state["position"],
                        cdog_clock,
                        gps_time,
                        state["transponder_est"],
                        state["offset"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    state["position"] = np.asarray(result, dtype=float)
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
                        cdog_clock,
                        gps_time,
                        state["transponder_est"],
                        state["offset"],
                        state["esv_bias"],
                        state["time_bias"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    arr = np.asarray(result, dtype=float)
                    state["position"] = arr[:3]
                    state["time_bias"] = float(arr[3])
                    state["esv_bias"] = float(arr[4])
                else:
                    result, *_ = final_fn(
                        state["position"],
                        cdog_clock,
                        gps_time,
                        state["transponder_est"],
                        state["offset"],
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data=args.data_type == "real",
                    )
                    state["position"] = np.asarray(result, dtype=float)
                logger.info(
                    "[STEP] Gauss–Newton final: position=%s", _arrfmt(state["position"])
                )
                return state

            steps.append(step_final)

    return steps


def run_inversion(
    args: argparse.Namespace,
    data,
    geometry: Geometry,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Execute the selected inversion workflow."""

    workflow = WorkflowData.from_tuple(data)

    logger.info("==== Inversion setup ====")
    logger.info("data_type=%s", args.data_type)
    logger.info("Initial CDOG reference: %s", _arrfmt(workflow.cdog_reference))
    logger.info("Default/initial DOG offset: %s", _arrfmt(DEFAULT_INITIAL_DOG_OFFSET))
    logger.info("Using GPS grid (solve): %s", _arrfmt(geometry.gps1_to_others))
    logger.info("Using lever guess (solve): %s", _arrfmt(geometry.gps1_to_transponder))

    if args.solve_cdog_init is not None:
        initial_guess = np.asarray(args.solve_cdog_init, dtype=float).reshape(-1)
    else:
        initial_guess = (
            np.asarray(workflow.cdog_reference, dtype=float).reshape(-1)
            + DEFAULT_INITIAL_DOG_OFFSET
        )
    logger.info("Using initial DOG position guess: %s", _arrfmt(initial_guess))

    offset = float(args.solve_offset)
    logger.info("Initial scalar offset: %.6g", offset)

    solver_geometry = Geometry(
        np.array(geometry.gps1_to_others, dtype=float, copy=True),
        np.array(geometry.gps1_to_transponder, dtype=float, copy=True),
    )

    transponder_est = findTransponder(
        workflow.gps_coordinates,
        solver_geometry.gps1_to_others,
        solver_geometry.gps1_to_transponder,
    )
    logger.info(
        "Transponder estimate from lever & GPS grid: %s", _arrfmt(transponder_est)
    )

    time_bias_guess = (
        float(args.solve_time_bias) if args.gauss_newton == "biased" else None
    )
    esv_bias_guess = (
        float(args.solve_esv_bias) if args.gauss_newton == "biased" else None
    )

    setup = SolverSetup(
        initial_position=np.array(initial_guess, dtype=float, copy=True),
        offset=offset,
        geometry=solver_geometry,
        time_bias=time_bias_guess,
        esv_bias=esv_bias_guess,
    )

    funcs = choose_inversion_functions(args)

    state = {
        "initial_guess": setup.initial_position.copy(),
        "position": None,
        "offset": setup.offset,
        "time_bias": setup.time_bias,
        "esv_bias": setup.esv_bias,
        "lever": setup.geometry.gps1_to_transponder.copy(),
        "transponder_est": transponder_est,
    }

    steps = _build_solver_steps(
        args,
        funcs,
        workflow,
        setup,
        dz_array,
        angle_array,
        esv_matrix,
    )

    state = solve_pipeline(steps, state)

    if state["position"] is None:
        state["position"] = np.asarray(state["initial_guess"], dtype=float)[:3]

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

    if not logging.getLogger().handlers():
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

    generation_geometry = resolve_generation_geometry(args)
    solver_geometry = resolve_solver_geometry(args)
    logger.info("  generation gps grid=%s", _arrfmt(generation_geometry.gps1_to_others))
    logger.info(
        "  generation lever=%s", _arrfmt(generation_geometry.gps1_to_transponder)
    )
    logger.info("  solver gps grid=%s", _arrfmt(solver_geometry.gps1_to_others))
    logger.info("  solver lever=%s", _arrfmt(solver_geometry.gps1_to_transponder))

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
        dataset = generate_synthetic(
            args, dz_gen, angle_gen, esv_gen, generation_geometry
        )
        logger.info("Synthetic data generated.")
    else:
        dataset = load_real_data(args)
        logger.info("Real data loaded.")

    res = run_inversion(args, dataset, solver_geometry, dz_inv, angle_inv, esv_inv)

    pos_err_vec = None
    pos_err_norm = 0.0
    lever_err_vec = None
    lever_err_norm = 0.0
    transp_err_vec = None
    transp_err_norm = 0.0
    time_bias_err = None
    esv_bias_err = None

    if args.data_type == "synthetic":
        cdog_ref = np.asarray(dataset.cdog_reference, dtype=float)[:3]
        pos_err_vec = np.asarray(res.position, dtype=float) - cdog_ref
        pos_err_norm = float(np.linalg.norm(pos_err_vec))

        if args.gen_lever is not None:
            lever_truth = np.asarray(
                generation_geometry.gps1_to_transponder, dtype=float
            )
            lever_err_vec = np.asarray(res.lever, dtype=float) - lever_truth
            lever_err_norm = float(np.linalg.norm(lever_err_vec))

        if res.time_bias is not None:
            time_bias_err = float(res.time_bias - float(args.gen_time_bias))
        if res.esv_bias is not None:
            esv_bias_err = float(res.esv_bias - float(args.gen_esv_bias))

        transponder_est_m = findTransponder(
            dataset.gps_coordinates,
            solver_geometry.gps1_to_others,
            res.lever,
        )
        transp_err_vec = np.asarray(transponder_est_m, dtype=float) - np.asarray(
            dataset.transponder_coordinates, dtype=float
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

    print("Inversion estimate:", np.round(res.position, 3))
    print("Estimated lever:", np.round(res.lever, 3))
    print("Estimated offset:", round(float(res.offset), 6))
    if res.time_bias is not None or res.esv_bias is not None:
        print("Time bias:", None if res.time_bias is None else round(res.time_bias, 6))
        print("ESV bias:", None if res.esv_bias is None else round(res.esv_bias, 6))
    if args.data_type == "synthetic" and pos_err_vec is not None:
        print(
            "Position error (m):",
            np.round(pos_err_vec, 4),
            " | ||e|| =",
            round(pos_err_norm, 6),
        )
        if lever_err_vec is not None:
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
        if res.time_bias is not None and time_bias_err is not None:
            print("Time bias error (s):", round(time_bias_err, 9))
        if res.esv_bias is not None and esv_bias_err is not None:
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
