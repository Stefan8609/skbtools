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

from data import gps_data_path, gps_output_path
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder

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
    """Generate a synthetic trajectory according to command line options.

    Notes
    -----
    `Generate_Unaligned.generateUnaligned` currently accepts offset, esv_bias,
    time_bias but not lever/gps grid/CDOG. We accept those generation args now
    for CLI symmetry; they will be plumbed through here in the future if/when
    the generator supports them.
    """
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
    path = gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{args.dog_num}.npz")
    data = np.load(path)
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]
    CDOG_data = data["CDOG_data"]
    CDOG_guess = data["CDOG_guess"]

    # True transponder coordinates are unknown for real data; provide zeros
    transponder = np.zeros(3)
    return CDOG_data, CDOG_guess, GPS_Coordinates, GPS_data, transponder


def choose_inversion_functions(args: argparse.Namespace):
    """Select inversion callables based on user options."""
    funcs: dict[str, object] = {}

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

    elif args.annealing:
        # Simulated annealing without Gauss–Newton (rare but allowed)
        from Inversion_Workflow.Inversion.Numba_xAline_Annealing import (
            simulated_annealing,
        )

        funcs["annealing"] = simulated_annealing

    return funcs


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

    # -------------------- SOLVING-SIDE CONFIG --------------------
    # GPS antenna geometry (relative to antenna 1)
    default_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.39341409, -4.22350344, 0.02941493],
            [-12.09568416, -0.94568462, 0.0043972],
            [-8.68674054, 5.16918806, 0.02499322],
        ]
    )
    gps1_to_others = (
        args.solve_gps_grid if args.solve_gps_grid is not None else default_grid
    )

    # Initial lever guess (transponder relative to GPS1)
    lever_guess = (
        args.solve_lever
        if args.solve_lever is not None
        else np.array([-10.0, 3.0, -15.0])
    )

    # Optional override of CDOG initial location (absolute)
    if args.solve_cdog_init is not None:
        initial_guess = args.solve_cdog_init.copy()
    else:
        initial_guess = CDOG + np.array([100.0, 100.0, 200.0])

    # Scalar offset for solver stages
    offset = float(args.solve_offset)

    # -------------------------------------------------------------

    # Transponder estimate from the current lever guess
    transponder_est = findTransponder(GPS_Coordinates, gps1_to_others, lever_guess)

    funcs = choose_inversion_functions(args)

    result = initial_guess

    if "annealing" in funcs:
        anneal = funcs["annealing"]
        # Keep kwargs conservative to avoid signature mismatches
        lever_guess, offset, result = anneal(
            args.anneal_iter,
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            gps1_to_others,
            initial_guess,
            lever_guess,
            dz_array,
            angle_array,
            esv_matrix,
            initial_offset=offset,
            real_data=args.data_type == "real",
        )

        # Update transponder estimate with the annealed lever
        transponder_est = findTransponder(GPS_Coordinates, gps1_to_others, lever_guess)
        initial_guess = result[:3] if result.size > 3 else result

    if "gauss_newton" in funcs:
        initial_fn, transition_fn, final_fn = funcs["gauss_newton"]

        result, offset = initial_fn(
            initial_guess,
            CDOG_data,
            GPS_data,
            transponder_est,
            dz_array,
            angle_array,
            esv_matrix,
            real_data=args.data_type == "real",
        )

        if args.alignment:
            if result.size == 5:
                # Biased path: [x, y, z, time_bias, esv_bias]
                guess = result[:3]
                time_bias = result[3]
                esv_bias = result[4]
                result, offset = transition_fn(
                    guess,
                    CDOG_data,
                    GPS_data,
                    transponder_est,
                    offset,
                    esv_bias,
                    time_bias,
                    dz_array,
                    angle_array,
                    esv_matrix,
                    real_data=args.data_type == "real",
                )
                result, *_ = final_fn(
                    result[:3],
                    CDOG_data,
                    GPS_data,
                    transponder_est,
                    offset,
                    esv_bias,
                    time_bias,
                    dz_array,
                    angle_array,
                    esv_matrix,
                    real_data=args.data_type == "real",
                )
            else:
                guess = result
                result, offset = transition_fn(
                    guess,
                    CDOG_data,
                    GPS_data,
                    transponder_est,
                    offset,
                    dz_array,
                    angle_array,
                    esv_matrix,
                    real_data=args.data_type == "real",
                )
                result, *_ = final_fn(
                    result,
                    CDOG_data,
                    GPS_data,
                    transponder_est,
                    offset,
                    dz_array,
                    angle_array,
                    esv_matrix,
                    real_data=args.data_type == "real",
                )

    return result, offset


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

    dz_inv, angle_inv, esv_inv = load_esv_table(args.inv_esv_table)

    if args.data_type == "synthetic":
        dz_gen, angle_gen, esv_gen = load_esv_table(args.gen_esv_table)
        data = generate_synthetic(args, dz_gen, angle_gen, esv_gen)
    else:
        data = load_real_data(args)

    result, offset = run_inversion(args, data, dz_inv, angle_inv, esv_inv)

    print("Inversion estimate:", np.round(result, 3))
    print("Estimated offset:", round(float(offset), 6))

    if args.save_output:
        out_path = gps_output_path(f"{args.output_name}.npz")
        np.savez(out_path, estimate=result, offset=offset)
        print("Saved output to", out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
