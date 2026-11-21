from __future__ import annotations

from pathlib import Path

import numpy as np

# Path to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Path to the GPS data directory located at the repository root
GPS_DATA_PATH = PROJECT_ROOT / "Bermuda_Data"


def gps_data_path(*paths: str | Path) -> Path:
    """Return absolute path inside the ``Data`` directory.

    Parameters may be provided either as separate path segments or as a
    single string containing forward slashes. For example, the following
    calls are equivalent::

        gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz")
        gps_data_path("GPS_Data", "Processed_GPS_Receivers_DOG_1.npz")
    """

    return GPS_DATA_PATH.joinpath(*map(str, paths))


def gps_output_path(*paths: str | Path) -> Path:
    """Return absolute path inside the ``Data/Output`` directory.

    The path is created if it does not already exist. Example usage::

        gps_output_path("mcmc_chain_good.npz")
        gps_output_path("Chains", "mcmc_chain_good.npz")
    """

    path = GPS_DATA_PATH.joinpath("Output", *map(str, paths))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Bermuda site constants. DOG positions are stored as offsets (m, ECEF) from
# CDOG_REFERENCE.
CDOG_REFERENCE = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

# Receivers 2-4 relative to receiver 1 in the vessel frame (m). Rigid grid fitted
# by geometry/GPS_Grid.py (fit_rigid_grid) to the PRIDE_final positions in
# Processed_GPS_Receivers_DOG_1.npz (2026-11-21): the per-antenna median after
# aligning every epoch to the grid, rotated onto the frame of the previous grid so
# the lever-arm axes keep their meaning, receiver 1 at the origin. Rigid-body misfit
# rms 1.8-2.2 cm per antenna (2.3-3.5 cm with the previous grid). Previous grid,
# fitted to the submitted GNSS solutions: [-2.39341409, -4.22350344, 0.02941493],
# [-12.09568416, -0.94568462, 0.0043972], [-8.68674054, 5.16918806, 0.02499322].
GPS1_TO_OTHERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.435242, -4.237408, 0.005141],
        [-12.125877, -0.967643, 0.000789],
        [-8.694497, 5.143601, 0.004453],
    ]
)

# Gauss-Newton starting guesses (offsets from CDOG_REFERENCE, m) and the integer
# clock offsets (s) found by the integer search, by DOG.
CDOG_GUESS_AUGMENTS = {
    1: np.array([-398.16, 371.90, 773.02]),
    3: np.array([825.182985, -111.05670221, -734.10011698]),
    4: np.array([236.428385, -1307.98390221, -2189.21991698]),
}
OFFSET_HINTS_S = {1: 1866.0, 3: 3175.0, 4: 1939.0}

# Survey legs (GPS hours) kept by the leg-mask sensitivity test. Paper I uses the
# full trajectory.
LEG_WINDOWS_H = [(9.0, 11.0), (12.4, 15.0)]


def leg_mask(GPS_data):
    """Boolean mask of the GPS epochs (s) inside LEG_WINDOWS_H."""
    hours = GPS_data / 3600
    mask = np.zeros(len(GPS_data), dtype=bool)
    for start, end in LEG_WINDOWS_H:
        mask |= (hours >= start) & (hours <= end)
    return mask
