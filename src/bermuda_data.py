from __future__ import annotations

from pathlib import Path

import numpy as np

# Path to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Path to the GPS data directory located at the repository root
GPS_DATA_PATH = PROJECT_ROOT / "Bermuda_Data"


def gps_data_path(*paths: str | Path) -> Path:
    """Absolute path inside the ``Data`` directory, from segments or a "a/b" string."""

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

# Receivers 1-4 relative to receiver 1 in the vessel frame (m): rigid grid fitted
# to the PRIDE_final positions by geometry/GPS_Grid.py (fit_rigid_grid), rotated
# onto the previous grid's frame so the lever-arm axes keep their meaning.
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

# Survey legs (GPS hours) kept by the leg-mask sensitivity test.
LEG_WINDOWS_H = [(9.0, 11.0), (12.4, 15.0)]


def leg_mask(GPS_data, windows_h=LEG_WINDOWS_H):
    """Boolean mask of the GPS epochs (s) inside any of ``windows_h`` (GPS hours)."""
    hours = np.asarray(GPS_data) / 3600
    mask = np.zeros(len(hours), dtype=bool)
    for start, end in windows_h:
        mask |= (hours >= start) & (hours <= end)
    return mask
