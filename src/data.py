from __future__ import annotations

from pathlib import Path

# Path to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Path to the GPS data directory located at the repository root
GPS_DATA_PATH = PROJECT_ROOT / "Data"


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
