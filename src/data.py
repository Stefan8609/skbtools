from __future__ import annotations

from pathlib import Path

# Path to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Path to the GPS data directory located at the repository root
GPS_DATA_PATH = PROJECT_ROOT / "Data"


def gps_data_path(*paths: str | Path) -> Path:
    """Return absolute path inside the ``Data`` directory."""
    return GPS_DATA_PATH.joinpath(*map(str, paths))
