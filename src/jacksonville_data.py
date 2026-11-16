from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPS_DATA_PATH = PROJECT_ROOT / "Jacksonville_Data"


def gps_data_path(*paths: str | Path) -> Path:
    """Return a path inside the repository's Jacksonville_Data directory."""
    return GPS_DATA_PATH.joinpath(*map(str, paths))


def gps_output_path(*paths: str | Path) -> Path:
    """Return a Jacksonville_Data/Output path, creating its parent directory."""
    path = GPS_DATA_PATH.joinpath("Output", *map(str, paths))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
