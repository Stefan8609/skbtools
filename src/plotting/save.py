import os
from data import gps_data_path


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe version of *name* for use in filenames.

    Replaces any path separators and backslashes with underscores and trims
    surrounding whitespace. Leaves other characters intact to preserve
    readability.
    """
    if name is None:
        return ""
    # Replace common path separators to avoid accidental subdirectories
    safe = name.replace("/", "_").replace("\\", "_")
    return safe.strip()


def save_plot(
    fig,
    func_name,
    chain_name=None,
    subdir="Figs",
    ext="pdf",
):
    """Save ``fig`` to ``subdir`` inside the Data directory.

    The file name is built from ``chain_name`` and ``func_name`` and the figure
    is saved as a PDF by default.
    """
    if chain_name is None:
        chain_name = "chain"
    safe_chain = _safe_filename(chain_name)
    if not chain_name:
        fname = f"{func_name}.{ext}"
    else:
        fname = f"{safe_chain}_{func_name}.{ext}"
    dirpath = gps_data_path(subdir)
    os.makedirs(dirpath, exist_ok=True)
    fullpath = dirpath / fname
    fig.savefig(fullpath, format=ext, bbox_inches="tight")
