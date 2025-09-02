import os
from typing import Optional
from data import gps_data_path


def save_plot(
    fig,
    chain_name: Optional[str],
    func_name: str,
    subdir: str = "Figs",
    ext: str = "pdf",
):
    """Save ``fig`` to ``subdir`` inside the Data directory.

    The file name is built from ``chain_name`` and ``func_name`` and the figure
    is saved as a PDF by default.
    """
    if chain_name is None:
        chain_name = "chain"
    # Set a descriptive title if one isn't already present
    if not fig._suptitle:
        fig.suptitle(f"{chain_name}: {func_name}")
    fname = f"{chain_name}_{func_name}.{ext}"
    dirpath = gps_data_path(subdir)
    os.makedirs(dirpath, exist_ok=True)
    fullpath = dirpath / fname
    fig.savefig(fullpath, format=ext, bbox_inches="tight")
