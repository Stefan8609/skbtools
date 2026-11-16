import os
from bermuda_data import gps_data_path


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
    fname = f"{func_name}.{ext}"
    dirpath = gps_data_path(subdir)
    os.makedirs(dirpath, exist_ok=True)
    fullpath = dirpath / fname
    fig.savefig(fullpath, format=ext, bbox_inches="tight")
