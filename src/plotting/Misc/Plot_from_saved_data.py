import numpy as np
import matplotlib.pyplot as plt

from bermuda_data import gps_output_path
from plotting.Plot_Modular import time_series_plot


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 20,
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
            "\n"
            r"\usepackage{textcomp}",
        }
    )

    DOG_num = 4
    data_file = gps_output_path(f"Plot_Data/integer_pick_metrics_DOG{DOG_num}.npz")
    data = np.load(data_file)

    DOG_name = {1: "DOG1", 3: "DOG3", 4: "DOG4"}
    save_path = "Figs/Inversion_Workflow"
    file_tag = f"{DOG_num}" + "_SimAnn"

    time_series_plot(
        data["CDOG_clock"],
        data["CDOG_full"],
        data["GPS_clock"],
        data["GPS_full"],
        path=save_path,
        save=True,
        zoom_start=35000,
        DOG_num=file_tag,
    )
