import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from data import gps_output_path, gps_data_path
from plotting.Plot_Modular import range_residual

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 20,  # change this freely
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)

DOG_num = 1
data_file = gps_output_path(f"Plot_Data/integer_pick_metrics_DOG{DOG_num}.npz")
data = np.load(data_file)

# plot_integer_pick_metrics_dog(
#     float(data["best_offset"]),
#     data["CDOG_data"],
#     data["GPS_data"],
#     data["times_guess"],
#     data["transponder_coordinates"],
#     data["esv"],
#     half_window=50,
#     step=0.1,
# )

# plt.show()

DOG_name = {1: "DOG1", 3: "DOG3", 4: "DOG4"}
save_path = "Figs/Inversion_Workflow"
file_tag = f"{DOG_num}" + "_SimAnn"

# time_series_plot(data["CDOG_clock"], data["CDOG_full"],
# data["GPS_clock"], data["GPS_full"], path = save_path,
# save=True, zoom_start=35000, DOG_num=file_tag)

esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))
dz_array = esv_table["distance"].flatten()
angle_array = esv_table["angle"].flatten()
esv_matrix = esv_table["matrice"]


inversion_guess = data["inversion_result"][:3]
transponder_coordinates = data["transponder_coordinates"]
esv_bias = data["esv_bias"]
best_offset = data["best_offset"]
CDOG_data = data["CDOG_data"]
GPS_data = data["GPS_data"]


times_guess, esv = calculateTimesRayTracing_Bias_Real(
    inversion_guess,
    transponder_coordinates,
    esv_bias,
    dz_array,
    angle_array,
    esv_matrix,
)

CDOG_clock_ex, CDOG_full_ex, GPS_clock_ex, GPS_full_ex, trans_coords_full, esv_full = (
    two_pointer_index(
        best_offset,
        0.6,
        CDOG_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )
)
GPS_full_ex = GPS_full_ex - data["time_bias"]
range_residual(
    trans_coords_full,
    esv_full,
    inversion_guess,
    CDOG_full_ex,
    GPS_full_ex,
    GPS_clock_ex,
    save=True,
    path=save_path,
    DOG_num=file_tag,
)
