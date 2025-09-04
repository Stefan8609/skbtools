import numpy as np
import scipy.io as sio
from data import gps_data_path, gps_output_path

"""Main function for running the workflow with modular components."""

"""Type"""
real_data = True  # True: Real Data, False: Synthetic Data

"""Toggles"""
plotting = True  # Toggle plotting of results
save_output = True  # Toggle saving output to file
save_path = gps_output_path("Modular_Synthetic_Output")  # Output path

"""Inversion Toggles"""
ray_tracing = True  # Toggle ray tracing calculation of travel times
alignment = True  # Toggle alignment correction
bias = True  # Toggle bias terms in inversion
annealing = True  # Toggle simulated annealing inversion

if real_data is False:
    """Inputs for Synthetic Data"""
    time_noise = 0.0
    position_noise = 0.0
    input_esv_bias = 0.0
    input_time_bias = 0.0
    input_offset = 0.0

    generating_esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    generating_dz_array = generating_esv_table["distance"].flatten()
    generating_angle_array = generating_esv_table["angle"].flatten()
    generating_esv_matrix = generating_esv_table["matrice"]

    inversion_esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    inversion_dz_array = inversion_esv_table["distance"].flatten()
    inversion_angle_array = inversion_esv_table["angle"].flatten()
    inversion_esv_matrix = inversion_esv_table["matrice"]

    generate_type = "bermuda"  # "bermuda", "line", "random", "realistic", "cross"

placeholder = np.zeros(3)
