import numpy as np
import scipy.io as sio

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    initial_bias_geiger,
    transition_bias_geiger,
    final_bias_geiger,
)
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from data import gps_data_path

esv_table_generate = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
dz_array_generate = esv_table_generate["distance"].flatten()
angle_array_generate = esv_table_generate["angle"].flatten()
esv_matrix_generate = esv_table_generate["matrice"]

esv_table_inversion = sio.loadmat(
    gps_data_path("ESV_Tables/global_table_esv_realistic_perturbed.mat")
)
dz_array_inversion = esv_table_inversion["distance"].flatten()
angle_array_inversion = esv_table_inversion["angle"].flatten()
esv_matrix_inversion = esv_table_inversion["matrice"]

time_noise = 2 * 10**-5
position_noise = 2 * 10**-2

initial_lever = np.array([-12.58, 0.165, -11.345])
offset = 1991.01247

(
    CDOG_data,
    CDOG,
    GPS_Coordinates,
    GPS_data,
    true_transponder_coordinates,
) = bermuda_trajectory(
    time_noise,
    position_noise,
    dz_array_generate,
    angle_array_generate,
    esv_matrix_generate,
)
true_offset = 1991.01236648
gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.4054, -4.20905, 0.060621],
        [-12.1105, -0.956145, 0.00877],
        [-8.70446831, 5.165195, 0.04880436],
    ]
)
real_data = True
initial_guess = CDOG + [100, 100, 200]

for dz in np.linspace(-5.0, 2.0, 15):
    lever = initial_lever + np.array([0, 0, dz])
    print("Lever: ", lever)
    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    inversion_result, offset = initial_bias_geiger(
        initial_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array_inversion,
        angle_array_inversion,
        esv_matrix_inversion,
        real_data=real_data,
    )
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]

    inversion_result, offset = transition_bias_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        esv_bias,
        time_bias,
        dz_array_inversion,
        angle_array_inversion,
        esv_matrix_inversion,
        real_data=real_data,
    )
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]

    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        esv_bias,
        time_bias,
        dz_array_inversion,
        angle_array_inversion,
        esv_matrix_inversion,
        real_data=real_data,
    )
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("Offset: {:.4f}".format(offset), "DIFF: {:.4f}".format(offset - true_offset))
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion_Workflow:", np.round(inversion_result, 3))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
    print("\n")
