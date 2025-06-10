import numpy as np
from numba import njit
from Ray_Tracing_Iter_Locate import ray_trace_locate, ray_tracing

depth = np.ascontiguousarray(np.genfromtxt('GPSData/depth_cast2_smoothed.txt')[::20])
cz = np.ascontiguousarray(np.genfromtxt('GPSData/cz_cast2_smoothed.txt')[::20])

"""Build for depths -125 to [-5150 to -5275]

The SVP limits us to 5250 meters
    May need some extrapolation method to get to depths below 5250 meters
    The SVP appears linear at such depths (velocity changes approximately with pressure)
"""

import matplotlib.pyplot as plt

plt.plot(cz, depth, label="CTD Bermuda SVP")
# Create a perturbation that starts at ~5 m/s at surface and diminishes below 100m
# surface_effect = 5.0 * np.exp(-depth/40)  # Exponential decay with depth
# deep_variation = 0.3 * np.sin(depth/200)  # Small sinusoidal variation for deep water
# cz_perturbation = surface_effect + deep_variation
# cz = cz + cz_perturbation

# plt.plot(cz, depth, label="Realistically Perturbed SVP")
# plt.gca().invert_yaxis()
# plt.xlabel("Sound Speed (m/s)")
# plt.ylabel("Depth (m)")
# plt.title("Sound Speed Profile with Perturbation (-0.001 * depth)")
# plt.legend()
# plt.show()
#
# plt.plot(cz_perturbation, depth, label="Realistic SVP Perturbation")
# plt.gca().invert_yaxis()
# plt.xlabel("Sound Speed (m/s)")
# plt.ylabel("Depth (m)")
# plt.title("Sound Speed Profile Perturbation")
# plt.legend()
# plt.show()

@njit()
def construct_esv(depth, cz):
    """Builds a table of effective sound velocities for a range of beta angles and dz values for given SVP
    Inputs
    depth: Depth array (m)
    cz: Sound speed array (m/s) corresponding to depth array
    Outputs
    beta_array: Array of beta angles (degrees)
    z_array: Array of dz values (m)
    esv_matrix: Matrix of effective sound velocities (m/s) corresponding to beta angles and dz values
    """
    beta_array = np.linspace(20, 90, 400)
    z_array = np.linspace(5150, 5250, 101)

    esv_matrix = np.zeros((len(z_array), len(beta_array)))
    z_a = 35

    # Pre-calculate constants
    beta_rad = beta_array * np.pi / 180
    tan_beta = np.tan(beta_rad)


    for i in range(len(z_array)):
        print("Progress: ", i, "/", len(z_array))
        z = z_array[i]
        x_values = (z - z_a) / tan_beta

        for j in range(len(beta_array)):
            x = x_values[j]
            alpha = ray_trace_locate(z_a, z, x, depth, cz)
            x_found, z_dist, time = ray_tracing(alpha, z_a, z, depth, cz)
            dist = np.sqrt(x ** 2 + (z - z_a) ** 2)
            esv_matrix[i, j] = dist / time

    return beta_array, z_array, esv_matrix

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from time import time

    z_a = 35

    # Time the execution
    start_time = time()
    beta_array, z_array, esv_matrix = construct_esv(depth, cz)
    end_time = time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    plt.figure(figsize=(10, 6))
    plt.contourf(beta_array, z_array, esv_matrix, levels=10, cmap='viridis')
    plt.colorbar(label='ESV (m/s)')
    plt.gca().invert_yaxis()
    plt.xlabel("Elevation Angle (degrees)")
    plt.ylabel("Depth (m)")
    plt.title("Effective Sound Velocity (m/s)")
    plt.show()

    dz_array = z_array - z_a

    data_to_save = {
        "angle": beta_array,
        "distance": dz_array,
        "matrice": esv_matrix
    }
    sio.savemat('GPSData/global_table_esv_normal.mat', data_to_save)