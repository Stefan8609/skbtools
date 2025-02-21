import numpy as np
from numba import njit
from Ray_Tracing_Iter_Locate import ray_trace_locate, ray_tracing

depth = np.ascontiguousarray(np.genfromtxt('GPSData/depth_cast2_smoothed.txt')[::20])
cz = np.ascontiguousarray(np.genfromtxt('GPSData/cz_cast2_smoothed.txt')[::20])

"""Build for depths -50 to [-5200 to -5250]

The SVP limits us to 5250 meters
    May need some extrapolation method to get to depths below 5250 meters
    The SVP appears linear at such depths (velocity changes approximately with pressure)
"""

import matplotlib.pyplot as plt

plt.plot(cz, depth, label="Normal SVP")
cz_perturbation = -0.001 * (5250-depth)
cz = cz + cz_perturbation

plt.plot(cz, depth, label="Perturbed SVP")
plt.gca().invert_yaxis()
plt.xlabel("Sound Speed (m/s)")
plt.ylabel("Depth (m)")
plt.title("Sound Speed Profile with Perturbation (-0.001 * depth)")
plt.legend()
plt.show()

@njit
def construct_esv(depth, cz):
    """Builds a table of effective sound velocities for a range of beta angles and dz values for given SVP"""
    beta_array = np.linspace(20, 90, 152)
    z_array = np.linspace(5200, 5250, 51)

    esv_matrix = np.zeros((len(z_array), len(beta_array)))
    z_a = 52

    i=0
    for z in z_array:
        print(i, "/", len(z_array))
        j=0
        for beta in beta_array:
            x = (z-z_a) / np.tan(beta * np.pi / 180)
            alpha = ray_trace_locate(z_a, z, x, depth, cz)
            x_found, z_dist, time = ray_tracing(alpha, z_a, z, depth, cz)
            dist = np.sqrt(x**2 + (z - z_a)**2)
            esv_matrix[i, j] = dist / time
            j+=1
        i+=1
    return beta_array, z_array, esv_matrix

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio

    z_a = 52
    beta_array, z_array, esv_matrix = construct_esv(depth, cz)
    print(beta_array)
    print(z_array)
    print(esv_matrix)

    plt.figure(figsize=(10, 6))
    plt.contourf(beta_array, z_array, esv_matrix, levels=10)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel("Elevation Angle (degrees)")
    plt.ylabel("Depth (m)")
    plt.title("Effective Sound Velocity (m/s)")
    plt.show()

    dist_array = z_array - z_a

    data_to_save = {"angle": beta_array, "distance": dist_array, "matrice": esv_matrix}
    sio.savemat('GPSData/global_table_esv_perturbed.mat', data_to_save)







