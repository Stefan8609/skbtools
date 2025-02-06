import numpy as np
from numba import njit
from scipy import stats

depth = np.ascontiguousarray(np.genfromtxt('GPSData/depth_cast2_smoothed.txt'))
cz = np.ascontiguousarray(np.genfromtxt('GPSData/cz_cast2_smoothed.txt'))

@njit
def ray_tracing(iga, z_a, z_b, depth, cz):
    """Ray Tracing Algorithm for a give initial grazing angle, source and receiver depths, and sound speed profile."""
    z_b_closest = np.abs(depth - z_b).argmin()
    z_a_closest = np.abs(depth - z_a).argmin()
    depth = depth[z_a_closest:z_b_closest]
    cz = cz[z_a_closest:z_b_closest]
    dz = np.diff(depth)

    #Initialize variables
    x_arr = np.zeros(len(dz))
    z_arr = np.zeros(len(dz))

    x = 0
    z = depth[0]
    c = cz[0]
    time = 0
    theta = (90-iga) * np.pi / 180

    for i in range(len(dz)):
        c_next = cz[i]
        theta = np.arcsin(c_next * np.sin(theta) / c)

        if theta > np.pi/2 - 0.01:
            print("Error: Total Internal Reflection")
            return np.NaN, np.NaN, np.NaN, x_arr[:i], z_arr[:i]
        dx = dz[i] * np.tan(theta)
        ds = np.sqrt(dx**2 + dz[i]**2)
        dt = ds / c
        c = c_next
        x = x + dx
        z = z + dz[i]

        x_arr[i] = x
        z_arr[i] = z

        time = time + dt
    return x, z, time, x_arr, z_arr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(0, 91, 5):
        x, z, time, x_arr, z_arr = ray_tracing(i, 52, 5250, depth, cz)
        plt.plot(x_arr, z_arr, label = f"{i} degrees")
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("depth (m)")
    plt.title("Ray Tracing")
    plt.legend()
    plt.show()