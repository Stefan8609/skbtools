import numpy as np
from numba import njit

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

    x = 0
    z = depth[0]
    c = cz[0]
    time = 0
    theta = (90-iga) * np.pi / 180

    for i in range(len(dz)):
        c_next = cz[i]
        theta = np.arcsin(c_next * np.sin(theta) / c)

        if theta > np.pi/2 - 0.05:
            # print("Error: Total Internal Reflection")
            return np.NaN, np.NaN, np.NaN
        dx = dz[i] * np.tan(theta)
        ds = np.sqrt(dx**2 + dz[i]**2)
        dt = ds / c
        c = c_next
        x = x + dx
        z = z + dz[i]
        time = time + dt

    z_dist = z_b - z_a

    return x, z_dist, time

@njit
def ray_trace_locate(z_a, z_b, x, depth, cz):
    """Iterative ray-tracing to find the ray that gets close to a given receiver location"""
    alpha = 50.0
    alpha_step = 10.0
    direction = 0

    while alpha_step > 0.001:
        if alpha < 0 or alpha > 90:
            print("Error: No Solution Found")
            return np.NaN

        x_curr, z, time = ray_tracing(alpha, z_a, z_b, depth, cz)
        if np.abs(x_curr - x) < .01:
            break

        if x_curr < x:
            alpha = max(alpha - alpha_step, 1.0)
            if direction == 1:
                alpha_step = alpha_step / 2
            direction = -1
        else:
            alpha = min(alpha + alpha_step, 90.0)
            if direction == -1:
                alpha_step = alpha_step / 2
            direction = 1
    return alpha

if __name__ == "__main__":
    alpha = ray_trace_locate(100, 5000, 20000, depth, cz)
    x, z, time = ray_tracing(alpha, 100, 5000, depth, cz)
    print(f"Ray Tracing for alpha = {alpha} degrees")
    print(f"Travel Time: {time} s")
    print(f"Source-Receiver Distance: {x} m")
    print(f"Receiver Depth: {z} m")