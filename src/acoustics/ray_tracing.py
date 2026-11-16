import numpy as np
from numba import njit
from time import time
from bermuda_data import gps_data_path


@njit
def ray_tracing(iga, z_a, z_b, depth, cz):
    """Return horizontal range, vertical separation (m), and travel time (s)."""
    if depth.ndim != 1 or cz.ndim != 1 or depth.size != cz.size or depth.size < 2:
        raise ValueError(
            "depth and cz must be matching 1D arrays with at least 2 points"
        )
    if not depth[0] <= z_a < z_b <= depth[-1]:
        raise ValueError("Source and receiver depths must lie within the SVP")

    z_b_closest = 0
    diff = abs(depth[0] - z_b)
    for i in range(1, depth.size):
        d = abs(depth[i] - z_b)
        if d < diff:
            diff = d
            z_b_closest = i

    z_a_closest = 0
    diff = abs(depth[0] - z_a)
    for i in range(1, depth.size):
        d = abs(depth[i] - z_a)
        if d < diff:
            diff = d
            z_a_closest = i

    if depth[z_a_closest] > z_a:
        z_a_closest -= 1
    if depth[z_b_closest] < z_b:
        z_b_closest += 1

    depth_slice = depth[z_a_closest : z_b_closest + 1].astype(np.float64)
    cz_slice = cz[z_a_closest : z_b_closest + 1].astype(np.float64)
    depth_slice[0], depth_slice[-1] = z_a, z_b
    cz_slice[0] = np.interp(z_a, depth, cz)
    cz_slice[-1] = np.interp(z_b, depth, cz)
    dz = depth_slice[1:] - depth_slice[:-1]

    # Initial conditions
    theta = (90 - iga) * np.pi / 180
    c = cz_slice[0]

    # Pre-calculate sine of theta
    sin_theta = np.sin(theta)

    # Check for total internal reflection
    if np.any(cz_slice * sin_theta / c >= 1.0):
        return np.nan, z_b - z_a, np.nan

    c_ratios = 0.5 * (cz_slice[1:] + cz_slice[:-1]) / c
    thetas = np.arcsin(c_ratios * sin_theta)

    # Vectorized calculations for distance and time
    dx = dz * np.tan(thetas)
    ds = np.sqrt(dx**2 + dz**2)
    dt = ds / (c_ratios * c)

    # Sum up the results
    x = np.sum(dx)
    time = np.sum(dt)

    return x, z_b - z_a, time


@njit
def ray_trace_locate(z_a, z_b, target_x, depth, cz):
    """Locate the launch angle (degrees) for a horizontal range in metres."""
    if not np.isfinite(target_x) or target_x < 0.0:
        raise ValueError("Horizontal range must be finite and nonnegative")
    if target_x == 0.0:
        ray_tracing(90.0, z_a, z_b, depth, cz)
        return 90.0

    left = 0.0
    right = 90.0
    tolerance = 0.0001
    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        alpha = (left + right) / 2
        x_curr, _, _ = ray_tracing(alpha, z_a, z_b, depth, cz)

        if np.isnan(x_curr):
            left = alpha
        elif abs(x_curr - target_x) < tolerance:
            return alpha
        elif x_curr < target_x:
            right = alpha
        else:
            left = alpha

        iteration += 1

    return np.nan


if __name__ == "__main__":
    depth = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("SVP_Data/depth_cast2_smoothed.txt"))
    )
    cz = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("SVP_Data/cz_cast2_smoothed.txt"))
    )

    start_time = time()
    for _ in range(10):
        alpha = ray_trace_locate(100, 5000, 20000, depth, cz)
    end_time = time()
    print(f"Execution time: {(end_time - start_time) / 1:.2f} seconds")

    if not np.isnan(alpha):
        x, z, time = ray_tracing(alpha, 100, 5000, depth, cz)
        print(f"Ray Tracing 1 for alpha = {alpha:.2f} degrees")
        print(f"Travel Time: {time:.3f} s")
        print(f"Source-Receiver Distance: {x:.2f} m")
        print(f"Receiver Depth: {z:.2f} m")
    else:
        print("No solution found")
