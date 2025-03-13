import numpy as np
from numba import njit
import numpy.typing as npt

@njit
def ray_tracing(iga: float, z_a: float, z_b: float,
                depth: npt.NDArray, cz: npt.NDArray) -> tuple:
    """Optimized Ray Tracing Algorithm using vectorized operations.
    Inputs:
    iga: Initial grazing angle in degrees
    z_a: Source depth in meters
    z_b: Receiver depth in meters
    depth: Depth array in meters
    cz: Sound speed array in m/s
    Outputs:
    x: Source-Receiver distance in meters
    z: Receiver depth in meters
    time: Travel time in seconds
    """
    z_b_closest = np.abs(depth - z_b).argmin()
    z_a_closest = np.abs(depth - z_a).argmin()

    # Pre-calculate all required arrays
    depth_slice = depth[z_a_closest:z_b_closest]
    cz_slice = cz[z_a_closest:z_b_closest]
    dz = np.diff(depth_slice)

    # Initial conditions
    theta = (90 - iga) * np.pi / 180
    c = cz_slice[0]

    # Pre-calculate sine of theta
    sin_theta = np.sin(theta)

    # Vectorized calculation of next angles
    c_ratios = cz_slice[1:] / c
    thetas = np.arcsin(c_ratios * sin_theta)

    # Check for total internal reflection
    if np.any(thetas > np.pi / 2):
        return np.NaN, np.NaN, np.NaN

    # Vectorized calculations for distance and time
    dx = dz * np.tan(thetas)
    ds = np.sqrt(dx ** 2 + dz ** 2)
    dt = ds / cz_slice[:-1]

    # Sum up the results
    x = np.sum(dx)
    time = np.sum(dt)

    return x, z_b - z_a, time


@njit
def ray_trace_locate(z_a: float, z_b: float, target_x: float,
                     depth: npt.NDArray, cz: npt.NDArray) -> float:
    """Binary search implementation for finding the correct angle.
    Inputs:
    z_a: Source depth in meters
    z_b: Receiver depth in meters
    target_x: Target source-receiver distance in meters
    depth: Depth array in meters
    cz: Sound speed array in m/s
    Outputs:
    alpha: Angle in degrees
    """
    left = 1.0
    right = 90.0
    tolerance = 0.0001
    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        alpha = (left + right) / 2
        x_curr, _, _ = ray_tracing(alpha, z_a, z_b, depth, cz)

        if np.isnan(x_curr):
            right = alpha
        elif abs(x_curr - target_x) < tolerance:
            return alpha
        elif x_curr < target_x:
            right = alpha
        else:
            left = alpha

        if right - left < tolerance:
            return alpha

        iteration += 1

    return np.NaN


if __name__ == "__main__":
    from time import time
    depth = np.ascontiguousarray(np.genfromtxt('GPSData/depth_cast2_smoothed.txt'))
    cz = np.ascontiguousarray(np.genfromtxt('GPSData/cz_cast2_smoothed.txt'))

    start_time = time()
    for i in range(1):
        alpha = ray_trace_locate(100, 5000, 20000, depth, cz)
    end_time = time()
    print(f"Execution time: {(end_time - start_time)/1:.2f} seconds")

    if not np.isnan(alpha):
        x, z, time = ray_tracing(alpha, 100, 5000, depth, cz)
        print(f"Ray Tracing 1 for alpha = {alpha:.2f} degrees")
        print(f"Travel Time: {time:.3f} s")
        print(f"Source-Receiver Distance: {x:.2f} m")
        print(f"Receiver Depth: {z:.2f} m")
    else:
        print("No solution found")