import numpy as np
import numpy.typing as npt


def ray_tracing(
    iga: float, z_a: float, z_b: float, depth: npt.NDArray, cz: npt.NDArray
) -> tuple:
    """Ray trace a path through a stratified sound speed profile.

    Parameters
    ----------
    iga : float
        Initial grazing angle in degrees.
    z_a, z_b : float
        Source and receiver depths in metres.
    depth, cz : ndarray
        Discrete depth array and corresponding sound speed.

    Returns
    -------
    tuple
        ``(x, z, time)`` where ``x`` is the horizontal distance in metres,
        ``z`` is the final depth and ``time`` is the travel time in seconds.
    """
    # Find nearest indices without using NumPy helpers that are unsupported by numba
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

    # Pre-calculate all required arrays
    depth_slice = depth[z_a_closest:z_b_closest]
    cz_slice = cz[z_a_closest:z_b_closest]
    dz = depth_slice[1:] - depth_slice[:-1]

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
        return np.nan, np.nan, np.nan

    # Vectorized calculations for distance and time
    dx = dz * np.tan(thetas)
    ds = np.sqrt(dx**2 + dz**2)
    dt = ds / cz_slice[:-1]

    # Sum up the results
    x = np.sum(dx)
    time = np.sum(dt)

    return x, z_b - z_a, time


def ray_trace_locate(
    z_a: float, z_b: float, target_x: float, depth: npt.NDArray, cz: npt.NDArray
) -> float:
    """Locate the launch angle that yields the desired range using bisection.

    Parameters
    ----------
    z_a, z_b : float
        Source and receiver depths in metres.
    target_x : float
        Horizontal range between source and receiver in metres.
    depth, cz : ndarray
        Discrete depth array and corresponding sound speed profile.

    Returns
    -------
    float
        Angle in degrees that achieves ``target_x`` or ``NaN`` if none is found.
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

    return np.nan


if __name__ == "__main__":
    from time import time
    from data import gps_data_path

    depth = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("depth_cast2_smoothed.txt"))
    )
    cz = np.ascontiguousarray(np.genfromtxt(gps_data_path("cz_cast2_smoothed.txt")))

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
