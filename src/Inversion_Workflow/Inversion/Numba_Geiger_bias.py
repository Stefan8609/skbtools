import numpy as np
from numba import njit


@njit(cache=True)
def compute_Jacobian_biased(guess, transponder_coordinates, times, esv):
    """Jacobian of the travel-time residual with respect to (x, y, z, dt, dc).

    ``esv`` must already include the ESV bias (c_eff + dc), as returned by
    ``calculateTimesRayTracing_Bias(_Real)``. The residual is
    (T_mod - time_bias) - T_obs with T_mod = range / esv.
    """
    diffs = transponder_coordinates - guess

    J = np.zeros((len(transponder_coordinates), 5))

    # Compute different partial derivatives
    J[:, 0] = -diffs[:, 0] / (times[:] * (esv[:] ** 2))
    J[:, 1] = -diffs[:, 1] / (times[:] * (esv[:] ** 2))
    J[:, 2] = -diffs[:, 2] / (times[:] * (esv[:] ** 2))
    J[:, 3] = -1.0
    J[:, 4] = -times[:] / esv[:]

    return J
