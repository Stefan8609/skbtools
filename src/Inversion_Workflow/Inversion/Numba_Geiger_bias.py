import numpy as np
from numba import njit
from geometry.ECEF_Geodetic import ECEF_Geodetic


@njit
def find_esv(beta, dz, dz_array, angle_array, esv_matrix):
    """Interpolate effective sound velocities from a lookup table.

    Parameters
    ----------
    beta : ndarray
        Takeoff angles in degrees.
    dz : ndarray
        Vertical separation of receiver and source.
    dz_array, angle_array, esv_matrix : ndarray
        Discrete lookup grids defining the ESV table.

    Returns
    -------
    ndarray
        Interpolated effective sound velocities.
    """
    idx_closest_dz = np.empty_like(dz, dtype=np.int64)
    idx_closest_beta = np.empty_like(beta, dtype=np.int64)

    for i in range(len(dz)):
        idx_closest_dz[i] = np.searchsorted(dz_array, dz[i], side="left")
        if idx_closest_dz[i] < 0:
            idx_closest_dz[i] = 0
        elif idx_closest_dz[i] >= len(dz_array):
            idx_closest_dz[i] = len(dz_array) - 1

        idx_closest_beta[i] = np.searchsorted(angle_array, beta[i], side="left")
        if idx_closest_beta[i] < 0:
            idx_closest_beta[i] = 0
        elif idx_closest_beta[i] >= len(angle_array):
            idx_closest_beta[i] = len(angle_array) - 1

    closest_esv = np.empty_like(dz, dtype=np.float64)
    for i in range(len(dz)):
        closest_esv[i] = esv_matrix[idx_closest_dz[i], idx_closest_beta[i]]

    return closest_esv


@njit
def calculateTimesRayTracing_Bias(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Compute travel times with an optional ESV bias term.

    Parameters
    ----------
    guess : array-like, shape (3,)
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of transponder positions.
    esv_bias : float or ndarray
        Bias added to the effective sound velocity.
    dz_array, angle_array, esv_matrix : ndarray
        Lookup table for computing the ESV.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and sound velocities.
    """
    hori_dist = np.sqrt(
        (transponder_coordinates[:, 0] - guess[0]) ** 2
        + (transponder_coordinates[:, 1] - guess[1]) ** 2
    )
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv


@njit
def calculateTimesRayTracing_Bias_Real(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Calculate times for real data using geodetic depths."""
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    depth_arr = ECEF_Geodetic(transponder_coordinates)[2]

    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)
    dz = depth_arr - depth
    beta = np.arcsin(dz / abs_dist) * 180 / np.pi
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv


@njit(cache=True)
def compute_Jacobian_biased(guess, transponder_coordinates, times, esv, esv_bias):
    """Jacobian of travel times with respect to state and bias terms."""
    diffs = transponder_coordinates - guess

    J = np.zeros((len(transponder_coordinates), 5))

    # Compute different partial derivatives
    J[:, 0] = -diffs[:, 0] / (times[:] * (esv[:] + esv_bias) ** 2)
    J[:, 1] = -diffs[:, 1] / (times[:] * (esv[:] + esv_bias) ** 2)
    J[:, 2] = -diffs[:, 2] / (times[:] * (esv[:] + esv_bias) ** 2)
    J[:, 3] = -1.0
    J[:, 4] = -times[:] / (esv[:] + esv_bias)

    return J


@njit
def numba_bias_geiger(
    guess,
    CDog,
    transponder_coordinates_Actual,
    transponder_coordinates_Found,
    esv_bias_input,
    time_bias_input,
    dz_array,
    angle_array,
    esv_matrix,
    dz_array_gen=None,
    angle_array_gen=None,
    esv_matrix_gen=None,
    time_noise=0,
):
    """Perform Gauss–Newton inversion including ESV and time bias terms."""
    if dz_array_gen is None:
        dz_array_gen = np.empty(0)
    if angle_array_gen is None:
        angle_array_gen = np.empty(0)
    if esv_matrix_gen is None:
        esv_matrix_gen = np.empty(0)

    epsilon = 10**-5

    # Calculate and apply noise for known times. Also apply time bias term
    if dz_array_gen.size > 0:
        times_known, esv = calculateTimesRayTracing_Bias(
            CDog,
            transponder_coordinates_Actual,
            esv_bias_input,
            dz_array_gen,
            angle_array_gen,
            esv_matrix_gen,
        )
    else:
        times_known, esv = calculateTimesRayTracing_Bias(
            CDog,
            transponder_coordinates_Actual,
            esv_bias_input,
            dz_array,
            angle_array,
            esv_matrix,
        )
    times_known += np.random.normal(0, time_noise, len(transponder_coordinates_Actual))
    times_known += time_bias_input

    time_bias = 0.0
    esv_bias = 0.0

    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    k = 0
    delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    while np.linalg.norm(delta) > epsilon and k < 100:
        times_guess, esv = calculateTimesRayTracing_Bias(
            guess,
            transponder_coordinates_Found,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
        J = compute_Jacobian_biased(
            guess, transponder_coordinates_Found, times_guess, esv, esv_bias
        )
        delta = (
            -1
            * np.linalg.inv(J.T @ J)
            @ J.T
            @ ((times_guess - time_bias) - times_known)
        )
        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        k += 1
    return estimate, times_known


if __name__ == "__main__":
    print("Deprecated")
    """Function was deprecated so it was removed"""
