import numpy as np
from numba import njit

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)

"""
Function that tests a Gauss-Newton inversion including ESV and
time bias terms. Toy box that requires knowledge of CDOG and
actual transducer locations"""


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
