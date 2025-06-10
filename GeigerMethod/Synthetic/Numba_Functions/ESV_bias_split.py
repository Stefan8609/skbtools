import numpy as np

from Numba_time_bias import find_esv

"""
Split the ESV bias term into multiple parts (based on time, range, location, etc.) and see if it improves the results
    Built for the MCMC sampler. Will have to figure out how to split the ESV bias term for the Geiger method
"""


# @njit
def calculateTimesRayTracing_split(
    guess, transponder_coordinates, esv_biases, dz_array, angle_array, esv_matrix
):
    """
    Ray Tracing calculation of times using ESV
        Capable of handling an array of ESV biases split by time (approximated by indices).
    :param guess:
        The guess for the position of the source.
    :param transponder_coordinates:
        The transponder coordinates to calculate the times for.
    :param esv_biases:
        The array of ESV biases to be applied to the transponder coordinates.
    :param dz_array:
        The array of depth differences to be used for the ESV calculation.
    :param angle_array:
        The array of angles to be used for the ESV calculation.
    :param esv_matrix:
        The ESV matrix to be used for the ESV calculation.
    """
    times = np.zeros(len(transponder_coordinates))
    esv = np.zeros(len(transponder_coordinates))
    l, r = 0, 0
    for n in range(len(esv_biases)):
        # Split the transponder coordinates into chunks based on the number of ESV biases
        l = r
        r = l + len(transponder_coordinates) // len(esv_biases)
        if n == len(esv_biases) - 1:
            r = len(transponder_coordinates)
        curr_transponder = transponder_coordinates[l:r]
        hori_dist = np.sqrt(
            (curr_transponder[:, 0] - guess[0]) ** 2
            + (curr_transponder[:, 1] - guess[1]) ** 2
        )
        abs_dist = np.sqrt(np.sum((curr_transponder - guess) ** 2, axis=1))
        beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
        dz = np.abs(guess[2] - curr_transponder[:, 2])
        esv[l:r] = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_biases[n]
        print(
            len(find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_biases[n])
        )
        times[l:r] = abs_dist / esv
    print(len(times), len(esv))
    return times, esv
