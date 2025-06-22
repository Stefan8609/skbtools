import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Numba_Geiger import generateRealistic


def find_esv_generate(beta, dz, dz_array, angle_array, esv_matrix):
    """Look up effective sound velocity for generation.

    Parameters
    ----------
    beta : ndarray
        Takeoff angles in degrees.
    dz : ndarray
        Vertical separation between source and receiver.
    dz_array, angle_array, esv_matrix : ndarray
        Discrete ESV table lookup grids.

    Returns
    -------
    ndarray
        Effective sound velocities matching ``beta`` and ``dz``.
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


def calculateTimesRayTracingGenerate(
    guess,
    transponder_coordinates,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Ray trace travel times for the generation phase.

    Parameters
    ----------
    guess : array-like, shape (3,)
        Source location estimate.
    transponder_coordinates : ndarray
        ``(N, 3)`` coordinates of the transponder.
    esv_bias : float
        Bias applied to the effective sound velocity.
    time_bias : float
        Bias applied directly to the travel times.
    dz_array, angle_array, esv_matrix : ndarray
        ESV lookup tables used for interpolation.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and effective sound speeds.
    """
    hori_dist = np.sqrt(
        (transponder_coordinates[:, 0] - guess[0]) ** 2
        + (transponder_coordinates[:, 1] - guess[1]) ** 2
    )
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv_generate(beta, dz, dz_array, angle_array, esv_matrix)
    esv += esv_bias
    times = abs_dist / esv
    times += time_bias
    return times, esv


# Function to generate the unaligned time series for a realistic trajectory
# Function to generate the unaligned time series for a realistic trajectory
def generateUnalignedRealistic(
    n,
    time_noise,
    offset,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    main=False,
):
    """Generate noisy travel times for a realistic trajectory.

    Parameters
    ----------
    n : int
        Number of points along the synthetic path.
    time_noise : float
        Standard deviation of arrival time noise.
    offset : float
        Base offset applied to DOG times.
    esv_bias, time_bias : float
        Bias values applied to ESV and times.
    dz_array, angle_array, esv_matrix : ndarray
        Lookup table for the ESV.
    main : bool, optional
        If ``True`` return arrays useful for plotting.

    Returns
    -------
    tuple
        Either ``(CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates)``
        or a longer tuple if ``main`` is ``True``.
    """
    (
        CDOG,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    ) = generateRealistic(n)

    GPS_time = np.arange(len(GPS_Coordinates))

    """Can change ray option to have a incorrect soundspeed to investigate outcome"""
    true_travel_times, true_esv = calculateTimesRayTracingGenerate(
        CDOG,
        transponder_coordinates,
        esv_bias,
        time_bias,
        dz_array,
        angle_array,
        esv_matrix,
    )

    CDOG_time = (
        GPS_time
        + true_travel_times
        + np.random.normal(0, time_noise, len(GPS_time))
        + offset
    )
    CDOG_remain, CDOG_int = np.modf(CDOG_time)

    CDOG_unwrap = np.unwrap(CDOG_remain * 2 * np.pi) / (
        2 * np.pi
    )  # Numpy page describes how unwrap works

    CDOG_mat = np.stack((CDOG_int, CDOG_remain), axis=0)
    CDOG_mat = CDOG_mat.T

    removed_CDOG = np.array([])
    removed_travel_times = np.array([])
    temp_travel_times = np.copy(true_travel_times)

    # Remove random indices from CDOG data
    for _ in range(5):
        length_to_remove = np.random.randint(200, 500)
        start_index = np.random.randint(
            0, len(CDOG_mat) - length_to_remove + 1
        )  # Start index cannot exceed len(array) - max_length
        indices_to_remove = np.arange(start_index, start_index + length_to_remove)
        removed_CDOG = np.append(
            removed_CDOG,
            CDOG_mat[indices_to_remove, 0] + CDOG_mat[indices_to_remove, 1],
        )
        removed_travel_times = np.append(
            removed_travel_times, temp_travel_times[indices_to_remove]
        )
        CDOG_mat = np.delete(CDOG_mat, indices_to_remove, axis=0)
        temp_travel_times = np.delete(temp_travel_times, indices_to_remove, axis=0)

    if main:
        return (
            CDOG_mat,
            CDOG,
            CDOG_time,
            CDOG_unwrap,
            CDOG_remain,
            true_travel_times,
            temp_travel_times,
            GPS_Coordinates,
            GPS_time,
            transponder_coordinates,
            removed_CDOG,
            removed_travel_times,
        )

    return CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates


if __name__ == "__main__":
    (
        CDOG_mat,
        CDOG,
        CDOG_time,
        CDOG_unwrap,
        CDOG_remain,
        true_travel_times,
        temp_travel_times,
        GPS_Coordinates,
        GPS_time,
        transponder_coordinates,
        removed_CDOG,
        removed_travel_times,
    ) = generateUnalignedRealistic(20000, 1200, True)

    mat_unwrap = np.unwrap(CDOG_mat[:, 1] * 2 * np.pi) / (
        2 * np.pi
    )  # Numpy page describes how unwrap works

    # Save the CDOG to a matlabfile
    from data import gps_data_path

    sio.savemat(
        gps_data_path("Realistic_CDOG_noise_subint_new.mat"),
        {"tags": CDOG_mat},
    )
    sio.savemat(
        gps_data_path("Realistic_CDOG_loc_noise_subint_new.mat"),
        {"xyz": CDOG},
    )

    # Save transponder + GPS data
    sio.savemat(
        gps_data_path("Realistic_transponder_noise_subint_new.mat"),
        {"time": GPS_time, "xyz": transponder_coordinates},
    )
    sio.savemat(
        gps_data_path("Realistic_GPS_noise_subint_new.mat"),
        {"time": GPS_time, "xyz": GPS_Coordinates},
    )

    # Plots below

    plt.scatter(
        CDOG_time, true_travel_times, s=1, marker="o", label="True Travel Times"
    )
    plt.scatter(CDOG_time, CDOG_unwrap, s=1, marker="x", label="True Unwrapped Times")
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.show()

    plt.scatter(list(range(len(CDOG_time))), CDOG_time, s=1)
    plt.xlabel("Time Index")
    plt.ylabel("True CDOG Pulse Arrival Time")
    plt.show()

    plt.scatter(
        list(range(len(CDOG_remain) - 1)),
        CDOG_remain[1:] - CDOG_remain[: len(CDOG_remain) - 1],
        s=1,
    )
    plt.xlabel("Time Index")
    plt.ylabel("Difference between i and i-1 true CDOG nanosecond clock times")
    plt.show()

    plt.scatter(
        list(range(len(true_travel_times) - 1)),
        true_travel_times[1:] - true_travel_times[: len(true_travel_times) - 1],
        s=1,
    )
    plt.xlabel("Time Index")
    plt.ylabel("Difference between i and i-1 true travel times")
    plt.show()

    plt.scatter(
        CDOG_mat[:, 0] + CDOG_mat[:, 1],
        temp_travel_times,
        s=1,
        label="Corrupted Travel Times",
    )
    plt.scatter(removed_CDOG, removed_travel_times, s=1, label="Removed Travel Times")
    plt.scatter(
        CDOG_mat[:, 0] + CDOG_mat[:, 1], mat_unwrap, s=1, label="Corrupted Unwrapping"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.show()

    plt.scatter(
        CDOG_mat[:, 0] + CDOG_mat[:, 1],
        mat_unwrap,
        marker="x",
        s=4,
        label="Corrupted Unwrapping",
    )
    plt.scatter(CDOG_time, CDOG_unwrap, s=1, label="True Unwrapping")
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.show()

    plt.scatter(
        list(range(len(CDOG_mat) - 1)),
        CDOG_mat[1:, 1] - CDOG_mat[: len(CDOG_mat) - 1, 1],
        s=1,
    )
    plt.xlabel("Index")
    plt.ylabel("Difference between i and i-1 corrupted nanosecond clock times")
    plt.show()
