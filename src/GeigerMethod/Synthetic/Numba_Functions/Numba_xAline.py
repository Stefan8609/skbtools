import numpy as np
from numba import njit
from scipy import signal
import matplotlib.pyplot as plt


@njit(cache=True)
def two_pointer_index(
    offset,
    threshhold,
    CDOG_data,
    GPS_data,
    GPS_travel_times,
    transponder_coordinates,
    esv,
    exact=False,
):
    """Synchronize DOG and GPS data streams given a time offset.

    Parameters
    ----------
    offset : float
        Candidate time offset between the DOG and GPS records.
    threshhold : float
        Maximum time difference allowed when pairing samples.
    CDOG_data : ndarray
        ``(N, 2)`` DOG integer and fractional times.
    GPS_data : ndarray
        ``(M,)`` GPS absolute times.
    GPS_travel_times : ndarray
        Modeled travel times from GPS to transponder.
    transponder_coordinates : ndarray
        Positions of the transponder for each GPS sample.
    esv : ndarray
        Effective sound speeds for each GPS sample.
    exact : bool, optional
        If ``True`` return the arrays without trimming.

    Returns
    -------
    tuple
        ``(CDOG_clock, GPS_clock, GPS_full, transponder_coords, esv_full)``.
    """
    CDOG_times = CDOG_data[:, 0] + CDOG_data[:, 1] - offset
    GPS_times = GPS_data + GPS_travel_times

    # Preallocate arrays
    max_len = len(CDOG_data) + len(GPS_data)
    CDOG_clock = np.zeros(max_len)
    GPS_clock = np.zeros(max_len)
    GPS_full = np.zeros(max_len)
    transponder_coordinates_full = np.zeros((max_len, 3))
    esv_full = np.zeros(max_len)

    CDOG_pointer, GPS_pointer, count = 0, 0, 0

    # Main loop
    while CDOG_pointer < len(CDOG_data) and GPS_pointer < len(GPS_data):
        if abs(GPS_times[GPS_pointer] - CDOG_times[CDOG_pointer]) < threshhold:
            CDOG_clock[count] = CDOG_times[CDOG_pointer]
            GPS_clock[count] = GPS_times[GPS_pointer]
            GPS_full[count] = GPS_travel_times[GPS_pointer]
            transponder_coordinates_full[count] = transponder_coordinates[GPS_pointer]
            esv_full[count] = esv[GPS_pointer]

            CDOG_pointer += 1
            GPS_pointer += 1
            count += 1
        elif GPS_times[GPS_pointer] < CDOG_times[CDOG_pointer]:
            GPS_pointer += 1
        else:
            CDOG_pointer += 1

    # Trim arrays to actual size
    CDOG_clock = CDOG_clock[:count]
    GPS_clock = GPS_clock[:count]
    GPS_full = GPS_full[:count]
    transponder_coordinates_full = transponder_coordinates_full[:count]
    esv_full = esv_full[:count]

    # Best travel times for known offset
    if exact:
        CDOG_full = CDOG_clock - (GPS_clock - GPS_full)
    else:
        CDOG_full = CDOG_clock + GPS_travel_times[0] - CDOG_clock[0]
        for i in range(len(CDOG_full)):
            diff = GPS_full[i] - CDOG_full[i]
            if abs(diff) >= 0.9:
                CDOG_full[i] += np.round(diff)

    return (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    )


@njit(cache=True)
def find_subint_offset(
    offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
):
    """Search for the fractional time offset with minimum RMSE."""
    # Initialize values for loop
    lower, upper = offset - 0.5, offset + 0.5
    intervals = np.array(
        [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    )
    best_offset = offset
    best_RMSE = np.inf

    for interval in intervals:
        for lag in np.arange(lower, upper + interval, interval):
            # Round to prevent numpy float errors
            lag = np.round(lag, 8)  # Adjust precision based on your interval

            # Index data using lag
            CDOG_full, GPS_clock, GPS_full = two_pointer_index(
                lag,
                0.5,
                CDOG_data,
                GPS_data,
                travel_times,
                transponder_coordinates,
                esv,
                True,
            )[1:4]

            # Adjust CDOG_full to match GPS_full
            for i in range(len(CDOG_full)):
                diff = GPS_full[i] - CDOG_full[i]
                if abs(diff) >= 0.9:
                    CDOG_full[i] += np.round(diff)

            # Compute RMSE
            diff_data = GPS_full - CDOG_full
            RMSE = np.sqrt(np.nanmean(diff_data**2))

            # Update best offset if RMSE is improved
            if RMSE < best_RMSE:
                best_offset = lag
                best_RMSE = RMSE
        # Narrow search bounds for the next iteration
        lower, upper = best_offset - interval, best_offset + interval
    return best_offset


def find_int_offset(
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    start=0,
    best=0,
    best_RMSE=np.inf,
):
    """Find the integer offset between two time series."""
    # Set initial parameters
    offset = start
    err_int = 1000
    k = 0
    lag = np.inf

    while lag != 0 and k < 10:
        # Get indexed data according to offset
        CDOG_full, GPS_clock, GPS_full = two_pointer_index(
            offset, 0.5, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
        )[1:4]
        # Get fractional parts of the data
        GPS_fp = np.modf(GPS_full)[0]
        CDOG_fp = np.modf(CDOG_full)[0]
        # Find the cross-correlation between the fractional parts of the time series
        correlation = signal.correlate(
            CDOG_fp - np.mean(CDOG_fp),
            GPS_fp - np.mean(GPS_fp),
            mode="full",
            method="fft",
        )
        lags = signal.correlation_lags(len(CDOG_fp), len(GPS_fp), mode="full")
        lag = lags[np.argmax(abs(correlation))]
        # Adjust the offset by the optimal lag
        offset += lag
        k += 1
        # Conditional check to prevent false positives
        if offset < 0:
            offset = err_int
            err_int += 500
            lag = np.inf

    # Conditional check to ensure the resulting value
    # is reasonable (and to prevent stack overflows)
    if start > 10000:
        print(f"Error - No true offset found: {offset}")
        return best

    # If RMSE is too high, rerun the algorithm to see if it can be improved
    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset, 0.5, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )[1:4]
    RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100
    if RMSE < best_RMSE:
        best = offset
        best_RMSE = RMSE
    if RMSE > 10000:
        start += 500
        return find_int_offset(
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            start,
            best,
            best_RMSE,
        )
    return best


if __name__ == "__main__":
    from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import (
        calculateTimesRayTracing,
        findTransponder,
    )
    from GeigerMethod.Synthetic.Generate_Unaligned_Realistic import (
        generateUnalignedRealistic,
    )

    n = 10000
    true_offset = np.random.rand() * 10000
    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    (
        CDOG_data,
        CDOG,
        GPS_Coordinates,
        GPS_data,
        true_transponder_coordinates,
    ) = generateUnalignedRealistic(n, time_noise, true_offset)
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )
    travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

    """BEGIN TESTING"""
    CDOG_full_test, GPS_clock_test, GPS_full_test = two_pointer_index(
        np.round(true_offset),
        0.9,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )[1:4]
    RMSE = np.sqrt(np.nanmean((CDOG_full_test - GPS_full_test) ** 2)) * 1515 * 100
    print("Closest INT RMSE:", RMSE, "cm", np.round(true_offset))
    """END TESTING"""
    # Find the derived offset
    offset = find_int_offset(
        CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )
    print("int offset:", offset)
    offset = find_subint_offset(
        offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )

    print("True offset:", true_offset, "\nDerived offset:", offset)

    [
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ] = two_pointer_index(
        offset,
        0.9,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
        True,
    )

    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100

    print("RMSE:", RMSE, "cm")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].scatter(
        CDOG_clock,
        CDOG_full,
        s=10,
        marker="x",
        label="Unwrapped/Adjusted Synthetic Dog Travel Time",
    )
    axes[0].scatter(
        GPS_clock, GPS_full, s=1, marker="o", label="Calculated GPS Travel Times"
    )
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("Arrivals in Absolute Time (s)")
    axes[0].set_ylabel("Travel Times (s)")
    axes[0].set_title(
        f"Synthetic travel times with offset: {offset} and RMSE: {np.round(RMSE, 3)}"
    )

    diff_data = CDOG_full - GPS_full
    axes[1].scatter(CDOG_clock, diff_data, s=1)
    axes[1].set_xlabel("Absolute Time (s)")
    axes[1].set_ylabel("Difference between calculated and unwrapped times (s)")
    axes[1].set_title("Residual Plot")

    print("Mean of residuals: ", np.mean(diff_data) * 1000, "ms")
    print("Diff between found and true offset: ", (offset - true_offset) * 1000, "ms")

    plt.show()
