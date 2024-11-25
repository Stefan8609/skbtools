import numpy as np
from numba import njit
from xAline import index_data

"""
Need to try to fix this function to work with the numba library
Need subint to work fast

Can get sub-int integration with 2-pointer approach with both being optimized by numba
"""

@njit
def two_pointer_index(offset, threshhold, CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv):
    """Optimized module to index closest data points against each other given correct offset."""
    # Precompute values
    CDOG_unwrap = (CDOG_data[:, 1] * 2 * np.pi) % (2 * np.pi)  # Manual unwrap equivalent
    CDOG_travel_times = CDOG_unwrap + GPS_travel_times[0] - CDOG_unwrap[0]

    CDOG_times = CDOG_data[:, 0] + CDOG_data[:, 1] - offset
    GPS_times = GPS_data + GPS_travel_times

    # Preallocate lists (dynamic) or arrays
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
    CDOG_full = CDOG_clock - (GPS_clock - GPS_full)

    return CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full


@njit
def find_subint_offset(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv):
    """Optimized function to find the best sub-integer offset."""
    # Initialize values for loop
    l, u = offset - 0.5, offset + 0.5
    intervals = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001])
    best_offset = offset
    best_RMSE = np.inf

    for interval in intervals:
        for lag in np.arange(l, u + interval, interval):
            # Round to prevent numpy float errors
            lag = np.round(lag, 8)  # Adjust precision based on your interval

            # Index data using lag
            CDOG_full, GPS_clock, GPS_full = two_pointer_index(lag, 0.6, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)[1:4]

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
        l, u = best_offset - interval, best_offset + interval

    return best_offset