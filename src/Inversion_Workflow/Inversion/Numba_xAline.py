import numpy as np
from numba import njit


OFFSET_SEARCH_THRESHOLD_SECONDS = 0.5


@njit(cache=True)
def two_pointer_index(
    offset,
    threshold,
    CDOG_data,
    GPS_data,
    GPS_travel_times,
    transponder_coordinates,
    esv,
):
    """Pair each GPS emission with the nearest DOG arrival within ``threshold`` (s).

    Eqs. A19-A20: arrival t' pairs with emission t when |(t' - offset) - (t + T_mod)|
    < gamma; of several arrivals in the gate the nearest is taken, and each arrival is
    used once. The paper's gamma is 0.4 s; the call sites use 0.05-0.6 s.
    ``CDOG_data`` is (N, 2) integer and fractional seconds. Returns ``(CDOG_clock,
    CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full)``: shifted
    arrival time, observed travel time (Eq. A21), predicted arrival time and modelled
    travel time of each pair, with the pair's transducer position and ESV.
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

    CDOG_pointer, count = 0, 0

    for GPS_pointer in range(len(GPS_data)):
        lower = GPS_times[GPS_pointer] - threshold
        upper = GPS_times[GPS_pointer] + threshold

        while CDOG_pointer < len(CDOG_data) and CDOG_times[CDOG_pointer] <= lower:
            CDOG_pointer += 1
        if CDOG_pointer == len(CDOG_data):
            break

        candidate = CDOG_pointer
        best_pointer = -1
        best_difference = np.inf
        while candidate < len(CDOG_data) and CDOG_times[candidate] < upper:
            difference = abs(GPS_times[GPS_pointer] - CDOG_times[candidate])
            if difference < best_difference:
                best_pointer = candidate
                best_difference = difference
            candidate += 1

        if best_pointer >= 0:
            GPS_clock[count] = GPS_times[GPS_pointer]
            GPS_full[count] = GPS_travel_times[GPS_pointer]
            transponder_coordinates_full[count] = transponder_coordinates[GPS_pointer]
            esv_full[count] = esv[GPS_pointer]

            CDOG_clock[count] = CDOG_times[best_pointer]
            CDOG_pointer = best_pointer + 1
            count += 1

    # Trim arrays to actual size
    CDOG_clock = CDOG_clock[:count]
    GPS_clock = GPS_clock[:count]
    GPS_full = GPS_full[:count]
    transponder_coordinates_full = transponder_coordinates_full[:count]
    esv_full = esv_full[:count]

    # Observed travel times (Eq. A21)
    CDOG_full = CDOG_clock - (GPS_clock - GPS_full)

    return (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    )


@njit(cache=True)
def filter_close_receptions(CDOG_data, minimum_interval=0.95):
    """Remove both receptions bordering intervals shorter than expected."""
    keep = np.ones(len(CDOG_data), dtype=np.bool_)
    times = CDOG_data[:, 0] + CDOG_data[:, 1]
    for index in range(len(CDOG_data) - 1):
        if times[index + 1] - times[index] < minimum_interval:
            keep[index] = False
            keep[index + 1] = False
    return CDOG_data[keep]


@njit(cache=True)
def _coherence_score_from_pairs(CDOG_full, GPS_full):
    """Circular-variance score (1 - R) / sqrt(N) of the fractional residuals.

    R = |mean(exp(2 pi i w))| for w = CDOG_full - GPS_full wrapped into [-0.5, 0.5)
    s; smaller is better. Eq. A22 uses 2 (1 - R); the 1 / sqrt(N) factor favours
    offsets that pair more arrivals.
    """
    N = len(CDOG_full)
    if N == 0:
        return np.inf

    # Wrapped residuals in [-0.5, 0.5)
    w = (CDOG_full - GPS_full + 0.5) % 1.0 - 0.5

    # Circular mean resultant length
    # mean(exp(i*theta)) = mean(cos)+ i mean(sin)
    c = 0.0
    s = 0.0
    for i in range(N):
        theta = 2.0 * np.pi * w[i]
        c += np.cos(theta)
        s += np.sin(theta)
    c /= N
    s /= N
    R = np.sqrt(c * c + s * s)

    return (1.0 - R) / np.sqrt(N)


@njit(cache=True)
def find_int_offset(
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    offset=0,
    halfwindow=10000,
):
    """Offset (s) minimizing the circular-variance score, on ever finer steps.

    Steps of 10 s down to 0.1 ms within ``offset +- halfwindow``. The default
    window, -10000 to 10000 s, covers negative DOG clock offsets (Puerto Rico: -19 s
    and -7459 s) as well as positive ones (Bermuda: 1866-3175 s).
    """
    search_lower = offset - halfwindow
    search_upper = offset + halfwindow
    lower = search_lower
    upper = search_upper
    intervals = np.array([10.0, 1.0, 0.1, 0.01, 0.001, 0.0001])
    best_offset = offset
    for interval in intervals:
        if interval > 2.0 * halfwindow:
            continue

        best_score = np.inf
        level_best_offset = best_offset

        # scan window
        lag = lower
        while lag <= upper + 1e-15:
            lag_r = np.round(lag, 8)

            CDOG_full, GPS_clock, GPS_full = two_pointer_index(
                lag_r,
                OFFSET_SEARCH_THRESHOLD_SECONDS,
                CDOG_data,
                GPS_data,
                travel_times,
                transponder_coordinates,
                esv,
            )[1:4]

            score = _coherence_score_from_pairs(CDOG_full, GPS_full)

            if score < best_score:
                best_score = score
                level_best_offset = lag_r

            lag += interval

        best_offset = level_best_offset
        lower = max(search_lower, best_offset - interval)
        upper = min(search_upper, best_offset + interval)

    return best_offset


@njit(cache=True)
def find_subint_offset(
    offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
):
    """Offset (s) within 0.5 s of ``offset`` minimizing the RMSE (Eq. A23)."""
    # Initialize values for loop
    search_lower, search_upper = offset - 0.5, offset + 0.5
    lower, upper = search_lower, search_upper
    intervals = np.array(
        [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    )
    best_offset = offset

    for interval in intervals:
        best_RMSE = np.inf
        level_best_offset = best_offset
        for lag in np.arange(lower, upper + interval, interval):
            # Round away the float error of np.arange
            lag = np.round(lag, 8)

            CDOG_full, GPS_clock, GPS_full = two_pointer_index(
                lag,
                OFFSET_SEARCH_THRESHOLD_SECONDS,
                CDOG_data,
                GPS_data,
                travel_times,
                transponder_coordinates,
                esv,
            )[1:4]

            # Compute RMSE
            diff_data = GPS_full - CDOG_full
            RMSE = np.sqrt(np.nanmean(diff_data**2))

            if RMSE < best_RMSE:
                level_best_offset = lag
                best_RMSE = RMSE
        best_offset = level_best_offset
        lower = max(search_lower, best_offset - interval)
        upper = min(search_upper, best_offset + interval)
    return best_offset


@njit(cache=True)
def align_offset(
    CDOG_data, GPS_data, travel_times, transponder_coordinates, esv, initial_offset=None
):
    """Clock offset (s): integer second by circular variance, then fraction by RMSE.

    With ``initial_offset`` the integer search stays within 0.5 s of it.
    """
    if initial_offset is None:
        offset = find_int_offset(
            CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
        )
    else:
        offset = find_int_offset(
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            offset=np.round(initial_offset),
            halfwindow=0.5,
        )
    return find_subint_offset(
        offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )
