import numpy as np
from numba import njit


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
def refine_offset(
    offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
):
    print("REFINING OFFSET")
    """Search for the fractional time offset with minimum RMSE."""
    # Initialize values for loop
    lower, upper = offset - 5, offset + 5
    intervals = np.array([0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001])
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


@njit(cache=True)
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
    """
    Robust integer-offset finder for real data with travel-time model error.

    Score(lag): minimize robust dispersion of (GPS_clock - CDOG_clock)
      - Use median (location) and MAD (scale) with 3*MAD trimming.
      - Tie-break by larger inlier count, then smaller trimmed RMSE.
    """
    coarse_step = 50
    coarse_span_hi = 20000
    coarse_span_lo = 5000
    coarse_thresh = 3.0

    fine_window = 180
    fine_step = 1
    fine_thresh = 1.0

    min_pairs_coarse = 5
    eps_mad = 1e-3
    trim_k = 3.0

    s_int = int(start)
    lower = s_int - coarse_span_lo
    if lower < 0:
        lower = 0
    upper = s_int + coarse_span_hi
    if upper < lower + coarse_step:
        upper = lower + coarse_step

    # Best trackers
    best_lag = -1
    best_mad = np.inf
    best_inl = -1
    best_trm = np.inf

    lag = lower
    while lag <= upper:
        tup = two_pointer_index(
            lag,
            coarse_thresh,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            False,
        )
        CDOG_clock = tup[0]
        GPS_clock = tup[2]
        n = CDOG_clock.shape[0]

        if n >= min_pairs_coarse:
            d = np.empty(n)
            for i in range(n):
                d[i] = GPS_clock[i] - CDOG_clock[i]

            ds = d.copy()
            ds.sort()
            if n % 2 == 1:
                med = ds[n // 2]
            else:
                med = 0.5 * (ds[n // 2 - 1] + ds[n // 2])

            absdev = np.empty(n)
            for i in range(n):
                v = d[i] - med
                if v < 0.0:
                    v = -v
                absdev[i] = v
            as_ = absdev.copy()
            as_.sort()
            if n % 2 == 1:
                mad = as_[n // 2]
            else:
                mad = 0.5 * (as_[n // 2 - 1] + as_[n // 2])
            if mad < eps_mad:
                mad = eps_mad

            thr = trim_k * mad
            inliers = 0
            sumsq = 0.0
            for i in range(n):
                if absdev[i] <= thr:
                    inliers += 1
                    r = d[i] - med
                    sumsq += r * r
            trm = np.inf
            if inliers > 0:
                trm = np.sqrt(sumsq / inliers)

            better = False
            if mad < best_mad:
                better = True
            elif mad == best_mad:
                if inliers > best_inl:
                    better = True
                elif inliers == best_inl and trm < best_trm:
                    better = True

            if better:
                best_mad = mad
                best_inl = inliers
                best_trm = trm
                best_lag = lag

        lag += coarse_step

    # If coarse found nothing, fall back
    if best_lag < 0:
        return s_int

    f_low = best_lag - fine_window
    if f_low < 0:
        f_low = 0
    f_high = best_lag + fine_window

    fine_best_lag = best_lag
    fine_best_mad = best_mad
    fine_best_inl = best_inl
    fine_best_trm = best_trm

    lag = f_low
    while lag <= f_high:
        tup = two_pointer_index(
            lag,
            fine_thresh,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            False,
        )
        CDOG_clock = tup[0]
        GPS_clock = tup[2]
        n = CDOG_clock.shape[0]

        if n > 0:
            d = np.empty(n)
            for i in range(n):
                d[i] = GPS_clock[i] - CDOG_clock[i]

            ds = d.copy()
            ds.sort()
            if n % 2 == 1:
                med = ds[n // 2]
            else:
                med = 0.5 * (ds[n // 2 - 1] + ds[n // 2])

            absdev = np.empty(n)
            for i in range(n):
                v = d[i] - med
                if v < 0.0:
                    v = -v
                absdev[i] = v
            as_ = absdev.copy()
            as_.sort()
            if n % 2 == 1:
                mad = as_[n // 2]
            else:
                mad = 0.5 * (as_[n // 2 - 1] + as_[n // 2])
            if mad < eps_mad:
                mad = eps_mad

            thr = trim_k * mad
            inliers = 0
            sumsq = 0.0
            for i in range(n):
                if absdev[i] <= thr:
                    inliers += 1
                    r = d[i] - med
                    sumsq += r * r
            trm = np.inf
            if inliers > 0:
                trm = np.sqrt(sumsq / inliers)

            better = False
            if mad < fine_best_mad:
                better = True
            elif mad == fine_best_mad:
                if inliers > fine_best_inl:
                    better = True
                elif inliers == fine_best_inl and trm < fine_best_trm:
                    better = True

            if better:
                fine_best_mad = mad
                fine_best_inl = inliers
                fine_best_trm = trm
                fine_best_lag = lag

        lag += fine_step

    return int(fine_best_lag)


if __name__ == "__main__":
    from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
    from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
        bermuda_trajectory,
    )
    from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
    from Inversion_Workflow.Forward_Model.Calculate_Times import (
        calculateTimesRayTracing,
    )
    import scipy.io as sio
    from data import gps_data_path

    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    # Parameters
    n = 10000
    true_offset = np.random.rand() * 10000
    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5
    esv_bias = 1.0
    time_bias = 0.0

    print("True Offset: ", true_offset)

    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.39341409, -4.22350344, 0.02941493],
            [-12.09568416, -0.94568462, 0.0043972],
            [-8.68674054, 5.16918806, 0.02499322],
        ]
    )
    gps1_to_transponder = np.array([-12.4659, 9.6021, -13.2993])

    # Generate synthetic data
    type = "bermuda"
    if type == "bermuda":
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = bermuda_trajectory(
            time_noise,
            position_noise,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            true_offset,
            gps1_to_others,
            gps1_to_transponder,
        )
    else:
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = generateUnaligned(
            n,
            time_noise,
            position_noise,
            true_offset,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            gps1_to_others,
            gps1_to_transponder,
        )

    # Find transponder coordinates from noisy GPS data
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )
    travel_times, esv = calculateTimesRayTracing(
        CDOG + np.array([1000.0, 1000.0, 50.0]),
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
    )

    # Find integer offset
    offset = find_int_offset(
        CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )

    print("Integer Offset: ", offset)

    # Refine to sub-integer offset
    offset = refine_offset(
        offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )
    print("Final Offset: ", offset)
    print("Offset Error: ", abs(true_offset - offset))
