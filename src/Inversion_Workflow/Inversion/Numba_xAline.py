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
def _coherence_score_from_pairs(CDOG_full, GPS_full):
    """
    obj = (1 - R) / sqrt(N)   (smaller is better)
    where:
      w = wrap(CDOG_full - GPS_full) into [-0.5, 0.5)
      R = |mean(exp(i 2π w))|
    """
    N = len(CDOG_full)
    if N == 0:
        return -1.0  # worst possible

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
    offset=5000,
    halfwindow=5000,
):
    """
    Refine an offset estimate by minimizing (1 - R)/sqrt(N)
    over progressively smaller step sizes.
    """
    lower = offset - halfwindow
    upper = offset + halfwindow
    intervals = np.array([10.0, 1.0, 0.1, 0.01, 0.001])
    best_offset = offset
    best_score = np.inf
    for interval in intervals:
        # scan window
        lag = lower
        while lag <= upper + 1e-15:
            lag_r = np.round(lag, 8)

            CDOG_full, GPS_clock, GPS_full = two_pointer_index(
                lag_r,
                0.5,
                CDOG_data,
                GPS_data,
                travel_times,
                transponder_coordinates,
                esv,
            )[1:4]

            score = _coherence_score_from_pairs(CDOG_full, GPS_full)

            if score < best_score:
                best_score = score
                best_offset = lag_r

            lag += interval

        # tighten bounds for next level
        lower = best_offset - interval
        upper = best_offset + interval

    return best_offset


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


if __name__ == "__main__":
    import scipy.io as sio
    from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
        bermuda_trajectory,
    )
    from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
    from data import gps_data_path
    from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
    from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
        calculateTimesRayTracing_Bias_Real,
        calculateTimesRayTracing_Bias,
    )
    from plotting.Analysis_Plots.metric_plots import plot_integer_pick_metrics_dog
    import matplotlib.pyplot as plt

    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    position_noise = 2e-2
    time_noise = 2e-5

    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )
    gps1_to_transponder = np.array([-12.48862757, 0.22622633, -15.89601934])

    esv_bias = 3.0
    time_bias = 0
    true_offset = np.random.rand() * 10000

    # Options
    type = "bermuda"  # "bermuda" or "unaligned"
    metrics_plot = True  # Whether to plot integer pick metrics

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
            offset=true_offset,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=gps1_to_transponder,
        )
    if type == "unaligned":
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = generateUnaligned(
            20000,
            time_noise,
            position_noise,
            true_offset,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=gps1_to_transponder,
        )

    print("True Offset: ", true_offset)

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )

    guess = CDOG + [100, 100, 100]

    if type == "bermuda":
        travel_times, esv = calculateTimesRayTracing_Bias_Real(
            CDOG,
            transponder_coordinates,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
    if type == "unaligned":
        travel_times, esv = calculateTimesRayTracing_Bias(
            guess,
            transponder_coordinates,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
    travel_times = travel_times + time_bias

    int_offset = find_int_offset(
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )
    print("Integer Offset: ", int_offset)

    subint_offset = find_subint_offset(
        int_offset,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )
    print("Sub-integer Offset: ", subint_offset)
    print("Offset Error: ", abs(true_offset - subint_offset))

    # Validate final result
    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        subint_offset,
        0.5,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )
    for i in range(len(CDOG_full)):
        diff = GPS_full[i] - CDOG_full[i]
        if abs(diff) >= 0.9:
            CDOG_full[i] += np.round(diff)
    RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100

    plt.figure()
    plt.plot(CDOG_full - GPS_full)
    plt.title("Final Time Differences After Synchronization")
    plt.xlabel("Sample Index")
    plt.ylabel("Time Difference (s)")
    plt.grid()
    plt.show()

    # Plot metrics if desired
    if metrics_plot:
        plot_integer_pick_metrics_dog(
            true_offset,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            half_window=100,
            step=0.1,
        )
