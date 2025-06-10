import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import (
    calculateTimesRayTracing,
    computeJacobianRayTracing,
    findTransponder,
)
from Generate_Unaligned_Realistic import generateUnalignedRealistic
from xAline import index_data, find_int_offset, two_pointer_index

"""Have a far geiger where only integer offset matters - Approximating right offset

Might not be perfect getting to the right point because first point in time series is dictated by first travel time

Change up the geiger part -- When at right offset (presumably) run it with true-offset mode on (need to implement)
    Rewrite (or have some condition when close) -- still do this but with new alignment method!!

Then after running that have a close offset where sub-integer offset matters too

Write a Geiger to check alignment post verification (should be converging better and faster with everything correct?)
"""


def xAline_Geiger2(guess, CDOG_data, GPS_data, transponder_coordinates, offset):
    epsilon = 10**-5
    k = 0
    delta = 1
    inversion_guess = guess
    estimate_arr = np.array([])

    times_guess, esv = calculateTimesRayTracing(
        inversion_guess, transponder_coordinates
    )
    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
    )

    while np.linalg.norm(delta) > epsilon and k < 10:
        GPS_full, esv = calculateTimesRayTracing(
            inversion_guess, transponder_coordinates_full
        )

        jacobian = computeJacobianRayTracing(
            inversion_guess, transponder_coordinates_full, GPS_full, esv_full
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (GPS_full - CDOG_full)
        )
        inversion_guess += delta

        estimate_arr = np.append(estimate_arr, inversion_guess, axis=0)
        k += 1
        print(inversion_guess, offset, k)

    times_guess, esv = calculateTimesRayTracing(
        inversion_guess, transponder_coordinates
    )
    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
    )
    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    estimate_arr = np.reshape(estimate_arr, (-1, 3))
    return inversion_guess, estimate_arr, CDOG_full, GPS_full, CDOG_clock, GPS_clock


def xAline_Geiger(guess, CDOG_data, GPS_data, transponder_coordinates):
    epsilon = 10**-5
    k = 0
    delta = 1
    inversion_guess = guess
    estimate_arr = np.array([])

    while np.linalg.norm(delta) > epsilon and k < 10:
        times_guess, esv = calculateTimesRayTracing(
            inversion_guess, transponder_coordinates
        )
        offset = find_int_offset(
            CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )
        full_times, CDOG_full, GPS_full, transponder_full, esv_full = index_data(
            offset, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )

        abs_diff = np.abs(CDOG_full - GPS_full)
        indices = np.where(abs_diff >= 0.9)
        CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

        jacobian = computeJacobianRayTracing(
            inversion_guess, transponder_full, GPS_full, esv_full
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (GPS_full - CDOG_full)
        )
        inversion_guess += delta
        estimate_arr = np.append(estimate_arr, inversion_guess, axis=0)
        k += 1

        if np.linalg.norm(inversion_guess - guess) > 1000:
            print("ERROR: Inversion too far from starting value")
            estimate_arr = np.reshape(estimate_arr, (-1, 3))
            return inversion_guess, estimate_arr, offset

        print(inversion_guess, offset, k)

    estimate_arr = np.reshape(estimate_arr, (-1, 3))
    return inversion_guess, estimate_arr, offset


if __name__ == "__main__":
    true_offset = int(np.random.rand() * 9000) + 1000
    print(true_offset)

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5
    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = (
        generateUnalignedRealistic(20000, time_noise, true_offset)
    )
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )

    guess = CDOG + [100, 100, 0]

    result, estimate_arr, offset = xAline_Geiger(
        guess, CDOG_data, GPS_data, transponder_coordinates
    )
    print("CDOG:", CDOG)
    print("Inversion:", result)
    print("Distance:", np.linalg.norm(result - CDOG) * 100, "cm")

    plt.scatter(
        guess[0], guess[1], s=50, marker="o", color="g", zorder=1, label="Initial Guess"
    )
    plt.scatter(CDOG[0], CDOG[1], s=50, marker="x", color="k", zorder=3, label="C-DOG")
    plt.scatter(
        result[0], result[1], marker="o", color="r", zorder=4, label="Final Estimate"
    )
    plt.scatter(
        estimate_arr[:, 0],
        estimate_arr[:, 1],
        s=50,
        marker="o",
        color="b",
        zorder=2,
        label="Estimate Iterations",
    )
    plt.legend()
    plt.show()

    times_guess, esv = calculateTimesRayTracing(result, transponder_coordinates)
    (
        inversion_guess,
        estimate_arr2,
        CDOG_full_derived,
        GPS_full_derived,
        CDOG_clock_derived,
        GPS_clock_derived,
    ) = xAline_Geiger2(result, CDOG_data, GPS_data, transponder_coordinates, offset)
    abs_diff = np.abs(CDOG_full_derived - GPS_full_derived)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full_derived[indices] += np.round(
        GPS_full_derived[indices] - CDOG_full_derived[indices]
    )

    estimate_arr = np.append(estimate_arr, estimate_arr2, axis=0)
    dist = np.linalg.norm(guess - CDOG)
    # plt.scatter(guess[0], guess[1], s=50, marker="o", color="g", zorder=1, label="Initial Guess")
    plt.scatter(CDOG[0], CDOG[1], s=100, marker="x", color="k", zorder=3, label="C-DOG")
    plt.scatter(
        inversion_guess[0],
        inversion_guess[1],
        marker="o",
        color="r",
        zorder=4,
        label="Final Estimate",
    )
    plt.scatter(
        estimate_arr[:, 0],
        estimate_arr[:, 1],
        s=50,
        marker="o",
        color="b",
        zorder=2,
        label="Estimate Iterations",
    )
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    # plt.xlim(CDOG[0] - dist * 1.2, CDOG[0] + dist * 1.2)
    # plt.ylim(CDOG[1] - dist * 1.2, CDOG[1] + dist * 1.2)

    plt.xlim(CDOG[0] - 50, CDOG[0] + 50)
    plt.ylim(CDOG[1] - 50, CDOG[1] + 50)
    plt.legend()
    plt.show()

    travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)
    full_times_true, CDOG_full_true, GPS_full_true = index_data(
        true_offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
    )[:3]
    abs_diff = np.abs(CDOG_full_true - GPS_full_true)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full_true[indices] += np.round(
        GPS_full_true[indices] - CDOG_full_true[indices]
    )

    RMSE_derived = (
        np.sqrt(np.nanmean((CDOG_full_derived - GPS_full_derived) ** 2)) * 1515 * 100
    )
    RMSE_true = np.sqrt(np.nanmean((CDOG_full_true - GPS_full_true) ** 2)) * 1515 * 100

    print("True RMSE:", RMSE_true, "cm", "\nDerived RMSE:", RMSE_derived, "cm\n")
    print("CDOG:", CDOG)
    print("Inversion:", inversion_guess)
    print("Distance:", np.linalg.norm(inversion_guess - CDOG) * 100, "cm")

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    axes[0, 0].scatter(
        full_times_true,
        CDOG_full_true,
        s=10,
        marker="x",
        label="Unwrapped/Adjusted Synthetic Dog Travel Time",
    )
    axes[0, 0].scatter(
        full_times_true,
        GPS_full_true,
        s=1,
        marker="o",
        label="Calculated GPS Travel Times",
    )
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].set_xlabel("Arrivals in Absolute Time (s)")
    axes[0, 0].set_ylabel("Travel Times (s)")
    axes[0, 0].set_title(
        f"Synthetic travel times with offset: {true_offset} and RMSE: {np.round(RMSE_true, 3)}"
    )

    diff_data_true = (CDOG_full_true - GPS_full_true) * 1000
    std_true = np.std(diff_data_true)
    mean_true = np.mean(diff_data_true)
    axes[1, 0].scatter(full_times_true, diff_data_true, s=1)
    axes[1, 0].set_ylim(-3 * std_true, 3 * std_true)
    axes[1, 0].set_xlabel("Absolute Time (s)")
    axes[1, 0].set_ylabel("Difference between calculated and unwrapped times (ms)")
    axes[1, 0].set_title("Residual Plot")

    axes[0, 1].scatter(
        CDOG_clock_derived / 3600,
        CDOG_full_derived,
        s=10,
        marker="x",
        label="Unwrapped/Adjusted Synthetic Dog Travel Time",
    )
    axes[0, 1].scatter(
        GPS_clock_derived / 3600,
        GPS_full_derived,
        s=1,
        marker="o",
        label="Calculated GPS Travel Times",
    )
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_xlabel("Absolute Time (hours)")
    axes[0, 1].set_ylabel("Travel Times (s)")
    # axes[0, 1].set_title(f"Synthetic travel times with offset: {offset} and RMSE: {np.round(RMSE_derived, 3)}")

    diff_data_derived = (CDOG_full_derived - GPS_full_derived) * 1000
    std_derived = np.std(diff_data_derived)
    mean_derived = np.mean(diff_data_derived)
    axes[1, 1].scatter(CDOG_clock_derived / 3600, diff_data_derived, s=1)
    axes[1, 1].axhline(std_derived, color="k", linestyle="--")
    axes[1, 1].axhline(-std_derived, color="k", linestyle="--")
    axes[1, 1].set_ylim(-3 * std_derived, 3 * std_derived)
    axes[1, 1].set_xlabel("Absolute Time (hours)")
    axes[1, 1].set_ylabel("Model Misfit(ms)")
    # axes[1, 1].set_title("Residual Plot")

    plt.show()


"""
State the problem: Given a time, (another time) ... (times corresponding to the same event measured on different clocks)
(Why the alignment fails --> doing this in a geiger update)
(Testing when the location is exactly right)
Show where it happens
show what happens when it doesnt work
Show the solution
Show with testing data
"""

"""
Showing that corrupted and offset data can't be aligned with two-pointer (or can?)
    Finds the most likely gaps

    How does it behave when values are noisy and not near

    If its noiseless - the time series are identical - can it detect the offset? or corrupted portions?

    Solving the recorder did not record problem

Step 2:
    Compare something that is perfect to begin with, with something that has a slightly wrong location,
    and then is corrupted

Make a bunch of illustrations of off cases (dropped data only, perfect data, both mispositionign and corrupted data)
    - Only mispositioned data..

--> Show why rounding causes slight misputs in the implementation

Show the problem what the solves - show its occurrence - show solution - and show results
"""
