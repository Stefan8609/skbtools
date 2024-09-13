"""
Module to test alignment with a bunch of randomly generated trajectories to see
    if there are any cases where the alignment function fails for integer offsets

This is really promising - Next steps
    1) Do gauss-newton inversion (with best integer offset for each run)
        See if integer offset is able to isolate towards inversion
        Kinda like a coarse test to get in the correct region
    2) Once guess is close, start implementing a sub-integer offset and re-run Gauss-Newton
        a finer test
    3) If this process seems to work - start sending in the simulated annealing algorithm
        (prob just start with best guess of lever for coarse testing as well and then isolate down
            as the testing continues)

Maybe if <10000 rms is not found - just return the best found...

Could have a combined RMSE from local point distance alongside local derivative distance

See if you can align time series of diff lengths (use longest continous section of CDOG series and test against GPS)
    Could avoid using fractional part (just align first point in continous to integer of GPS)

Hidden Markov Models are sometimes used in alignment - prob too complicated for my usage
"""

import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import calculateTimesRayTracing, findTransponder
from Generate_Unaligned_Realistic import generateUnalignedRealistic
from xAline import index_data, find_int_offset, find_subint_offset
from xAline_plot import xAline_plot

def alignment_testing(iter, n, position_noise):
    # Loop the number of desired iterations
    for i in range(iter):
        # Generate a random offset
        true_offset = np.random.rand() * 10000

        # Generate the arrival time series for the generated offset (aswell as GPS Coordinates)
        CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(n, true_offset)
        GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

        """ Need to make the relative GPS and transducer positions modular at some point """
        # Find the transponder given the noisy GPS coordinates and get the resulting travel times
        gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
        gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
        transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
        travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

        # Find the derived offset
        offset = find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
        offset = find_subint_offset(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)

        print("True offset:", true_offset, "\nDerived offset:", offset)

        xAline_plot(offset, CDOG_data, GPS_data, travel_times)

        # Get the RMSE of the travel time inversion for the true offset and compare with the derived offset
        full_times_true, CDOG_full_true, GPS_full_true= index_data(true_offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)[:3]
        abs_diff = np.abs(CDOG_full_true - GPS_full_true)
        indices = np.where(abs_diff >= 0.9)
        CDOG_full_true[indices] += np.round(GPS_full_true[indices] - CDOG_full_true[indices])

        full_times_derived, CDOG_full_derived, GPS_full_derived = index_data(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)[:3]
        abs_diff = np.abs(CDOG_full_derived - GPS_full_derived)
        indices = np.where(abs_diff >= 0.9)
        CDOG_full_derived[indices] += np.round(GPS_full_derived[indices] - CDOG_full_derived[indices])

        RMSE_true = np.sqrt(np.nanmean((CDOG_full_true - GPS_full_true) ** 2)) * 1515 * 100
        RMSE_derived = np.sqrt(np.nanmean((CDOG_full_derived - GPS_full_derived) ** 2)) * 1515 * 100

        print("True RMSE:", RMSE_true, "cm", "\nDerived RMSE:", RMSE_derived, "cm\n")

        # If derived offset is different from true offset plot both
        if abs(offset - true_offset) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 8))

            axes[0, 0].scatter(full_times_true, CDOG_full_true, s=10, marker="x", label="Unwrapped/Adjusted Synthetic Dog Travel Time")
            axes[0, 0].scatter(full_times_true, GPS_full_true, s=1, marker="o", label="Calculated GPS Travel Times")
            axes[0, 0].legend(loc="upper right")
            axes[0, 0].set_xlabel("Arrivals in Absolute Time (s)")
            axes[0, 0].set_ylabel("Travel Times (s)")
            axes[0, 0].set_title(f"Synthetic travel times with offset: {true_offset} and RMSE: {np.round(RMSE_derived,3)}")

            diff_data_true = CDOG_full_true - GPS_full_true
            axes[1, 0].scatter(full_times_true, diff_data_true, s=1)
            axes[1, 0].set_xlabel("Absolute Time (s)")
            axes[1, 0].set_ylabel("Difference between calculated and unwrapped times (s)")
            axes[1, 0].set_title("Residual Plot")

            axes[0, 1].scatter(full_times_derived, CDOG_full_derived, s=10, marker="x", label="Unwrapped/Adjusted Synthetic Dog Travel Time")
            axes[0, 1].scatter(full_times_derived, GPS_full_derived, s=1, marker="o", label="Calculated GPS Travel Times")
            axes[0, 1].legend(loc="upper right")
            axes[0, 1].set_xlabel("Arrivals in Absolute Time (s)")
            axes[0, 1].set_ylabel("Travel Times (s)")
            axes[0, 1].set_title(f"Synthetic travel times with offset: {offset} and RMSE: {np.round(RMSE_derived,)}")

            diff_data_derived = CDOG_full_derived - GPS_full_derived
            axes[1, 1].scatter(full_times_derived, diff_data_derived, s=1)
            axes[1, 1].set_xlabel("Absolute Time (s)")
            axes[1, 1].set_ylabel("Difference between calculated and unwrapped times (s)")
            axes[1, 1].set_title("Residual Plot")

            plt.show()
    return

if __name__ == "__main__":
    alignment_testing(10, 20000, 2*10**-2)

