import numpy as np
import matplotlib.pyplot as plt

from GeigerMethod.Synthetic.xAline import two_pointer_index
from advancedGeigerMethod import calculateTimesRayTracing, findTransponder
from Generate_Unaligned_Realistic import *
from xAline import *

def no_offset_test(n, time_noise, position_noise):
    offset = 0
    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(n, time_noise, 0)

    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

    travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

    CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
        two_pointer_index(offset, 0.6, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
    )

    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    full_times, CDOG_full2, GPS_full2, transponder_full2, esv_full2 = index_data(0, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)

    abs_diff2 = np.abs(CDOG_full2 - GPS_full2)
    indices = np.where(abs_diff2 >= 0.9)
    CDOG_full2[indices] += np.round(GPS_full2[indices] - CDOG_full2[indices])

    # Plotting 1x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("No Offset Test")
    # axs[0, 0].scatter(CDOG_clock, CDOG_full, label="CDOG", marker='o', s=5)
    axs[0, 0].scatter(CDOG_clock, CDOG_full, label="CDOG", marker='o', s=5)
    axs[0, 0].scatter(GPS_clock, GPS_full, label="GPS", marker='x', s=1)
    axs[0, 0].set_title("CDOG and GPS match: Method 1")
    axs[0, 0].legend()

    diff_data = CDOG_full - GPS_full
    std = np.std(diff_data)
    axs[1, 0].scatter(CDOG_clock, diff_data, s=1)
    axs[1, 0].set_title("Difference between CDOG and GPS")
    axs[1, 0].set_ylim(-3 * std, 3*std)

    axs[0, 1].scatter(full_times, CDOG_full2, label="CDOG", marker='o', s=5)
    axs[0, 1].scatter(full_times, GPS_full2, label="GPS", marker='x', s=1)
    axs[0, 1].set_title("CDOG and GPS match: Method 2")
    axs[0, 1].legend()

    diff_data2 = CDOG_full2 - GPS_full2
    axs[1, 1].scatter(full_times, diff_data2, s=1)
    axs[1, 1].set_title("Difference between CDOG and GPS")
    axs[1, 1].set_ylim(-3 * std, 3*std)
    plt.show()

    print(np.mean(diff_data))
    print(np.mean(diff_data2))


no_offset_test(10000, 2*10**-5, 2*10**-2)

