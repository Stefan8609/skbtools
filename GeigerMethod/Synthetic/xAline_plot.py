import numpy as np
import matplotlib.pyplot as plt
from Generate_Unaligned_Realistic import generateUnalignedRealistic
from advancedGeigerMethod import findTransponder
from xAline import *

def xAline_plot(offset, CDOG_data, GPS_data, travel_times):

    CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)
    CDOG_travel_times = CDOG_unwrap + travel_times[0] - CDOG_unwrap[0]

    CDOG_times = CDOG_data[:,0] + CDOG_data[:,1] - offset
    GPS_times = GPS_data + travel_times

    two_pointer_index(offset, CDOG_data, GPS_data, travel_times)

    plt.scatter(CDOG_times, CDOG_travel_times, s=10, marker='x', label="CDOG Derived Travel Times")
    plt.scatter(GPS_times, travel_times, s=1, label="Inversion Travel Times")
    plt.legend()
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.title(f"Comparison of time series with offset: {offset}")

    plt.show()


if __name__ == "__main__":
    position_noise = 2*10**-2
    true_offset = 1325
    n = 10000
    offset = 0
    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(n, 2*10**-5, true_offset)

    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
    travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

    offset = find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
    offset = find_subint_offset(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)




