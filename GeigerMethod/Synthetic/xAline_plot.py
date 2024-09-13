import numpy as np
import matplotlib.pyplot as plt

def two_pointer_index(offset, CDOG_data, GPS_data, GPS_travel_times):
    """Module to index closest data points against each other given correct offset"""
    #initialize information
    CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)
    CDOG_travel_times = CDOG_unwrap + GPS_travel_times[0] - CDOG_unwrap[0]
    CDOG_times = CDOG_data[:,0] + CDOG_data[:,1] - offset
    GPS_times = GPS_data + GPS_travel_times

    #Initialize loop conditions
    CDOG_pointer = 0
    GPS_pointer = 0
    curr_idx = 0

    #Need to use append or have these have blank space to overwrite (currently accessing non-indexed values)
    CDOG_full = np.array([])
    CDOG_travel_full = np.array([])
    GPS_full = np.array([])
    GPS_travel_full = np.array([])

    while CDOG_pointer < len(CDOG_data) and GPS_pointer < len(GPS_data):
        if np.abs(GPS_travel_times[GPS_pointer] - CDOG_travel_times[CDOG_pointer]) < 0.4:
            CDOG_full[curr_idx] = CDOG_times[CDOG_pointer]
            CDOG_travel_full[curr_idx] = CDOG_travel_times[CDOG_pointer]
            GPS_full[curr_idx] = GPS_times[GPS_pointer]
            GPS_travel_full[curr_idx] = GPS_travel_times[GPS_pointer]

            CDOG_pointer += 1
            GPS_pointer += 1
            curr_idx += 1
        #Now iterate up CDOG_pointer when GPS time is currently greater and vice versa



def xAline_plot(offset, CDOG_data, GPS_data, travel_times):

    CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)
    CDOG_travel_times = CDOG_unwrap + travel_times[0] - CDOG_unwrap[0]

    CDOG_times = CDOG_data[:,0] + CDOG_data[:,1] - offset
    GPS_times = GPS_data + travel_times

    plt.scatter(CDOG_times, CDOG_travel_times, s=10, marker='x', label="CDOG Derived Travel Times")
    plt.scatter(GPS_times, travel_times, s=1, label="Inversion Travel Times")
    plt.legend()
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.title(f"Comparison of time series with offset: {offset}")

    plt.show()
