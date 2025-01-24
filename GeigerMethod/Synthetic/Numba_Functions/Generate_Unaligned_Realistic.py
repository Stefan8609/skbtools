"""
This is a version of the time alignment script that generates a trajectory that is completely synthetic
    contrary to the other version which uses the trajectory from the Bermuda experiment
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Numba_Geiger import find_esv, findTransponder, calculateTimesRayTracing, generateRealistic

#Function to generate the unaligned time series for a realistic trajectory
def generateUnalignedRealistic(n, time_noise, offset, ray=True, main=False):
    CDOG, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder = generateRealistic(n)

    GPS_time = np.arange(len(GPS_Coordinates))

    """Can change ray option to have a incorrect soundspeed to investigate outcome"""
    true_travel_times, true_esv = calculateTimesRayTracing(CDOG, transponder_coordinates, ray)

    CDOG_time = GPS_time + true_travel_times + np.random.normal(0, time_noise, len(GPS_time)) + offset
    CDOG_remain, CDOG_int = np.modf(CDOG_time)

    CDOG_unwrap = np.unwrap(CDOG_remain * 2 * np.pi) / (2*np.pi)  #Numpy page describes how unwrap works

    CDOG_mat = np.stack((CDOG_int, CDOG_remain), axis=0)
    CDOG_mat = CDOG_mat.T

    removed_CDOG = np.array([])
    removed_travel_times = np.array([])
    temp_travel_times = np.copy(true_travel_times)

    #Remove random indices from CDOG data
    for i in range(5):
        length_to_remove = np.random.randint(200, 500)
        start_index = np.random.randint(0, len(CDOG_mat) - length_to_remove + 1)  # Start index cannot exceed len(array) - max_length
        indices_to_remove = np.arange(start_index, start_index + length_to_remove)
        removed_CDOG = np.append(removed_CDOG, CDOG_mat[indices_to_remove, 0]+CDOG_mat[indices_to_remove, 1])
        removed_travel_times = np.append(removed_travel_times, temp_travel_times[indices_to_remove])
        CDOG_mat = np.delete(CDOG_mat, indices_to_remove, axis=0)
        temp_travel_times = np.delete(temp_travel_times, indices_to_remove, axis=0)

    if main==True:
        return (CDOG_mat, CDOG, CDOG_time, CDOG_unwrap, CDOG_remain, true_travel_times, temp_travel_times,
            GPS_Coordinates, GPS_time, transponder_coordinates, removed_CDOG, removed_travel_times)

    return CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder_coordinates


if __name__=="__main__":
    (CDOG_mat, CDOG, CDOG_time, CDOG_unwrap, CDOG_remain, true_travel_times, temp_travel_times, GPS_Coordinates,
     GPS_time, transponder_coordinates, removed_CDOG, removed_travel_times) = generateUnalignedRealistic(20000, 1200, True)

    mat_unwrap = np.unwrap(CDOG_mat[:,1] * 2 * np.pi) / (2*np.pi)  #Numpy page describes how unwrap works

    #Save the CDOG to a matlabfile
    sio.savemat("../../GPSData/Realistic_CDOG_noise_subint_new.mat", {"tags":CDOG_mat})
    sio.savemat("../../GPSData/Realistic_CDOG_loc_noise_subint_new.mat", {'xyz':CDOG})

    #Save transponder + GPS data
    sio.savemat("../../GPSData/Realistic_transponder_noise_subint_new.mat", {"time":GPS_time, "xyz": transponder_coordinates})
    sio.savemat("../../GPSData/Realistic_GPS_noise_subint_new.mat", {"time":GPS_time, "xyz": GPS_Coordinates})

    #Plots below

    plt.scatter(CDOG_time, true_travel_times, s=1, marker="o", label="True Travel Times")
    plt.scatter(CDOG_time, CDOG_unwrap, s=1, marker="x", label="True Unwrapped Times")
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.show()

    plt.scatter(list(range(len(CDOG_time))), CDOG_time, s=1)
    plt.xlabel("Time Index")
    plt.ylabel("True CDOG Pulse Arrival Time")
    plt.show()

    plt.scatter(list(range(len(CDOG_remain)-1)), CDOG_remain[1:]-CDOG_remain[:len(CDOG_remain)-1], s=1)
    plt.xlabel("Time Index")
    plt.ylabel("Difference between i and i-1 true CDOG nanosecond clock times")
    plt.show()

    plt.scatter(list(range(len(true_travel_times)-1)), true_travel_times[1:]-true_travel_times[:len(true_travel_times)-1], s=1)
    plt.xlabel("Time Index")
    plt.ylabel("Difference between i and i-1 true travel times")
    plt.show()

    plt.scatter(CDOG_mat[:,0] + CDOG_mat[:,1], temp_travel_times, s=1, label="Corrupted Travel Times")
    plt.scatter(removed_CDOG, removed_travel_times, s=1, label="Removed Travel Times")
    plt.scatter(CDOG_mat[:,0] + CDOG_mat[:,1], mat_unwrap, s=1, label="Corrupted Unwrapping")
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.show()

    plt.scatter(CDOG_mat[:,0] + CDOG_mat[:,1], mat_unwrap, marker="x", s=4, label="Corrupted Unwrapping")
    plt.scatter(CDOG_time, CDOG_unwrap, s=1, label="True Unwrapping")
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time")
    plt.ylabel("Travel Times")
    plt.show()

    plt.scatter(list(range(len(CDOG_mat)-1)), CDOG_mat[1:,1]-CDOG_mat[:len(CDOG_mat)-1,1], s=1)
    plt.xlabel("Index")
    plt.ylabel("Difference between i and i-1 corrupted nanosecond clock times")
    plt.show()