import numpy as np
import matplotlib.pyplot as plt
from Generate_Realistic_Transducer import generateRealistic_Transducer
from advancedGeigerMethod import *

cz = np.genfromtxt('../../GPSData/cz_cast2_smoothed.txt')[::100]
depth = np.genfromtxt('../../GPSData/depth_cast2_smoothed.txt')[::100]

def transducer_plot(n, time_noise, position_noise):
    CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder = generateRealistic_Transducer(n)

    GPS_Coordinates = GPS_Coordinates + np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    times_known = calculateTimesRayTracing(CDog, transponder_coordinates)[0]

    times_known = times_known + np.random.normal(0, time_noise, len(transponder_coordinates))

    off_lever = [-11, 2, -13]

    transponder_coordinates_Found_off = findTransponder(GPS_Coordinates, gps1_to_others, off_lever)
    times_found_off = calculateTimesRayTracing(CDog, transponder_coordinates_Found_off)[0]
    difference_data_off = times_found_off - times_known

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    difference_data_off = difference_data_off * 1000
    std = np.std(difference_data_off)
    plt.scatter(np.arange(len(difference_data_off))/3600, difference_data_off, s=1)
    plt.xlabel("Time (hours)")
    plt.ylabel("Misfit (ms)")
    plt.ylim(-3 * std, 3 * std)
    plt.show()

    transponder_coordinates_Found_right = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
    times_found_right = calculateTimesRayTracing(CDog, transponder_coordinates_Found_right)[0]
    difference_data_right = times_found_right - times_known

    plt.figure(figsize=(10, 6))
    difference_data_right = difference_data_right * 1000
    std_right = np.std(difference_data_right)
    plt.scatter(np.arange(len(difference_data_right))/3600, difference_data_right, s=1)
    plt.xlabel("Time (hours)")
    plt.ylabel("Misfit (ms)")
    plt.axhline(std_right, color='k', linestyle='--')
    plt.axhline(-std_right, color='k', linestyle='--')
    plt.ylim(-3 * std_right, 3 * std_right)
    plt.show()

    print(std/std_right)

def transducer_plot_all(n):
    CDog, GPS_Coordinates_init, transponder_coordinates_init, gps1_to_others, gps1_to_transponder = generateRealistic_Transducer(n)


    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    title = ["Just Geometry", "Geometry + Position Noise + Time Noise"]

    for i in range(2):
        if i == 0:
            GPS_Coordinates = np.copy(GPS_Coordinates_init)
            times_known = calculateTimesRayTracing(CDog, transponder_coordinates_init)[0]
        else:
            GPS_Coordinates = np.copy(GPS_Coordinates_init) + np.random.normal(0, 2*10**-2, (len(GPS_Coordinates_init), 4, 3))
            times_known = calculateTimesRayTracing(CDog, transponder_coordinates_init)[0]
            times_known = times_known + np.random.normal(0, 2*10**-5, len(transponder_coordinates_init))

        off_lever = [-11, 2, -13]

        transponder_coordinates_Found_off = findTransponder(GPS_Coordinates, gps1_to_others, off_lever)
        times_found_off = calculateTimesRayTracing(CDog, transponder_coordinates_Found_off)[0]
        difference_data_off = times_found_off - times_known

        difference_data_off = difference_data_off * 1000
        axes[i].scatter(np.arange(len(difference_data_off))/3600, difference_data_off, s=1)
        axes[i].set_ylabel("Misfit (ms)")
        if i == 0:
            std = np.std(difference_data_off)
            axes[i].get_xaxis().set_visible(False)
        if i == 1:
            std = np.std(difference_data_off)
            axes[i].set_xlabel("Time (hours)")
        axes[i].axhline(std, color='r', linestyle='--')
        axes[i].axhline(-std, color='r', linestyle='--')
        axes[i].set_title(f"{title[i]}")
        axes[i].set_ylim(-3 * std, 3 * std)
    plt.show()

if __name__ == "__main__":
    transducer_plot(10000, 2*10**-5, 2*10**-2)