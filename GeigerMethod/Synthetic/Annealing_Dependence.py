import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import *
from simulatedAnnealing_Synthetic import simulatedAnnealing

def point_dependence(time_noise, position_noise, lever_noise = 5):
    point_arr = []

    for n in np.logspace(1, 3, 10):
        n = int(n)
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
        initial_guess = np.random.normal(0, 10000, 3)

        annealing_guess, lever = simulatedAnnealing(n, 300, time_noise, position_noise, 0, False,
                                                    CDog, GPS_Coordinates, transponder_coordinates_Actual,
                                                    gps1_to_others, gps1_to_transponder)

        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        annealing_estimate, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found, time_noise)[:2]
        times_calc = calculateTimesRayTracing(annealing_estimate, transponder_coordinates_Found)[0]
        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)
        point_arr.append(std_diff)
    plt.scatter(np.logspace(1, 3, 10), point_arr)
    plt.xscale('log')
    plt.xlabel('Number of Points')
    plt.ylabel('Derived Uncertainty in Estimation Position')
    plt.ylim([0, (time_noise + position_noise/1515) * 1.1 ])
    plt.axhline(time_noise, color='y', label=f"Time Noise: {time_noise*1000} ms", alpha=0.5)
    plt.axhline(position_noise/1515, color='r', label=f"Position Noise: {position_noise*100} cm", alpha=0.5)
    plt.axhline(np.sqrt(time_noise**2 + (position_noise/1515)**2), color='b', label="Time and Position Noise", alpha=0.5)
    plt.legend(loc = "lower right")
    plt.show()

def noise_dependence(points, n):
    time_noise = np.random.normal(5*10**-5, 1*10**-5, points)
    space_noise = np.random.normal(5*10**-2, 1*10**-2, points)

    std_exp = np.sqrt(time_noise**2 + (space_noise/1515)**2)
    std_observed = np.zeros(points)

    for i in range(points):
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
        initial_guess = np.array(
            [random.uniform(-10000, 10000), random.uniform(-10000, 10000), random.uniform(-4000, -6000)])

        annealing_guess, lever = simulatedAnnealing(n, 300, time_noise[i], space_noise[i], 0, False,
                                                    CDog, GPS_Coordinates, transponder_coordinates_Actual,
                                                    gps1_to_others, gps1_to_transponder)
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        annealing_estimate, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                                        transponder_coordinates_Found, time_noise[i])[:2]
        times_calc = calculateTimesRayTracing(annealing_estimate, transponder_coordinates_Found)[0]
        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)
        std_observed[i] = std_diff

    plt.scatter(std_exp, std_observed)
    plt.show()


# point_dependence(2*10**-5, 2*10**-2)
noise_dependence(10, 500)