import numpy as np
import matplotlib.pyplot as plt
import random
from advancedGeigerMethod import geigersMethod, calculateTimesRayTracing, generateRealistic, findTransponder

def  Bayesian_Geiger(iterations, n, time_noise, position_noise):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)

    guess_arr = np.zeros((iterations, 3))
    for i in range(iterations):
        initial_guess = np.array([random.uniform(-10000, 10000), random.uniform(-10000,10000), random.uniform(-4000, -6000)])
        GPS_Coordinates_iter = GPS_Coordinates + np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
        transponder_coordinates_Found = findTransponder(GPS_Coordinates_iter, gps1_to_others, gps1_to_transponder)

        guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found, time_noise)

        guess_arr[i] = guess

    print(np.mean(guess_arr, axis=0))
    print(CDog)
    print(np.mean(guess_arr, axis=0) - CDog)
    print(np.linalg.norm(np.mean(guess_arr, axis=0) - CDog)*100)

    plt.scatter(CDog[0], CDog[1], s=100, color="r", marker="x", zorder=2, label = "CDOG Position")
    plt.scatter(guess_arr[:,0], guess_arr[:,1], color='b', marker='o', alpha=0.2, zorder=1, label="Guesses")
    plt.xlim([CDog[0]-0.05, CDog[0]+0.05])
    plt.ylim([CDog[1]-0.05, CDog[1]+0.05])
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"Distribution of CDOG position Guesses for {iterations} iterations")
    plt.legend(loc = "upper right")
    plt.show()

    dist_arr = np.linalg.norm(guess_arr-CDog, axis=1)*100
    plt.hist(dist_arr, bins=25, density=True)
    plt.title(f"Histogram of residual distance from {iterations} CDOG guesses")
    plt.xlabel("Distance from guess to CDOG (cm)")
    plt.ylabel("Distribution")
    plt.show()

Bayesian_Geiger(10000, 100, 2*10**-5, 2*10**-2)

#Make new a function which creates a trajectory of 10000 points with noise in time and space
#   Sample random sets of 100 points and run geiger's method
#   Sample consecutive sets of 100 points and run geiger's method
#       See how close the average final guess is from these 100 runs to the actual CDOG location