"""
File to use Bayesian Analysis to find the Posterior distribution of the CDOG and Lever-arm estimate

Prior is a uniform distribution between -10000 to 10000 in x,y and -4000 to -6000 in z.
    Lever arm prior is -15 to -5 in x, 0 to 10 in y, and -10 to -20 in z
Sampled likelihood found using simulated annealing and gauss-newton inversion

Need to plot the 1,2,3 std of the posterior distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from advancedGeigerMethod import geigersMethod, calculateTimesRayTracing, generateRealistic, findTransponder
from simulatedAnnealing_Synthetic import simulatedAnnealing

def Bayesian_Annealing(iterations, n, time_noise, position_noise, geom_noise):
    print('placeholder')
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)

    guess_arr = np.zeros((iterations, 3))
    lever_arr = np.zeros((iterations, 3))

    for i in range(iterations):
        guess, lever = simulatedAnnealing(n, 300, time_noise, position_noise, geom_noise, False)

        guess_arr[i] = guess
        lever_arr[i] = lever

    print(np.mean(guess_arr, axis=0))
    print(CDog)
    print(np.mean(guess_arr, axis=0) - CDog)
    print(np.linalg.norm(np.mean(guess_arr, axis=0) - CDog)*100)

    plt.scatter(CDog[0], CDog[1], s=100, color="r", marker="x", zorder=2, label = "CDOG Position")
    plt.scatter(guess_arr[:,0], guess_arr[:,1], color='b', marker='o', alpha=0.2, zorder=1, label="Guesses")
    plt.xlim([CDog[0]-0.05, CDog[0]+0.05])
    plt.ylim([CDog[1]-0.05, CDog[1]+0.05])
    plt.axis('equal')
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"Distribution of CDOG position Guesses for {iterations} iterations")
    plt.legend(loc = "upper right")
    plt.show()


Bayesian_Annealing(10, 1000, 2*10**-5, 2*10**-2, 0)