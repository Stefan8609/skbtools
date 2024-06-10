#TO DO: Add standard deviation points to plot, and error std to plot
#Make 4 plot with sections for each inclusion of error: make error a part of the geiger's method module
    #Make the plot modular with input noise too
#Plot observed uncertainty versus noise added
#Plot RMS contour plot
#Run 100 times to get statistical analysis of it

from advancedGeigerMethod import geigersMethod, generateCross

import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import generateCross, geigersMethod, calculateTimesRayTracing

def noise_dependence(n):
    #write function to find the dependence of geiger's method with noise
    #Returns array of observed standard deviations
    noise_arr = []
    for noise in np.logspace(-5, -1, 25):
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateCross(n)
        initial_guess = [-10000, 5000, -4000]

        transponder_coordinates_Found = transponder_coordinates_Actual

        guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, noise)

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)

        noise_arr.append(std_diff)

    plt.scatter(np.logspace(-5, -1, 25), noise_arr)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

noise_dependence(10000)


