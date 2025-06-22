import matplotlib.pyplot as plt
import numpy as np
from GeigerMethod.Synthetic.advancedGeigerMethod import (
    geigersMethod,
    findTransponder,
    calculateTimesRayTracing,
)
from GeigerMethod.Synthetic.geigerTimePlot import geigerTimePlot
from GeigerMethod.Synthetic.Generate_Realistic_Transducer import (
    generateRealistic_Transducer,
)
import random
from data import gps_data_path

cz = np.genfromtxt(gps_data_path("cz_cast2_smoothed.txt"))[::100]
depth = np.genfromtxt(gps_data_path("depth_cast2_smoothed.txt"))[::100]

"""This program feels inefficient recalculating times known a bunch of times"""


def simulatedAnnealing(
    n,
    iter,
    time_noise,
    position_noise,
    geom_noise=0,
    main=True,
    CDog=None,
    GPS_Coordinates_in=None,
    transponder_coordinates_Actual=None,
    gps1_to_others_in=None,
    gps1_to_transponder=None,
):
    if main:
        (
            CDog,
            GPS_Coordinates_in,
            transponder_coordinates_Actual,
            gps1_to_others_in,
            gps1_to_transponder,
        ) = generateRealistic_Transducer(n)

    # Apply position noise
    gps1_to_others = gps1_to_others_in + np.random.normal(0, geom_noise, (4, 3))
    GPS_Coordinates = GPS_Coordinates_in + np.random.normal(
        0, position_noise, (len(GPS_Coordinates_in), 4, 3)
    )

    # Get initial values
    times_known = calculateTimesRayTracing(CDog, transponder_coordinates_Actual)[0]

    old_lever = np.array([-11, 2, -13])

    transponder_coordinates_Found = findTransponder(
        GPS_Coordinates, gps1_to_others, old_lever
    )
    initial_guess = np.array(
        [
            random.uniform(-10000, 10000),
            random.uniform(-10000, 10000),
            random.uniform(-4000, -6000),
        ]
    )
    guess = geigersMethod(
        initial_guess,
        CDog,
        transponder_coordinates_Actual,
        transponder_coordinates_Found,
    )[0]

    # Apply time noise
    times_known += np.random.normal(0, time_noise, len(transponder_coordinates_Actual))

    # Calculate times from initial guess and lever arm
    # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    # Calculate the initial RMSE
    difference_data = times_calc - times_known
    old_RMS = np.sqrt(np.nanmean(difference_data**2))

    # Plot initial conditions
    # experimentPathPlot(transponder_coordinates_Actual, CDog)
    # leverHist(transponder_coordinates_Actual, transponder_coordinates_Found)
    if main:
        geigerTimePlot(
            initial_guess,
            GPS_Coordinates,
            CDog,
            transponder_coordinates_Actual,
            transponder_coordinates_Found,
            gps1_to_transponder,
            cz,
            depth,
            time_noise,
            position_noise,
            old_lever,
            sim=1,
        )

        plt.figure(figsize=(10, 4))
        difference_data = difference_data * 1000
        std = np.std(difference_data)
        plt.scatter(range(len(difference_data)), difference_data, s=1)
        plt.xlabel("Time Step (s)")
        plt.ylabel("Misfit (ms)")
        plt.ylim(-3 * std, 3 * std)
        plt.show()

    # Run simulated annealing
    k = 0
    RMS_arr = [0] * (iter - 1)
    while k < iter - 1:  # Figure out how to work temp threshold
        temp = np.exp(-k * 7 * (1 / (iter)))  # temp schdule
        displacement = ((np.random.rand(3) * 2) - [1, 1, 1]) * temp
        lever = old_lever + displacement

        # Find RMS
        transponder_coordinates_Found = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        guess = geigersMethod(
            guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found
        )[0]

        # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        difference_data = times_calc - times_known
        RMS = np.sqrt(np.nanmean(difference_data**2))
        if RMS - old_RMS < 0:  # What is acceptance condition?
            old_lever = lever
            old_RMS = RMS
        if k % 10 == 0:
            print(k, old_RMS * 100 * 1515, old_lever)
        RMS_arr[k] = RMS * 100 * 1515
        k += 1

    # leverHist(transponder_coordinates_Actual, transponder_coordinates_Final)

    if main:
        plt.plot(list(range(iter - 1)), RMS_arr)
        plt.xlabel("Simulated Annealing Iteration")
        plt.ylabel("RMSE from Inversion (cm)")
        plt.title("Simulated Annealing Inversion for GPS to Transducer Lever Arm")
        plt.show()

        print(old_lever, gps1_to_transponder)
        transponder_coordinates_Final = findTransponder(
            GPS_Coordinates, gps1_to_others, old_lever
        )
        geigerTimePlot(
            initial_guess,
            GPS_Coordinates,
            CDog,
            transponder_coordinates_Actual,
            transponder_coordinates_Final,
            gps1_to_transponder,
            cz,
            depth,
            time_noise,
            position_noise,
            old_lever,
            sim=2,
        )

        plt.figure(figsize=(10, 4))
        difference_data = difference_data * 1000
        std = np.std(difference_data)
        plt.scatter(range(len(difference_data)), difference_data, s=1)
        plt.xlabel("Time Step (s)")
        plt.ylabel("Misfit (ms)")
        plt.ylim(-3 * std, 3 * std)
        plt.show()

    transponder_coordinates_Final = findTransponder(
        GPS_Coordinates, gps1_to_others, old_lever
    )
    guess = geigersMethod(
        guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Final
    )[0]

    return guess, old_lever


if __name__ == "__main__":
    simulatedAnnealing(5000, 300, 2 * 10**-5, 2 * 10**-2)

# at each time step keep the cdog location and then calculate a bunch of deviations
#   and keep the best one at each time step --Could speed up
#   Also scale with RMSE (see if this works for temperature)
#   Genetic algorithm, Neighborhood algorithm
#   Plot the ones I keep on the plot
#   Markov Chain Monte Carlo - https://agupubs.onlinelibrary.wiley.com/doi/10.1029/94JB03097
#   Base off barycenter rather then GPS1
#   Solve for all 4 gps offsets and see if position matches
#   Derben-Watson test

#   Normalize histogram by normalizing area of bins to be 1
#   See what histogram noise looks like with no time noise
#   Then do the same with no gps noise with time noise to see how histograms change
#   Variance Reduction

# Add panel to time plot giving a zoomed window of points explaining the trajectory and
# On same axis show the same thing

# See what pathing I get if I have straight cross in x-y and no perturbation in z
# Break down the error

# Testing for github authentication
