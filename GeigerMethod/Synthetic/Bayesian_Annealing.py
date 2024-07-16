"""
File to use Bayesian Analysis to find the Posterior distribution of the CDOG and Lever-arm estimate

Prior is a uniform distribution between -10000 to 10000 in x,y and -4000 to -6000 in z.
    Lever arm prior is -15 to -5 in x, 0 to 10 in y, and -10 to -20 in z
Sampled likelihood found using simulated annealing and gauss-newton inversion

Need to plot the 1,2,3 std of the posterior distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random
from advancedGeigerMethod import geigersMethod, calculateTimesRayTracing, generateRealistic, findTransponder
from simulatedAnnealing_Synthetic import simulatedAnnealing

def Bayesian_Annealing(iterations, n, time_noise, position_noise, geom_noise):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)

    guess_arr = np.zeros((iterations, 3))
    lever_arr = np.zeros((iterations, 3))

    for i in range(iterations):
        guess, lever = simulatedAnnealing(n, 300, time_noise, position_noise, geom_noise, False,
                                          CDog, GPS_Coordinates, transponder_coordinates_Actual,
                                          gps1_to_others, gps1_to_transponder)

        guess_arr[i] = guess
        lever_arr[i] = lever
        print(i)

    print(np.mean(guess_arr, axis=0))
    print(CDog)
    print(np.mean(guess_arr, axis=0) - CDog)
    print(np.linalg.norm(np.mean(guess_arr, axis=0) - CDog)*100)

    expected_std = np.sqrt(0.00103**2 * position_noise**2 + time_noise**2) * 1515
    print('expected', expected_std)

    # Plot points and contours
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ellipses of expected standard deviation from input noise
    for i in range(1, 4):
        ell = Ellipse(xy=(CDog[0], CDog[1]),
                      width=expected_std * i * 2, height=expected_std * i * 2,
                      angle=0, color='k', zorder=3)
        ell.set_facecolor('none')
        ax.add_artist(ell)

    # Plot ellipses derived from covariance matrix of estimates
    cov = np.cov(guess_arr[:, :2], rowvar=False)
    eigval, eigvec = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    w, h = np.sqrt(eigval) * 2
    for i in range(1, 4):
        ell = Ellipse(xy=(CDog[0], CDog[1]), width=i * w, height=i * h, angle=angle,
                      color='r', linewidth=2 - i / 2, zorder=3)
        ell.set_facecolor('none')
        ax.add_artist(ell)

    # Scatter estimate points and C-DOG
    ax.scatter(CDog[0], CDog[1], s=100, color="r", marker="o", zorder=2, label="CDOG Position")
    ax.scatter(guess_arr[:, 0], guess_arr[:, 1], color='b', marker='o', alpha=0.2, zorder=1, label="Position Estimates")
    ax.set_xlim([CDog[0] - 3.1 * expected_std, CDog[0] + 3.1 * expected_std])
    ax.set_ylim([CDog[1] - 3.1 * expected_std, CDog[1] + 3.1 * expected_std])
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Distribution of CDOG position estimates for {iterations} iterations")
    ax.legend(loc="upper right")
    plt.show()

    # Plot histogram of how far estimates are from C-DOG location
    dist_arr = np.linalg.norm(guess_arr - CDog, axis=1) * 100
    plt.hist(dist_arr, bins=25, density=True)
    plt.title(f"Histogram of residual distance from {iterations} CDOG position estimates")
    plt.xlabel("Distance from guess to CDOG (cm)")
    plt.ylabel("Distribution")
    plt.show()

    #Add Histograms for lever arms


Bayesian_Annealing(100, 1000, 2*10**-5, 2*10**-2, 0)