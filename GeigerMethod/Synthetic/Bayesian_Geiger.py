"""
File to use Bayesian Analysis to find the Posterior distribution of the CDOG estimate

Prior is a uniform distribution between -10000 to 10000 in x,y and -4000 to -6000 in z.
Sampled likelihood found using Gauss-Newton Inversion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random
from advancedGeigerMethod import (
    geigersMethod,
    generateRealistic,
    findTransponder,
)


# Using same GPS Coords with random spatial noise added each iteration
def Bayesian_Geiger(iterations, n, time_noise, position_noise):
    (
        CDog,
        GPS_Coordinates,
        transponder_coordinates_Actual,
        gps1_to_others,
        gps1_to_transponder,
    ) = generateRealistic(n)

    guess_arr = np.zeros((iterations, 3))
    for i in range(iterations):
        initial_guess = np.array(
            [
                random.uniform(-10000, 10000),
                random.uniform(-10000, 10000),
                random.uniform(-4000, -6000),
            ]
        )
        GPS_Coordinates_iter = GPS_Coordinates + np.random.normal(
            0, position_noise, (len(GPS_Coordinates), 4, 3)
        )
        transponder_coordinates_Found = findTransponder(
            GPS_Coordinates_iter, gps1_to_others, gps1_to_transponder
        )

        guess, times_known = geigersMethod(
            initial_guess,
            CDog,
            transponder_coordinates_Actual,
            transponder_coordinates_Found,
            time_noise,
        )[:2]

        guess_arr[i] = guess

    print(np.mean(guess_arr, axis=0))
    print(CDog)
    print(np.mean(guess_arr, axis=0) - CDog)
    print(np.linalg.norm(np.mean(guess_arr, axis=0) - CDog) * 100, "cm")

    print("std", np.std((guess_arr - CDog) * 100), "cm")

    expected_std = np.sqrt(0.00103**2 * position_noise**2 + time_noise**2) * 1515
    print("expected", expected_std)

    # Plot points and contours
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ellipses of expected standard deviation from input noise
    for i in range(1, 4):
        ell = Ellipse(
            xy=(CDog[0], CDog[1]),
            width=expected_std * i * 2,
            height=expected_std * i * 2,
            angle=0,
            color="k",
            zorder=3,
        )
        ell.set_facecolor("none")
        ax.add_artist(ell)

    # Plot ellipses derived from covariance matrix of estimates
    cov = np.cov(guess_arr[:, :2], rowvar=False)
    eigval, eigvec = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    w, h = np.sqrt(eigval) * 2
    for i in range(1, 4):
        ell = Ellipse(
            xy=(CDog[0], CDog[1]),
            width=i * w,
            height=i * h,
            angle=angle,
            color="r",
            linewidth=2 - i / 2,
            zorder=3,
        )
        ell.set_facecolor("none")
        ax.add_artist(ell)

    # Scatter estimate points and C-DOG
    ax.scatter(
        CDog[0], CDog[1], s=100, color="r", marker="o", zorder=2, label="CDOG Position"
    )
    ax.scatter(
        guess_arr[:, 0],
        guess_arr[:, 1],
        color="b",
        marker="o",
        alpha=0.2,
        zorder=1,
        label="Position Estimates",
    )
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
    plt.title(
        f"Histogram of residual distance from {iterations} CDOG position estimates"
    )
    plt.xlabel("Distance from guess to CDOG (cm)")
    plt.ylabel("Distribution")
    plt.show()


def Sampled_Geiger(iterations, n, sample_size, time_noise, position_noise):
    (
        CDog,
        GPS_Coordinates,
        transponder_coordinates_Actual,
        gps1_to_others,
        gps1_to_transponder,
    ) = generateRealistic(n)
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    guess_arr = np.zeros((iterations, 3))

    for i in range(iterations):
        initial_guess = np.array(
            [
                random.uniform(-10000, 10000),
                random.uniform(-10000, 10000),
                random.uniform(-4000, -6000),
            ]
        )
        indices = np.random.choice(np.arange(n), sample_size, replace=False)
        GPS_Coordinates_iter = GPS_Coordinates[indices]
        transponder_coordinates_Actual_iter = transponder_coordinates_Actual[indices]
        transponder_coordinates_Found = findTransponder(
            GPS_Coordinates_iter, gps1_to_others, gps1_to_transponder
        )
        guess, times_known = geigersMethod(
            initial_guess,
            CDog,
            transponder_coordinates_Actual_iter,
            transponder_coordinates_Found,
            time_noise,
        )[:2]
        guess_arr[i] = guess

    print(np.mean(guess_arr, axis=0))
    print(CDog)
    print(np.mean(guess_arr, axis=0) - CDog)
    print(np.linalg.norm(np.mean(guess_arr, axis=0) - CDog) * 100, "cm")
    print("std", np.std((guess_arr - CDog) * 100), "cm")

    expected_std = np.sqrt(0.00103**2 * position_noise**2 + time_noise**2) * 1515
    print("expected", expected_std)

    # Plot points and contours
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ellipses of expected standard deviation from input noise
    for i in range(1, 4):
        ell = Ellipse(
            xy=(0, 0),
            width=expected_std * i * 2 * 100,
            height=expected_std * i * 2 * 100,
            angle=0,
            color="k",
            zorder=3,
        )
        ell.set_facecolor("none")
        ax.add_artist(ell)

    # Plot ellipses derived from covariance matrix of estimates
    cov = np.cov(guess_arr[:, :2], rowvar=False)
    eigval, eigvec = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    w, h = np.sqrt(eigval) * 2
    for i in range(1, 4):
        ell = Ellipse(
            xy=(0, 0),
            width=i * w * 100,
            height=i * h * 100,
            angle=angle,
            color="r",
            linewidth=2 - i / 2,
            zorder=3,
        )
        ell.set_facecolor("none")
        ax.add_artist(ell)

    # Scatter estimate points and C-DOG
    ax.scatter(0, 0, s=100, color="r", marker="o", zorder=2, label="CDOG Position")
    ax.scatter(
        (guess_arr[:, 0] - CDog[0]) * 100,
        (guess_arr[:, 1] - CDog[1]) * 100,
        color="b",
        marker="o",
        alpha=0.2,
        zorder=1,
        label="Position Estimates",
    )
    ax.set_xlim([-3.1 * expected_std * 100, 3.1 * expected_std * 100])
    ax.set_ylim([-3.1 * expected_std * 100, 3.1 * expected_std * 100])
    ax.set_xlabel("x distance (cm)", fontsize=12)
    ax.set_ylabel("y distance (cm)", fontsize=12)
    ax.set_title(f"Distribution of CDOG position estimates for {iterations} iterations")
    ax.legend(loc="upper right")
    plt.show()

    # Plot histogram of how far estimates are from C-DOG location
    dist_arr = np.linalg.norm(guess_arr - CDog, axis=1) * 100
    plt.hist(dist_arr, bins=25, density=True)
    plt.title(
        f"Histogram of residual distance from {iterations} CDOG position estimates"
    )
    plt.xlabel("Distance from guess to CDOG (cm)", fontsize=12)
    plt.ylabel("Distribution", fontsize=12)
    plt.show()


def Consecutive_Geiger(iterations, n, sample_size, time_noise, position_noise):
    (
        CDog,
        GPS_Coordinates,
        transponder_coordinates_Actual,
        gps1_to_others,
        gps1_to_transponder,
    ) = generateRealistic(n)
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    guess_arr = np.zeros((iterations, 3))

    for i in range(iterations):
        initial_guess = np.array(
            [
                random.uniform(-10000, 10000),
                random.uniform(-10000, 10000),
                random.uniform(-4000, -6000),
            ]
        )
        start_idx = np.random.randint(0, n - sample_size)
        indices = np.arange(start_idx, start_idx + sample_size)

        GPS_Coordinates_iter = GPS_Coordinates[indices]
        transponder_coordinates_Actual_iter = transponder_coordinates_Actual[indices]
        transponder_coordinates_Found = findTransponder(
            GPS_Coordinates_iter, gps1_to_others, gps1_to_transponder
        )
        guess, times_known = geigersMethod(
            initial_guess,
            CDog,
            transponder_coordinates_Actual_iter,
            transponder_coordinates_Found,
            time_noise,
        )[:2]
        guess_arr[i] = guess

    print(np.mean(guess_arr, axis=0))
    print(CDog)
    print(np.mean(guess_arr, axis=0) - CDog)
    print(np.linalg.norm(np.mean(guess_arr, axis=0) - CDog) * 100, "cm")

    expected_std = np.sqrt(0.00103**2 * position_noise**2 + time_noise**2) * 1515
    print("expected", expected_std)

    # Plot points and contours
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ellipses of expected standard deviation from input noise
    for i in range(1, 4):
        ell = Ellipse(
            xy=(CDog[0], CDog[1]),
            width=expected_std * i * 2,
            height=expected_std * i * 2,
            angle=0,
            color="k",
            zorder=3,
        )
        ell.set_facecolor("none")
        ax.add_artist(ell)

    # Plot ellipses derived from covariance matrix of estimates
    cov = np.cov(guess_arr[:, :2], rowvar=False)
    eigval, eigvec = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    w, h = np.sqrt(eigval) * 2
    for i in range(1, 4):
        ell = Ellipse(
            xy=(CDog[0], CDog[1]),
            width=i * w,
            height=i * h,
            angle=angle,
            color="r",
            linewidth=2 - i / 2,
            zorder=3,
        )
        ell.set_facecolor("none")
        ax.add_artist(ell)

    # Scatter estimate points and C-DOG
    ax.scatter(
        CDog[0], CDog[1], s=100, color="r", marker="o", zorder=2, label="CDOG Position"
    )
    ax.scatter(
        guess_arr[:, 0],
        guess_arr[:, 1],
        color="b",
        marker="o",
        alpha=0.2,
        zorder=1,
        label="Position Estimates",
    )
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
    plt.title(
        f"Histogram of residual distance from {iterations} CDOG position estimates"
    )
    plt.xlabel("Distance from guess to CDOG (cm)")
    plt.ylabel("Distribution")
    plt.show()


# Bayesian_Geiger(10000, 100, 2*10**-5, 2*10**-2)
Sampled_Geiger(10000, 10000, 100, 2 * 10**-5, 2 * 10**-2)
# Consecutive_Geiger(10, 10000, 1000, 2*10**-5, 2*10**-2)
