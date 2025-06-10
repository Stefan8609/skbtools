import numpy as np
import matplotlib.pyplot as plt
import random
from advancedGeigerMethod import (
    generateRealistic,
    calculateTimesRayTracing,
    findTransponder,
    computeJacobianRayTracing,
)


def RMSE_iter_plot(n, runs, time_noise, position_noise):
    for i in range(runs):
        (
            CDog,
            GPS_Coordinates,
            transponder_coordinates_Actual,
            gps1_to_others,
            gps1_to_transponder,
        ) = generateRealistic(n)
        guess = np.array(
            [
                random.uniform(-10000, 10000),
                random.uniform(-10000, 10000),
                random.uniform(-4000, -6000),
            ]
        )
        GPS_Coordinates += np.random.normal(
            0, position_noise, (len(GPS_Coordinates), 4, 3)
        )
        transponder_coordinates_Found = findTransponder(
            GPS_Coordinates, gps1_to_others, gps1_to_transponder
        )

        epsilon = 10**-5
        times_known, esv = calculateTimesRayTracing(
            CDog, transponder_coordinates_Actual
        )
        times_known += np.random.normal(
            0, time_noise, len(transponder_coordinates_Actual)
        )

        k = 0
        delta = 1
        RMSE_arr = np.array([])
        # Loop until change in guess is less than the threshold
        while np.linalg.norm(delta) > epsilon and k < 100:
            times_guess, esv = calculateTimesRayTracing(
                guess, transponder_coordinates_Found
            )
            jacobian = computeJacobianRayTracing(
                guess, transponder_coordinates_Found, times_guess, esv
            )
            delta = (
                -1
                * np.linalg.inv(jacobian.T @ jacobian)
                @ jacobian.T
                @ (times_guess - times_known)
            )
            guess = guess + delta

            difference_data = times_guess - times_known
            RMSE_arr = np.append(RMSE_arr, np.sqrt(np.nanmean(difference_data**2)))
            k += 1
        plt.plot(np.arange(len(RMSE_arr)), RMSE_arr, label=f"Run {i + 1}")
    plt.title(
        f"RMSE by iteration of Geiger's Method: \n"
        f"Time noise:{round(time_noise * 10**6, 3)} µs, Position noise: {round(position_noise * 100, 3)} cm"
    )
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (s)")
    plt.legend()
    plt.show()

    return


RMSE_iter_plot(10000, 10, 2 * 10**-5, 2 * 10**-2)
