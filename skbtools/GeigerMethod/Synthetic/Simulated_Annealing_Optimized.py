import numpy as np
from advancedGeigerMethod import (
    geigersMethod,
    calculateTimesRayTracing,
    findTransponder,
    generateRealistic_Transducer,
)
import random


cz = np.genfromtxt("../../GPSData/cz_cast2_smoothed.txt")[::100]
depth = np.genfromtxt("../../GPSData/depth_cast2_smoothed.txt")[::100]


def RMSE_function(
    lever,
    GPS_Coordinates,
    guess,
    gps1_to_others,
    CDog,
    transponder_coordinates_Actual,
    times_known,
):
    # Find RMS
    transponder_coordinates_Found = findTransponder(
        GPS_Coordinates, gps1_to_others, lever
    )
    guess = geigersMethod(
        guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found
    )[0]

    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
    difference_data = times_calc - times_known
    RMSE = np.sqrt(np.nanmean(difference_data**2))

    return RMSE


def simulated_annealing_optimized(
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

    lever_old = np.array([-11, 2, -13])

    transponder_coordinates_Found = findTransponder(
        GPS_Coordinates, gps1_to_others, lever_old
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
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    # Calculate the initial RMSE
    difference_data = times_calc - times_known
    RMSE_old = np.sqrt(np.nanmean(difference_data**2))

    k = 0
    while k < iter - 1:
        temp = np.exp(-k * 7 * (1 / (iter)))  # temp schdule
        levers = lever_old + ((np.random.rand(100, 3) * 2 - np.array([1, 1, 1])) * temp)

        RMSE_arr = np.array(
            [
                RMSE_function(
                    lever,
                    GPS_Coordinates,
                    guess,
                    gps1_to_others,
                    CDog,
                    transponder_coordinates_Actual,
                    times_known,
                )
                for lever in levers
            ]
        )

        RMSE_diff = RMSE_arr - RMSE_old
        mask = np.where(RMSE_diff < 0)

        total = 0
        direction = np.array([0.0, 0.0, 0.0])
        if mask[0].size == 0:
            k += 1
            continue
        for index in mask[0]:
            total += np.abs(RMSE_diff[index])
            direction += lever_old - levers[index]
        direction /= 100

        print(lever_old, gps1_to_transponder, lever_old - gps1_to_transponder)
        RMSE_reduction = -np.inf
        while RMSE_reduction < 0:
            lever_new = lever_old - direction
            RMSE_new = RMSE_function(
                lever_new,
                GPS_Coordinates,
                guess,
                gps1_to_others,
                CDog,
                transponder_coordinates_Actual,
                times_known,
            )
            RMSE_reduction = RMSE_new - RMSE_old

            if RMSE_reduction < 0:
                lever_old = lever_new
                RMSE_old = RMSE_new
        print(RMSE_old * 100 * 1515)
        k += 1
    print(lever_old - gps1_to_transponder)


if __name__ == "__main__":
    # I have 10 cpu's on this computer
    simulated_annealing_optimized(1000, 10, 2 * 10**-5, 2 * 10**-2)
