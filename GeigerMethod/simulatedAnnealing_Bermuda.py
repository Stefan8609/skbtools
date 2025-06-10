import numpy as np
import matplotlib.pyplot as plt
from geigerMethod_Bermuda import (
    findTransponder,
    geigersMethod,
    calculateTimesRayTracing,
)
from Synthetic.experimentPathPlot import experimentPathPlot
from timePlot_Bermuda import geigerTimePlot


def simulatedAnnealing_Bermuda(
    n,
    GPS_Coordinates,
    initial_guess,
    times_known,
    gps1_to_others,
    old_lever,
    timestamps,
):
    """Estimate the GPS-to-transponder lever arm via simulated annealing.

    Parameters
    ----------
    n : int
        Number of annealing iterations.
    GPS_Coordinates : ndarray
        Array of GPS positions for each time step.
    initial_guess : array-like, shape (3,)
        Initial guess of the platform location.
    times_known : ndarray
        Measured travel times.
    gps1_to_others : ndarray
        Initial offsets between GPS receivers.
    old_lever : ndarray
        Starting lever-arm guess.
    timestamps : ndarray
        Time values associated with ``times_known``.

    Returns
    -------
    numpy.ndarray
        Estimated lever arm between GPS1 and the transponder.
    """

    # Get initial values
    transponder_coordinates_Found = findTransponder(
        GPS_Coordinates, gps1_to_others, old_lever
    )

    guess = geigersMethod(initial_guess, times_known, transponder_coordinates_Found)

    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    difference_data = times_calc - times_known
    old_RMS = np.sqrt(np.nanmean(difference_data**2))

    # Plot initial conditions
    # experimentPathPlot(transponder_coordinates_Found, initial_guess)
    geigerTimePlot(
        initial_guess, times_known, transponder_coordinates_Found, timestamps
    )

    # Run simulated annealing
    k = 0
    RMS_arr = [0] * (n - 1)
    while k < n - 1:
        temp = np.exp(-k * 7 * (1 / (n)))  # temp schdule
        displacement = ((np.random.rand(3) * 2) - [1, 1, 1]) * 10 * temp
        lever = old_lever + displacement

        # Find RMS
        transponder_coordinates_Found = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        guess = geigersMethod(initial_guess, times_known, transponder_coordinates_Found)

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        difference_data = times_calc - times_known
        RMS = np.sqrt(np.nanmean(difference_data**2))
        if RMS - old_RMS < 0:
            old_lever = lever
            old_RMS = RMS
        print(k, old_RMS * 100 * 1515, temp)
        RMS_arr[k] = RMS * 100 * 1515
        k += 1
    plt.plot(list(range(n - 1)), RMS_arr)
    plt.xlabel("Simulated Annealing Iteration")
    plt.ylabel("RMSE from Inversion (cm)")
    plt.title("Simulated Annealing Inversion for GPS to Transducer Lever Arm")
    plt.show()
    print(old_lever)

    transponder_coordinates_Final = findTransponder(
        GPS_Coordinates, gps1_to_others, old_lever
    )
    experimentPathPlot(transponder_coordinates_Final, guess)

    geigerTimePlot(
        initial_guess, times_known, transponder_coordinates_Final, timestamps
    )

    plt.scatter(GPS_Coordinates[:, 0, 0], GPS_Coordinates[:, 0, 1], color="r", s=3)
    plt.scatter(GPS_Coordinates[:, 1, 0], GPS_Coordinates[:, 1, 1], color="y", s=3)
    plt.scatter(GPS_Coordinates[:, 2, 0], GPS_Coordinates[:, 2, 1], color="g", s=3)
    plt.scatter(GPS_Coordinates[:, 3, 0], GPS_Coordinates[:, 3, 1], color="b", s=3)
    plt.scatter(
        transponder_coordinates_Final[:, 0],
        transponder_coordinates_Final[:, 1],
        color="k",
        s=3,
    )
    plt.show()

    print(old_lever)

    return old_lever
