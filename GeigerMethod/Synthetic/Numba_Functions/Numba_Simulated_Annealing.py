from Numba_Geiger import *
from numba import njit


@njit
def simulated_annealing(n, iter, time_noise, position_noise):
    CDog, GPS_Coordinates_in, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)

    #Apply position noise
    GPS_Coordinates = GPS_Coordinates_in + np.random.normal(0, position_noise, (len(GPS_Coordinates_in), 4, 3))

    times_known = calculateTimesRayTracing(CDog, transponder_coordinates_Actual)[0]

    old_lever = np.array([-11.0, 2.0, -13.0])

    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
    initial_guess = np.array(
        [random.uniform(-10000, 10000), random.uniform(-10000, 10000), random.uniform(-4000, -6000)])
    guess = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                       transponder_coordinates_Found)[0]

    #Apply time noise
    times_known+=np.random.normal(0, time_noise, len(transponder_coordinates_Actual))

    #Calculate times from initial guess and lever arm
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    #Calculate the initial RMSE
    difference_data = times_calc - times_known
    old_RMS = np.sqrt(np.nanmean(difference_data ** 2))

    k=0
    while k < iter - 1:  # Figure out how to work temp threshold
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / (iter)))  # temp schdule
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = old_lever + displacement

        # Find RMS
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        guess = geigersMethod(guess, CDog, transponder_coordinates_Actual,
                              transponder_coordinates_Found)[0]

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        difference_data = times_calc - times_known
        RMS = np.sqrt(np.nanmean(difference_data ** 2))
        if RMS - old_RMS < 0:
            old_lever = lever
            old_RMS = RMS
        if k % 10 == 0:
            print(k, old_RMS * 100 * 1515, old_lever)
        k += 1

    transponder_coordinates_Final = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
    guess = geigersMethod(guess, CDog, transponder_coordinates_Actual,
                          transponder_coordinates_Final)[0]
    return guess, old_lever


if __name__ == "__main__":
    simulated_annealing(1000, 300, 2*10**-5, 2*10**-2)


