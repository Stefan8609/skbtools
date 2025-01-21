import numpy as np
import matplotlib.pyplot as plt

from Numba_xAline import *
from Numba_xAline_Geiger import *

"""
Annealing Schematic

    Find Alignment? (How often do I do this?) (Every iteration?)
    
    Find Seafloor Receiver and RMSE
    
    Perturb Lever Arm
    
    [When do we transition to the next geiger?]
    
What part of the algorithm do we use Numba for?
"""

def simulated_annealing(iter, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others, initial_guess):
    """Algorithm to determine the best lever arm, offset, and seafloor receiver position"""

    #Initialize variables
    status = "int"
    old_offset = 0
    inversion_guess = initial_guess
    best_rmse = np.inf
    best_lever = np.array([-11.0, 2.0, -13.0], dtype=np.float64)
    k=0
    while k<300:
        temp = np.exp(-np.float64(k) * 7.0 * (1.0 / (iter)))
        displacement = ((np.random.rand(3) * 2.0) - np.array([1.0, 1.0, 1.0])) * temp
        lever = best_lever + displacement

        transponder_coordinates_found = findTransponder(GPS_Coordinates, gps1_to_others, lever)

        if status == "int":
            inversion_guess, offset = initial_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates_found)
            if offset == old_offset:
                status = "subint"
        elif status == "subint":
            inversion_guess, offset = transition_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates_found, offset)
            status = "constant"
        else:
            if k == 100 or k == 200:
                inversion_guess, offset = transition_geiger(inversion_guess, CDOG_data, GPS_data,
                                                            transponder_coordinates_found, offset)
            inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(inversion_guess, CDOG_data, GPS_data,
                                                                                       transponder_coordinates_found, offset)

        times_guess, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates_found)
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
            two_pointer_index(offset, 0.6, CDOG_data, GPS_data, times_guess, transponder_coordinates_found, esv)
        )

        RMSE = np.sqrt(np.nanmean((GPS_full - CDOG_full) ** 2))
        if RMSE < best_rmse:
            best_rmse = RMSE
            best_lever = lever

        if k%10 == 0:
            print(k, RMSE*100*1515, offset, lever)
        old_offset = offset
        k+=1

    return best_lever, offset, inversion_guess

if __name__ == "__main__":
    true_offset = np.random.rand() * 9000 + 1000
    print(true_offset)
    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5

    CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(
        50000, time_noise, true_offset
    )
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    intitial_guess = CDOG + np.array([100, -100, -50], dtype=np.float64)

    lever, offset, inversion_guess = simulated_annealing(300, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others, intitial_guess)

    transponder_coordinates_found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    times_found, esv = calculateTimesRayTracing(inversion_guess, transponder_coordinates_found)

    offset = find_subint_offset(offset, CDOG_data, GPS_data, times_found, transponder_coordinates_found, esv)

    inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(inversion_guess, CDOG_data, GPS_data,
                                                                               transponder_coordinates_found, offset)

    print(np.linalg.norm(inversion_guess - CDOG) * 100, 'cm')
    print("Found Lever", lever, "Found Offset", offset)
    print("Actual Lever", gps1_to_transponder, "Actual Offset", true_offset)