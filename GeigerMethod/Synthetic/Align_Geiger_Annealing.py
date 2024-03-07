import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from advancedGeigerMethod import calculateTimesRayTracing

from Unwrap_Fix import align

"""
Goal of this file is to iterate through finding the CDOG with the following steps:
    1) Align the travel time series for a CDOG guess as close as possible with the unwrapped CDOG data
    2) Apply Geiger's method to the aligned data to find an improved guess of CDOG position
    3) Use Synthetic Annealing to find the best guess for the transducer location (with a constricted range of possibilities)
    4) Iterate back to the alignment phase with the new CDOG guess and travel time series
    5) After a certain number of iterations take the given CDOG location (determine validity by looking at residual distribution)
    
Gotta rename indices data_DOG and dog_data are not discernible names lol

Need a Geiger's Method which can accept data in the form that I have
    I.E. Aligned time series data (with corresponding GPS Travel times)
    
One thing about alignment is the scaling of the unwrapping with the first index of travel time. In reality, this first index
    will not correspond with the first unwrapping index. Hence, it will be scaled off by some factor. This may not be a problem
    and actually might make it easier for my algorithm to recognize the correct alignment
    
Problem with the fact that first index of dog and first index of travel times are always a match
    Need to find a way to counteract this - maybe compare difference between consecutive travel times?
    
IMPORTANT NOTE
GNSS time series is always corresponding with transducer output!
"""

data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG_noise_subint.mat')['tags'].astype(float)

transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder_noise_subint.mat')['xyz'].astype(float)
GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder_noise_subint.mat')['time'].astype(float)

CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])

# TEST (I like it!) -> This starts the unwrapping at a reasonable value instead of 0 time

#Next add Geiger's Method suited for the given data

def computeJacobianRayTracing(guess, transponder_coordinates, times, sound_speed):
    # Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    diffs = transponder_coordinates - guess
    jacobian = -diffs / (times[:, np.newaxis] * (sound_speed[:, np.newaxis] ** 2))
    return jacobian

def geigersMethod(guess, transponder_coordinates, data_DOG, GPS_time):
    epsilon = 10**-5
    k=0
    delta = 1

    #add noise to transponder_coordinates
    # transponder_coordinates += np.random.normal(0, 2*10**-2, (len(transponder_coordinates), 3))

    # Find initial times
    travel_times, esv = calculateTimesRayTracing(guess, transponder_coordinates)

    # Align Data Initially
    full_times, dog_data, GPS_data, transponder_data = align(data_DOG, GPS_time, travel_times,
                                                                       transponder_coordinates)
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<10:
        #Find times
        travel_times_guess, esv = calculateTimesRayTracing(guess, transponder_data)

        #Do Gauss-Newton Iteration
        jacobian = computeJacobianRayTracing(guess, transponder_data, travel_times_guess, esv)

        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (travel_times_guess - dog_data)
        guess = guess + delta
        print(np.sqrt(np.sum((CDOG - guess)**2)) * 100, "cm")
        k+=1
    return guess

#If guess is too far - iterate gauss newton a couple times
guess = np.array([-1971.9094551, 4505.73551826, -2007.85148619])

new_guess = geigersMethod(guess, transponder_coordinates, data_DOG, GPS_time)

new_guess = geigersMethod(new_guess, transponder_coordinates, data_DOG, GPS_time)


print(new_guess)
print(CDOG)

travel_times, esv = calculateTimesRayTracing(new_guess, transponder_coordinates)
full_times, dog_data, GPS_data, transponder_data = align(data_DOG, GPS_time, travel_times,
                                                         transponder_coordinates)
