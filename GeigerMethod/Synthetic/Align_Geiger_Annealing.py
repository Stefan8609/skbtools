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
"""

data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG_noise.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:, 1] * 2 * np.pi) / (2 * np.pi)

transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder_noise.mat')['xyz'].astype(float)
GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder_noise.mat')['time'].astype(float)

CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])

travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

# TEST (I like it!) -> This starts the unwrapping at a reasonable value instead of 0 time
acoustic_DOG += travel_times[0] - acoustic_DOG[0]

full_times, dog_data, GPS_data, transponder_coordinates, esv_data = align(data_DOG, acoustic_DOG, GPS_time, travel_times, esv)

#Next add Geiger's Method suited for the given data

#I need to propogate transponder coordinates through as well

def computeJacobianRayTracing(guess, transponder_coordinates, times, sound_speed):
    # Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    diffs = transponder_coordinates - guess
    jacobian = -diffs / (times[:, np.newaxis] * (sound_speed[:, np.newaxis] ** 2))
    return jacobian

# def geigersMethod(guess, times_known, transponder_coordinates_Found):
#     epsilon = 10**-5
#     k=0
#     delta = 1
#     #Loop until change in guess is less than the threshold
#     while np.linalg.norm(delta) > epsilon and k<100:
#         jacobian = computeJacobianRayTracing(guess, transponder_coordinates_Found, times_guess, esv)
#         delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
#         guess = guess + delta
#         k+=1
#     return guess




