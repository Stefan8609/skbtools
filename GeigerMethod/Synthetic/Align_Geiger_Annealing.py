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
"""

data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG_noise.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:, 1] * 2 * np.pi) / (2 * np.pi)

transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder_noise.mat')['xyz'].astype(float)
GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder_noise.mat')['time'].astype(float)

CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])
travel_times = calculateTimesRayTracing(CDOG, transponder_coordinates)[0]

# TEST (I like it!) -> This starts the unwrapping at a reasonable value instead of 0 time
acoustic_DOG += travel_times[0] - acoustic_DOG[0]

full_times, dog_data, GPS_data = align(data_DOG, acoustic_DOG, GPS_time, travel_times)


