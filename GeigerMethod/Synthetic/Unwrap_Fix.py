"""
Goal of this file is to fix the unwrapping issue that occurs when there are big time jumps
    Also going to see if I can position the correct CDOG given the synthetic CDOG data
    with missing data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from advancedGeigerMethod import calculateTimesRayTracing

data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:,1] *2*np.pi) / (2*np.pi)

transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder.mat')['xyz'].astype(float)
GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder.mat')['time'].astype(float)

CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])
travel_times = calculateTimesRayTracing(CDOG, transponder_coordinates)[0]

plt.scatter(GPS_time + travel_times, travel_times, s=1, color='b', label="GPS Estimated Travel Time")
plt.scatter(data_DOG[:,0] + data_DOG[:,1], acoustic_DOG, s=1, color='r', label="Synthetic Dog Travel Time")
plt.legend(loc="upper right")
plt.xlabel("Absolute Time (s)")
plt.ylabel("Travel Time (s)")
plt.title("Synthetic travel times without offset or unwrapping fix")
plt.show()

times = data_DOG[:,0]
full_times = np.arange(min(times), max(times)+1)
dog_data = np.full(full_times.shape, np.nan)
indices = np.where(np.isin(full_times, times))
dog_data[indices] = acoustic_DOG[:len(indices[0])]

print(dog_data[14000:14500])
print(len(dog_data))

plt.scatter(full_times, dog_data, s=1)
plt.show()

#I want an algorithm that can work with non-integers on the time-axis
#   That can find an offset to sub-integer precision
#   That can run very fast
#   Can iterate to update with an ever better position of the CDOG for finding the travel times

#Could align by finding similar points of derivatives in each time series
#   Align with derivative closest in scale, then scale according to unwrapping error
#   I think minimizing derivative difference would work

#Gonna start with cross-correlation
    #Find a good way to window data -- Maybe by minimum slope and then interpolate and align that window

#   Also look at dynamic time warping algorithm - May not work because it warps original time series