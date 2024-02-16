"""
Goal of this file is to fix the unwrapping issue that occurs when there are big time jumps
    Also going to see if I can position the correct CDOG given the synthetic CDOG data
    with missing data
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from advancedGeigerMethod import calculateTimesRayTracing

data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG_noise.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:,1] *2*np.pi) / (2*np.pi)

transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder_noise.mat')['xyz'].astype(float)
GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder_noise.mat')['time'].astype(float)

CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])
travel_times = calculateTimesRayTracing(CDOG, transponder_coordinates)[0]

#TEST (I like it!)
acoustic_DOG += travel_times[0] - acoustic_DOG[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.scatter(data_DOG[:,0] + data_DOG[:,1], acoustic_DOG, s=1, color='r', label="Synthetic Dog Travel Time")
ax1.legend(loc="upper right")
ax1.set_xlabel("Arrivals in Absolute Time (s)")
ax1.set_ylabel("Unwrapped Travel Time (s)")
ax1.set_title("Synthetic travel times")

diff_plot_data = np.diff(acoustic_DOG)
ax2.scatter(data_DOG[:len(data_DOG)-1,0] + data_DOG[:len(data_DOG)-1,1], diff_plot_data, s=1)
ax2.set_xlabel("Index")
ax2.set_ylabel("Difference between unwrapped times (s)")
ax2.set_title("Relative Change in DOG data")

plt.show()

#Make this capable of handling vectorization
def RMSE_offset(offset, travel_times, acoustic_DOG):
    # Get DOG indexed approximately with time
    times_DOG = np.round(data_DOG[:, 0] + data_DOG[:, 1])
    times_GPS = np.round(GPS_time[0] + travel_times + offset)

    # Get unique times and corresponding acoustic data for DOG
    unique_times_DOG, indices_DOG = np.unique(times_DOG, return_index=True)
    unique_acoustic_DOG = acoustic_DOG[indices_DOG]

    # Get unique times and corresponding travel times for GPS
    unique_times_GPS, indices_GPS = np.unique(times_GPS, return_index=True)
    unique_travel_times = travel_times[indices_GPS]

    # Create arrays for DOG and GPS data
    full_times = np.arange(min(unique_times_GPS.min(), unique_times_DOG.min()), max(unique_times_GPS.max(), unique_times_DOG.max()) + 1)
    dog_data = np.full(full_times.shape, np.nan)
    GPS_data = np.full(full_times.shape, np.nan)

    # Assign values to DOG and GPS data arrays
    dog_data[np.isin(full_times, unique_times_DOG)] = unique_acoustic_DOG
    GPS_data[np.isin(full_times, unique_times_GPS)] = unique_travel_times

    # Calculate RMSE
    RMSE = np.sqrt(np.nansum((np.modf(GPS_data)[0] - np.modf(dog_data)[0]) ** 2))
    return RMSE

#Vectorize this function to make it faster (and clean it up to make it more readable)
def find_offset():
    l, u = 0, 10000
    for interval in [100, 10, 1, .1]:
        mini = np.inf
        best_offset = 0
        while l<u:
            val = RMSE_offset(l, travel_times, acoustic_DOG)
            if val < mini:
                mini = val
                best_offset = l
            l += interval
        print(best_offset)
        l, u = best_offset - interval, best_offset + interval
    return best_offset

print(find_offset())



# mini = 10000
# mino=0
# for o in np.arange(4050, 7050, 100):
#     # if find_offset(o, travel_times, acoustic_DOG) < mini:
#     #     mini = find_offset(o, travel_times, acoustic_DOG)
#     #     print(mini, o)
#     #     mino = o
#     print(o, RMSE_offset(o, travel_times,acoustic_DOG))
# print(mino)


"""comments below"""

#There are repeated indices in data_DOG[:,0]
#   Unfortunate when trying to handle how to align data
#   For alignment I want to account for the fact that the offset could be something that is not an integer

#Could try to use absolute time and minimize the distance between nearest in GPS and DOG
#   This would allow for sub-integer offsets
#   Have a coupled algorithm that minimizes the time difference + corresponding arrival time?

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