"""
Goal of this file is to fix the unwrapping issue that occurs when there are big time jumps
    Also going to see if I can position the correct CDOG given the synthetic CDOG data
    with missing data

Should rename variables because right now its confusing

Should make this alignment process more efficient (right now it too slow)
    Possibly could do this using cross correlation ? Cuz this can use FFT
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
#Make functions more flexible with parameters (basically just parameterize all variables that are inputs)
def RMSE_offset(offset):
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
    min_time = min(unique_times_GPS.min(), unique_times_DOG.min())
    max_time = max(unique_times_GPS.max(), unique_times_DOG.max())
    full_times = np.arange(min_time, max_time+1)
    dog_data = np.full(full_times.shape, np.nan)
    GPS_data = np.full(full_times.shape, np.nan)

    # Assign values to DOG and GPS data arrays
    dog_data[np.isin(full_times, unique_times_DOG)] = unique_acoustic_DOG
    GPS_data[np.isin(full_times, unique_times_GPS)] = unique_travel_times

    #Remove nan values from the three arrays
    mask = ~np.isnan(dog_data) & ~np.isnan(GPS_data)
    dog_data = dog_data[mask]
    GPS_data = GPS_data[mask]
    full_times = full_times[mask]

    # Calculate RMSE
    RMSE = np.sqrt(np.nanmean((np.modf(GPS_data)[0] - np.modf(dog_data)[0]) ** 2))
    return RMSE, full_times, dog_data, GPS_data

#Only use this up to integer differences
def find_offset():
    l, u = 0, 10000
    intervals = np.array([100, 10, 1])
    best_offset = 0
    for interval in intervals:
        offsets = np.arange(l, u, interval)
        vals = np.array([RMSE_offset(offset)[0] for offset in offsets])
        best_index = np.argmin(vals)
        best_offset = offsets[best_index]
        l, u = best_offset - interval, best_offset + interval
    return best_offset

#Steps to alignment: Find offset, adjust for integer differences
def align():

    return


offset = find_offset()

#Use offset found from find_offset in the future
full_times, dog_data, GPS_data = RMSE_offset(offset)[1:]

for i in range(len(full_times)):
    #If off by around an integer, adjust by that integer amount
    if np.abs(dog_data[i] - GPS_data[i]) >= .9:
        dog_data[i] += np.round(GPS_data[i] - dog_data[i])

print(np.sqrt(np.nanmean((dog_data - GPS_data)**2))*1515*100, "cm")

diff_data = dog_data - GPS_data

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

ax1.scatter(full_times, dog_data, s=10, marker="x", label="Unwrapped/Adjusted Synthetic Dog Travel Time")
ax1.scatter(full_times, GPS_data, s=1, marker="o", label="Calculated GPS Travel Times")
ax1.legend(loc="upper right")
ax1.set_xlabel("Arrivals in Absolute Time (s)")
ax1.set_ylabel("Travel Times (s)")
ax1.set_title("Synthetic travel times")

ax2.scatter(full_times, diff_data, s=1)
ax2.set_xlabel("Absolute Time (s)")
ax2.set_ylabel("Difference between calculated and unwrapped times (s)")
ax2.set_title("Residual Plot")

plt.show()


sio.savemat("../../GPSData/Aligned_Synthetic.mat", {"full_times": full_times, "dog_data": dog_data, "GPS_data": GPS_data})


"""comments below"""

#Now add alignment to an iterative process with Gauss-Newton
#   After that add it to the iterative process of finding the transponder coordinates

#Correlation/convolution exercise [1,2,3,4,5,6] and [2,3,4,5,6,7]
#   Normal and FFT to frequency domain multiplication