"""
Goal of this file is to fix the unwrapping issue that occurs when there are big time jumps
    Also going to see if I can position the correct CDOG given the synthetic CDOG data
    with missing data

Should rename variables because right now its confusing

ESV not properly aligned
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from advancedGeigerMethod import calculateTimesRayTracing

#Function to index data in both sets according to an absolute time
def index_data(offset, data_DOG, GPS_time, travel_times, transponder_coordinates, true_offset = False):

    #Get the unwrapping of DOG data to start at the right initial value
    acoustic_DOG = np.unwrap(data_DOG[:, 1] * 2 * np.pi) / (2 * np.pi)

    #Determine the initial travel time
    if true_offset == True:
        #When offset is found -> the initial travel time is the first arrival time minus the offset
        acoustic_DOG += data_DOG[0,0] - offset
    else:
        #When finding alignment -> estimate that the initial travel time is the first calculated time
        acoustic_DOG += travel_times[0] - acoustic_DOG[0]

    # Get DOG/GPS indexed approximately with time (either floor or round)
    # times_DOG = np.round(data_DOG[:, 0] + data_DOG[:, 1])
    # times_GPS = np.round(GPS_time[0] + travel_times + offset)

#MOST RECENT ATTEMPTS AT SOMETHING NEW
    # acoustic_DOG = np.delete(acoustic_DOG, np.where(data_DOG < 0.01)[0], axis=0)
    # data_DOG = np.delete(data_DOG, np.where(data_DOG < 0.01)[0], axis=0)
    # acoustic_DOG = np.delete(acoustic_DOG, np.where(data_DOG[:, 1] > 0.99), axis=0)
    # data_DOG = np.delete(data_DOG, np.where(data_DOG[:, 1] > 0.99), axis=0)
    #
    # transponder_coordinates = np.delete(transponder_coordinates, np.where((GPS_time[0] + travel_times + offset) % 1 < 0.01), axis=0)
    # temp_travel = np.delete(travel_times, np.where((GPS_time[0] + travel_times + offset) % 1 < 0.01))
    # GPS_time = np.delete(GPS_time, np.where((GPS_time[0] + travel_times + offset) % 1 < 0.01))
    # travel_times = temp_travel
    # transponder_coordinates = np.delete(transponder_coordinates, np.where((GPS_time[0] + travel_times + offset) % 1 > 0.99), axis=0)
    # temp_travel = np.delete(travel_times, np.where((GPS_time[0] + travel_times + offset) % 1 > 0.99))
    # GPS_time = np.delete(GPS_time, np.where((GPS_time[0] + travel_times + offset) % 1 > 0.99))
    # travel_times = temp_travel
    #
    # GPS_time = GPS_time[np.newaxis,:]

    times_DOG = np.floor(data_DOG[:, 0] + data_DOG[:, 1])
    times_GPS = np.floor(GPS_time[0] + travel_times + offset)

    # Get unique times and corresponding acoustic data for DOG
    unique_times_DOG, indices_DOG = np.unique(times_DOG, return_index=True)
    unique_acoustic_DOG = acoustic_DOG[indices_DOG]

    # Get unique times and corresponding travel times for GPS (perhaps get rid of both repeated points)
    unique_times_GPS, indices_GPS = np.unique(times_GPS, return_index=True)
    unique_travel_times = travel_times[indices_GPS]
    unique_transponder_coordinates = transponder_coordinates[indices_GPS]

    # Create arrays for DOG and GPS data
    min_time = min(unique_times_GPS.min(), unique_times_DOG.min())
    max_time = max(unique_times_GPS.max(), unique_times_DOG.max())
    full_times = np.arange(min_time, max_time+1)
    dog_data = np.full(full_times.shape, np.nan)
    GPS_data = np.full(full_times.shape, np.nan)
    transponder_data = np.full((len(full_times),3), np.nan)

    # For dog_data
    indices_dog = np.searchsorted(full_times, unique_times_DOG)
    mask_dog = (indices_dog < len(full_times)) & (full_times[indices_dog] == unique_times_DOG)
    dog_data[indices_dog[mask_dog]] = unique_acoustic_DOG[mask_dog]

    # For GPS_data
    indices_gps = np.searchsorted(full_times, unique_times_GPS)
    mask_gps = (indices_gps < len(full_times)) & (full_times[indices_gps] == unique_times_GPS)
    GPS_data[indices_gps[mask_gps]] = unique_travel_times[mask_gps]
    transponder_data[indices_gps[mask_gps]] = unique_transponder_coordinates[mask_gps]

    #Remove nan values from the three arrays
    mask = ~np.isnan(dog_data) & ~np.isnan(GPS_data)
    dog_data = dog_data[mask]
    GPS_data = GPS_data[mask]
    transponder_data = transponder_data[mask]
    full_times = full_times[mask]

    return full_times, dog_data, GPS_data, transponder_data

def find_int_offset(data_DOG, GPS_time, travel_times, transponder_coordinates):
    #Set initial parameters
    offset = 0
    k = 0
    lag = np.inf

    #Loop through
    while lag != 0 and k<10:
        #Get indexed data according to offset
        dog_data, GPS_data = index_data(offset, data_DOG, GPS_time, travel_times, transponder_coordinates)[1:3]

        #Get fractional parts of the data
        GPS_fp = np.modf(GPS_data)[0]
        dog_fp = np.modf(dog_data)[0]

        #Find the cross-correlation between the fractional parts of the time series
        correlation = signal.correlate(dog_fp - np.mean(dog_fp), GPS_fp - np.mean(GPS_fp), mode="full", method="fft")
        lags = signal.correlation_lags(len(dog_fp), len(GPS_fp), mode="full")
        lag = lags[np.argmax(abs(correlation))]

        #Adjust the offset by the optimal lag
        offset+=lag
        k+=1
    return offset

#This works but is slow...
#If gets too small, then can run into issues with Geiger's Method
#   Could have an iteration technique where I add more decimals as the guess approaches a good value.
#but most importantly this is slow and need to become quicker
def find_subint_offset(offset, data_DOG, GPS_time, travel_times, transponder_coordinates):
    #Initalize values for loop
    l, u = offset-0.5, offset+0.5
    intervals = np.array([0.1, 0.01])
    best_offset = offset
    best_RMSE = np.inf

    #Iterate through each decimal offset (check this function to make sure it works properly)
    for interval in intervals:
        for lag in np.arange(l, u+interval, interval):
            #Round to prevent numpy float errors
            lag = np.round(lag, 4)
            dog_data, GPS_data = index_data(lag, data_DOG, GPS_time, travel_times, transponder_coordinates, True)[1:3]

            #If off by around an int then add that int in
            abs_diff = np.abs(dog_data - GPS_data)
            indices = np.where(abs_diff >= 0.9)
            dog_data[indices] += np.round(GPS_data[indices] - dog_data[indices])

            #Find minimum RMSE and set that as best offset
            diff_data = GPS_data - dog_data
            RMSE = np.sqrt(np.nanmean(diff_data**2))
            if RMSE < best_RMSE:
                best_offset = lag
                best_RMSE = RMSE
        l, u = best_offset - interval, best_offset + interval
    return best_offset

#Steps to alignment: Find offset, adjust for integer differences, return time series
def align(data_DOG, GPS_time, travel_times, transponder_coordinates):
    #Find offset, and get new time series corresponding with that offset
    offset = find_int_offset(data_DOG, GPS_time, travel_times, transponder_coordinates)
    sub_offset = find_subint_offset(offset, data_DOG, GPS_time, travel_times, transponder_coordinates)
    offset = sub_offset

    full_times, dog_data, GPS_data, transponder_data = index_data(offset, data_DOG, GPS_time, travel_times, transponder_coordinates, True)

    #If off by around an integer, adjust by that integer amount
    abs_diff = np.abs(dog_data - GPS_data)
    indices = np.where(abs_diff >= 0.9)
    dog_data[indices] += np.round(GPS_data[indices] - dog_data[indices])

    #Plot overlaying time series and the residual plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    """Make the bottom plot for 3 std around mean (say in paper that we plot 99% of data)"""

    ax1.scatter(full_times, dog_data, s=10, marker="x", label="Unwrapped/Adjusted Synthetic Dog Travel Time")
    ax1.scatter(full_times, GPS_data, s=1, marker="o", label="Calculated GPS Travel Times")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Arrivals in Absolute Time (s)")
    ax1.set_ylabel("Travel Times (s)")
    ax1.set_title(f"Synthetic travel times with offset: {offset}")

    diff_data = dog_data - GPS_data
    ax2.scatter(full_times, diff_data, s=1)
    ax2.set_xlabel("Absolute Time (s)")
    ax2.set_ylabel("Difference between calculated and unwrapped times (s)")
    ax2.set_title("Residual Plot")

    plt.show()

    return full_times, dog_data, GPS_data, transponder_data, offset

if __name__ == "__main__":
    data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG_noise_subint.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] * 2 * np.pi) / (2 * np.pi)

    transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder_noise_subint.mat')['xyz'].astype(float)
    GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder_noise_subint.mat')['time'].astype(float)

    CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])
    travel_times = calculateTimesRayTracing(CDOG, transponder_coordinates)[0]

    acoustic_DOG += travel_times[0] - acoustic_DOG[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.scatter(data_DOG[:, 0] + data_DOG[:, 1], acoustic_DOG, s=1, color='r', label="Synthetic Dog Travel Time")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Arrivals in Absolute Time (s)")
    ax1.set_ylabel("Unwrapped Travel Time (s)")
    ax1.set_title("Synthetic travel times")

    diff_plot_data = np.diff(acoustic_DOG)
    ax2.scatter(data_DOG[:len(data_DOG) - 1, 0] + data_DOG[:len(data_DOG) - 1, 1], diff_plot_data, s=1)
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Difference between unwrapped times (s)")
    ax2.set_title("Relative Change in DOG data")

    plt.show()

    plt.scatter(data_DOG[:, 0] + data_DOG[:, 1], acoustic_DOG, s=1, color='r', label="Synthetic Dog Travel Time")
    plt.scatter(GPS_time, travel_times, s=1, color='b', label="Calculated GPS travel times")
    plt.legend(loc="upper right")
    plt.xlabel("Absolute Time (s)")
    plt.ylabel("Travel Times (s)")
    plt.title("Pre-Alignment Time Series")
    plt.show()

    full_times, dog_data, GPS_data, transponder_data, offset = align(data_DOG, GPS_time, travel_times, transponder_coordinates)

    print(np.sqrt(np.nanmean((dog_data - GPS_data) ** 2)) * 100, "cm")


# sio.savemat("../../GPSData/Aligned_Synthetic.mat", {"full_times": full_times, "dog_data": dog_data, "GPS_data": GPS_data})


"""comments below"""

#Now add alignment to an iterative process with Gauss-Newton
#   After that add it to the iterative process of finding the transponder coordinates

#Correlation/convolution exercise [1,2,3,4,5,6] and [2,3,4,5,6,7]
#   Normal and FFT to frequency domain multiplication


"""deprecated functions"""