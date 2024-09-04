"""
Module to align corrupted CDOG and GPS time series to a sub-integer lag
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from advancedGeigerMethod import calculateTimesRayTracing, findTransponder

# Function to extend the time series so that they can be aligned (given an offset)
def index_data(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates):

    # Get unwrapped CDOG data
    CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)

    # Set initial CDOG observed travel time is same as first calculated travel time
    CDOG_travel_times = CDOG_unwrap + travel_times[0] - CDOG_unwrap[0]

    # Format CDOG and GPS times to the closest integer below
    CDOG_times = np.floor(CDOG_data[:,0] + CDOG_data[:,1] - offset)
    GPS_times = np.floor(GPS_data + travel_times)

    # Get unique times and corresponding acoustic data for DOG
    unique_CDOG_times, CDOG_indices = np.unique(CDOG_times, return_index=True)
    unique_CDOG_travel_times = CDOG_travel_times[CDOG_indices]

    # Get unique times and corresponding travel times for GPS
    unique_GPS_times, indices_GPS = np.unique(GPS_times, return_index=True)
    unique_travel_times = travel_times[indices_GPS]
    unique_transponder_coordinates = transponder_coordinates[indices_GPS]

    # Create a full size array that has indices for all times covered by CDOG or GPS times
    min_time = min(unique_GPS_times.min(), unique_CDOG_times.min())
    max_time = max(unique_GPS_times.max(), unique_CDOG_times.max())
    full_times = np.arange(min_time, max_time+1)
    CDOG_full = np.full(full_times.shape, np.nan)
    GPS_full = np.full(full_times.shape, np.nan)
    transponder_full = np.full((len(full_times),3), np.nan)

    # Get CDOG travel times into the full time array at the correct indices
    CDOG_match = np.searchsorted(full_times, unique_CDOG_times)
    CDOG_mask = (CDOG_match < len(full_times)) & (full_times[CDOG_match] == unique_CDOG_times)
    CDOG_full[CDOG_match[CDOG_mask]] = unique_CDOG_travel_times[CDOG_mask]

    # Get GPS data into the respective full time arrays at the correct indices
    GPS_match = np.searchsorted(full_times, unique_GPS_times)
    GPS_mask = (GPS_match < len(full_times)) & (full_times[GPS_match] == unique_GPS_times)
    GPS_full[GPS_match[GPS_mask]] = unique_travel_times[GPS_mask]
    transponder_full[GPS_match[GPS_mask]] = unique_transponder_coordinates[GPS_mask]

    # Remove nan values from all arrays
    nan_mask = ~np.isnan(CDOG_full) & ~np.isnan(GPS_full)
    CDOG_full = CDOG_full[nan_mask]
    GPS_full = GPS_full[nan_mask]
    transponder_full = transponder_full[nan_mask]
    full_times = full_times[nan_mask]

    return full_times, CDOG_full, GPS_full, transponder_full

def find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates):
    # Set initial parameters
    offset = 0
    k = 0
    lag = np.inf

    # Loop through
    while lag != 0 and k<10:
        # Get indexed data according to offset
        CDOG_full, GPS_full = index_data(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates)[1:3]

        # Get fractional parts of the data
        GPS_fp = np.modf(GPS_full)[0]
        CDOG_fp = np.modf(CDOG_full)[0]

        # Find the cross-correlation between the fractional parts of the time series
        correlation = signal.correlate(CDOG_fp - np.mean(CDOG_fp), GPS_fp - np.mean(GPS_fp), mode="full", method="fft")
        lags = signal.correlation_lags(len(CDOG_fp), len(GPS_fp), mode="full")
        lag = lags[np.argmax(abs(correlation))]

        # Adjust the offset by the optimal lag
        offset+=lag
        k+=1
        print(offset)
    return offset


CDOG = sio.loadmat('../../GPSData/Realistic_CDOG_loc_noise_subint_new.mat')['xyz'][0].astype(float)
CDOG_data = sio.loadmat('../../GPSData/Realistic_CDOG_noise_subint_new.mat')['tags'].astype(float)
# transponder_coordinates = sio.loadmat('../../GPSData/Realistic_transponder_noise_subint_new.mat')['xyz'].astype(float)
GPS_data = sio.loadmat('../../GPSData/Realistic_GPS_noise_subint_new.mat')['time'][0].astype(float)
GPS_Coordinates = sio.loadmat('../../GPSData/Realistic_GPS_noise_subint_new.mat')['xyz'].astype(float)

# Add noise to GPS Coordinates
position_noise = 2*10**-2
GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

# Find transponder coordinates from noisy GPS
gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

# Calculate travel times given noise in GPS position
travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

# Obtain offset
offset = find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates)

full_times, CDOG_full, GPS_full, transponder_full = index_data(offset, CDOG_data, GPS_data,
                                                               travel_times, transponder_coordinates)

plt.scatter(full_times, CDOG_full, s=1)
plt.scatter(full_times, GPS_full, s=1)
plt.show()