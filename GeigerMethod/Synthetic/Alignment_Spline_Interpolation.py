"""
This is a module for testing the validity of using spline interpolation of the time series
    to estimate missing data points and then using the alignment algorithm with the filled in
    data - this might help resolve the issue missing data in the CDOG time series

This might not be a needed path atm - may return here later
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal, interpolate
from advancedGeigerMethod import calculateTimesRayTracing, findTransponder

def index_data_interpolate(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates):

    # Get unwrapped CDOG data
    CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)

    # Set initial CDOG observed travel time is same as first calculated travel time
    CDOG_travel_times = CDOG_unwrap + travel_times[0] - CDOG_unwrap[0]

    # Format CDOG and GPS times to the closest integer below
    CDOG_times = np.round(CDOG_data[:,0] + CDOG_data[:,1] - offset)
    GPS_times = np.round(GPS_data + travel_times)

    # Get unique times and corresponding acoustic data for DOG
    unique_CDOG_times, CDOG_indices = np.unique(CDOG_times, return_index=True)
    unique_CDOG_travel_times = CDOG_travel_times[CDOG_indices]

    # Get unique times and corresponding travel times for GPS
    unique_GPS_times, indices_GPS = np.unique(GPS_times, return_index=True)
    unique_travel_times = travel_times[indices_GPS]
    unique_transponder_coordinates = transponder_coordinates[indices_GPS]

    """
    start interpolation test
    Might want to make my own interpolator based on first and second derivatives at break point
    (make assumption that they are relatively constant)
    """
    #Figure out how to adjust the data integer wise without messing everything up...
    # abs_diff = np.abs(unique_CDOG_travel_times - unique_GPS_times)
    # indices = np.where(abs_diff >= 0.9)
    # CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    cs = interpolate.Akima1DInterpolator(unique_CDOG_times, unique_CDOG_travel_times)

    xs=np.arange(min(unique_CDOG_times), max(unique_CDOG_times), 0.1)

    plt.scatter(unique_CDOG_times, unique_CDOG_travel_times, marker="x", s=1)
    plt.scatter(unique_GPS_times, unique_travel_times, s=1)
    plt.plot(xs, cs(xs), color="r")
    plt.show()

    exit()
    """
    End interpolation test
    """

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


if __name__ == "__main__":
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

    full_times, CDOG_full, GPS_full, transponder_full = index_data_interpolate(0, CDOG_data, GPS_data,
                                                                   travel_times, transponder_coordinates)
