"""
Module to align corrupted CDOG and GPS time series to a sub-integer lag
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from advancedGeigerMethod import calculateTimesRayTracing, findTransponder

# Function to extend the time series so that they can be aligned (given an offset)
# Need a true offset version
def index_data(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv):

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
    unique_esv = esv[indices_GPS]

    # Create a full size array that has indices for all times covered by CDOG or GPS times
    min_time = min(unique_GPS_times.min(), unique_CDOG_times.min())
    max_time = max(unique_GPS_times.max(), unique_CDOG_times.max())
    full_times = np.arange(min_time, max_time+1)
    CDOG_full = np.full(full_times.shape, np.nan)
    GPS_full = np.full(full_times.shape, np.nan)
    transponder_full = np.full((len(full_times),3), np.nan)
    esv_full = np.full(full_times.shape, np.nan)

    # Get CDOG travel times into the full time array at the correct indices
    CDOG_match = np.searchsorted(full_times, unique_CDOG_times)
    CDOG_mask = (CDOG_match < len(full_times)) & (full_times[CDOG_match] == unique_CDOG_times)
    CDOG_full[CDOG_match[CDOG_mask]] = unique_CDOG_travel_times[CDOG_mask]

    # Get GPS data into the respective full time arrays at the correct indices
    GPS_match = np.searchsorted(full_times, unique_GPS_times)
    GPS_mask = (GPS_match < len(full_times)) & (full_times[GPS_match] == unique_GPS_times)
    GPS_full[GPS_match[GPS_mask]] = unique_travel_times[GPS_mask]
    transponder_full[GPS_match[GPS_mask]] = unique_transponder_coordinates[GPS_mask]
    esv_full[GPS_match[GPS_mask]] = unique_esv[GPS_mask]

    # Remove nan values from all arrays
    nan_mask = ~np.isnan(CDOG_full) & ~np.isnan(GPS_full)
    CDOG_full = CDOG_full[nan_mask]
    GPS_full = GPS_full[nan_mask]
    transponder_full = transponder_full[nan_mask]
    esv_full = esv_full[nan_mask]
    full_times = full_times[nan_mask]

    return full_times, CDOG_full, GPS_full, transponder_full, esv_full

def find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates, esv, start=0, best=0, best_RMSE=np.inf):
    # Set initial parameters
    offset = start
    err_int = 1000
    k = 0
    lag = np.inf

    while lag != 0 and k < 10:
        # Get indexed data according to offset
        CDOG_full, GPS_full = index_data(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)[1:3]
        # Get fractional parts of the data
        GPS_fp = np.modf(GPS_full)[0]
        CDOG_fp = np.modf(CDOG_full)[0]
        # Find the cross-correlation between the fractional parts of the time series
        correlation = signal.correlate(CDOG_fp - np.mean(CDOG_fp), GPS_fp - np.mean(GPS_fp), mode="full", method="fft")
        lags = signal.correlation_lags(len(CDOG_fp), len(GPS_fp), mode="full")
        lag = lags[np.argmax(abs(correlation))]
        # Adjust the offset by the optimal lag
        offset += lag
        k += 1
        # Conditional check to prevent false positives
        if offset < 0:
            offset = err_int
            err_int += 500
            lag = np.inf

    # Conditional check to ensure the resulting value is reasonable (and to prevent stack overflows)
    if start > 20000:
        print("Error - No true offset found")
        return best

    # If RMSE is too high, rerun the algorithm to see if it can be improved
    CDOG_full, GPS_full = index_data(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)[1:3]
    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])
    RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100
    if RMSE > 5000:
        if RMSE < best_RMSE:
            best = offset
            best_RMSE = RMSE
        start += 1000
        return find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates, esv, start, best, best_RMSE)
    return offset

def find_subint_offset(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv):
    #Initalize values for loop
    l, u = offset-0.5, offset+0.5
    intervals = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001])
    best_offset = offset
    best_RMSE = np.inf

    #Iterate through each decimal offset (check this function to make sure it works properly)
    for interval in intervals:
        for lag in np.arange(l, u+interval, interval):
            #Round to prevent numpy float errors
            lag = np.round(lag, 10)
            CDOG_full, GPS_full = index_data(lag, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)[1:3]

            #If off by around an int then add that int in
            abs_diff = np.abs(CDOG_full - GPS_full)
            indices = np.where(abs_diff >= 0.9)
            CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

            #Find minimum RMSE and set that as best offset
            diff_data = GPS_full - CDOG_full
            RMSE = np.sqrt(np.nanmean(diff_data**2))
            if RMSE < best_RMSE:
                best_offset = lag
                best_RMSE = RMSE
        l, u = best_offset - interval, best_offset + interval
    return best_offset

def two_pointer_index(offset, threshhold, CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv):
    """Module to index closest data points against each other given correct offset"""
    # Initialize information
    CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)
    CDOG_travel_times = CDOG_unwrap + GPS_travel_times[0] - CDOG_unwrap[0]

    CDOG_times = CDOG_data[:, 0] + CDOG_data[:, 1] - offset
    GPS_times = GPS_data + GPS_travel_times

    # Initialize loop conditions
    CDOG_pointer = 0
    GPS_pointer = 0

    # Initialize lists to store results
    CDOG_clock = np.array([])
    GPS_clock = np.array([])
    GPS_full = np.array([])
    transponder_coordinates_full = np.empty((0,3))
    esv_full = np.array([])

    while CDOG_pointer < len(CDOG_data) and GPS_pointer < len(GPS_data):
        if np.abs(GPS_times[GPS_pointer] - CDOG_times[CDOG_pointer]) < threshhold:
            CDOG_clock = np.append(CDOG_clock, CDOG_times[CDOG_pointer])
            GPS_clock = np.append(GPS_clock, GPS_times[GPS_pointer])
            GPS_full = np.append(GPS_full, GPS_travel_times[GPS_pointer])
            transponder_coordinates_full = np.vstack((transponder_coordinates_full, transponder_coordinates[GPS_pointer]))
            esv_full = np.append(esv_full, esv[GPS_pointer])

            CDOG_pointer += 1
            GPS_pointer += 1
        elif GPS_times[GPS_pointer] < CDOG_times[CDOG_pointer]:
            GPS_pointer += 1
        else:
            CDOG_pointer += 1

    #Best travel times for known offset
    CDOG_full = CDOG_clock - (GPS_clock - GPS_full)

    return CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full


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

    # Obtain offset
    offset = find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
    print(offset)

    full_times, CDOG_full, GPS_full, transponder_full, esv_full = index_data(offset, CDOG_data, GPS_data,
                                                                             travel_times, transponder_coordinates, esv)

    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    plt.scatter(full_times, CDOG_full, s=5, marker='x')
    plt.scatter(full_times, GPS_full, s=1)
    plt.show()

    print(np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100, "cm")

    full_times, CDOG_full, GPS_full, transponder_full, esv_full = index_data(1200, CDOG_data, GPS_data,
                                                                   travel_times, transponder_coordinates, esv)

    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])
    print(np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100, "cm")

    """
    Occationally getting offset wrong by 1 - I believe to be an issue with indexing (removing points
        might misaline by 1 sometimes because its not able to index or find a better solution cause important points
        are missing) - RMSE is still the best when the offset is absolutely correct
        
        Maybe round is better than floor? Sometimes one point is really far off raising residuals
            This is probably a situation where the sub-int is a gonna freak out (maybe find a way to remove)
            We can empirically tell this point is wrong so removal is not bad
            
    Investigate using cubic spline for interpolation then using the algorithm to find alignment
        May remove issues with missing data points?
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    """Make the bottom plot for 3 std around mean (say in paper that we plot 99% of data)"""

    ax1.scatter(full_times, CDOG_full, s=10, marker="x", label="Unwrapped/Adjusted Synthetic Dog Travel Time")
    ax1.scatter(full_times, GPS_full, s=1, marker="o", label="Calculated GPS Travel Times")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Arrivals in Absolute Time (s)")
    ax1.set_ylabel("Travel Times (s)")
    ax1.set_title(f"Synthetic travel times with offset: {offset}")

    diff_data = CDOG_full - GPS_full
    ax2.scatter(full_times, diff_data, s=1)
    ax2.set_xlabel("Absolute Time (s)")
    ax2.set_ylabel("Difference between calculated and unwrapped times (s)")
    ax2.set_title("Residual Plot")

    plt.show()