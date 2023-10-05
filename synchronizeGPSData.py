'''
This script performs the following tasks:
1. Reads multiple GPS data files in .mat format.
2. Aligns all the GPS data to a common time vector, interpolating missing values as NaN.
3. Saves each aligned matrix to a new .mat file.
4. Calculates the average of the aligned matrices and saves it as a new .mat file.

Dependencies: numpy, scipy.io

Written by Thalia Gueroult
'''

import scipy.io as sio
import numpy as np

# File paths for the input .mat files
files = [
    'GPSData/Unit1-camp_bis.mat',
    'GPSData/Unit2-camp_bis.mat',
    'GPSData/Unit3-camp_bis.mat',
    'GPSData/Unit4-camp_bis.mat'
]

all_combined_times = []

# Load all 'days' and 'times' data from each file and combine them
for file in files:
    data = sio.loadmat(file)
    days = data['days'].flatten()
    times = data['times'].flatten()
    combined_times = days * 24 * 3600 + times  # Convert days to seconds and add to times
    all_combined_times.extend(combined_times)

# Identify the minimum and maximum time for alignment
min_time = int(np.min(all_combined_times))
max_time = int(np.max(all_combined_times))

# Create a common time matrix
common_times = np.arange(min_time, max_time + 1)

# Initialize lists to store aligned data
aligned_lats, aligned_lons, aligned_elevs, aligned_xs, aligned_ys, aligned_zs = [], [], [], [], [], []

# For each file, create an aligned matrix
for file in files:
    data = sio.loadmat(file)
    days = data['days'].flatten()
    times = data['times'].flatten()
    combined_times = days * 24 * 3600 + times

    # Create empty arrays for each variable
    aligned_lat = np.full(common_times.shape, np.nan)
    aligned_lon = np.full(common_times.shape, np.nan)
    aligned_elev = np.full(common_times.shape, np.nan)
    aligned_x = np.full(common_times.shape, np.nan)
    aligned_y = np.full(common_times.shape, np.nan)
    aligned_z = np.full(common_times.shape, np.nan)

    # Indices where the data exist
    indices = np.isin(common_times, combined_times)

    # Fill the aligned arrays
    aligned_lat[indices] = data['lat'].flatten()
    aligned_lon[indices] = data['lon'].flatten()
    aligned_elev[indices] = data['elev'].flatten()
    aligned_x[indices] = data['x'].flatten()
    aligned_y[indices] = data['y'].flatten()
    aligned_z[indices] = data['z'].flatten()

    # Add to global list
    aligned_lats.append(aligned_lat)
    aligned_lons.append(aligned_lon)
    aligned_elevs.append(aligned_elev)
    aligned_xs.append(aligned_x)
    aligned_ys.append(aligned_y)
    aligned_zs.append(aligned_z)

# Convert the common aligned times to days and seconds
aligned_days = common_times // (24 * 3600)
aligned_secs = common_times % (24 * 3600)

# Re-save each aligned matrix
for idx, file in enumerate(files):
    base_name = 'gpsData/' + file.split('/')[-1].split('.')[0]
    new_file_name = base_name + "_GPStime_synchro.mat"
    data_dict = {
        'days': aligned_days,
        'times': aligned_secs,
        'lat': aligned_lats[idx],
        'lon': aligned_lons[idx],
        'elev': aligned_elevs[idx],
        'x': aligned_xs[idx],
        'y': aligned_ys[idx],
        'z': aligned_zs[idx]
    }
    sio.savemat(new_file_name, data_dict)
    print(f"Saved data to {new_file_name}")

# Create a 5th matrix that is the average of the 4 aligned matrices
average_lat = np.nanmean(np.array(aligned_lats), axis=0)
average_lon = np.nanmean(np.array(aligned_lons), axis=0)
average_elev = np.nanmean(np.array(aligned_elevs), axis=0)
average_x = np.nanmean(np.array(aligned_xs), axis=0)
average_y = np.nanmean(np.array(aligned_ys), axis=0)
average_z = np.nanmean(np.array(aligned_zs), axis=0)

# Save this 5th matrix to a .mat file
average_data_dict = {
    'days': aligned_days,
    'times': aligned_secs,
    'lat': average_lat,
    'lon': average_lon,
    'elev': average_elev,
    'x': average_x,
    'y': average_y,
    'z': average_z
}

sio.savemat("GPSData/Average_GPStime_synchro.mat", average_data_dict)
print("Saved average data to Average_GPStime_synchro.mat")

print(aligned_xs[1])