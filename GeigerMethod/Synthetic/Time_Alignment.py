"""
Python file to investigate the affects of the time alignment file
Trying to mimic the CDOG file that we have, and the GPS data that we have

GPS will be offset from CDOG by travel time
Data sets will be offset by some random amount of time
Swaths of CDOG data points are removed at random

Could use GPS data from our experiment at some point
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from RigidBodyMovementProblem import findRotationAndDisplacement
from pymap3d import geodetic2ecef

esv_table = sio.loadmat('../../GPSData/global_table_esv.mat')
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

def find_esv(beta, dz):
    idx_closest_dz = np.searchsorted(dz_array, dz, side="left")
    idx_closest_dz = np.clip(idx_closest_dz, 0, len(dz_array)-1)
    idx_closest_beta = np.searchsorted(angle_array, beta, side="left")
    idx_closest_beta = np.clip(idx_closest_beta, 0, len(angle_array)-1)
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv

def calculateTimesRayTracing(guess, transponder_coordinates):
    hori_dist = np.sqrt((transponder_coordinates[:, 0] - guess[0])**2 + (transponder_coordinates[:, 1] - guess[1])**2)
    abs_dist = np.linalg.norm(transponder_coordinates - guess, axis=1)
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz)
    times = abs_dist / esv
    return times, esv

def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    xs, ys, zs = gps1_to_others.T
    initial_transponder = gps1_to_transponder
    n = len(GPS_Coordinates)
    transponder_coordinates = np.zeros((n, 3))
    for i in range(n):
        new_xs, new_ys, new_zs = GPS_Coordinates[i].T
        R_mtrx, d = findRotationAndDisplacement(np.array([xs,ys,zs]), np.array([new_xs, new_ys, new_zs]))
        transponder_coordinates[i] = np.matmul(R_mtrx, initial_transponder) + d
    return transponder_coordinates

#Initialize Bermuda GPS
def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    # condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)
    # time_GNSS = datetimes[condition_GNSS]/3600
    # x,y,z = data['x'].flatten()[condition_GNSS], data['y'].flatten()[condition_GNSS], data['z'].flatten()[condition_GNSS]

    time_GNSS = datetimes/3600
    x,y,z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return time_GNSS, x,y,z

paths = [
    '../../GPSData/Unit1-camp_bis.mat',
    '../../GPSData/Unit2-camp_bis.mat',
    '../../GPSData/Unit3-camp_bis.mat',
    '../../GPSData/Unit4-camp_bis.mat'
]

all_data = [load_and_process_data(path) for path in paths]
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

filtered_data = []
for datetimes, x, y, z in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append([np.array(datetimes)[mask], np.array(x)[mask], np.array(y)[mask], np.array(z)[mask]])
filtered_data = np.array(filtered_data)

#Initialize Coordinates in form of Geiger's Method
GPS_Coordinates = np.zeros((len(filtered_data[0,0]),4,3))
for i in range(len(filtered_data[0,0])):
    for j in range(4):
        GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
        GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
        GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

CDOG = [31.46356091, 291.29859266, -5271.47395559]
CDOG = np.array(geodetic2ecef(CDOG[0], CDOG[1], CDOG[2])) - np.mean(GPS_Coordinates, axis=0)[0]

GPS_Coordinates = GPS_Coordinates - np.mean(GPS_Coordinates, axis=0)
gps1_to_others = np.array([[0,0,0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195, 0.04880436]])
initial_lever_guess = np.array([-12.4, 15.46, -15.24])
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)

# CDOG = [-1000, -1000, -5000]

# offset = 1000
travel_times = calculateTimesRayTracing(CDOG, transponder_coordinates)[0]
GPS_time = filtered_data[0,0] * 3600 - 85185

#Add some noise too
CDOG_time = travel_times + GPS_time + np.random.normal(0,2*10**-5, len(GPS_time)) #+ offset
CDOG_remain, CDOG_int = np.modf(CDOG_time)

"""Remove values at random indices"""
# removal_indices = np.random.choice(len(CDOG_time), 10000, replace=False)
# removal_indices = np.s_[10000:10500]
# print(removal_indices)
# removed_unwrapped = np.unwrap(CDOG_remain[10000:10500] * 2 * np.pi)/(2*np.pi)
# travel_times = np.delete(travel_times, removal_indices)
# CDOG_remain = np.delete(CDOG_remain, removal_indices)
# CDOG_time = np.delete(CDOG_time, removal_indices)

acoustic_DOG = np.unwrap(CDOG_remain * 2 * np.pi) / (2*np.pi)  #Numpy page describes how unwrap works

test = acoustic_DOG + travel_times[0] - CDOG_remain[0]

# print(test[57900:57925])
# print(travel_times[57900:57925])
# print(CDOG_time[57900:57925])
# print(GPS_Coordinates[57900:57925])
plt.scatter(CDOG_time, test - travel_times, s=1, color='b', label = "Unwrapped CDOG data minus travel time")
plt.scatter(CDOG_time, test, s=1, color='r', label = "Unwrapped CDOG data")
plt.legend(loc="upper right")
plt.xlabel("Absolute Time (s)")
plt.ylabel("Travel Time (s)")

# plt.scatter(list(range(len(CDOG_time))), CDOG_time, s=1)
# plt.scatter(GPS_Coordinates[:,0,0],GPS_Coordinates[:,0,1], s=1)
# plt.scatter(CDOG[0], CDOG[1])
plt.show()

offset = 5600
CDOG_int += offset

CDOG_mat = np.stack((CDOG_int, CDOG_remain), axis=0)
CDOG_mat = CDOG_mat.T

#Remove random indices from CDOG data
for i in range(10):
    length_to_remove = np.random.randint(50, 500)
    start_index = np.random.randint(0, len(CDOG_mat) - length_to_remove + 1)  # Start index cannot exceed len(array) - max_length
    indices_to_remove = np.arange(start_index, start_index + length_to_remove)
    CDOG_mat = np.delete(CDOG_mat, indices_to_remove, axis=0)

mat_unwrapped = np.unwrap(CDOG_mat[:,1] * 2 * np.pi) / (2*np.pi)  #Numpy page describes how unwrap works
print(mat_unwrapped)
plt.scatter(CDOG_mat[:,0] + CDOG_mat[:,1], mat_unwrapped, s=1)
plt.show()

#Save the synthetic to a matlabfile
sio.savemat("../../GPSData/Synthetic_CDOG_noise.mat", {"tags":CDOG_mat})

#Save transponder data
sio.savemat("../../GPSData/Synthetic_transponder_noise.mat", {"time":GPS_time, "xyz": transponder_coordinates})

"""
Find a way to save CDOG_time to a matlab file in the same way that the CDOG data is actually saved
Try to get the data to mimic what we see in the real data

Currently x,y,z of dog is not very good (the ecef coords don't have z correspondign to depth)

Sometime when indices are removed the CDOG data - travel time is given integers instead of 0
This is due to wrapping -- Jumps are too big i.e. 5.7 to 5.01 goes to 6.01

    This also sometimes happens when we are missing large swaths of GPS data (This should only be a
        synthetic problem because synthetic CDOG is based on the travel times given)

    How do I automatically find whether these jumps are too big and correct for it when data is missing?
    
    Could interpolate data for missing CDOG points -- Find the closest that matches all GPS positions
    
    Plot wrapped time versus absolute time -- From here should be able to find a way to correct for 
    false movements
    
How do I account for the wrapping when there are big jumps from missing data points?
"""