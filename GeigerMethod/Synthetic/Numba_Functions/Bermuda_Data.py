import numpy as np
import scipy.io as sio

from Numba_Geiger import findTransponder, calculateTimesRayTracing, calculateTimesRayTracingReal, find_esv
from Numba_xAline_Geiger import initial_geiger, transition_geiger, final_geiger
from Numba_xAline import two_pointer_index, find_subint_offset, find_int_offset
from Numba_xAline_Annealing import simulated_annealing

import matplotlib.pyplot as plt
from pymap3d import geodetic2ecef, ecef2geodetic

esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')


"""
Process:
    Load the data in the form of CDOG_data, GPS_data, GPS_coordinates
    Need to figure out how to find the depth and angle for ray-tracing
    
    Need to configure what the coordinate systems are for each different type of data
        GET absolute distance using ECEF, and vertical distance from geodetic
        
    Need to get elevation of transducer (how?) Convert ecef to geodetic for transducer?
"""

#Load GNSS Data during the time of expedition (25 through 40.9) hours
def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    # condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)

    time_GNSS = datetimes
    x,y,z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()

    return time_GNSS, x,y,z

paths = [
    '../../../GPSData/Unit1-camp_bis.mat',
    '../../../GPSData/Unit2-camp_bis.mat',
    '../../../GPSData/Unit3-camp_bis.mat',
    '../../../GPSData/Unit4-camp_bis.mat'
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

#Initialize time-tagged data for GPS and CDOG
GPS_data = filtered_data[0, 0, :]
CDOG_data = sio.loadmat('../../../GPSData/DOG3-camp.mat')['tags'].astype(float)

lat = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lat'].flatten()
lon = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lon'].flatten()
elev = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['elev'].flatten()


CDOG_guess_geodetic = np.array([np.mean(lat), np.mean(lon), np.mean(elev)]) + np.array([0, 0, -5200])
CDOG_guess = np.array(geodetic2ecef(CDOG_guess_geodetic[0], CDOG_guess_geodetic[1], CDOG_guess_geodetic[2]))
print("ECEF", CDOG_guess, "GEODETIC", CDOG_guess_geodetic)

gps1_to_others = np.array([[0.0,0.0,0.0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195, 0.04880436]])

#Scale GPS Clock slightly
GPS_data = GPS_data - 68826

initial_lever_guess = np.array([-12.4, 15.46, -15.24])
offset = 2000

print("GPS DATA:", GPS_data)
print("CDOG DATA:", CDOG_data[:,0])

transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)
GPS_travel_times, esv= calculateTimesRayTracingReal(CDOG_guess, transponder_coordinates)

print("MODELLED TRAVEL TIMES:", GPS_travel_times)

CDOG_data[:, 1] = CDOG_data[:, 1]/1e9

GPS_travel_times, esv = calculateTimesRayTracingReal(CDOG_guess, transponder_coordinates)

CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = two_pointer_index(offset, 0.4,
                                                                                                       CDOG_data, GPS_data, GPS_travel_times,
                                                                                                       transponder_coordinates, esv, exact=True)
offset = find_int_offset(CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv)
print("INT OFFSET", offset)
offset = find_subint_offset(offset, CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv)
print("SUBINT OFFSET", offset)

CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = two_pointer_index(offset, 0.4,
                                                                                                       CDOG_data, GPS_data, GPS_travel_times,
                                                                                                       transponder_coordinates, esv, exact=True)

RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100
print(f"RMSE: {RMSE} cm for offset {offset} and initial CDOG")

inversion_guess, offset = initial_geiger(CDOG_guess, CDOG_data, GPS_data, transponder_coordinates)

print(offset)
GPS_travel_times, esv = calculateTimesRayTracingReal(inversion_guess, transponder_coordinates)

CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = two_pointer_index(offset, 0.4,
                                                                                                       CDOG_data, GPS_data, GPS_travel_times,
                                                                                                       transponder_coordinates, esv, exact=True)

RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100
print(f"RMSE: {RMSE} cm for offset {offset} and inversion CDOG")

#Figure with 2 plots arranged vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
# axes[0].scatter(CDOG_clock, CDOG_full, s=10, marker="x", label="CDOG")
axes[0].scatter(CDOG_clock, CDOG_clock - (GPS_clock - GPS_full), s=10, marker="x", label="CDOG")
axes[0].scatter(GPS_clock, GPS_full, s=10, marker="x", label="GPS")
axes[0].set_title("Travel Times")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Travel Time (s)")
axes[0].legend()

unwrap = np.unwrap(CDOG_data[:,1]*2*np.pi) / (2*np.pi)
axes[1].scatter(CDOG_data[:,0], unwrap, s=10, label="CDOG")
axes[1].scatter(GPS_data + offset, GPS_travel_times, s=10, label="GPS")
axes[1].set_title("Travel Times")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Travel Time (s)")
axes[1].legend()
plt.show()

"""Offset is opposite from desired... :0"""

"""Real Data
CalculateTimesRayTracing
Add Real Data flag to the Numba functions
"""

