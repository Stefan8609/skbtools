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
    condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)

    datetimes = datetimes[condition_GNSS]
    time_GNSS = datetimes
    x,y,z = data['x'].flatten()[condition_GNSS], data['y'].flatten()[condition_GNSS], data['z'].flatten()[condition_GNSS]
    # x,y,z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()

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

"""Best estimates of where CDOG3 is to add to the thing
[0, -1000, -200]
[862.12, -537.32, -372.171]
[978.19841247, -555.40502669, -390.82455075]
[816.85365039, -84.07699622, -753.42690995] with offset 1992
[816.87969888,  -85.77315484, -752.19279771] with offset 1998.001
"""
CDOG_guess_geodetic = np.array([np.mean(lat), np.mean(lon), np.mean(elev)]) + np.array([0, 0, -5200])
CDOG_guess_base = np.array(geodetic2ecef(CDOG_guess_geodetic[0], CDOG_guess_geodetic[1], CDOG_guess_geodetic[2]))
CDOG_guess = CDOG_guess_base + np.array([824.07262209, -109.48995844, -736.00540406])
print("ECEF", CDOG_guess, "GEODETIC", CDOG_guess_geodetic)

gps1_to_others = np.array([[0.0,0.0,0.0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195, 0.04880436]])

#Scale GPS Clock slightly and scale CDOG clock to nanoseconds
GPS_data = GPS_data - 68826
CDOG_data[:, 1] = CDOG_data[:, 1]/1e9

# initial_lever_guess = np.array([-12.4, 15.46, -15.24])
initial_lever_guess = np.array([-6.68026498, 9.21708333, -12.45692575])
offset = 1998

print("GPS DATA:", GPS_data)
print("CDOG DATA:", CDOG_data[:,0])

"""Running Geiger"""
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)
# inversion_guess, best_offset = initial_geiger(CDOG_guess, CDOG_data, GPS_data, transponder_coordinates, real_data=True)
# print("Initial Complete:", best_offset)
# inversion_guess, best_offset = transition_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset, real_data=True)
# print("Transition Complete:", best_offset)
inversion_guess = CDOG_guess
best_offset = offset
inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset, real_data=True)
print("Final Complete")
best_lever = initial_lever_guess

"""Running Simulated Annealing
1st run results: Best Lever: [ -6.68026498   9.21708333 -12.45692575], 
                 Offset: 2010.01544813, 
                 Inversion Guess: [ 824.07262209 -109.48995844 -736.00540406]
"""
# best_lever, best_offset, inversion_guess = simulated_annealing(300, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others, CDOG_guess, initial_lever_guess, initial_offset=offset, real_data = True)
# transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, best_lever)
# GPS_travel_times, esv = calculateTimesRayTracingReal(inversion_guess, transponder_coordinates)
# CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = two_pointer_index(best_offset, 0.6, CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv, exact=True)

print(f"Best Lever: {best_lever}, Offset: {best_offset}, Inversion Guess: {inversion_guess-CDOG_guess_base}")
diff_data = (CDOG_full - GPS_full)
RMSE = np.sqrt(np.nanmean(diff_data**2)) * 100 * 1515
print("RMSE:", RMSE, "cm")


#Figure with 2 plots arranged vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
# axes[0].scatter(CDOG_clock, CDOG_full, s=10, marker="x", label="CDOG")
axes[0].scatter(CDOG_clock, CDOG_clock - (GPS_clock - GPS_full), s=10, marker="x", label="CDOG")
axes[0].scatter(GPS_clock, GPS_full, s=1, marker="x", label="GPS")
axes[0].set_title("Travel Times")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Travel Time (s)")
axes[0].legend()

axes[1].scatter(GPS_clock, diff_data, s=1)
axes[1].set_title("Difference between CDOG and GPS")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Residual (ms)")
plt.tight_layout()
plt.show()

"""Offset is opposite from desired... :0"""

"""Real Data
CalculateTimesRayTracing
Add Real Data flag to the Numba functions
"""

