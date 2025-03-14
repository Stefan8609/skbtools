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
    condition_GNSS = (datetimes/3600 >= 35.3) & (datetimes / 3600 <= 37.6)
    # condition_GNSS = (datetimes/3600 >= 31.9) & (datetimes / 3600 <= 34.75)

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
CDOG_guess_augment = np.array([ 974.12667502,  -80.98121315, -805.07870249])
# initial_lever_guess = np.array([-30.22391079,  -0.22850613, -21.97254162])
initial_lever_guess = np.array([-12.48862757, 0.22622633, -15.89601934])
offset = 1991.01236648
# offset = 2076.0242

CDOG_guess_geodetic = np.array([np.mean(lat), np.mean(lon), np.mean(elev)]) + np.array([0, 0, -5200])
CDOG_guess_base = np.array(geodetic2ecef(CDOG_guess_geodetic[0], CDOG_guess_geodetic[1], CDOG_guess_geodetic[2]))
CDOG_guess = CDOG_guess_base + CDOG_guess_augment
print("ECEF", CDOG_guess, "GEODETIC", CDOG_guess_geodetic)

print(CDOG_guess)
print(ecef2geodetic(CDOG_guess[0], CDOG_guess[1], CDOG_guess[2]))

gps1_to_others = np.array([[0.0,0.0,0.0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195, 0.04880436]])

#Scale GPS Clock slightly and scale CDOG clock to nanoseconds
GPS_data = GPS_data - 68826
CDOG_data[:, 1] = CDOG_data[:, 1]/1e9

print("GPS DATA:", GPS_data)
print("CDOG DATA:", CDOG_data[:,0])

"""Running Geiger"""
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)
inversion_guess, best_offset = initial_geiger(CDOG_guess, CDOG_data, GPS_data, transponder_coordinates, real_data=True)
print("Initial Complete:", best_offset)
inversion_guess, best_offset = transition_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset, real_data=True)
print("Transition Complete:", best_offset)

inversion_guess = CDOG_guess
best_offset = offset
inversion_guess, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset, real_data=True)
print("Final Complete")
best_lever = initial_lever_guess

"""Running Simulated Annealing
1st run results: Best Lever: [ -6.68026498   9.21708333 -12.45692575], 
                 Offset: 2010.01544813, 
                 Inversion Guess: [ 824.07262209 -109.48995844 -736.00540406]
                 
Subset results: Best Lever: [ -6.14227704   8.87509882 -13.50000437], 
                Offset: 2303, 
                Inversion Guess: [-1204.03979944   287.08544063   518.85686073]
                
sub_set(late) results: Best Lever: [-12.48862757   0.22622633 -15.89601934], 
                       Offset: 1994.01275663,
                       Inversion Guess: [ 946.28801027  -82.90482082 -780.04869985]
                      
2nd full run results: Best Lever: [-22.67388371   4.21094505 -17.73045142], 
                      Offset: 1997.01520138, 
                      Inversion Guess: [ 823.23140004 -107.17257696 -738.77824697]
                      
Best sub_set(late) results: Best Lever: [-12.48862757   0.22622633 -15.89601934], 
                            Offset: 1991.0163564, 
                            Inversion Guess: [ 883.39039202  -89.15279875 -739.79800661]

sub_set(early) results: Best Lever: [ -6.59657956   3.94814215 -13.94312983], 
                             Offset: 1991.01688373, 
                             Inversion Guess: [ 829.64866938 -130.02010894 -757.1188929 ]
                             
Best sub_set (early) results: Best Lever: [-11.07118568   3.45761614 -13.73010071], 
                              Offset: 1991.01700163, 
                              Inversion Guess: [ 824.07645653 -127.75081484 -752.17779813]
"""
# best_lever, best_offset, inversion_guess = simulated_annealing(300, CDOG_data, GPS_data, GPS_Coordinates, gps1_to_others, CDOG_guess, initial_lever_guess, initial_offset=offset, real_data = True)
# transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, best_lever)
# GPS_travel_times, esv = calculateTimesRayTracingReal(inversion_guess, transponder_coordinates)
# CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = two_pointer_index(best_offset, 0.6, CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv, exact=True)

print(f"Best Lever: {best_lever}, Offset: {best_offset}, Inversion Guess: {inversion_guess-CDOG_guess_base}")
diff_data = (CDOG_full - GPS_full) * 1000
RMSE = np.sqrt(np.nanmean(diff_data**2))/1000 * 1515 * 100
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

"""
Adjust sound speed with a constant
Make some minimal inversion

Run on three hours of the data that is continuous in the residual

Isolate a geometric effect from the part that nearly fits
    Adjust the sound speed by a constant because its not constant
        Need to rewrite Thalia's code to do this

    If no way I can make this better than it has to be the sound speed
    
    
Make Synthetic with the incorrect sound speed profile to see the variation
"""