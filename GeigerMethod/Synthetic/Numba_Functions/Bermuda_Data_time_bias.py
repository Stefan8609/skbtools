from idlelib.pyparse import trans

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ECEF_Geodetic import ECEF_Geodetic
from pymap3d import geodetic2ecef, ecef2geodetic
from Numba_Geiger import findTransponder
from Numba_xAline_bias import initial_bias_geiger, transition_bias_geiger, final_bias_geiger

"""
Bud's Algorithm gives another drastic drop in RMSE and definitely has room for improvement
    Down to 41 cm RMSE for later stretch of data
    
    
Run synthetic with the real data location and see what happens

Write modular codes
Run the inversion on all the dogs and see if we get similar parameters

Run the numba time bias with multople different severities of esv offset to see how the noise propogates

Make the algorithms easier to run with options (turning on and off certain parts of the algorithm)
    Sound speed
    Time_noise
    Lever_arm
    trajectory
    etc...
    
How wrong can sound speed be for us to still recover it (build out various variations)
    Especially the uppermost part of the ocean
"""

#Load GNSS Data during the time of expedition (25 through 40.9) hours
def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)
    # condition_GNSS = (datetimes/3600 >= 35.3) & (datetimes / 3600 <= 37.6)
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

#Load ESV Table
esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

#Initialize time-tagged data for GPS and CDOG
GPS_data = filtered_data[0, 0, :]
CDOG_data = sio.loadmat('../../../GPSData/DOG3-camp.mat')['tags'].astype(float)

lat = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lat'].flatten()
lon = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lon'].flatten()
elev = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['elev'].flatten()

CDOG_guess_augment = np.array([ 974.12667502,  -80.98121315, -805.07870249])
# initial_lever_guess = np.array([-30.22391079,  -0.22850613, -21.97254162])
initial_lever_guess = np.array([-12.48862757, 0.22622633, -15.89601934])
# offset = 1991.01236648
offset = 2003

CDOG_guess_geodetic = np.array([np.mean(lat), np.mean(lon), np.mean(elev)]) + np.array([0, 0, -5200])
CDOG_guess_base = np.array(geodetic2ecef(CDOG_guess_geodetic[0], CDOG_guess_geodetic[1], CDOG_guess_geodetic[2]))
CDOG_guess = CDOG_guess_base + CDOG_guess_augment

gps1_to_others = np.array([[0.0,0.0,0.0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195, 0.04880436]])

#Scale GPS Clock slightly and scale CDOG clock to nanoseconds
GPS_data = GPS_data - 68826
CDOG_data[:, 1] = CDOG_data[:, 1]/1e9

transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)

inversion_result, best_offset = initial_bias_geiger(CDOG_guess, CDOG_data, GPS_data, transponder_coordinates, dz_array,
                        angle_array, esv_matrix, real_data=True)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
print("Initial Complete:", best_offset)

inversion_result, best_offset = transition_bias_geiger(inversion_guess, CDOG_data, GPS_data, transponder_coordinates, best_offset,
                                                      esv_bias, time_bias, dz_array, angle_array, esv_matrix, real_data=True)
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
print("Transition Complete:", best_offset)

# inversion_result = CDOG_guess
inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]

print("offsets: ", best_offset, offset)


"""If we don't want offset found by our method"""
best_offset = offset


inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(inversion_guess, CDOG_data, GPS_data,
                                                                                     transponder_coordinates, best_offset, esv_bias, time_bias,
                                                                                     dz_array, angle_array, esv_matrix, real_data=True)
print("Final Complete")
best_lever = initial_lever_guess

inversion_guess = inversion_result[:3]
time_bias = inversion_result[3]
esv_bias = inversion_result[4]
GPS_full = GPS_full - time_bias

print(f"Estimate: {inversion_result}")
print(f"Best Lever: {best_lever}, Offset: {best_offset}, Inversion Guess: {inversion_guess-CDOG_guess_base}")
diff_data = (CDOG_full - GPS_full) * 1000
RMSE = np.sqrt(np.nanmean(diff_data**2))/1000 * 1515 * 100
print("RMSE:", RMSE, "cm")

#Figure with 2 plots arranged vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
# axes[0].scatter(CDOG_clock, CDOG_full, s=10, marker="x", label="CDOG")
axes[0].scatter(CDOG_clock, CDOG_full, s=10, marker="x", label="Observed Data")
axes[0].scatter(GPS_clock, GPS_full, s=1, marker="x", label="Model Data")
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
Later Stretch with offset 1991.01236648:
    Estimate: [ 1.97756330e+06 -5.06971291e+06  3.30558847e+06 -6.15213141e-03
      3.45269699e-01]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 1991.01236648, Inversion Guess: [ 891.68438411  -90.36853403 -742.22868432]
    RMSE: 41.76412114430489 cm

Early stetch with offset 1991.01236648:
    Estimate: [ 1.97749469e+06 -5.06975002e+06  3.30557765e+06 -4.69472577e-03
     -8.25014171e-03]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 1991.01236648, Inversion Guess: [ 823.07273856 -127.4784012  -753.04546293]
    RMSE: 34.92721519014112 cm

Full data with found offset:
    Estimate: [ 1.97749415e+06 -5.06972687e+06  3.30558970e+06 -5.71262003e-03
     -1.42138328e+00]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 2002.0024736817998, Inversion Guess: [ 822.53498628 -104.33636845 -740.99398228]
    RMSE: 346.6423130226219 cm
    
Full data with 2003 offset:
    Estimate: [ 1.97749420e+06 -5.06972741e+06  3.30558995e+06 -8.76245851e-03
     -1.33482911e+00]
    Best Lever: [-12.48862757   0.22622633 -15.89601934], Offset: 2003, Inversion Guess: [ 822.58142272 -104.87236322 -740.7419129 ]
    RMSE: 302.33173949545716 cm

Note that the ESV bias vary greatly between the two stretches of data
"""