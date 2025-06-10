"""
Looks at the shape of the C-DOG files to get an understanding of what they look like
"""

import scipy.io as sio
import numpy as np
from geigerMethod_Bermuda import findTransponder, calculateTimesRayTracing
import matplotlib.pyplot as plt
from pymap3d import geodetic2ecef


# Load GNSS Data during the time of expedition (25 through 40.9) hours
def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data["days"].flatten() - 59015
    times = data["times"].flatten()
    datetimes = (days * 24 * 3600) + times
    # condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)
    # time_GNSS = datetimes[condition_GNSS]/3600
    # x,y,z = data['x'].flatten()[condition_GNSS], data['y'].flatten()[condition_GNSS], data['z'].flatten()[condition_GNSS]

    time_GNSS = datetimes / 3600
    x, y, z = data["x"].flatten(), data["y"].flatten(), data["z"].flatten()

    return time_GNSS, x, y, z


paths = [
    "../GPSData/Unit1-camp_bis.mat",
    "../GPSData/Unit2-camp_bis.mat",
    "../GPSData/Unit3-camp_bis.mat",
    "../GPSData/Unit4-camp_bis.mat",
]

all_data = [load_and_process_data(path) for path in paths]
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

filtered_data = []
for datetimes, x, y, z in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append(
        [
            np.array(datetimes)[mask],
            np.array(x)[mask],
            np.array(y)[mask],
            np.array(z)[mask],
        ]
    )
filtered_data = np.array(filtered_data)

# Initialize Coordinates in form of Geiger's Method
GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
for i in range(len(filtered_data[0, 0])):
    for j in range(4):
        GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
        GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
        GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

# Initialize GNSS time
time_GNSS = filtered_data[0, 0] * 3600 - 68126
print(time_GNSS)

# Initialize CDOG Guess
CDOG = [31.46356091, 291.29859266, -5271.47395559]
CDOG = np.array(geodetic2ecef(CDOG[0], CDOG[1], CDOG[2]))

# Initialize Dog Acoustic Data
# offset:RMSE, 68116:222.186, 68126:165.453, 68136:219.04, 68130:184.884, 68128: 170.04, 68124: 168.05, 68125:167
offset = 68126  # 66828#68126 This is approximately overlaying them now

DOG = 4
data_DOG = sio.loadmat(f"../GPSData/DOG{DOG}-camp.mat")["tags"].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (
    2 * np.pi
)  # Numpy page describes how unwrap works
# I don't think the periodicity for unwrap function is 2*pi as what is set now
time_DOG = (data_DOG[:, 0] + offset) / 3600
condition_DOG = (time_DOG >= 25) & (time_DOG <= 40.9)
time_DOG, acoustic_DOG, data_DOG = (
    time_DOG[condition_DOG],
    acoustic_DOG[condition_DOG],
    data_DOG[condition_DOG],
)

# Find travel times from GPS to CDOG guess
gps1_to_others = np.array(
    [
        [0, 0, 0],
        [-2.4054, -4.20905, 0.060621],
        [-12.1105, -0.956145, 0.00877],
        [-8.70446831, 5.165195, 0.04880436],
    ]
)
initial_lever_guess = np.array([-12.4, 15.46, -15.24])
transponder_coordinates = findTransponder(
    GPS_Coordinates, gps1_to_others, initial_lever_guess
)
travel_times = calculateTimesRayTracing(CDOG, transponder_coordinates)[0]

print(acoustic_DOG + data_DOG[:, 0])
print(time_GNSS + travel_times)

plt.rcParams.update({"font.size": 14})  # Set the default font size to 14
plt.figure(figsize=(8, 4.8))
plt.scatter(time_DOG, acoustic_DOG, s=1, color="r", label=f"DOG {DOG} Data")
# plt.scatter(time_GNSS + travel_times, travel_times, s=1, color='r', label="GPS Data + travel times")
plt.legend()
plt.xlabel("Absolute Time (hours)")
plt.ylabel(f"Travel Time for DOG {DOG} (s)")
plt.xlim(25, 40.9)
plt.ylim(1, 9)
# plt.scatter(list(range(len(acoustic_DOG))),acoustic_DOG + data_DOG[:,0], s=1)
# plt.scatter(list(range(len(acoustic_DOG))),acoustic_DOG, s=1)
plt.show()


"""
Good next step -- overlay plot of best CDOG guess and calculated travel times from GPS on top of plot of
    wrapped dog versus absolute dog time (can figure out offset and scaling).

Coincidence that the wrapping time matches up with the actual travel time

Need to create an algorithm to automatically find the best time offset
"""
