"""
Looks at the shape of the C-DOG files to get an understanding of what they look like
"""


import scipy.io as sio
import numpy as np
from geigerMethod_Bermuda import calculateTimesRayTracing, find_esv
import matplotlib.pyplot as plt

#Load GNSS Data during the time of expedition (25 through 40.9) hours
def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)
    time_GNSS = datetimes[condition_GNSS]/3600
    x,y,z = data['x'].flatten()[condition_GNSS], data['y'].flatten()[condition_GNSS], data['z'].flatten()[condition_GNSS]
    return time_GNSS, x,y,z

paths = [
    '../GPSData/Unit1-camp_bis.mat',
    '../GPSData/Unit2-camp_bis.mat',
    '../GPSData/Unit3-camp_bis.mat',
    '../GPSData/Unit4-camp_bis.mat'
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

#Initialize Dog Acoustic Data

#offset:RMSE, 68116:222.186, 68126:165.453, 68136:219.04, 68130:184.884, 68128: 170.04, 68124: 168.05, 68125:167
offset = 68126#66828#68126 This is approximately overlaying them now
data_DOG = sio.loadmat('../GPSData/DOG1-camp.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:,1] / 1e9*2*np.pi) / (2*np.pi)  #Numpy page describes how unwrap works
    #I don't think the periodicity for unwrap function is 2*pi as what is set now
time_DOG = (data_DOG[:, 0] + offset) / 3600
# condition_DOG = (time_DOG >=25) & (time_DOG <= 40.9)
# time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]


plt.scatter(acoustic_DOG + data_DOG[:,0], acoustic_DOG, s=1)
# plt.scatter(list(range(len(acoustic_DOG))),acoustic_DOG + data_DOG[:,0], s=1)
# plt.scatter(list(range(len(acoustic_DOG))),acoustic_DOG, s=1)
plt.show()


"""
Good next step -- overlay plot of best CDOG guess and calculated travel times from GPS on top of plot of 
    wrapped dog versus absolute dog time (can figure out offset and scaling).
"""