import scipy.io as sio
import numpy as np
from GeigerMethod.simulatedAnnealing_Bermuda import simulatedAnnealing_Bermuda
from GeigerMethod.GPS_Lever_Arms import GPS_Lever_arms
from data import gps_data_path


# Load GNSS Data during the time of expedition (25 through 40.9) hours
def load_and_process_data(path):
    """Load GNSS data for the Bermuda experiment period.

    Parameters
    ----------
    path : str or Path
        MATLAB ``.mat`` file containing receiver data.

    Returns
    -------
    tuple
        Time array in hours and the associated ``x``, ``y`` and ``z`` coordinates.
    """

    data = sio.loadmat(path)
    days = data["days"].flatten() - 59015
    times = data["times"].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)
    time_GNSS = datetimes[condition_GNSS] / 3600
    x, y, z = (
        data["x"].flatten()[condition_GNSS],
        data["y"].flatten()[condition_GNSS],
        data["z"].flatten()[condition_GNSS],
    )
    return time_GNSS, x, y, z


paths = [
    gps_data_path("Unit1-camp_bis.mat"),
    gps_data_path("Unit2-camp_bis.mat"),
    gps_data_path("Unit3-camp_bis.mat"),
    gps_data_path("Unit4-camp_bis.mat"),
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

# Initialize Dog Acoustic Data

offset = 68126  # 66828#68126 This is approximately overlaying them now
data_DOG = sio.loadmat(gps_data_path("DOG1-camp.mat"))["tags"].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (
    2 * np.pi
)  # Numpy page describes how unwrap works
# I don't think the periodicity for unwrap function is 2*pi as what is set now
time_DOG = (data_DOG[:, 0] + offset) / 3600
condition_DOG = (time_DOG >= 25) & (time_DOG <= 40.9)
time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

# Get data at matching time stamps between acoustic data and GNSS data
time_GNSS = filtered_data[0, 0]
valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
valid_timestamp = np.full(time_GNSS.shape, np.nan)

common_indices = np.isin(time_GNSS, time_DOG)
time_GNSS = time_GNSS[common_indices]
GPS_Coordinates = GPS_Coordinates[common_indices]

# Find repeated timestamps and remove them
repeat = np.full(len(time_DOG), False)
for i in range(1, len(time_DOG)):
    if time_DOG[i - 1] == time_DOG[i]:
        print(time_DOG[i] * 3600 - offset)
        print(acoustic_DOG[i], acoustic_DOG[i - 1])
        repeat[i] = True

time_DOG = time_DOG[~repeat]
acoustic_DOG = acoustic_DOG[~repeat]

common_indices2 = np.isin(time_DOG, time_GNSS)
time_DOG = time_DOG[common_indices2]
acoustic_DOG = acoustic_DOG[common_indices2]

valid_acoustic_DOG = acoustic_DOG
valid_timestamp = time_DOG

# Take every 30th coordinate (reduce computation time for testing)
valid_acoustic_DOG = valid_acoustic_DOG[0::30]
valid_timestamp = valid_timestamp[0::30]
GPS_Coordinates = GPS_Coordinates[0::30]

print("\n")
GPS_Lever_arms(GPS_Coordinates)
print("\n")

print(valid_acoustic_DOG)

initial_dog_guess = np.mean(GPS_Coordinates[:, 0], axis=0)
initial_dog_guess[2] += 5000

# gps1_to_others = np.array([[0,0,0],[0, -4.93, 0], [-10.2,-7.11,0],[-10.1268,0,0]])
gps1_to_others = np.array(
    [
        [0, 0, 0],
        [-2.4054, -4.20905, 0.060621],
        [-12.1105, -0.956145, 0.00877],
        [-8.70446831, 5.165195, 0.04880436],
    ]
)
# Design a program to find the optimal gps1_to_others

initial_lever_guess = np.array([-12.4, 15.46, -15.24])
# initial_lever_guess = np.array([-10.43, 2.58, -3.644])

simulatedAnnealing_Bermuda(
    300,
    GPS_Coordinates,
    initial_dog_guess,
    valid_acoustic_DOG,
    gps1_to_others,
    initial_lever_guess,
    valid_timestamp,
)
