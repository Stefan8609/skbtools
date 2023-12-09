import scipy.io as sio
import numpy as np
from simulatedAnnealing_Bermuda import simulatedAnnealing_Bermuda
from pymap3d import geodetic2ecef


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
offset = 68126#66828#68126 #Why? This is approximately overlaying them now
data_DOG = sio.loadmat('../GPSData/DOG1-camp.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:,1] / 1e9*2*np.pi) / (2*np.pi) #Why?
time_DOG = (data_DOG[:, 0] + offset) / 3600
condition_DOG = (time_DOG >=25) & (time_DOG <= 40.9)
time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

#Get data at matching time stamps between acoustic data and GNSS data
time_GNSS = filtered_data[0,0]
valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
valid_timestamp = np.full(time_GNSS.shape, np.nan)

common_indices = np.isin(time_GNSS, time_DOG)
time_GNSS = time_GNSS[common_indices]
GPS_Coordinates = GPS_Coordinates[common_indices]

#Find repeated timestamps and remove them
repeat = np.full(len(time_DOG), False)
for i in range(1,len(time_DOG)):
    if time_DOG[i-1] == time_DOG[i]:
        repeat[i] = True

time_DOG = time_DOG[~repeat]
acoustic_DOG = acoustic_DOG[~repeat]

common_indices2 = np.isin(time_DOG, time_GNSS)
time_DOG = time_DOG[common_indices2]
acoustic_DOG = acoustic_DOG[common_indices2]

valid_acoustic_DOG = acoustic_DOG
valid_timestamp = time_DOG

#Take every 30th coordinate (reduce computation time for testing)
valid_acoustic_DOG=valid_acoustic_DOG[0::30]
valid_timestamp=valid_timestamp[0::30]
GPS_Coordinates = GPS_Coordinates[0::30]

# initial_dog_guess = np.mean(GPS_Coordinates[:,0], axis=0)
# initial_dog_guess[2] += 5000
sound_speed = 1515 #Changing sound speed gets better fit
initial_dog_guess=np.array([1979509.5631926274, -5077550.411986372, 3312551.0725191827]) #Thalia's guess for CDOG3
initial_dog_guess[2] += 5000

## For testing the potential of deleting outlying items in GPS coordinates
# valid_acoustic_DOG = np.delete(valid_acoustic_DOG, 859, 0)
# GPS_Coordinates = np.delete(GPS_Coordinates, 859, 0)
# valid_timestamp = np.delete(valid_timestamp, 859, 0)
# valid_acoustic_DOG = np.delete(valid_acoustic_DOG, 1154, 0)
# GPS_Coordinates = np.delete(GPS_Coordinates, 1154, 0)
# valid_timestamp = np.delete(valid_timestamp, 1154, 0)
# valid_acoustic_DOG = np.delete(valid_acoustic_DOG, 1154, 0)
# GPS_Coordinates = np.delete(GPS_Coordinates, 1154, 0)
# valid_timestamp = np.delete(valid_timestamp, 1154, 0)

# gps1_to_others = np.array([[0,0,0],[0, -4.93, 0], [-10.2,-7.11,0],[-10.1268,0,0]]
gps1_to_others = np.array([[0,0,0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195,0.04880436]])
initial_lever_guess = np.array([-12.4, 15.46, -15.24])
# initial_lever_guess = np.array([-10.43, 2.58, -3.644])

simulatedAnnealing_Bermuda(300, GPS_Coordinates, initial_dog_guess, valid_acoustic_DOG, gps1_to_others, initial_lever_guess, valid_timestamp, sound_speed)

#Test use this to gps1_to_others lever arm
# from GPS_Lever_Arms import GPS_Lever_arms
# GPS_Lever_arms(GPS_Coordinates)

#Change findxyzt
#check if time offset is good.

#Try modifying lever arms between gps and see what happens

#Next step - Build out synthetic where actual travel times are determined from ray tracing
#   Then using a constant sound velocity find calc times and find best RMSE and compare to what I'm getting here
#   If they are similar then the high RMSE can possibly be due to not adequately accomodating sound velocity
#   then implement ray tracing in determination of calc times to see if it lowers RMSE
#   then implement ray tracing in this analysis of data.

# [ 1979507.8868552  -5077545.65205964  3312550.35393797]
# [ 1979508.02136779 -5077545.99643553  3312550.57786073]

