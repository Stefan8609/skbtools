import scipy.io as sio
import numpy as np
from simulatedAnnealing_Bermuda import simulatedAnnealing_Bermuda
from GPS_Lever_Arms import GPS_Lever_arms
from timePlot_Bermuda import geigerTimePlot
from geigerMethod_Bermuda import findTransponder
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
offset = 68126#66828#68126 This is approximately overlaying them now
data_DOG = sio.loadmat('../GPSData/DOG1-camp.mat')['tags'].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:,1] / 1e9*2*np.pi) / (2*np.pi)  #Numpy page describes how unwrap works
    #I don't think the periodicity for unwrap function is 2*pi as what is set now
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
        print(time_DOG[i] * 3600 - offset)
        print(acoustic_DOG[i], acoustic_DOG[i-1])
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

print('\n')
GPS_Lever_arms(GPS_Coordinates)
print('\n')

print(valid_acoustic_DOG)

initial_dog_guess = np.mean(GPS_Coordinates[:,0], axis=0)
initial_dog_guess[2] += 5000

# gps1_to_others = np.array([[0,0,0],[0, -4.93, 0], [-10.2,-7.11,0],[-10.1268,0,0]])
gps1_to_others = np.array([[0,0,0],[-2.4054, -4.20905, 0.060621], [-12.1105,-0.956145,0.00877],[-8.70446831,5.165195, 0.04880436]])
#Design a program to find the optimal gps1_to_others

initial_lever_guess = np.array([-12.4, 15.46, -15.24])
# initial_lever_guess = np.array([-10.43, 2.58, -3.644])

simulatedAnnealing_Bermuda(300, GPS_Coordinates, initial_dog_guess, valid_acoustic_DOG, gps1_to_others, initial_lever_guess, valid_timestamp)

# transponder_coordinates_found = findTransponder(GPS_Coordinates, gps1_to_others, [-7.56446, 7.54666, -3.661])
# transponder_coordinates_found = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever_guess)
# geigerTimePlot([1979507.95, -5077545.81, 3312550.46], valid_acoustic_DOG, transponder_coordinates_found, valid_timestamp, sound_speed)


#Test use this to gps1_to_others lever arm
# from GPS_Lever_Arms import GPS_Lever_arms
# GPS_Lever_arms(GPS_Coordinates)

#Change findxyzt
#check if time offset is good.

#Try modifying lever arms between gps and see what happens

# [ 1979507.8868552  -5077545.65205964  3312550.35393797]
# [ 1979508.02136779 -5077545.99643553  3312550.57786073]

#Does the curvature of the Earth mess up the transducer displacement??



#Model is not adequately capturing the real scenario
#First source of error is no ray tracing implemented.
#
#Look at residuals as a function of range -> Bias (Look in dissertation) Manuscripts 2
#Third and Fourth manuscripts are not relevant right now
#First and Second are relevant to seafloor geodesy

#Add an offset into the synthetic to see how it affects things

#I think their is still a major problem with matching the emission arrival with the point where emitted
#   Need to take the (arrival time - travel time) to find the GPS coordinates to use for emission point
#   This seems like a potentially difficult problem as travel time is a considerable unknown
#   Maybe can estimate with the approximate esv from known GPS at time of arrival
#       Idea: Take the travel time from the given location at arrival time and subtract from given time
#           To find the time of emission (then can do further corrections by seeing how travel time changes)
#           Kinda will descend towards the actual emission time

#Use the 'elev' tag in the MATLAB file to get the z-distance when calculating sound speed.

#Check periodicity of unwrap function and make sure that it is right

#Get absolute distance (diff in xyzs) and the vertical diff (diff in elev), then find the
#   Hori distance using Hi = sqrt(absDist^2 - vertDist^2)

