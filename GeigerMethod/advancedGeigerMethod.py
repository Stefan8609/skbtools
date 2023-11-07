"""
[add description]

Written by Stefan Kildal-Brandt
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from findPointByPlane import initializeFunction, findXyzt
np.set_printoptions(precision = 6, suppress = True)

def generateRandomData(n): #Generate the random data in the form of numpy arrays
    #Generate CDog
    CDog = np.array([random.uniform(-2000, 2000), random.uniform(-2000,2000), random.uniform(-5500, -4500)])

    #Generate and initial GPS point to base all others off of
    xyz_point = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-100, 100)])

    #Generate the translations from initial point (random x,y,z translation with z/100) for each time step
    translations = (np.random.rand(n,3) * 5000) - 2500
    translations = np.matmul(translations, np.array([[1,0,0],[0,1,0],[0,0,1/100]]))

    #Generate rotations from initial point for each time step (yaw, pitch, roll) between -pi/2 to pi/2
    rot = (np.random.rand(n, 3) * np.pi) - np.pi/2

    #Have GPS coordinates for all 4 GPS at each time step. Also have transponder for each time step
    GPS_Coordinates = np.zeros((n,4, 3))
    transponder_coordinates = np.zeros((n,3))

    #Have a displacement vectors to find other GPS from first GPS. Also displacement from first GPS to transponder
    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n):
        #Build rotation matrix at each time step
        xRot = np.array([[1, 0, 0], [0, np.cos(rot[i,0]), -np.sin(rot[i,0])], [0, np.sin(rot[i,0]), np.cos(rot[i,0])]])
        yRot = np.array([[np.cos(rot[i,1]), 0, np.sin(rot[i,1])], [0, 1, 0], [-np.sin(rot[i,1]), 0, np.cos(rot[i,1])]])
        zRot = np.array([[np.cos(rot[i,2]), -np.sin(rot[i,2]), 0], [np.sin(rot[i,2]), np.cos(rot[i,2]), 0], [0, 0, 1]])
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

        for j in range(1, 4): #Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = xyz_point + np.matmul(totalRot, gps1_to_others[j])
            GPS_Coordinates[i, j] += translations[i]

        #Put in known transponder location to get simulated times
        transponder_coordinates[i] = xyz_point + np.matmul(totalRot, gps1_to_transponder)
        transponder_coordinates[i] += translations[i]

        GPS_Coordinates[i, 0] = xyz_point + translations[i] #translate original point

    return CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder

def generateLine(n):
    #Initialize CDog and GPS locations
    CDog = np.array([random.uniform(-2000, 2000), random.uniform(-2000,2000), random.uniform(-5500, -4500)])
    x_coords = (np.random.rand(n) * 5000) - 2500
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    z_coords = (np.random.rand(n) * 50) - 25
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    GPS1_Coordinates = sorted(GPS1_Coordinates, key=lambda k: [k[0], k[1], k[2]])

    GPS_Coordinates = np.zeros((n, 4, 3))
    transponder_coordinates = np.zeros((n, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    #Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n):
        #Build rotation matrix at each time step
        xRot = np.array([[1, 0, 0], [0, np.cos(rot[i,0]), -np.sin(rot[i,0])], [0, np.sin(rot[i,0]), np.cos(rot[i,0])]])
        yRot = np.array([[np.cos(rot[i,1]), 0, np.sin(rot[i,1])], [0, 1, 0], [-np.sin(rot[i,1]), 0, np.cos(rot[i,1])]])
        zRot = np.array([[np.cos(rot[i,2]), -np.sin(rot[i,2]), 0], [np.sin(rot[i,2]), np.cos(rot[i,2]), 0], [0, 0, 1]])
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))
        for j in range(1, 4): #Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i,0] + np.matmul(totalRot, gps1_to_others[j])
        #Initialize transponder location
        transponder_coordinates[i] = GPS_Coordinates[i,0] + np.matmul(totalRot, gps1_to_transponder)
    return CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder

def generateCross(n):
    # Initialize CDog and GPS locations
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(-5500, -4500)])
    x_coords = (np.random.rand(n) * 5000) - 2500
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    y_coords[n//2:] *= -1
    z_coords = (np.random.rand(n) * 50) - 25
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    GPS1_Coordinates = sorted(GPS1_Coordinates, key=lambda k: [k[0], k[1], k[2]])

    GPS_Coordinates = np.zeros((n, 4, 3))
    transponder_coordinates = np.zeros((n, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    # Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n):
        # Build rotation matrix at each time step
        xRot = np.array(
            [[1, 0, 0], [0, np.cos(rot[i, 0]), -np.sin(rot[i, 0])], [0, np.sin(rot[i, 0]), np.cos(rot[i, 0])]])
        yRot = np.array(
            [[np.cos(rot[i, 1]), 0, np.sin(rot[i, 1])], [0, 1, 0], [-np.sin(rot[i, 1]), 0, np.cos(rot[i, 1])]])
        zRot = np.array(
            [[np.cos(rot[i, 2]), -np.sin(rot[i, 2]), 0], [np.sin(rot[i, 2]), np.cos(rot[i, 2]), 0], [0, 0, 1]])
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))
        for j in range(1, 4):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i, 0] + np.matmul(totalRot, gps1_to_others[j])
        # Initialize transponder location
        transponder_coordinates[i] = GPS_Coordinates[i, 0] + np.matmul(totalRot, gps1_to_transponder)
    return CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder

def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    #Add some noise to initial information
    gps1_to_others += np.random.normal(0, 2*10**-2, (4,3))
    gps1_to_transponder += np.random.normal(0, 2*10**-2, 3)

    print('here', gps1_to_others, gps1_to_transponder)

    # Given initial information relative GPS locations and transponder and GPS Coords at each timestep
    xs, ys, zs = gps1_to_others.T
    initial_transponder = gps1_to_transponder
    theta, phi, length, orientation = initializeFunction(xs, ys, zs, 0, initial_transponder)

    n = len(GPS_Coordinates)
    transponder_coordinates = np.zeros((n, 3))
    for i in range(n):
        new_xs, new_ys, new_zs = GPS_Coordinates[i].T
        xyzt_vector, barycenter = findXyzt(new_xs, new_ys, new_zs, 0, length, theta, phi, orientation)
        transponder_coordinates[i] = xyzt_vector + barycenter

    return transponder_coordinates

def calculateTimes(guess, transponder_coordinates, sound_speed):
    times = np.zeros(len(transponder_coordinates))
    for i in range(len(transponder_coordinates)):
        distance = np.linalg.norm(transponder_coordinates[i] - guess)
        times[i] = distance / sound_speed
    return times

def computeJacobian(guess, transponder_coordinates, times, sound_speed):
    #Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    jacobian = np.zeros((len(transponder_coordinates), 3))
    for i in range(len(transponder_coordinates)):
        jacobian[i, 0] = (-1 * transponder_coordinates[i, 0] + guess[0]) / (times[i]*(sound_speed**2))
        jacobian[i, 1] = (-1 * transponder_coordinates[i, 1] + guess[1]) / (times[i]*(sound_speed**2))
        jacobian[i, 2] = (-1 * transponder_coordinates[i, 2] + guess[2]) / (times[i]*(sound_speed**2))
    return jacobian

#Goal is to minimize sum of the difference of times squared
def geigersMethod(guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found):
    #Use Geiger's method to find the guess of CDOG location which minimizes sum of travel times squared
    #Define threshold
    epsilon = 10**-5

    #Sound Speed of water (right now constant, later will use ray tracing)
    sound_speed = 1515

    #Get known times
    times_known = calculateTimes(CDog, transponder_coordinates_Actual, sound_speed)
    #Apply noise to known times on scale of 20 microseconds
    times_known+=np.random.normal(0,2*10**-5,len(transponder_coordinates_Actual))

    k=0
    delta = 1
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        print(guess, CDog)
        times_guess = calculateTimes(guess, transponder_coordinates_Found, sound_speed)
        jacobian = computeJacobian(guess, transponder_coordinates_Found, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1
    return guess, times_known

CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateCross(1000)

#Add noise to GPS on scale of 2 cm
GPS_Coordinates += np.random.normal(0, 2*10**-2, (len(GPS_Coordinates), 4, 3))

transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

# differences = np.sum(np.abs(transponder_coordinates_Found-transponder_coordinates_Actual)**2,axis=-1)
# print(differences)
# plt.hist(differences, orientation='horizontal', bins=30, alpha=0.5)
# plt.show()

#Plot histograms of coordinate differences between found transponder and actual transponder
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Coordinate differences between calculated transponder and actual transponder", y=0.92)

axs[0].hist(transponder_coordinates_Found[:,0]-transponder_coordinates_Actual[:,0], bins=30, color="blue", alpha=0.7)
axs[0].set_xlabel('X-difference(m)')

axs[1].hist(transponder_coordinates_Found[:,1]-transponder_coordinates_Actual[:,1], bins=30, color="blue", alpha=0.7)
axs[1].set_xlabel('Y-difference(m)')

axs[2].hist(transponder_coordinates_Found[:,2]-transponder_coordinates_Actual[:,2], bins=30, color="blue", alpha=0.7)
axs[2].set_xlabel('Z-difference(m)')
fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
plt.show()

guess, times_known = geigersMethod([-5000, 5000, -5000], CDog, transponder_coordinates_Actual, transponder_coordinates_Found)

print(np.linalg.norm(CDog - guess))

#Plot path of experiment
plt.scatter(GPS_Coordinates[:,0,0], GPS_Coordinates[:,0,1], label="GPS1")
plt.scatter(GPS_Coordinates[:,1,0], GPS_Coordinates[:,1,1], label="GPS2")
plt.scatter(GPS_Coordinates[:,2,0], GPS_Coordinates[:,2,1], label="GPS3")
plt.scatter(GPS_Coordinates[:,3,0], GPS_Coordinates[:,3,1], label="GPS4")
plt.scatter(transponder_coordinates_Actual[:,0], transponder_coordinates_Actual[:,1], label="Transponder")
plt.scatter(CDog[0], CDog[1], label="CDog")

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('GPS and transducer coordinates over course of experimental path')
plt.legend(loc = "upper right")
plt.show()

times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)

difference_data = times_calc - times_known
RMS = np.sqrt(np.nanmean(difference_data ** 2))
print(RMS)

# Prepare label and plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
fig.suptitle("Comparison of calculated arrival times and actual arrival times", y=0.92)
fig.text(0.05, 0.85, "Noise: \n GPS: 2cm \n Arrival time: 20\u03BCs \n Lever arms: 2cm",
         fontsize=12, bbox=dict(facecolor='yellow', alpha=0.8))
fig.text(0.05, 0.7, f"Distance between \npredicted and actual \nCDog location:\n{np.round(np.linalg.norm(CDog-guess)*100, 4)}cm",
         fontsize=12, bbox=dict(facecolor='green', alpha=0.8))

# Acoustic vs GNSS plot
GPS_Coord_Num = list(range(len(GPS_Coordinates)))

axes[0, 1].scatter(GPS_Coord_Num, times_known, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
axes[0, 1].scatter(GPS_Coord_Num, times_calc, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
axes[0, 1].set_ylabel('Travel Time (s)')
axes[0, 1].text(25, max(times_known), "actual arrival times versus estimated times",bbox=dict(facecolor='yellow', alpha=0.8))
axes[0, 1].legend(loc="upper right")

# Difference plot
axes[1, 1].scatter(GPS_Coord_Num, difference_data, s=1)
axes[1, 1].set_xlabel('Position Index')
axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')

# Histogram
axes[1, 0].hist(difference_data, orientation='horizontal', bins=30, alpha=0.5)
axes[1, 0].set_ylabel('Difference (s)')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].invert_xaxis()
axes[1, 0].set_title(f"RMS: {round(RMS * 1515, 4)*100} cm")
axes[0, 0].axis('off')

plt.show()


#Need to correct functions for generating line data and cross data
#Need a good function to plot experimental paths (maybe dual plot comparing calculated transponder with actual location)

#Interesting things to note:
    # Being off in terms of initial information given to findTransponder function introduces skew in the x,y,z coordinate
    # difference histograms. This greatly affects the quality of the data
    # This skew is not seen in the Acoustic vs GNNS arrival time plots, meaning that it is hidden beneath the RMS