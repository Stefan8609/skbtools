"""
[add description]

Written by Stefan Kildal-Brandt
"""

import numpy as np
import random
from findPointByPlane import initializeFunction, findXyzt
from RigidBodyMovementProblem import findRotationAndDisplacement
import scipy.io as sio

esv_table = sio.loadmat('../../GPSData/global_table_esv.mat')
cz = np.genfromtxt('../../GPSData/cz_cast2_smoothed.txt')[::100]
depth = np.genfromtxt('../../GPSData/depth_cast2_smoothed.txt')[::100]
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

def generateRandomData(n): #Generate the random data in the form of numpy arrays
    #Generate CDog
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5225, -5235)])

    #Generate and initial GPS point to base all others off of
    xyz_point = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-10, 10)])

    #Generate the translations from initial point (random x,y,z translation with z/100) for each time step
    translations = (np.random.rand(n,3) * 15000) - 7500
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
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5225, -5235)])
    x_coords = (np.random.rand(n) * 15000) - 7500
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    z_coords = (np.random.rand(n) * 5) - 10
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
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(-5225, -5235)])
    x_coords1 = (np.random.rand(n//2) * 15000) - 7500
    x_coords2 = (np.random.rand(n//2) * 15000) - 7500
    x_coords = np.concatenate((np.sort(x_coords1), np.sort(x_coords2)))
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    y_coords[n//2:] *= -1
    z_coords = (np.random.rand(n) * 5) - 10
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))

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

def generateRealistic(n):
    # Initialize CDog and GPS locations
    CDog = np.array([random.uniform(-5000, 5000), random.uniform(-5000, 5000), random.uniform(-5225, -5235)])
    x_coords1 = np.sort((np.random.rand(n//4) * 15000) - 7500)
    x_coords2 = -1*np.sort(-1*((np.random.rand(n//4) * 15000) - 7500))
    x_coords3 = np.sort((np.random.rand(n//4) * 15000) - 7500)
    x_coords4 = -1*np.sort(-1*((np.random.rand(n//4) * 15000) - 7500))
    y_coords1 = x_coords1 + (np.random.rand(n//4) * 50) - 25
    y_coords2 = 7500 + (np.random.rand(n//4) * 50) - 25
    y_coords3 = - x_coords1 + (np.random.rand(n//4) * 50) - 25
    y_coords4 = -7500 + (np.random.rand(n//4) * 50) - 25
    x_coords = np.concatenate((x_coords1, x_coords2, x_coords3, x_coords4))
    y_coords = np.concatenate((y_coords1, y_coords2, y_coords3, y_coords4))
    z_coords = (np.random.rand(n//4 * 4) * 5) - 10
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))

    GPS_Coordinates = np.zeros((n//4 * 4, 4, 3))
    transponder_coordinates = np.zeros((n//4 * 4, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    # Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n//4 * 4):
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
    # gps1_to_others += np.random.normal(0, 2*10**-3, (4,3))
    # gps1_to_transponder += np.random.normal(0, 2*10**-2, 3)
    # gps1_to_transponder += np.array([0,0,0])

    # Given initial information relative GPS locations and transponder and GPS Coords at each timestep
    xs, ys, zs = gps1_to_others.T
    initial_transponder = gps1_to_transponder
    n = len(GPS_Coordinates)
    transponder_coordinates = np.zeros((n, 3))
    for i in range(n):
        new_xs, new_ys, new_zs = GPS_Coordinates[i].T
        R_mtrx, d = findRotationAndDisplacement(np.array([xs,ys,zs]), np.array([new_xs, new_ys, new_zs]))
        transponder_coordinates[i] = np.matmul(R_mtrx, initial_transponder) + d
    return transponder_coordinates
    #Next step in speed is vectorizing this function

def calculateTimes(guess, transponder_coordinates, sound_speed):
    times = np.zeros(len(transponder_coordinates))
    for i in range(len(transponder_coordinates)):
        distance = np.linalg.norm(transponder_coordinates[i] - guess)
        times[i] = distance / sound_speed
    return times

#This is to test vectorization
def find_esv(beta, dz):
    idx_closest_dz = np.searchsorted(dz_array, dz, side="left")
    idx_closest_dz = np.clip(idx_closest_dz, 0, len(dz_array)-1)
    idx_closest_beta = np.searchsorted(angle_array, beta, side="left")
    idx_closest_beta = np.clip(idx_closest_beta, 0, len(angle_array)-1)
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv

def calculateTimesRayTracing(guess, transponder_coordinates):
    hori_dist = np.sqrt((transponder_coordinates[:, 0] - guess[0])**2 + (transponder_coordinates[:, 1] - guess[1])**2)
    abs_dist = np.linalg.norm(transponder_coordinates - guess, axis=1)
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz)
    times = abs_dist / esv
    return times, esv

def computeJacobian(guess, transponder_coordinates, times, sound_speed):
    # Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    diffs = transponder_coordinates - guess
    jacobian = -diffs / (times[:, np.newaxis] * (sound_speed ** 2))
    return jacobian

def computeJacobianRayTracing(guess, transponder_coordinates, times, sound_speed):
    # Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    diffs = transponder_coordinates - guess
    jacobian = -diffs / (times[:, np.newaxis] * (sound_speed[:, np.newaxis] ** 2))
    return jacobian

#Goal is to minimize sum of the difference of times squared
def geigersMethod(guess, CDog, transponder_coordinates_Actual,
                  transponder_coordinates_Found, time_noise=0):
    #Use Geiger's method to find the guess of CDOG location which minimizes sum of travel times squared
    #Define threshold
    epsilon = 10**-5

    #Sound Speed of water (right now constant, later will use ray tracing)
    sound_speed = 1515

    #Get known times
    # times_known = calculateTimes(CDog, transponder_coordinates_Actual, sound_speed)
    times_known, esv = calculateTimesRayTracing(CDog, transponder_coordinates_Actual)

    #Apply noise to known times
    times_known+=np.random.normal(0, time_noise, len(transponder_coordinates_Actual))
    # times_known+=noise

    k=0
    delta = 1
    estimate_arr = np.array([])
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        # times_guess = calculateTimes(guess, transponder_coordinates_Found, sound_speed)
        times_guess, esv = calculateTimesRayTracing(guess, transponder_coordinates_Found)
        # jacobian = computeJacobian(guess, transponder_coordinates_Found, times_guess, sound_speed)
        jacobian = computeJacobianRayTracing(guess, transponder_coordinates_Found, times_guess, esv)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        estimate_arr = np.append(estimate_arr, guess, axis=0)
        k+=1
    estimate_arr = np.reshape(estimate_arr, (-1, 3))
    return guess, times_known, estimate_arr

if __name__ == "__main__":
    from geigerTimePlot import geigerTimePlot
    from leverHist import leverHist

    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(10000)

    #Plot histograms of coordinate differences between found transponder and actual transponder
    # leverHist(transponder_coordinates_Actual,transponder_coordinates_Found)

    #Define noise
    time_noise = 2*10**-5
    position_noise = 2*10**-2

    #Apply noise to position
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

    #Make plot
    initial_guess = [-10000, 5000, -4000]
    geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual, transponder_coordinates_Found,
                   gps1_to_transponder, cz, depth, time_noise, position_noise)


# Geometric Dilulion of Precision is the square root of the trace of (J.t*J)^_1

#Evaluate error distribution at the truth and compare with the best guess

#Next steps to include:
#   Allow for an offset in the matching of GPS and arrival times (see how much of an impact this has on performance)
#   Write in code that makes it so the match GPS coordinates are at the time when emission arrives at receiver
#       Hence, the boat is actually displaced from the place where it emitted the acoustic pulse...