"""
[add description]

Written by Stefan Kildal-Brandt
"""

import numpy as np
import random
from findPointByPlane import initializeFunction, findXyzt

def generateRandomData(n): #Generate the random data in the form of numpy arrays
    #Generate CDog
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])

    #Generate and initial GPS point to base all others off of
    xyz_point = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-100, 100)])

    #Generate the translations from initial point (random x,y,z translation with z/100) for each time step
    translations = (np.random.rand(n,3) * 6000) - 3000
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
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])
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
    x_coords1 = (np.random.rand(n//2) * 5000) - 2500
    x_coords2 = (np.random.rand(n//2) * 5000) - 2500
    x_coords = np.concatenate((np.sort(x_coords1), np.sort(x_coords2)))
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    y_coords[n//2:] *= -1
    z_coords = (np.random.rand(n) * 50) - 25
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

def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    #Add some noise to initial information
    # gps1_to_others += np.random.normal(0, 2*10**-3, (4,3))
    # gps1_to_transponder += np.random.normal(0, 2*10**-3, 3)
    # gps1_to_transponder += np.array([0,0,0])

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
def geigersMethod(guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, noise):
    #Use Geiger's method to find the guess of CDOG location which minimizes sum of travel times squared
    #Define threshold
    epsilon = 10**-5

    #Sound Speed of water (right now constant, later will use ray tracing)
    sound_speed = 1515

    #Get known times
    times_known = calculateTimes(CDog, transponder_coordinates_Actual, sound_speed)
    #Apply noise to known times on scale of 20 microseconds
    # times_known+=np.random.normal(0,2*10**-5,len(transponder_coordinates_Actual))
    times_known+=noise

    k=0
    delta = 1
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        times_guess = calculateTimes(guess, transponder_coordinates_Found, sound_speed)
        jacobian = computeJacobian(guess, transponder_coordinates_Found, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1
    return guess, times_known

if __name__ == "__main__":
    from experimentPathPlot import experimentPathPlot
    from geigerTimePlot import geigerTimePlot
    from leverHist import leverHist

    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateCross(2000)


    #Add noise to GPS on scale of 2 cm
    GPS_Coordinates += np.random.normal(0, 2*10**-2, (len(GPS_Coordinates), 4, 3))

    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

    #Plot histograms of coordinate differences between found transponder and actual transponder
    leverHist(transponder_coordinates_Actual,transponder_coordinates_Found)

    #Plot path of experiment
    experimentPathPlot(transponder_coordinates_Actual, CDog)

    #Plot comparison of times
    initial_guess = [-2000, 3000, -5000]
    geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, gps1_to_transponder)



#Need to correct functions for generating line data and cross data
#Need a good function to plot experimental paths (maybe dual plot comparing calculated transponder with actual location)

#Interesting things to note:
    # Being off in terms of initial information given to findTransponder function introduces skew in the x,y,z coordinate
    # difference histograms. This greatly affects the quality of the data
    # This skew is not seen in the Acoustic vs GNNS arrival time plots, meaning that it is hidden beneath the RMS


#Potential areas of exploration. With noise of lever arms and gps locations,
#   create an inversion to find the best lever arm to minimize error further down the line
#   See if this has the potential to work because this could fix skew with
#   high noise in the lever arm (real scenario for our gps)

#Save as pdf or eps, scale plot 1x1 in x and y, just transponder, smaller dots

#Put standard deviation in histograms, make them all in the same scale, also plot normal distribution
#   Color them different. Plot the noise gaussian as well.

#Get an understanding of the systematic offset (I'm off in the leverarms)
    # Error should be completely structured
    # Next algorithm needs to be able to find the systematic offset in lever arms
    # Inversion technique to place transducer
    # Figure out how to untrend residuals...

    #Simulated annealing could work may be expensive in time though