"""
This file creates random coordinates for GPS points and random coordinates for a CDOG.
    Then using a known (constant) sound speed, it finds the time it would take for an
    acoustic signal to reach the CDOG from the GPS points. It then adds noise to both
    the times calculated and the locations of the GPS points. Finally it applies
    Gauss-Newton Optimization (Geiger's method) using an initial guess to estimate the
    position of the CDOG using the noisy data.

Current Goals for this file:
    Add in parameters for the transducer (Have to find a way to orient the 'lever-arms')
        They can go any directions (requirement is distance stays the same from point to transducer)
        parameters are length, theta, and phi?

        Part of this needs to be adding rigor to findPointByPlane (as this is susceptible to noise)

        Steps needed to be taken:
            1)Make 4 GPS points, instead of one, for each time step (still have 1 CDOG)
                (The other 4 have to remain relative to eachother) basically rotations of the same plane
            2)Have the known times be set by a transducer with location relative to the GPS plane
                This transducer will have position determined by parameters of length, theta, phi from plane
            3)Use Gauss-Newton Optimization to estimate both CDOG location and transducer parameters
                Hard part is finding equation for times (as we need a new one that is dependent on new
                parameters in order to properly calculate for the Jacobians)
            4)Add noise to GPS positions, transducer position, CDOG position, etc

Written by Stefan Kildal-Brandt
"""

import numpy as np
import random
from fitPlane import fitPlane
from projectToPlane import projectToPlane

np.set_printoptions(precision = 2, suppress = True)


def generateRandomData():
    #Generate the random data in the form of numpy arrays
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])

    xyz_point = np.array([10, -10 , 10])
    translations = (np.random.rand(100,3) * 5000) - 2500
    translations = np.matmul(translations, np.array([[1,0,0],[0,1,0],[0,0,1/100]]))

    rot = (np.random.rand(100, 3) * np.pi) - np.pi/2

    #add in a another dimension that will contain the other GPS, and assign a vector to each GPS
    #Calculate these vectors with the rotation matrix
    GPS_Coordinates = np.zeros((100,4, 3))
    transponder_Coordinates = np.zeros((100,3))
    gps1_to_other = np.array([[0,0,0],[10, 1, -1],[11, 9, 1],[-1, 11, 0]])
    gps1_to_transponder = np.array([-10, 3, -15])

    for i in range(len(GPS_Coordinates)):
        xRot = np.array([[1, 0, 0], [0, np.cos(rot[i,0]), -np.sin(rot[i,0])], [0, np.sin(rot[i,0]), np.cos(rot[i,0])]])
        yRot = np.array([[np.cos(rot[i,1]), 0, np.sin(rot[i,1])], [0, 1, 0], [-np.sin(rot[i,1]), 0, np.cos(rot[i,1])]])
        zRot = np.array([[np.cos(rot[i,2]), -np.sin(rot[i,2]), 0], [np.sin(rot[i,2]), np.cos(rot[i,2]), 0], [0, 0, 1]])
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))
        for j in range(1, 4): #Add in other GPS with their rotations and translations
            GPS_Coordinates[i, j] = xyz_point + np.matmul(totalRot, gps1_to_other[j])
            GPS_Coordinates[i, j] += translations[i]

        #Put in transponder to get simulated times
        transponder_Coordinates[i] = xyz_point + np.matmul(totalRot, gps1_to_transponder)
        transponder_Coordinates[i] += translations[i]

        GPS_Coordinates[i, 0] = xyz_point + translations[i] #translate original point

    return CDog, GPS_Coordinates, transponder_Coordinates

def calculateTimes(guess, GPS_Coordinates, sound_speed): #Guess in form [x,y,z,L,theta,phi]
    #Calculate the time for the acoustic wave to travel the distance between a GPS point
    #   and a guess of the CDOG'S locaton
    times = np.zeros(len(GPS_Coordinates))
    barycenters = np.mean(GPS_Coordinates, axis=1 )

    for i in range(len(GPS_Coordinates)):
        normVect = fitPlane(GPS_Coordinates[i,:,0], GPS_Coordinates[i,:,1], GPS_Coordinates[i,:,2])

        #Choose reference GPS to rotate around
        reference = GPS_Coordinates[i,0]

        #Project reference onto plane and find orthogonal vector on plane to rotate around
        dot = np.dot(reference, normVect)
        reference = reference - (dot * normVect)
        reference_cross = np.cross(normVect, reference)
        reference_cross = reference_cross / np.sqrt(reference_cross.dot(reference_cross))

        #Lengthen then rotate via rodrigues rotation formula
        transponder_vect = normVect * guess[3]
        transponder_vect = transponder_vect*np.cos(guess[4])\
                   + np.cross(reference_cross, transponder_vect)*np.sin(guess[4])\
                   + reference_cross * np.dot(reference_cross, transponder_vect)*(1-np.cos(guess[4]))
        transponder_vect = transponder_vect * np.cos(guess[5]) \
                   + np.cross(reference_cross, transponder_vect) * np.sin(guess[5]) \
                   + reference_cross * np.dot(reference_cross, transponder_vect) * (1 - np.cos(guess[5]))

        #Calculate distance from transponder to cdog and then find the acoustic wave travel time
        transponder_to_cdog = guess[:3] - (barycenters[i] + transponder_vect)
        distance = np.sqrt(transponder_to_cdog.dot(transponder_to_cdog))
        times[i] = distance / sound_speed
    return times

def computeJacobian(guess, GPS_Coordinates, times, sound_speed):
    #Computes the Jacobian, parameters are xyz coordinates and functions are the travel times

    #Update jacobian for new parameters
    jacobian = np.zeros((len(GPS_Coordinates), 3))
    for i in range(len(GPS_Coordinates)):
        jacobian[i, 0] = (-1 * GPS_Coordinates[i, 0] + guess[0]) / (times[i]*(sound_speed**2))
        jacobian[i, 1] = (-1 * GPS_Coordinates[i, 1] + guess[1]) / (times[i]*(sound_speed**2))
        jacobian[i, 2] = (-1 * GPS_Coordinates[i, 2] + guess[2]) / (times[i]*(sound_speed**2))
    return jacobian


#Goal is to minimize sum of the difference of times squared
def geigersMethod(guess, CDog, GPS_Coordinates, transponder_Coordinates):
    #Use Geiger's method to find the guess of CDOG location
    #   which minimizes sum of travel times squared

    #Define threshold
    epsilon = 10**-4

    #Sound Speed of water (right now constant, later will use ray tracing)
    sound_speed = 1515

    #Apply Noise to times and GPS Coordinates (noise removed for now)
    times_known = calculateTimes(CDog, transponder_Coordinates, sound_speed)#+ np.random.normal(0,10**-3,len(GPS_Coordinates))
    # GPS_Coordinates = GPS_Coordinates + np.random.normal(0, 10**-2, (len(GPS_Coordinates), 3))

    k=0
    delta = 1
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        print(guess, CDog)
        times_guess = calculateTimes(guess, GPS_Coordinates, sound_speed)
        jacobian = computeJacobian(guess, GPS_Coordinates, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1

    return guess

CDog, GPS_Coordinates, transponder_Coordinates = generateRandomData()

print(CDog, '\n \n', GPS_Coordinates[:5])

times = calculateTimes(np.array([0,0,-5000, 6, 0, 0]), GPS_Coordinates, 1515)
print(times)
