"""
[add description]

Written by Stefan Kildal-Brandt
"""

import numpy as np
from findPointByPlane import initializeFunction, findXyzt

def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
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
def geigersMethod(guess, times_known, transponder_coordinates_Found, sound_speed):
    #Use Geiger's method to find the guess of CDOG location which minimizes sum of travel times squared
    #Define threshold
    epsilon = 10**-5
    #Sound Speed of water (right now constant, later will use ray tracing)

    k=0
    delta = 1
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        times_guess = calculateTimes(guess, transponder_coordinates_Found, sound_speed)
        jacobian = computeJacobian(guess, transponder_coordinates_Found, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1
    return guess