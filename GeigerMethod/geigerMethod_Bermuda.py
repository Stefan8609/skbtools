"""
[add description]

Written by Stefan Kildal-Brandt
"""

import numpy as np
from findPointByPlane import initializeFunction, findXyzt
import scipy.io as sio
from pymap3d import geodetic2ecef, ecef2geodetic
from geopy import distance

esv_table = sio.loadmat('../GPSData/global_table_esv.mat')
dz_array = esv_table['distance'].flatten()
angle_array = esv_table['angle'].flatten()
esv_matrix = esv_table['matrice']

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

def haversine_distance(lat1, lon1, lat2, lon2): #Modified from Thalia
    lat1, lon1, lat2, lon2 = lat1*np.pi/180, lon1*np.pi/180, lat2*np.pi/180, lon2*np.pi/180
    #Radius of Earth
    R = 6371e3
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    #Haversine formula for horizontal distance
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
#Make sure that this is actually right - Might be more accurate ways to do this

def find_esv(beta, dz): #Modified from Thalia
    idx_closest_dz = np.argmin(np.abs(dz_array[:, None] - dz), axis=0)
    idx_closest_beta = np.argmin(np.abs(angle_array[:, None] - beta), axis=0)
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv[0]

def calculateTimesRayTracing(guess, transponder_coordinates):
    times = np.zeros(len(transponder_coordinates))
    for i in range(len(transponder_coordinates)):
        lat_guess, lon_guess, depth_guess = ecef2geodetic(guess[0], guess[1], guess[2])
        lat_transponder, lon_transponder, depth_transponder = ecef2geodetic(transponder_coordinates[i,0],transponder_coordinates[i,1],transponder_coordinates[i,2])
        dz = abs(depth_guess - depth_transponder)
        dh = haversine_distance(lat_transponder, lon_transponder, lat_guess, lon_guess)
        print(distance.distance((lat_guess, lon_guess), (lat_transponder, lon_transponder)).km * 1000, dh)
        dh = distance.distance((lat_guess, lon_guess), (lat_transponder, lon_transponder)).km * 1000
        beta = 180/np.pi * np.arctan2(dz, dh)
        esv = find_esv(beta, dz)

        abs_dist = np.linalg.norm(guess - transponder_coordinates[i])
        times[i] = abs_dist/esv
        # print(esv, beta, dz)
    return times
    #Make sure that this is absolutely right, make it faster by using lat/lon coordinates
    #From the start. Consider how much the curvature of the earth affects our results
    #Instead of working in lat/lon, can use depth from conversion, then just use trig to
    #estimate dh

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
        # times_guess = calculateTimesRayTracing(guess, transponder_coordinates_Found)

        jacobian = computeJacobian(guess, transponder_coordinates_Found, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1
    return guess

#XYZ in ECEF does not coordinate with z being depth and xy corresponding to horizontal!
#Need to account for this when getting the dz and horizontal distance.

#What is the best way to calculate horizontal distance?
#There is a considerable (8m) curvature of the earth of 10 km
#Could I project out the depth to be the same? Then get horizontal dist?

#Important to consider curvature effects. Make sure ray-tracing is correct
#Implement Bud's algorithm for ray-tracing Gauss-Newton

#Vectorize everything -- Should make code a decent amount faster (too slow as is)