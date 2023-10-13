"""
This file creates random coordinates for GPS points (orientations include line, cross,
    and a completely random distribution) and random coordinates for a CDOG.
    Then using a known (constant) sound speed, it finds the time it would take for an
    acoustic signal to reach the CDOG from the GPS points. It then adds noise to both
    the times calculated and the locations of the GPS points. Finally it applies
    Gauss-Newton Optimization (Geiger's method) using an initial guess to estimate the
    position of the CDOG using the noisy data.

Written by Stefan Kildal-Brandt

Try different Geometries (gps in line, cross, etc)
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def generateRandomData(n):
    #Generate the random data in the form of numpy arrays
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])
    x_coords = (np.random.rand(n) * 5000) - 2500
    y_coords = (np.random.rand(n) * 5000) - 2500
    z_coords = (np.random.rand(n) * 50) - 25
    GPS_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    return CDog, GPS_Coordinates

def generateLine(n):
    #Generates GPS data for a line of points
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])
    x_coords = (np.random.rand(n) * 5000) - 2500
    y_coords = x_coords + (np.random.rand(n) * 20) - 10
    z_coords = (np.random.rand(n) * 50) - 25
    GPS_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    return CDog, GPS_Coordinates

def generateCross(n):
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(-5500, -4500)])
    line1 = generateLine(int(n/2))[1]
    line2 = line1 @ np.array([[1,0,0],[0,-1,0],[0,0,1]])
    GPS_Coordinates = np.concatenate((line1,line2), axis=0)
    return CDog, GPS_Coordinates



def calculateTimes(guess, GPS_Coordinates, sound_speed):
    #Calculate the time for the acoustic wave to travel the distance between a GPS point
    #   and a guess of the CDOG'S locaton
    times = np.zeros(len(GPS_Coordinates))
    for i in range(len(GPS_Coordinates)):
        distance = np.linalg.norm(GPS_Coordinates[i] - guess)
        times[i] = distance / sound_speed
    return times

def computeJacobian(guess, GPS_Coordinates, times, sound_speed):
    #Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    jacobian = np.zeros((len(GPS_Coordinates), 3))
    for i in range(len(GPS_Coordinates)):
        jacobian[i, 0] = (-1 * GPS_Coordinates[i, 0] + guess[0]) / (times[i]*(sound_speed**2))
        jacobian[i, 1] = (-1 * GPS_Coordinates[i, 1] + guess[1]) / (times[i]*(sound_speed**2))
        jacobian[i, 2] = (-1 * GPS_Coordinates[i, 2] + guess[2]) / (times[i]*(sound_speed**2))
    return jacobian


#Goal is to minimize sum of the difference of times squared
def geigersMethod(guess, CDog, GPS_Coordinates):
    #Use Geiger's method to find the guess of CDOG location
    #   which minimizes sum of travel times squared

    #Define threshold
    epsilon = 10**-6

    #Sound Speed of water (right now constant, later will use ray tracing)
    sound_speed = 1515

    #Apply Noise to times and GPS Coordinates
    times_known = calculateTimes(CDog, GPS_Coordinates, sound_speed) + np.random.normal(0,10**-5,len(GPS_Coordinates))
    GPS_Coordinates = GPS_Coordinates + np.random.normal(0, 10**-2, (len(GPS_Coordinates), 3))

    k=0
    delta = 1
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        times_guess = calculateTimes(guess, GPS_Coordinates, sound_speed)
        jacobian = computeJacobian(guess, GPS_Coordinates, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1

    return guess

CDog, GPS_Coordinates = generateLine(100)

# Define the dimensions of the grid
x_min, x_max = -1000, 1000  # Define your x-coordinate range
y_min, y_max = -1000, 1000  # Define your y-coordinate range
num_rows, num_columns = 21, 21  # Define the number of rows and columns

# Generate coordinate grids
x = np.linspace(x_min, x_max, num_columns)
y = np.linspace(y_min, y_max, num_rows)
X, Y = np.meshgrid(x, y)

data = np.zeros((num_rows, num_columns))
idx1 = 0
idx2 = 0
for i in x:
    print(i)
    for j in y:
        result = geigersMethod(np.array([i,j,-5000]), CDog, GPS_Coordinates)
        data[idx2, idx1] = np.linalg.norm(result - CDog)
        idx2 +=1
    idx2=0
    idx1+=1

print(data)
# Create the color plot
plt.imshow(data, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')#, interpolation='bilinear')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Color Plot')
plt.show()