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
import time

def generateRandomData(n):
    #Generate the random data in the form of numpy arrays
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])
    x_coords = (np.random.rand(n) * 5000) - 2500
    y_coords = (np.random.rand(n) * 5000) - 2500
    z_coords = (np.random.rand(n) * 50) - 25
    GPS_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    return CDog, GPS_Coordinates #Output CDog [x,y,z] and GPS_Coordinates [x len(n), ylen(n), z len(n)]

def generateLine(n):
    #Generates GPS data for a line of points
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000,1000), random.uniform(-5500, -4500)])
    x_coords = (np.random.rand(n) * 5000) - 2500
    y_coords = x_coords + (np.random.rand(n) * 20) - 10 #variation around x-coord
    z_coords = (np.random.rand(n) * 50) - 25
    GPS_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    GPS_Coordinates = sorted(GPS_Coordinates, key = lambda k: [k[0],k[1],k[2]])
    return CDog, GPS_Coordinates #Output CDog [x,y,z] and GPS_Coordinates [x len(n), ylen(n), z len(n)]

def generateCross(n):
    CDog = np.array([random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(-5500, -4500)])
    line1 = generateLine(int(n/2))[1]
    line2 = line1 @ np.array([[1,0,0],[0,-1,0],[0,0,1]])
    GPS_Coordinates = np.concatenate((line1,line2), axis=0)
    return CDog, GPS_Coordinates #Output CDog [x,y,z] and GPS_Coordinates [x len(n), ylen(n), z len(n)]



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
    # GPS_Coordinates = GPS_Coordinates + np.random.normal(0, 10**-2, (len(GPS_Coordinates), 3))

    k=0
    delta = 1
    #Loop until change in guess is less than the threshold
    while np.linalg.norm(delta) > epsilon and k<100:
        times_guess = calculateTimes(guess, GPS_Coordinates, sound_speed)
        jacobian = computeJacobian(guess, GPS_Coordinates, times_guess, sound_speed)
        delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (times_guess-times_known)
        guess = guess + delta
        k+=1

    return guess, times_known

if __name__ == '__main__':
    CDog, GPS_Coordinates = generateCross(500)

    GPS_Coordinates = GPS_Coordinates + np.random.normal(0, 10**-2, (len(GPS_Coordinates), 3)) #Add noise to GPS

    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('GPS Coordinates from which arrival times are calculated')
    plt.scatter(GPS_Coordinates[:,0], GPS_Coordinates[:,1], GPS_Coordinates[:,2])
    plt.show()
    # result = geigersMethod([0,10,-5000], CDog, GPS_Coordinates)
    #
    # print(result , CDog, np.linalg.norm(result - CDog))

    result, times_act = geigersMethod(np.array([100,-100,-5000]), CDog, GPS_Coordinates)
    times_calc = calculateTimes(result, GPS_Coordinates, 1515)

    difference_data = times_calc - times_act
    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(RMS)

    # Prepare label and plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
    fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Antenna', y=0.92)

    # Acoustic vs GNSS plot
    GPS_Coord_Num = list(range(len(GPS_Coordinates)))

    axes[0, 1].scatter(GPS_Coord_Num, times_act, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    axes[0, 1].scatter(GPS_Coord_Num, times_calc, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].text(25, max(times_act), "actual arrival times versus estimated times", bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend(loc="upper right")

    # Difference plot
    axes[1, 1].scatter(GPS_Coord_Num, difference_data, s=1)
    axes[1, 1].set_xlabel('Position Index')
    axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')
    axes[1, 1].legend()

    # Histogram
    axes[1, 0].hist(difference_data, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (s)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title(f"RMS: {round(RMS*1515,4)} m")
    axes[0, 0].axis('off')

    plt.show()

    times = np.zeros(49)
    for n in range(100, 5000, 100):
        CDog, GPS_Coordinates = generateCross(n)
        start = time.time()
        geigersMethod([0,0,-5000], CDog, GPS_Coordinates)
        end = time.time()
        times[int(n/100)-1]=(end-start)*100
    plt.plot(list(range(100,5000,100)), times)
    plt.xlabel("Number of iterations")
    plt.ylabel("Run Time (ms)")
    plt.title("Run time of Geiger's Method Algorithm")
    plt.show()




    # # Define the dimensions of the grid
    # x_min, x_max = -1000, 1000  # Define your x-coordinate range
    # y_min, y_max = -1000, 1000  # Define your y-coordinate range
    # num_rows, num_columns = 21, 21  # Define the number of rows and columns
    #
    # # Generate coordinate grids
    # x = np.linspace(x_min, x_max, num_columns)
    # y = np.linspace(y_min, y_max, num_rows)
    # X, Y = np.meshgrid(x, y)
    #
    # data = np.zeros((num_rows, num_columns))
    # idx1 = 0
    # idx2 = 0
    # for i in x:
    #     print(i)
    #     for j in y:
    #         result = geigersMethod(np.array([i,j,-5000]), CDog, GPS_Coordinates)
    #         data[idx2, idx1] += np.linalg.norm(result - CDog)
    #         idx2 +=1
    #     idx2=0
    #     idx1+=1
    #
    # # Create the color plot
    # plt.imshow(data, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')#, interpolation='bilinear')
    # plt.colorbar()
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('2D Color Plot')
    # plt.show()