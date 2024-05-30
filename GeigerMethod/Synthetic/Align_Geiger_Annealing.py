import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from advancedGeigerMethod import calculateTimesRayTracing

from Unwrap_Fix import align

"""
Goal of this file is to iterate through finding the CDOG with the following steps:
    1) Align the travel time series for a CDOG guess as close as possible with the unwrapped CDOG data
    2) Apply Geiger's method to the aligned data to find an improved guess of CDOG position
    3) Use Synthetic Annealing to find the best guess for the transducer location (with a constricted range of possibilities)
    4) Iterate back to the alignment phase with the new CDOG guess and travel time series
    5) After a certain number of iterations take the given CDOG location (determine validity by looking at residual distribution)
    
Gotta rename indices data_DOG and dog_data are not discernible names lol

Need a Geiger's Method which can accept data in the form that I have
    I.E. Aligned time series data (with corresponding GPS Travel times)
    
One thing about alignment is the scaling of the unwrapping with the first index of travel time. In reality, this first index
    will not correspond with the first unwrapping index. Hence, it will be scaled off by some factor. This may not be a problem
    and actually might make it easier for my algorithm to recognize the correct alignment
    
Problem with the fact that first index of dog and first index of travel times are always a match
    Need to find a way to counteract this - maybe compare difference between consecutive travel times?
    
IMPORTANT NOTE
GNSS time series is always corresponding with transducer output!

Something is wrong with the alignment I believe, sub-integer wise. resulting in an incorrect local minimium
"""

data_DOG = sio.loadmat('../../GPSData/Synthetic_CDOG_noise_subint.mat')['tags'].astype(float)

transponder_coordinates = sio.loadmat('../../GPSData/Synthetic_transponder_noise_subint.mat')['xyz'].astype(float)
GPS_time = sio.loadmat('../../GPSData/Synthetic_transponder_noise_subint.mat')['time'].astype(float)

CDOG = np.array([-1979.9094551, 4490.73551826, -2011.85148619])

# TEST (I like it!) -> This starts the unwrapping at a reasonable value instead of 0 time

#Next add Geiger's Method suited for the given data

def computeJacobianRayTracing(guess, transponder_coordinates, times, sound_speed):
    # Computes the Jacobian, parameters are xyz coordinates and functions are the travel times
    diffs = transponder_coordinates - guess
    jacobian = -diffs / (times[:, np.newaxis] * (sound_speed[:, np.newaxis] ** 2))
    return jacobian

def geigersMethod(guess, transponder_coordinates, data_DOG, GPS_time):
    epsilon = 10**-7
    k=0
    delta = 1

    #add noise to transponder_coordinates
    # transponder_coordinates += np.random.normal(0, 2*10**-2, (len(transponder_coordinates), 3))

    # Find initial times
    travel_times, esv = calculateTimesRayTracing(guess, transponder_coordinates)

    # Align Data Initially
    full_times, dog_data, GPS_data, transponder_data, offset = align(data_DOG, GPS_time, travel_times,
                                                                       transponder_coordinates)
    old_offset = 0
    iterations = 0
    while old_offset != offset and iterations < 5:
    #Loop until change in guess is less than the threshold
        print(offset)
        k=0
        while np.linalg.norm(delta) > epsilon and k<10:
            #Find times
            travel_times_guess, esv = calculateTimesRayTracing(guess, transponder_data)

            #Do Gauss-Newton Iteration
            jacobian = computeJacobianRayTracing(guess, transponder_data, travel_times_guess, esv)

            delta = -1 * np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ (travel_times_guess - dog_data)
            guess = guess + delta
            # print(np.sqrt(np.sum((CDOG - guess)**2)) * 100, "cm")
            k+=1

        #Update based on new guess
        old_offset = offset
        travel_times, esv = calculateTimesRayTracing(guess, transponder_coordinates)
        full_times, dog_data, GPS_data, transponder_data, offset = align(data_DOG, GPS_time, travel_times,
                                                                     transponder_coordinates)
        iterations +=1
    return guess

#If guess is too far - iterate gauss newton a couple times
guess = np.array([-1971.9094551, 4475.73551826, -2007.85148619])

#Not converging when guess too far (prob some errors somewhere)
    #If it is hitting local min that would be sad :(


new_guess = geigersMethod(guess, transponder_coordinates, data_DOG, GPS_time)

# new_guess = geigersMethod(new_guess, transponder_coordinates, data_DOG, GPS_time)


print(new_guess)
print(CDOG)

print(np.sqrt(np.sum((new_guess - CDOG)**2)) * 100)


"""Time to verify that this will work with an incorrectly located transducer :O"""
from advancedGeigerMethod import findTransponder
#
# GPS_Coordinates = sio.loadmat('../../GPSData/Synthetic_GPS_noise_subint.mat')['xyz'].astype(float)
# GPS_time = sio.loadmat('../../GPSData/Synthetic_GPS_noise_subint.mat')['time'].astype(float)

#
# def simulatedAnnealing(n, GPS_Coordinates, CDOG, data_dog, GPS_time):
#     # gps1_to_others += np.random.normal(0, 2*10**-2, (4,3))
#     GPS_Coordinates += np.random.normal(0, 2 * 10 ** -2, (len(GPS_Coordinates), 4, 3))
#
#     #Set true values
#     gps1_to_others = np.array(
#         [[0, 0, 0], [-2.4054, -4.20905, 0.060621], [-12.1105, -0.956145, 0.00877], [-8.70446831, 5.165195, 0.04880436]])
#     true_lever = np.array([-12.4, 15.46, -15.24])
#
#     #Get initial values
#     old_lever = np.array([-7.5079, 6.411, -13.033])
#     transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
#     guess = np.array([-1971.9094551, 4505.73551826, -2007.85148619])
#
#     #align time series with initial conditions
#     #Get initial RMSE from geiger's method
#     #Adjust with random walk via simulated annealing
#     #change if RMSE is better
#     #   How often should I run the alignment algorithm (its slow) every 10 steps or so?
#
#     #Verify if it works if the CDOG is guessed right, and the offset is accurate
#
#     # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
#     times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
#
#
#     difference_data = times_calc - times_known
#     old_RMS = np.sqrt(np.nanmean(difference_data ** 2))
#
#     #Run simulated annealing
#     k=0
#     RMS_arr = [0]*(n-1)
#     while k<n-1: #Figure out how to work temp threshold
#         temp = np.exp(-k*7*(1/(n))) #temp schdule
#         displacement = ((np.random.rand(3)*2)-[1,1,1]) * temp
#         lever = old_lever + displacement
#
#         #Find RMS
#         transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
#         guess, times_known = geigersMethod(guess, CDog, transponder_coordinates_Actual,
#                                            transponder_coordinates_Found)
#
#         # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
#         times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
#
#         difference_data = times_calc - times_known
#         RMS = np.sqrt(np.nanmean(difference_data ** 2))
#         if RMS - old_RMS < 0: #What is acceptance condition?
#             old_lever = lever
#             old_RMS = RMS
#         print(k, old_RMS*100*1515, old_lever)
#         RMS_arr[k]=RMS*100*1515
#         k+=1
#     plt.plot(list(range(n-1)), RMS_arr)
#     plt.xlabel("Simulated Annealing Iteration")
#     plt.ylabel("RMSE from Inversion (cm)")
#     plt.title("Simulated Annealing Inversion for GPS to Transducer Lever Arm")
#     plt.show()
#     print(old_lever, gps1_to_transponder)
#
#     return old_lever
#
# simulatedAnnealing(300)