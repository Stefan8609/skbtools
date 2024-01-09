import matplotlib.pyplot as plt
import numpy as np
from advancedGeigerMethod import *
from geigerTimePlot import geigerTimePlot
from experimentPathPlot import experimentPathPlot
from leverHist import leverHist

def tempSchedule(RMSE):
    if RMSE * 100 * 1515 > 100:
        temp = 1
    elif RMSE * 100 * 1515 > 50:
        temp = .5
    elif RMSE * 100 * 1515 > 10:
        temp = .25
    elif RMSE * 100 * 1515 > 6:
        temp = .1
    elif RMSE * 100 * 1515 > 5.1:
        temp = .01
    elif RMSE * 100 * 1515 > 4.9:
        temp = .002
    else:
        temp = .0007
    return temp


def simulatedAnnealing(n):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateCross(1000)

    # gps1_to_others += np.random.normal(0, 2*10**-2, (4,3))
    GPS_Coordinates += np.random.normal(0, 2 * 10 ** -2, (len(GPS_Coordinates), 4, 3))

    #Get initial values
    old_lever = np.array([-7.5079, 6.411, -13.033])
    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
    initial_guess = [-2000, 3000, -4000]
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                       transponder_coordinates_Found)

    # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)


    difference_data = times_calc - times_known
    old_RMS = np.sqrt(np.nanmean(difference_data ** 2))

    #Plot initial conditions
    experimentPathPlot(transponder_coordinates_Actual, CDog)
    leverHist(transponder_coordinates_Actual, transponder_coordinates_Found)
    geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, gps1_to_transponder, old_lever)

    #Run simulated annealing
    k=0
    RMS_arr = [0]*(n-1)
    while k<n-1: #Figure out how to work temp threshold
        # if k<n/2:
        #     temp = 1-(k+1)/n
        # else:
        temp = np.exp(-k*7*(1/(n))) #temp schdule
        # temp = tempSchedule(old_RMS)
        print(temp)
        best_displacement=[0,0,0]
        best_RMSE = np.inf
        for i in range(1):
            displacement = ((np.random.rand(3)*2)-[1,1,1]) * temp
            print(old_lever + displacement)
            transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, old_lever+displacement)

            # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
            times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)

            difference_data = times_calc - times_known
            RMSE = np.sqrt(np.nanmean(difference_data ** 2))
            if best_RMSE > RMSE:
                best_RMSE = RMSE
                best_displacement = displacement

        lever = old_lever + best_displacement

        #Find RMS
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        guess, times_known = geigersMethod(guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found)

        # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)

        difference_data = times_calc - times_known
        RMS = np.sqrt(np.nanmean(difference_data ** 2))
        if RMS - old_RMS < 0: #What is acceptance condition?
            old_lever = lever
            old_RMS = RMS
        print(k, old_RMS*100*1515, old_lever)
        RMS_arr[k]=RMS*100*1515
        k+=1
    plt.plot(list(range(n-1)), RMS_arr)
    plt.xlabel("Simulated Annealing Iteration")
    plt.ylabel("RMSE from Inversion (cm)")
    plt.title("Simulated Annealing Inversion for GPS to Transducer Lever Arm")
    plt.show()
    print(old_lever, gps1_to_transponder)

    transponder_coordinates_Final = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
    leverHist(transponder_coordinates_Actual, transponder_coordinates_Final)
    geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual,
                   transponder_coordinates_Final, gps1_to_transponder, old_lever)

    return old_lever

simulatedAnnealing(300)

#at each time step keep the cdog location and then calculate a bunch of deviations
#   and keep the best one at each time step --Could speed up
#   Also scale with RMSE (see if this works for temperature)
#   Genetic algorithm, Neighborhood algorithm
#   Plot the ones I keep on the plot
#   Markov Chain Monte Carlo - https://agupubs.onlinelibrary.wiley.com/doi/10.1029/94JB03097
#   Base off barycenter rather then GPS1
#   Solve for all 4 gps offsets and see if position matches
#   Derben-Watson test

#   Normalize histogram by normalizing area of bins to be 1
#   See what histogram noise looks like with no time noise
#   Then do the same with no gps noise with time noise to see how histograms change
#   Variance Reduction

#Add panel to time plot giving a zoomed window of points explaining the trajectory and
#On same axis show the same thing

#See what pathing I get if I have straight cross in x-y and no perturbation in z
#Break down the error