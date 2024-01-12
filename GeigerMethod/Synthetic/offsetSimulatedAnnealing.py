import matplotlib.pyplot as plt
from advancedGeigerMethod import *
from offsetTimePlot import geigerTimePlotOffset
from experimentPathPlot import experimentPathPlot
from leverHist import leverHist

def simulatedAnnealingOffset(n, points, offset):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateCross(points)

    # gps1_to_others += np.random.normal(0, 2*10**-2, (4,3))
    GPS_Coordinates += np.random.normal(0, 2 * 10 ** -2, (len(GPS_Coordinates), 4, 3))

    #Get initial values
    old_lever = np.array([-7.5079, 6.411, -13.033])
    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
    initial_guess = [-2000, 3000, -4000]
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                       transponder_coordinates_Found)

    GPS_Coordinates = GPS_Coordinates[offset:]
    transponder_coordinates_Found = transponder_coordinates_Found[offset:]
    transponder_coordinates_Actual = transponder_coordinates_Actual[offset:]
    times_known = times_known[:len(transponder_coordinates_Found)]

    # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    difference_data = times_calc - times_known
    difference_data[points//2 - offset:points//2] = 0

    old_RMS = np.sqrt(np.nanmean(difference_data ** 2))

    #Plot initial conditions
    experimentPathPlot(transponder_coordinates_Actual, CDog)
    leverHist(transponder_coordinates_Actual, transponder_coordinates_Found)
    geigerTimePlotOffset(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual,
                         transponder_coordinates_Found, gps1_to_transponder, times_known, offset, points, old_lever)

    #Run simulated annealing
    k=0
    RMS_arr = [0]*(n-1)
    while k<n-1: #Figure out how to work temp threshold
        temp = np.exp(-k*7*(1/(n))) #temp schdule
        displacement = ((np.random.rand(3)*2)-[1,1,1]) * temp
        lever = old_lever + displacement

        #Find RMS
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        guess = geigersMethod(guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found)[0]

        # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        difference_data = times_calc - times_known
        difference_data[points // 2 - offset:points // 2] = 0

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
    geigerTimePlotOffset(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual,
                   transponder_coordinates_Final, gps1_to_transponder, times_known, offset, points, old_lever)

    return old_lever

simulatedAnnealingOffset(300, 10000, 2)

#Seems that an offset has a major effect on the shape and ability to zone in on a graph
#   Also the gps lands in the wrong spot with an offset