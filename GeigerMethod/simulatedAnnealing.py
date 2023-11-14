import matplotlib.pyplot as plt
import numpy as np
from advancedGeigerMethod import *
from geigerTimePlot import geigerTimePlot
from experimentPathPlot import experimentPathPlot
from leverHist import leverHist

#Currently under assumption that arms b/w GPS are perfectly known

def simulatedAnnealing(n):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateCross(1000)

    # gps1_to_others += np.random.normal(0, 2*10**-2, (4,3))
    GPS_Coordinates += np.random.normal(0, 2 * 10 ** -2, (len(GPS_Coordinates), 4, 3))

    #Get initial values
    old_lever = [-10, 6, -13]
    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, old_lever)
    initial_guess = [-2000, 3000, -5000]
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                       transponder_coordinates_Found)
    times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
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
        displacement = ((np.random.rand(3)*2)-[1,1,1]) * temp
        lever = old_lever + displacement

        #Find RMS
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
        guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found)
        times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
        difference_data = times_calc - times_known
        RMS = np.sqrt(np.nanmean(difference_data ** 2))
        if RMS - old_RMS < 0: #What is acceptance condition?
            old_lever = lever
            old_RMS = RMS
        print(k, old_RMS*100*1515)
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
    geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual, transponder_coordinates_Final, gps1_to_transponder, old_lever)

    return gps1_to_transponder

simulatedAnnealing(300)

#Good plot to make is the RMS at each iteration of Simulated Annealing











# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle("Coordinate differences between calculated transponder and actual transponder, 2mm noise in lever arms", y=0.92)
# xmin, xmax = -15, 15
# colors = ["blue", 'red', 'green']
# axis = ["x", "y", "z"]
# mu_arr, std_arr, n_arr = [0]*3, [0]*3, [0]*3
# for i in range(3):
#     axs[i].set_xlim(xmin, xmax)
#     axs[i].set_ylim(0, 275)
#     n, bins, patches = axs[i].hist((transponder_coordinates_Found[:, i] - transponder_coordinates_Actual[:, i])*100,
#                                    bins=30, color=colors[i], alpha=0.7)
#     n_arr[i]=n.max()
#     mu_arr[i], std_arr[i] = norm.fit((transponder_coordinates_Found[:, i] - transponder_coordinates_Actual[:, i])*100)
# for i in range(3):
#     xmin, xmax = -3*max(std_arr), 3*max(std_arr)
#     x = np.linspace(xmin, xmax, 100)
#     axs[i].set_xlim(xmin, xmax)
#     axs[i].set_ylim(0, max(n_arr))
#     p = norm.pdf(x, mu_arr[i], std_arr[i])
#     p_noise = norm.pdf(x, 0, 2)
#     p *= n_arr[i] / p.max()
#     p_noise *= n_arr[i] / p_noise.max()
#     axs[i].plot(x, p, 'k', linewidth=2)
#     axs[i].plot(x, p_noise, color='y', linewidth=2)
#     axs[i].set_xlabel(f'{axis[i]}-difference(cm) std={round(std_arr[i], 4)} (cm)')
# plt.show()
