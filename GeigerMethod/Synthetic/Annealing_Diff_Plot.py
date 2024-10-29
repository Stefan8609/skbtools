from advancedGeigerMethod import *
from simulatedAnnealing_Synthetic import simulatedAnnealing
import matplotlib.pyplot as plt

def annealing_plot(n, time_noise, position_noise, lever_noise = 5, geom_noise = 0):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
    times_known, esv = calculateTimesRayTracing(CDog, transponder_coordinates_Actual)

    #Find transponder with noise and wrong lever
    starting_lever = gps1_to_transponder + np.random.normal(0, lever_noise, 3)
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, starting_lever)

    #Run Geiger's method with wrong lever
    offset_estimate = geigersMethod(starting_lever, CDog, transponder_coordinates_Actual,
                                    transponder_coordinates_Found, time_noise)[0]
    offset_times = calculateTimesRayTracing(offset_estimate, transponder_coordinates_Found)[0]

    #Run simulated annealing a make a corresponding plot below
    annealing_guess, lever = simulatedAnnealing(n, 300, time_noise, position_noise, geom_noise, False,
                                      CDog, GPS_Coordinates, transponder_coordinates_Actual,
                                      gps1_to_others, gps1_to_transponder)

    transponder_coordinates_annealing = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    annealing_estimate = geigersMethod(annealing_guess, CDog, transponder_coordinates_Actual,
                                        transponder_coordinates_annealing, time_noise)[0]
    annealing_times = calculateTimesRayTracing(annealing_estimate, transponder_coordinates_annealing)[0]

    print(annealing_estimate, offset_estimate, CDog)

    offset_diff = (offset_times - times_known)*100
    annealing_diff = (annealing_times - times_known)*100

    offset_std = np.std(offset_diff)
    annealing_std = np.std(annealing_diff)

    #Make a vertical figure with 2 plots of the residual time series
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [3, 1]})
    axs[0, 0].scatter(np.arange(len(offset_times)), offset_diff, marker='o', s=1)
    axs[0, 0].set_ylim(-3*offset_std, 3*offset_std)
    axs[0, 0].set_title('Geiger Method')
    axs[0, 0].set_ylabel('Time Residuals (ms)')
    axs[0, 0].set_xlabel('Time Known (ms)')
    axs[0, 0].grid()

    axs[1, 0].scatter(np.arange(len(annealing_times)), annealing_diff, marker='o', s=1)
    axs[1, 0].set_ylim(-3*offset_std, 3*offset_std)
    axs[1, 0].set_title('Simulated Annealing')
    axs[1, 0].set_ylabel('Time Residuals (ms)')
    axs[1, 0].set_xlabel('Time Known (ms)')
    axs[1, 0].grid()

    # Transducer trajectory plot
    axs[0, 1].scatter(transponder_coordinates_Actual[:, 0], transponder_coordinates_Actual[:, 1],
                      label='Transducer Trajectory', color='b', marker='o', s=5)
    axs[0, 1].scatter(CDog[0], CDog[1], label='CDOG Location', color='k', marker='x', s=20)
    axs[0, 1].set_title('Transducer Trajectory')
    axs[0, 1].set_xlabel('Easting (m)')
    axs[0, 1].set_ylabel('Northing (m)')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Hide the empty subplot
    fig.delaxes(axs[1, 1])
    axs[0, 1].set_aspect('equal', 'box')

    plt.tight_layout()
    plt.show()

    return

annealing_plot(10000, 0, 0)