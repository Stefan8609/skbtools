import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from advancedGeigerMethod import geigersMethod, generateRealistic, calculateTimesRayTracing, findTransponder

def fourPlot(n, time_noise, position_noise):
    #Initialize the same conditions for each plot
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
    initial_guess = [-10000, 5000, -4000]

    #Run 1 - No noise:
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Actual, 0)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Actual)[0]
    diff_data1 = times_calc - times_known
    mu1, std1 = norm.fit(diff_data1 * 1000)

    #Run 2 - Time noise only:
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Actual, time_noise)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Actual)[0]
    diff_data2 = times_calc - times_known
    mu2, std2 = norm.fit(diff_data2 * 1000)

    #Run 3 - Position noise only:
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, 0)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
    diff_data3 = times_calc - times_known
    mu3, std3 = norm.fit(diff_data3 * 1000)

    #Run 4 - Position and Time noise
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, time_noise)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
    diff_data4 = times_calc - times_known
    mu4, std4 = norm.fit(diff_data4 * 1000)

    #Plot the normal curves for each run
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    input_noise = time_noise*1000 + position_noise/1500 * 1000
    min_std = min([std2, std3, std4])
    height = norm.pdf(0, 0, min_std) + 1

    #Plot 1
    x = np.linspace(mu1-3*std1, mu1+3*std1, 100)
    p = norm.pdf(x, mu1, std1)
    axes[0, 0].hist(diff_data1 * 1000, orientation='vertical', bins=30, alpha=0.5, density=True)
    axes[0, 0].set_xlim([-3*input_noise, 3*input_noise])
    axes[0, 0].set_ylim([0, height])
    axes[0, 0].plot(x, p, 'k', linewidth=2, label="Normal Distribution of Differences")
    axes[0, 0].set_xlabel("Noise (ms)")
    axes[0, 0].set_ylabel("Probability Density")
    axes[0, 0].set_title("No Time or Position Noise")

    #Plot 2
    x = np.linspace(mu2-3*std2, mu2+3*std2, 100)
    p = norm.pdf(x, mu2, std2)
    axes[0, 1].hist(diff_data2 * 1000, orientation='vertical', bins=30, alpha=0.5, density=True)
    axes[0, 1].set_xlim([-3*input_noise, 3*input_noise])
    axes[0, 1].set_ylim([0, height])
    axes[0, 1].plot(x, p, 'k', linewidth=2, label="Normal Distribution")
    axes[0, 1].set_xlabel("Noise (ms)")
    axes[0, 1].set_ylabel("Probability Density")
    axes[0, 1].set_title(f"Input Time Noise: {time_noise*1000} ms")

    std_height, noise_height = norm.pdf(std2, mu2, std2), norm.pdf(time_noise*1000, 0, std2)
    axes[0, 1].arrow(mu2, std_height, std2, 0, length_includes_head = True, color='k', label="Observed Noise")
    axes[0 ,1].arrow(mu2, noise_height, time_noise*1000, 0, length_includes_head = True, color='r', label="Time Noise")
    axes[0, 1].vlines([mu2-std2, mu2+std2, -time_noise*1000, time_noise*1000], [0,0,0,0],
                      [std_height, std_height, noise_height, noise_height], colors=['k','k','r','r'], linestyles='dashed')
    axes[0, 1].legend(loc='upper right', fontsize = 'small')


    #Plot 3
    x = np.linspace(mu3-3*std3, mu3+3*std3, 100)
    p = norm.pdf(x, mu3, std3)
    axes[1, 0].hist(diff_data3 * 1000, orientation='vertical', bins=30, alpha=0.5, density=True)
    axes[1, 0].set_xlim([-3*input_noise, 3*input_noise])
    axes[1, 0].set_ylim([0, height])
    axes[1, 0].plot(x, p, 'k', linewidth=2, label="Normal Distribution")
    axes[1, 0].set_xlabel("Noise (ms)")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].set_title(f"Input Position Noise: {position_noise*100} cm")

    std_height, noise_height = norm.pdf(std3, mu3, std3), norm.pdf(position_noise/1515*1000, 0, std3)
    axes[1, 0].arrow(mu3, std_height, std3, 0, length_includes_head = True, color='k', label="Observed Noise")
    axes[1 ,0].arrow(mu3, noise_height, position_noise/1515*1000, 0, length_includes_head = True, color='g', label="Position Noise")
    axes[1, 0].vlines([mu3-std3, mu3+std3, -position_noise/1515*1000, position_noise/1515*1000], [0,0,0,0],
                      [std_height, std_height, noise_height, noise_height], colors=['k','k','g','g'], linestyles='dashed')
    axes[1, 0].legend(loc='upper right', fontsize = 'small')


    #Plot 4
    x = np.linspace(mu4-3*std4, mu4+3*std4, 100)
    p = norm.pdf(x, mu4, std4)
    axes[1, 1].hist(diff_data4 * 1000, orientation='vertical', bins=30, alpha=0.5, density=True)
    axes[1, 1].set_xlim([-3*input_noise, 3*input_noise])
    axes[1, 1].set_ylim([0, height])
    axes[1, 1].plot(x, p, 'k', linewidth=2, label="Normal Distribution")
    axes[1, 1].set_xlabel("Noise (ms)")
    axes[1, 1].set_ylabel("Probability Density")
    axes[1, 1].set_title(f"Input Time Noise: {time_noise*1000} ms, Input Position Noise: {position_noise*100} cm")

    std_height, time_noise_height, position_noise_height = (norm.pdf(std4, mu4, std4), norm.pdf(time_noise*1000, 0, std4),
                                                            norm.pdf(position_noise/1515*1000, 0, std4))
    axes[1, 1].arrow(mu4, std_height, std4, 0, length_includes_head = True, color='k', label="Observed Noise")
    axes[1 ,1].arrow(mu4, time_noise_height, time_noise*1000, 0, length_includes_head = True, color='r', label="Time Noise")
    axes[1 ,1].arrow(mu4, position_noise_height, position_noise/1515*1000, 0, length_includes_head = True, color='g', label="Position Noise")
    axes[1, 1].vlines([mu4-std4, mu4+std4, -time_noise*1000, time_noise*1000, -position_noise/1515*1000, position_noise/1515*1000], [0,0,0,0,0,0],
                      [std_height, std_height, time_noise_height, time_noise_height, position_noise_height, position_noise_height],
                      colors=['k','k','r','r','g','g'], linestyles='dashed')
    axes[1, 1].legend(loc='upper right', fontsize = 'small')

    plt.show()

fourPlot(10000, 2*10**-5, 2*10**-2)



