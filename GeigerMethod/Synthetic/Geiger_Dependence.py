import numpy as np
import matplotlib.pyplot as plt
import random
from advancedGeigerMethod import generateRealistic, geigersMethod, calculateTimesRayTracing, findTransponder

def time_dependence(n):
    #write function to find the dependence of geiger's method with noise
    #Returns array of observed standard deviations
    noise_arr = []
    for noise in np.logspace(-5, -1, 25):
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
        initial_guess = [-10000, 5000, -4000]

        transponder_coordinates_Found = transponder_coordinates_Actual

        guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, noise)[:2]

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)

        noise_arr.append(std_diff)

    #Plot y=x line
    x = np.logspace(-5.5, -.5, 100)
    plt.plot(x, x, color='k', label="Line y=x")

    plt.scatter(np.logspace(-5, -1, 25), noise_arr, label="Observed Noise")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10**-5.1, .09])
    plt.xlabel('Input Time Noise')
    plt.ylabel('Derived Uncertainty in Estimation Position')
    plt.legend()
    plt.show()

def spatial_dependence(n):
    noise_arr = []
    for noise in np.logspace(-3, 0, 25):
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
        initial_guess = [-10000, 5000, -4000]

        GPS_Coordinates += np.random.normal(0, noise, (len(GPS_Coordinates), 4, 3))
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

        guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found, 0)[:2]

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)

        noise_arr.append(std_diff)

    #Find and plot regression line
    m,b = np.polyfit(np.logspace(-3, 0, 25), noise_arr, deg=1)
    x = np.logspace(-3, 0, 25)
    plt.plot(x, x*m, color='k', label=f"Line y={np.round(m,5)}*x")
    print(f"Slope = {np.round(m,5)} and intercept = {np.round(b,5)}")

    plt.scatter(np.logspace(-3, 0, 25), noise_arr, label="Observed Noise")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input GPS Noise')
    plt.ylabel('Derived Uncertainty in Estimation Position')
    plt.legend()
    plt.show()

def point_dependence(time_noise, position_noise):
    point_arr = []
    for n in np.logspace(1, 5, 25):
        n = int(n)
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
        initial_guess = [-10000, 5000, -4000]

        GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

        guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found, time_noise)[:2]

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)

        point_arr.append(std_diff)

    plt.scatter(np.logspace(1, 5, 25), point_arr)
    plt.xscale('log')
    plt.xlabel('Number of Points')
    plt.ylabel('Derived Uncertainty in Estimation Position')
    plt.ylim([0, (time_noise + position_noise/1515) * 1.1 ])
    plt.axhline(time_noise, color='y', label=f"Time Noise: {time_noise*1000} ms")
    plt.axhline(position_noise/1515, color='r', label=f"Position Noise: {position_noise*100} cm")
    plt.axhline(time_noise + position_noise/1515, color='b', label="Time and Position Noise")
    plt.legend(loc = "lower right")
    plt.show()

def combined_dependence(n):
    space_axis = np.linspace(2*10**-2, 2*10**-1, 50)
    time_axis = np.linspace(2*10**-5, 2*10**-4, 50)

    [X, Y] = np.meshgrid(space_axis, time_axis)

    Z = np.zeros((len(Y[:,0]), len(X[0])))

    for i in range(len(X[0])):
        print(i)
        for j in range(len(Y[:,0])):
            CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
            initial_guess = np.array([random.uniform(-10000, 10000), random.uniform(-10000, 10000), random.uniform(-4000, -6000)])

            GPS_Coordinates += np.random.normal(0, X[0,i], (len(GPS_Coordinates), 4, 3))
            transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

            guess, times_known, est = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                               transponder_coordinates_Found, Y[j,0])

            times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
            diff_data = times_calc - times_known
            std_diff = np.std(diff_data)
            Z[j, i] = std_diff

    #Plot expected uncertainty contours
    Z_exp = np.sqrt(0.00103**2*np.square(X) + np.square(Y))

    CS = plt.contour(X*100, Y*1000, Z_exp*1515*100, colors="k", levels=np.arange(0,50,5))
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.contourf(X*100, Y*1000, Z*1515*100, levels=np.arange(0,50,5))
    cbar = plt.colorbar()
    cbar.set_label("Estimate Position Noise (cm)", rotation= 270, labelpad = 15)
    plt.xlabel("GPS Position Noise (cm)")
    plt.ylabel("C-DOG Time Noise (ms)")
    plt.title("Combined dependence of GPS position and C-DOG time noise")
    plt.show()

def random_dependence(points, n):
    time_noise = np.random.normal(5*10**-5, 1*10**-5, points)
    position_noise = np.random.normal(5*10**-2, 1*10**-2, points)

    std_expected = np.sqrt(time_noise**2 + (position_noise/1515)**2)
    std_observed = np.zeros(points)
    for i in range(points):
        if i%10 == 0:
            print(i)
        CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)
        initial_guess = np.array(
            [random.uniform(-10000, 10000), random.uniform(-10000, 10000), random.uniform(-4000, -6000)])

        GPS_Coordinates += np.random.normal(0, position_noise[i], (len(GPS_Coordinates), 4, 3))
        transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

        try:
            guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual,
                                           transponder_coordinates_Found, time_noise[i])[:2]
        except:
            print(initial_guess, time_noise[i], position_noise[i])
            std_observed[i] = 0
            continue

        times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
        diff_data = times_calc - times_known
        std_diff = np.std(diff_data)
        std_observed[i] = std_diff


    plt.scatter(time_noise * 1515 * 100, position_noise*100, c=(std_observed/std_expected))
    cbar = plt.colorbar()
    cbar.set_label('Ratio of observed to expected noise', fontsize=12)
    plt.xlabel("Time Noise times sound speed (cm)")
    plt.ylabel("Position Noise (cm)")
    plt.show()




# time_dependence(1000)
# spatial_dependence(10000)
# point_dependence(2*10**-5, 2*10**-2)
combined_dependence(1000)

# random_dependence(500, 1000)
