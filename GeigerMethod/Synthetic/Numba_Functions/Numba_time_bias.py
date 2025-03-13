import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from scipy.stats import norm
from Numba_Geiger import generateRealistic, findTransponder
from matplotlib.patches import Ellipse


"""
Need an short algorithm to estimate the derivative of the ESV in x,y, and z
    According to the algorithm listed by bud (short and simple numerical method)
    Remove timing bias term from bud's algorithm (or investigate how it works as sub-integer offset)
    Implement Bud's algorithm for finding ESV bias
    
This algorithm finds a fixed bias (that is a constant in time and depth)

 - Stop the program from diverging
 
 Next Steps:
  - Implement this alongside alignment (How can the sound bias term be used to improve alignment precision??)
  - Implement the combination with simulated annealing for transducer offset
  
Tau vs P plot for the rays in the ocean
"""

@njit
def find_esv(beta, dz, dz_array, angle_array, esv_matrix):
    idx_closest_dz = np.empty_like(dz, dtype=np.int64)
    idx_closest_beta = np.empty_like(beta, dtype=np.int64)

    for i in range(len(dz)):
        idx_closest_dz[i] = np.searchsorted(dz_array, dz[i], side="left")
        if idx_closest_dz[i] < 0:
            idx_closest_dz[i] = 0
        elif idx_closest_dz[i] >= len(dz_array):
            idx_closest_dz[i] = len(dz_array) - 1

        idx_closest_beta[i] = np.searchsorted(angle_array, beta[i], side="left")
        if idx_closest_beta[i] < 0:
            idx_closest_beta[i] = 0
        elif idx_closest_beta[i] >= len(angle_array):
            idx_closest_beta[i] = len(angle_array) - 1

    closest_esv = np.empty_like(dz, dtype=np.float64)
    for i in range(len(dz)):
        closest_esv[i] = esv_matrix[idx_closest_dz[i], idx_closest_beta[i]]

    return closest_esv

@njit
def calculateTimesRayTracing_Bias(guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix):
    """
    Ray Tracing calculation of times using ESV
        Capable of handling an ESV bias input term
        (whether a constant or an array with same length as transponder_coordinates)
    """
    hori_dist = np.sqrt((transponder_coordinates[:, 0] - guess[0])**2 + (transponder_coordinates[:, 1] - guess[1])**2)
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess)**2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv

@njit(cache=True)
def compute_Jacobian_biased(guess, transponder_coordinates, times, esv, esv_bias):
    """Compute the Jacobian of the system with respect to the ESV bias term
    According to Bud's algorithm for Gauss-Newton Inversion"""
    diffs = transponder_coordinates - guess

    J = np.zeros((len(transponder_coordinates), 5))

    #Compute different partial derivatives
    J[:, 0] = -diffs[:, 0] / (times[:] * (esv[:] + esv_bias)**2)
    J[:, 1] = -diffs[:, 1] / (times[:] * (esv[:] + esv_bias)**2)
    J[:, 2] = -diffs[:, 2] / (times[:] * (esv[:] + esv_bias)**2)
    J[:, 3] = -1.0
    J[:, 4] = -times[:] / (esv[:] + esv_bias)

    return J

@njit
def numba_bias_geiger(guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found,
                      esv_bias_input, time_bias_input, dz_array, angle_array, esv_matrix,
                      dz_array_gen = np.array([]), angle_array_gen = np.array([]), esv_matrix_gen = np.array([]),  time_noise=0):
    """Geiger method with estimation of ESV bias term"""
    epsilon = 10**-5

    #Calculate and apply noise for known times. Also apply time bias term
    if dz_array_gen.size > 0:
        times_known, esv = calculateTimesRayTracing_Bias(CDog, transponder_coordinates_Actual, esv_bias_input, dz_array_gen, angle_array_gen, esv_matrix_gen)
    else:
        times_known, esv = calculateTimesRayTracing_Bias(CDog, transponder_coordinates_Actual, esv_bias_input, dz_array, angle_array, esv_matrix)
    times_known+=np.random.normal(0, time_noise, len(transponder_coordinates_Actual))
    times_known+=time_bias_input

    time_bias = 0.0
    esv_bias = 0.0

    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    k = 0
    delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    while np.linalg.norm(delta) > epsilon and k < 100:
        times_guess, esv = calculateTimesRayTracing_Bias(guess, transponder_coordinates_Found, esv_bias, dz_array, angle_array, esv_matrix)
        J = compute_Jacobian_biased(guess, transponder_coordinates_Found, times_guess, esv, esv_bias)
        delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ ((times_guess - time_bias)-times_known)

        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        k += 1
    return estimate, times_known

if __name__ == "__main__":
    perturbed = True

    #Load in the ESV table
    if perturbed == True:
        #Table to generate synthetic times
        esv_table_generate = sio.loadmat('../../../GPSData/global_table_esv.mat')
        dz_array_generate = esv_table_generate['distance'].flatten()
        angle_array_generate = esv_table_generate['angle'].flatten()
        esv_matrix_generate = esv_table_generate['matrice']

        #Perturbed table to use in simulation
        esv_table_sim = sio.loadmat('../../../GPSData/global_table_esv_perturbed.mat')
        dz_array_sim = esv_table_sim['distance'].flatten()
        angle_array_sim = esv_table_sim['angle'].flatten()
        esv_matrix_sim = esv_table_sim['matrice']

        # Generate synthetic data
        CDOG, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(20000)

        esv_bias = 0.0
        time_bias = 0.0
    else:
        esv_table_sim = sio.loadmat('../../../GPSData/global_table_esv.mat')
        dz_array_sim = esv_table_sim['distance'].flatten()
        angle_array_sim = esv_table_sim['angle'].flatten()
        esv_matrix_sim = esv_table_sim['matrice']

        # Generate synthetic data
        CDOG, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(20000)

        esv_bias = 0.0
        time_bias = 0.0

    # time_noise = 0
    # position_noise = 0
    time_noise = 2.0 * 10 ** -5
    position_noise = 2.0 * 10 ** -2

    #Apply noise to position
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
    CDOG_guess = np.array([random.uniform(-5000, 5000), random.uniform(-5000, 5000), random.uniform(-5225, -5235)])

    # Run the Geiger method with ESV bias estimation
    if perturbed == True:
        estimate, times_known = numba_bias_geiger(CDOG_guess, CDOG, transponder_coordinates_Actual,
                                                  transponder_coordinates_Found, esv_bias, time_bias, dz_array_sim,
                                                  angle_array_sim, esv_matrix_sim, dz_array_generate,
                                                  angle_array_generate, esv_matrix_generate, time_noise)
    else:
        estimate, times_known = numba_bias_geiger(CDOG_guess, CDOG, transponder_coordinates_Actual,
                                                  transponder_coordinates_Found, esv_bias, time_bias, dz_array_sim,
                                                  angle_array_sim, esv_matrix_sim, time_noise)

    print(f"Input: [{CDOG[0]:.2f}, {CDOG[1]:.2f}, {CDOG[2]:.2f}, {time_bias:.2f}, {esv_bias:.2f}]")
    print(f"Output: [{estimate[0]:.2f}, {estimate[1]:.2f}, {estimate[2]:.2f}, {estimate[3]:.2f}, {estimate[4]:.2f}]")
    print(f"Diff: [{(estimate[0]-CDOG[0]):.2f}, {(estimate[1]-CDOG[1]):.2f}, {(estimate[2]-CDOG[2]):.2f}, {(estimate[3]+time_bias):.2f}, {(estimate[4]-esv_bias):.2f}]")

    # Plot the results
    CDOG_found = estimate[:3]
    time_bias_found = -1 * estimate[3]
    esv_bias_found = estimate[4]

    print(f"CDOG Distance: {np.linalg.norm(CDOG_found - CDOG):.2f}")
    times_calc, esv = calculateTimesRayTracing_Bias(CDOG_found, transponder_coordinates_Found, esv_bias_found,
                                                    dz_array_sim, angle_array_sim, esv_matrix_sim)

    times_calc += time_bias_found
    difference_data = times_calc - times_known

    #Fit the residuals to a normal distribution
    mu, std = norm.fit(difference_data * 1000)
    position_std = std*1515/1000

    #Get range of times for zoom in
    zoom_idx = np.random.randint(0, len(GPS_Coordinates)-100)
    zoom_length = 100

    fig, axes = plt.subplots(3, 3, figsize=(17, 10), gridspec_kw={'width_ratios': [1, 4, 2], 'height_ratios': [2, 2, 1]})
    axes[0, 0].axis('off')

    axes[0, 1].scatter(transponder_coordinates_Actual[:, 0], transponder_coordinates_Actual[:, 1], s=3, marker="o",
                       label="Transponder")
    axes[0, 1].scatter(CDOG[0], CDOG[1], s=50, marker="x", color="k", label="C-DOG")
    axes[0, 1].set_xlabel('Easting (m)')
    axes[0, 1].set_ylabel('Northing (m)')
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].axis("equal")

    axes[0, 2].scatter(CDOG[0], CDOG[1], s=50, marker="x", color="k", zorder=3, label="C-DOG")
    axes[0, 2].scatter(CDOG_found[0], CDOG_found[1], s=50, marker="o", color="r", zorder=4, label="Final Estimate")
    axes[0, 2].scatter(CDOG_guess[0], CDOG_guess[1], s=50, marker="o", color="g", zorder=1, label="Initial Guess")
    axes[0, 2].set_xlim(CDOG[0]-(3.1*position_std), CDOG[0]+(3.1*position_std))
    axes[0, 2].set_ylim(CDOG[1]-(3.1*position_std), CDOG[1]+(3.1*position_std))

    for i in range(1,4):
        ell = Ellipse(xy=(CDOG[0], CDOG[1]),
                      width= position_std * i * 2, height= position_std * i * 2,
                      angle=0, color='k')
        ell.set_facecolor('none')
        axes[0, 2].add_artist(ell)
    axes[0, 2].legend(loc="upper right")

    sound_velocity = np.genfromtxt('../../../GPSData/cz_cast2_smoothed.txt')[::100]
    depth = np.genfromtxt('../../../GPSData/depth_cast2_smoothed.txt')[::100]

    axes[1, 0].plot(sound_velocity, depth, color='b', label="Initial SVP")
    axes[1, 0].plot(sound_velocity - 0.001 * (5250-depth), depth, color='r', label="True SVP")

    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(min(sound_velocity)-5, max(sound_velocity)+5)
    axes[1, 0].set_ylabel('Depth')
    axes[1, 0].set_xlabel('Sound Velocity (m/s)')
    axes[1, 0].legend(loc="upper right")

    # Acoustic vs GNSS plot
    GPS_Coord_Num = list(range(len(GPS_Coordinates)))

    axes[1, 1].scatter(GPS_Coord_Num, times_known, s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[1, 1].scatter(GPS_Coord_Num, times_calc, s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r',
                       zorder=1)
    axes[1, 1].axvline(zoom_idx, color='k', linestyle="--")
    axes[1, 1].axvline(zoom_idx + 100, color='k', linestyle="--")
    axes[1, 1].set_ylabel('Travel Time (s)')
    axes[1, 1].legend(loc="upper right")

    axes[1, 2].scatter(GPS_Coord_Num[zoom_idx:zoom_idx + zoom_length], times_known[zoom_idx:zoom_idx + zoom_length],
                       s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[1, 2].scatter(GPS_Coord_Num[zoom_idx:zoom_idx + zoom_length], times_calc[zoom_idx:zoom_idx + zoom_length],
                       s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r', zorder=1)

    axes[1, 2].legend(loc="upper right")

    # Histogram and normal distributions
    n, bins, patches = axes[2, 0].hist(difference_data * 1000, orientation='horizontal', bins=30, alpha=0.5,
                                       density=True)
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    axes[2, 0].set_xlim([n.min(), n.max()])
    axes[2, 0].set_ylim([mu - 3 * std, mu + 3 * std])
    p = norm.pdf(x, mu, std)
    point1, point2 = norm.pdf(np.array([-std, std]), mu, std)
    axes[2, 0].plot(p, x, 'k', linewidth=2, label="Normal Distribution of Differences")
    axes[2, 0].scatter([point1, point2], [-std, std], s=10, color='r', zorder=1)

    # add horizontal lines for the noise and uncertainty
    axes[2, 0].axhline(mu-std, color='r', label="Observed Noise")
    axes[2, 0].axhline(mu+std, color='r')
    axes[2, 0].text(-0.2, std * 1.2, "$\\sigma_p$", va="center", color='r')

    if position_noise != 0:
        axes[2, 0].axhline(-position_noise / 1515 * 1000, color='g', label="Input Position Noise")
        axes[2, 0].axhline(position_noise / 1515 * 1000, color='g')
        axes[2, 0].text(-0.2, position_noise / 1515 * 1000 * .5, "$\\sigma_x$", va="center", color='g')

    if time_noise != 0:
        axes[2, 0].axhline(-time_noise * 1000, color='y', label="Input Time Noise")
        axes[2, 0].axhline(time_noise * 1000, color='y')
        axes[2, 0].text(-0.2, time_noise * 1000, "$\\sigma_t$", va="center", color='y')

    # invert axis and plot
    axes[2, 0].set_ylabel(f'Difference (ms) \n Std: {np.round(std, 3)} ms')
    axes[2, 0].set_xlabel('Normalized Frequency')
    axes[2, 0].invert_xaxis()
    # axes[2, 0].axis('off')

    # Difference plot
    axes[2, 1].scatter(GPS_Coord_Num, difference_data * 1000, s=1)
    axes[2, 1].axvline(zoom_idx, color='k', linestyle="--")
    axes[2, 1].axvline(zoom_idx+100, color='k', linestyle="--")
    axes[2, 1].set_xlabel('Time(ms)')
    axes[2, 1].set_ylim([mu-3*std, mu+3*std])

    axes[2, 2].scatter(GPS_Coord_Num[zoom_idx:zoom_idx+zoom_length], difference_data[zoom_idx:zoom_idx+zoom_length] * 1000, s=1)
    axes[2, 2].set_xlabel('Time(ms)')
    axes[2, 2].set_ylim([mu-3*std, mu+3*std])

    plt.show()

