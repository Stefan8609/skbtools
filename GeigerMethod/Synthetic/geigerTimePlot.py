import matplotlib.pyplot as plt
import numpy as np
from advancedGeigerMethod import geigersMethod, calculateTimes, calculateTimesRayTracing
from scipy.stats import norm


def geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual, transponder_coordinates_Found,
                   gps1_to_transponder, sound_velocity, depth, time_noise, position_noise, lever=[None,None,None], sim = 0):
    if not lever[0]:
        lever = gps1_to_transponder

    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found, time_noise)

    # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    difference_data = times_calc - times_known
    print(np.std(difference_data))

    difference_data = np.round(difference_data, 10)

    RMS = np.sqrt(np.nanmean(difference_data ** 2))

    # Prepare label and plot
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), gridspec_kw={'width_ratios': [1, 4, 2], 'height_ratios': [2, 2, 1]})
    # fig.suptitle("Comparison of calculated arrival times and actual arrival times", y=0.92)

    fig.text(0.07, 0.85, f"Noise: \n GPS: {position_noise*100}cm \n Arrival time: {time_noise*10**6}\u03BCs",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    fig.text(0.07, 0.75,
             f"Distance between \npredicted and actual \nCDog location:\n{np.round(np.linalg.norm(CDog - guess) * 100, 4)}cm",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    fig.text(0.07, 0.61, f"Initial Guess (x,y,z):\n({initial_guess[0]}, {initial_guess[1]}, {initial_guess[2]})"
                         f"\nAverage of residuals:\n{round(np.average(difference_data) * 1000, 4)}ms"
                         f"\nActual vs. Found lever:\n({round(lever[0]-gps1_to_transponder[0],3)},{round(lever[1]-gps1_to_transponder[1],3)},{round(lever[2]-gps1_to_transponder[2],3)})m",
             fontsize=12, bbox=dict(facecolor="white", alpha=0.8))


    axes[0, 0].axis('off')

    axes[0, 1].scatter(transponder_coordinates_Actual[:, 0], transponder_coordinates_Actual[:, 1], s=3, marker="o",
                       label="Transponder")
    axes[0, 1].scatter(CDog[0], CDog[1], s=50, marker="x", color="k", label="C-DOG")
    axes[0, 1].set_xlabel('Easting (m)')
    axes[0, 1].set_ylabel('Northing (m)')
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].axis("equal")

    axes[0, 2].scatter(CDog[0], CDog[1], s=50, marker="x", color="k", label="C-DOG")
    axes[0, 2].scatter(guess[0], guess[1], s=50, marker="o", color="r", label="C-DOG Guess")
    axes[0, 2].set_xlim(CDog[0]-5, CDog[0]+5)
    axes[0, 2].set_ylim(CDog[1]-5, CDog[1]+5)
    axes[0, 2].legend(loc="upper right")

    axes[1, 0].plot(sound_velocity, depth, color='b')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(min(sound_velocity), max(sound_velocity))
    axes[1, 0].set_ylabel('Depth')
    axes[1, 0].set_xlabel('Sound Velocity')

    # Acoustic vs GNSS plot
    GPS_Coord_Num = list(range(len(GPS_Coordinates)))

    axes[1, 1].scatter(GPS_Coord_Num, times_known, s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[1, 1].scatter(GPS_Coord_Num, times_calc, s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r', zorder=1)
    axes[1, 1].set_ylabel('Travel Time (s)')
    # axes[1, 1].text(25, max(times_known), "actual arrival times versus estimated times",
    #                 bbox=dict(facecolor='yellow', alpha=0.8))
    axes[1, 1].legend(loc="upper right")

    axes[1, 2].scatter(GPS_Coord_Num[475:525], times_known[475:525], s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[1, 2].scatter(GPS_Coord_Num[475:525], times_calc[475:525], s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r', zorder=1)
    # axes[1, 2].text(25, max(times_known), "actual arrival times versus estimated times",
    #                 bbox=dict(facecolor='yellow', alpha=0.8))
    axes[1, 2].legend(loc="upper right")

    # Difference plot
    axes[2, 1].scatter(GPS_Coord_Num, difference_data * 1000, s=1)
    axes[2, 1].set_xlabel('Position Index')

    axes[2, 2].scatter(GPS_Coord_Num[475:525], difference_data[475:525] * 1000, s=1)
    axes[2, 2].set_xlabel('Position Index')

    # Histogram and normal distributions
    n, bins, patches = axes[2, 0].hist(difference_data * 1000, orientation='horizontal', bins=30, alpha=0.5, density=True)
    mu, std = norm.fit(difference_data * 1000)
    x = np.linspace(mu-3*std, mu+3*std, 100)
    axes[2, 0].set_xlim([n.min(), n.max()])
    axes[2, 0].set_ylim([mu-3*std, mu+3*std])
    p = norm.pdf(x, mu, std)
    point1, point2 = norm.pdf(np.array([-std, std]), mu, std)
    axes[2, 0].plot(p, x, 'k', linewidth=2, label="Normal Distribution of Differences")
    axes[2, 0].scatter([point1, point2], [-std, std], s=10, color='r', zorder=1)
    
    #add horizontal lines for the noise and uncertainty
    axes[2, 0].axhline(-std, color='r', label="Observed Noise")
    axes[2, 0].axhline(std, color='r')
    if position_noise!=0:
        axes[2, 0].axhline(-position_noise / 1515 * 1000, color='g', label="Input Position Noise")
        axes[2, 0].axhline(position_noise / 1515 * 1000, color='g')
    if time_noise!=0:
        axes[2, 0].axhline(-time_noise * 1000, color='y', label="Input Time Noise")
        axes[2, 0].axhline(time_noise * 1000, color='y')

    #invert axis and plot
    axes[2, 0].set_ylabel(f'Difference (ms) \n Std: {np.round(std, 3)} ms')
    axes[2, 0].set_xlabel('Normalized Frequency')
    axes[2, 0].invert_xaxis()
    # axes[2, 0].axis('off')

    # if sim == 1:
    #     plt.savefig('../../Figs/init_sim_noise_ray_tracing.png')
    # elif sim == 2:
    #     plt.savefig('../../Figs/final_sim_noise_ray_tracing.png')
    # else:
    #     plt.savefig('../../Figs/Noise_ray_tracing.png')
    plt.show()
    return axes