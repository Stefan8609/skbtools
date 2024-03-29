import matplotlib.pyplot as plt
import numpy as np
from advancedGeigerMethod import geigersMethod, calculateTimes, calculateTimesRayTracing
from scipy.stats import norm


def geigerTimePlot(initial_guess, GPS_Coordinates, CDog, transponder_coordinates_Actual,
                   transponder_coordinates_Found, gps1_to_transponder, lever=[None,None,None]):
    if not lever[0]:
        lever = gps1_to_transponder
    print(lever)
    guess, times_known = geigersMethod(initial_guess, CDog, transponder_coordinates_Actual, transponder_coordinates_Found)

    # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    difference_data = times_calc - times_known
    RMS = np.sqrt(np.nanmean(difference_data ** 2))

    # Prepare label and plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4, 2], 'height_ratios': [4, 1]})
    fig.suptitle("Comparison of calculated arrival times and actual arrival times", y=0.92)
    fig.text(0.05, 0.85, "Noise: \n GPS: 2cm \n Arrival time: 20\u03BCs",
             fontsize=12, bbox=dict(facecolor='yellow', alpha=0.8))
    fig.text(0.05, 0.7,
             f"Distance between \npredicted and actual \nCDog location:\n{np.round(np.linalg.norm(CDog - guess) * 100, 4)}cm",
             fontsize=12, bbox=dict(facecolor='green', alpha=0.8))
    fig.text(0.05, 0.5, f"Initial Guess (x,y,z):\n({initial_guess[0]}, {initial_guess[1]}, {initial_guess[2]})"
                         f"\nAverage of residuals:\n{round(np.average(difference_data) * 1000, 4)}ms"
                         f"\nActual vs. Found lever:\n({round(lever[0]-gps1_to_transponder[0],3)},{round(lever[1]-gps1_to_transponder[1],3)},{round(lever[2]-gps1_to_transponder[2],3)})m",
             fontsize=12, bbox=dict(facecolor='red', alpha=0.8))

    # Acoustic vs GNSS plot
    GPS_Coord_Num = list(range(len(GPS_Coordinates)))

    axes[0, 1].scatter(GPS_Coord_Num, times_known, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[0, 1].scatter(GPS_Coord_Num, times_calc, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].text(25, max(times_known), "actual arrival times versus estimated times",
                    bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend(loc="upper right")

    axes[0, 2].scatter(GPS_Coord_Num[475:525], times_known[475:525], s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[0, 2].scatter(GPS_Coord_Num[475:525], times_calc[475:525], s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 2].set_ylabel('Travel Time (s)')
    axes[0, 2].text(25, max(times_known), "actual arrival times versus estimated times",
                    bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 2].legend(loc="upper right")

    # Difference plot
    axes[1, 1].scatter(GPS_Coord_Num, difference_data * 1000, s=1)
    axes[1, 1].set_xlabel('Position Index')

    axes[1, 2].scatter(GPS_Coord_Num[475:525], difference_data[475:525] * 1000, s=1)
    axes[1, 2].set_xlabel('Position Index')

    # Histogram
    n, bins, patches = axes[1, 0].hist(difference_data * 1000, orientation='horizontal', bins=30, alpha=0.5, density=True)
    mu, std = norm.fit(difference_data * 1000)
    x = np.linspace(mu-3*std, mu+3*std, 100)
    axes[1, 0].set_xlim([n.min(), n.max()])
    axes[1, 0].set_ylim([mu-3*std, mu+3*std])
    p = norm.pdf(x, mu, std)
    axes[1, 0].plot(p, x, 'k', linewidth=2, label="Normal Distribution of Differences")
    axes[1, 0].set_ylabel('Difference (ms)')
    axes[1, 0].set_xlabel('Normalized Frequency')
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title(f"RMSE: {round(RMS * 1515 * 100, 3)} cm")
    axes[0, 0].axis('off')

    plt.show()
    return axes