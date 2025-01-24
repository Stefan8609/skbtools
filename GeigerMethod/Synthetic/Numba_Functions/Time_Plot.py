import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import norm
from matplotlib.patches import Ellipse

def time_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full, CDOG, true_transponder_coordinates, position_noise, time_noise,
              initial_guess, inversion_guess, lever, gps1_to_transponder):
    sound_velocity = np.genfromtxt('../../../GPSData/cz_cast2_smoothed.txt')[::100]
    depth = np.genfromtxt('../../../GPSData/depth_cast2_smoothed.txt')[::100]
    difference_data = CDOG_full - GPS_full

    # Fit the residuals to a normal distribution
    mu, std = norm.fit(difference_data * 1000)
    position_std = std * 1515 / 1000

    # Get range of times for zoom in
    zoom_idx = np.random.randint(0, len(CDOG_clock) - 100)
    zoom_length = 100

    # RMS = np.sqrt(np.nanmean(difference_data ** 2))

    # Prepare label and plot
    fig, axes = plt.subplots(3, 3, figsize=(17, 10),
                             gridspec_kw={'width_ratios': [1, 4, 2], 'height_ratios': [2, 2, 1]})
    # fig.suptitle("Comparison of calculated arrival times and actual arrival times", y=0.92)

    fig.text(0.07, 0.85, f"Noise: \n GPS: {position_noise * 100}cm \n Arrival time: {time_noise * 10 ** 6}\u03BCs",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    fig.text(0.07, 0.75,
             f"Distance between \npredicted and actual \nCDog location:\n{np.round(np.linalg.norm(CDOG - inversion_guess) * 100, 4)}cm",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    fig.text(0.07, 0.61,
             f"Initial Guess (x,y,z):\n({np.round(initial_guess[0], 2)}, {np.round(initial_guess[1], 2)}, {np.round(initial_guess[2]), 2})"
             f"\nAverage of residuals:\n{round(np.average(difference_data) * 1000, 4)}ms"
             f"\nActual vs. Found lever:\n({round(lever[0] - gps1_to_transponder[0], 3)},{round(lever[1] - gps1_to_transponder[1], 3)},{round(lever[2] - gps1_to_transponder[2], 3)})m",
             fontsize=12, bbox=dict(facecolor="white", alpha=0.8))

    axes[0, 0].axis('off')

    axes[0, 1].scatter(true_transponder_coordinates[:, 0], true_transponder_coordinates[:, 1], s=3, marker="o",
                       label="Transponder")
    axes[0, 1].scatter(CDOG[0], CDOG[1], s=50, marker="x", color="k", label="C-DOG")
    axes[0, 1].set_xlabel('Easting (m)')
    axes[0, 1].set_ylabel('Northing (m)')
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].axis("equal")

    axes[0, 2].scatter(CDOG[0], CDOG[1], s=50, marker="x", color="k", zorder=3, label="C-DOG")
    axes[0, 2].scatter(inversion_guess[0], inversion_guess[1], s=50, marker="o", color="r", zorder=4, label="Final Estimate")
    axes[0, 2].scatter(initial_guess[0], initial_guess[1], s=50, marker="o", color="g", zorder=1, label="Initial Guess")
    axes[0, 2].set_xlim(CDOG[0] - (3.1 * position_std), CDOG[0] + (3.1 * position_std))
    axes[0, 2].set_ylim(CDOG[1] - (3.1 * position_std), CDOG[1] + (3.1 * position_std))
    for i in range(1, 4):
        ell = Ellipse(xy=(CDOG[0], CDOG[1]),
                      width=position_std * i * 2, height=position_std * i * 2,
                      angle=0, color='k')
        ell.set_facecolor('none')
        axes[0, 2].add_artist(ell)
    axes[0, 2].legend(loc="upper right")

    axes[1, 0].plot(sound_velocity, depth, color='b')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(min(sound_velocity), max(sound_velocity))
    axes[1, 0].set_ylabel('Depth')
    axes[1, 0].set_xlabel('Sound Velocity (m/s)')

    # Acoustic vs GNSS plot
    axes[1, 1].scatter(CDOG_clock, CDOG_full, s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[1, 1].scatter(GPS_clock, GPS_full, s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r',
                       zorder=1)
    axes[1, 1].axvline(zoom_idx, color='k', linestyle="--")
    axes[1, 1].axvline(zoom_idx + 100, color='k', linestyle="--")
    axes[1, 1].set_ylabel('Travel Time (s)')
    axes[1, 1].legend(loc="upper right")

    axes[1, 2].scatter(CDOG_clock[zoom_idx:zoom_idx + zoom_length], CDOG_full[zoom_idx:zoom_idx + zoom_length],
                       s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[1, 2].scatter(GPS_clock[zoom_idx:zoom_idx + zoom_length], GPS_full[zoom_idx:zoom_idx + zoom_length],
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
    axes[2, 0].axhline(-std, color='r', label="Observed Noise")
    axes[2, 0].axhline(std, color='r')
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
    axes[2, 1].scatter(CDOG_clock, difference_data * 1000, s=1)
    axes[2, 1].axvline(zoom_idx, color='k', linestyle="--")
    axes[2, 1].axvline(zoom_idx + 100, color='k', linestyle="--")
    axes[2, 1].set_xlabel('Time(ms)')
    axes[2, 1].set_ylim([mu - 3 * std, mu + 3 * std])

    axes[2, 2].scatter(CDOG_clock[zoom_idx:zoom_idx + zoom_length],
                       difference_data[zoom_idx:zoom_idx + zoom_length] * 1000, s=1)
    axes[2, 2].set_xlabel('Time(ms)')
    axes[2, 2].set_ylim([mu - 3 * std, mu + 3 * std])

    plt.show()