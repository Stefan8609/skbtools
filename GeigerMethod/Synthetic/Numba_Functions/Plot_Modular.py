import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import norm

def time_series_plot(CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise=0, time_noise=0, block=True):
    difference_data = CDOG_full - GPS_full

    # convert ms residuals to a normal distribution
    mu, std = norm.fit(difference_data * 1000)

    # Get range of times for zoom in
    zoom_region = np.random.randint(min(CDOG_clock), max(CDOG_clock) - 100)
    zoom_idx = (np.abs(CDOG_clock - zoom_region)).argmin()
    zoom_length = 400

    # Plot axes to return
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), gridspec_kw={'width_ratios': [1, 4, 2], 'height_ratios': [2, 1]})
    axes[0, 0].axis('off')

    # Acoustic vs GNSS plot
    axes[0, 1].scatter(CDOG_clock, CDOG_full, s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[0, 1].scatter(GPS_clock, GPS_full, s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r',
                       zorder=1)
    axes[0, 1].axvline(zoom_region, color='k', linestyle="--")
    axes[0, 1].axvline(zoom_region + zoom_length, color='k', linestyle="--")
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].legend(loc="upper right")

    axes[0, 2].scatter(CDOG_clock[zoom_idx:zoom_idx + zoom_length], CDOG_full[zoom_idx:zoom_idx + zoom_length],
                       s=5, label='Observed Travel Times', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[0, 2].scatter(GPS_clock[zoom_idx:zoom_idx + zoom_length], GPS_full[zoom_idx:zoom_idx + zoom_length],
                       s=10, label='Modelled Travel Times', alpha=1, marker='x', color='r', zorder=1)

    axes[0, 2].legend(loc="upper right")

    # Histogram and normal distributions
    n, bins, patches = axes[1, 0].hist(difference_data * 1000, orientation='horizontal', bins=40, alpha=0.5,
                                       density=True)
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    axes[1, 0].set_xlim([n.min()*1.4, n.max()*1.4])
    axes[1, 0].set_ylim([mu - 3 * std, mu + 3 * std])
    p = norm.pdf(x, mu, std)
    point1, point2 = norm.pdf(np.array([-std, std]), mu, std)
    axes[1, 0].plot(p, x, 'k', linewidth=2, label="Normal Distribution of Differences")
    axes[1, 0].scatter([point1, point2], [-std, std], s=10, color='r', zorder=1)

    # add horizontal lines for the noise and uncertainty
    axes[1, 0].axhline(-std, color='r', label="Observed Noise")
    axes[1, 0].axhline(std, color='r')
    axes[1, 0].text(-0.1, std * 1.2, "$\\sigma_p$", va="center", color='r')

    if position_noise != 0:
        axes[1, 0].axhline(-position_noise / 1515 * 1000, color='g', label="Input Position Noise")
        axes[1, 0].axhline(position_noise / 1515 * 1000, color='g')
        axes[1, 0].text(-0.2, position_noise / 1515 * 1000 * .5, "$\\sigma_x$", va="center", color='g')

    if time_noise != 0:
        axes[1, 0].axhline(-time_noise * 1000, color='y', label="Input Time Noise")
        axes[1, 0].axhline(time_noise * 1000, color='y')
        axes[1, 0].text(-0.2, time_noise * 1000, "$\\sigma_t$", va="center", color='y')

    # invert axis and plot
    axes[1, 0].set_ylabel(f'Difference (ms) \n Std: {np.round(std, 3)} ms or {np.round(std * 1515 / 10,2)} cm')
    axes[1, 0].set_xlabel('Normalized Frequency')
    axes[1, 0].invert_xaxis()
    # axes[2, 0].axis('off')

    # Difference plot
    axes[1, 1].scatter(CDOG_clock, difference_data * 1000, s=1)
    axes[1, 1].axvline(zoom_region, color='k', linestyle="--")
    axes[1, 1].axvline(zoom_region + zoom_length, color='k', linestyle="--")
    axes[1, 1].set_xlabel('Time(ms)')
    axes[1, 1].set_ylim([mu - 3 * std, mu + 3 * std])

    axes[1, 2].scatter(CDOG_clock[zoom_idx:zoom_idx + zoom_length],
                       difference_data[zoom_idx:zoom_idx + zoom_length] * 1000, s=1)
    axes[1, 2].set_xlabel('Time(ms)')
    axes[1, 2].set_ylim([mu - 3 * std, mu + 3 * std])

    plt.show(block = block)

def trajectory_plot(coordinates, GPS_clock, CDOG, block=True):
    # Calculate time values in hours for proper colorbar range
    times_hours = GPS_clock / 3600  # Convert seconds to hours
    min_time = np.min(times_hours)
    max_time = np.max(times_hours)

    scatter = plt.scatter(coordinates[:,0], coordinates[:,1], s=1, c=times_hours, cmap='viridis', label='Surface Vessel')
    plt.scatter(CDOG[0], CDOG[1], marker='x', s=5, label='CDOG')
    plt.colorbar(scatter, label='Elapsed Time (hours)')
    plt.clim(min_time, max_time)  # Set the colorbar to actual time range
    plt.title('Plot of Trajectory and CDOG location')
    plt.xlabel('ECEF (x)')
    plt.ylabel('ECEF (y)')
    plt.legend()

    plt.show(block = block)
