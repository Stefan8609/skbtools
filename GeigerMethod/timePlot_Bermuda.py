import matplotlib.pyplot as plt
import numpy as np
from geigerMethod_Bermuda import geigersMethod, calculateTimes, calculateTimesRayTracing
from scipy.stats import norm


def geigerTimePlot(initial_guess, times_known, transponder_coordinates_Found, timestamps):
    guess = geigersMethod(initial_guess, times_known, transponder_coordinates_Found)
    print(guess)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    difference_data = times_calc - times_known
    RMS = np.sqrt(np.nanmean(difference_data ** 2))

    ## For identifying outlying items
    # for idx, item in enumerate(difference_data):
    #     if abs(item*1515*100) > 500:
    #         print(idx)

    # Prepare label and plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4, 2], 'height_ratios': [4, 1]})
    fig.suptitle("Comparison of calculated arrival times and actual arrival times", y=0.92)
    fig.text(0.05, 0.5, f"\nAverage of residuals:\n{round(np.average(difference_data) * 1000, 4)}ms",
             fontsize=12, bbox=dict(facecolor='red', alpha=0.8))

    # Acoustic vs GNSS plot
    axes[0, 1].scatter(timestamps, times_known, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[0, 1].scatter(timestamps, times_calc, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].text(25, max(times_known), "actual arrival times versus estimated times",
                    bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend(loc="upper right")
    axes[0, 2].scatter(timestamps[475:525], times_known[475:525], s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b',
                       zorder=2)
    axes[0, 2].scatter(timestamps[475:525], times_calc[475:525], s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 2].set_ylabel('Travel Time (s)')
    axes[0, 2].text(25, max(times_known), "actual arrival times versus estimated times",
                    bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 2].legend(loc="upper right")

    # Difference plot
    axes[1, 1].scatter(timestamps, difference_data * 1000, s=1)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 2].scatter(timestamps[475:525], difference_data[475:525] * 1000, s=1)
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
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title(f"RMS: {round(RMS * 1515 * 100, 3)} cm")
    axes[0, 0].axis('off')

    plt.show()
    return axes