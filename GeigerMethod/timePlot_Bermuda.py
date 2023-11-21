import matplotlib.pyplot as plt
import numpy as np
from geigerMethod_Bermuda import geigersMethod, calculateTimes


def geigerTimePlot(initial_guess, times_known, transponder_coordinates_Found, timestamps, sound_speed):
    guess = geigersMethod(initial_guess, times_known, transponder_coordinates_Found, sound_speed)

    times_calc = calculateTimes(guess, transponder_coordinates_Found, sound_speed)

    difference_data = times_calc - times_known
    RMS = np.sqrt(np.nanmean(difference_data ** 2))

    # Prepare label and plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
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

    # Difference plot
    axes[1, 1].scatter(timestamps, difference_data * 1000, s=1)
    axes[1, 1].set_xlabel('Time (hours)')

    # Histogram
    axes[1, 0].hist(difference_data * 1000, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (ms)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title(f"RMS: {round(RMS * sound_speed * 100, 3)} cm")
    axes[0, 0].axis('off')

    plt.show()
    return axes