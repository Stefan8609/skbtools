import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from GeigerMethod.Synthetic.advancedGeigerMethod import (
    geigersMethod,
    calculateTimesRayTracing,
)
from scipy.stats import norm


def geigerTimePlot(
    initial_guess,
    GPS_Coordinates,
    CDog,
    transponder_coordinates_Actual,
    transponder_coordinates_Found,
    gps1_to_transponder,
    sound_velocity,
    depth,
    time_noise,
    position_noise,
    lever=None,
    sim=0,
):
    if lever is None:
        lever = [None, None, None]
    if not lever[0]:
        lever = gps1_to_transponder

    guess, times_known, estimate_arr = geigersMethod(
        initial_guess,
        CDog,
        transponder_coordinates_Actual,
        transponder_coordinates_Found,
        time_noise,
    )

    # times_calc = calculateTimes(guess, transponder_coordinates_Found, 1515)
    times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]

    difference_data = times_calc - times_known
    print(np.std(difference_data))

    difference_data = np.round(difference_data, 10)

    # Fit the residuals to a normal distribution
    mu, std = norm.fit(difference_data * 1000)
    position_std = std * 1515 / 1000

    # Get range of times for zoom in
    zoom_idx = np.random.randint(0, len(GPS_Coordinates) - 100)
    zoom_length = 100

    # RMS = np.sqrt(np.nanmean(difference_data ** 2))

    # Prepare label and plot
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [1, 4, 2], "height_ratios": [2, 2, 1]},
    )

    fig.text(
        0.07,
        0.85,
        f"Noise: \n GPS: {position_noise * 100}cm "
        f"\n Arrival time: {time_noise * 10**6}\u03bcs",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    fig.text(
        0.07,
        0.75,
        f"Distance between \npredicted and actual \nCDog location:"
        f"\n{np.round(np.linalg.norm(CDog - guess) * 100, 4)}cm",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    fig.text(
        0.07,
        0.61,
        f"Initial Guess (x,y,z):\n({np.round(initial_guess[0], 2)}, "
        f"{np.round(initial_guess[1], 2)}, {np.round(initial_guess[2]), 2})"
        f"\nAverage of residuals:\n{round(np.average(difference_data) * 1000, 4)}ms"
        f"\nActual vs. Found lever:\n({round(lever[0] - gps1_to_transponder[0], 3)},"
        f"{round(lever[1] - gps1_to_transponder[1], 3)},"
        f"{round(lever[2] - gps1_to_transponder[2], 3)})m",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    axes[0, 0].axis("off")

    axes[0, 1].scatter(
        transponder_coordinates_Actual[:, 0],
        transponder_coordinates_Actual[:, 1],
        s=3,
        marker="o",
        label="Transponder",
    )
    axes[0, 1].scatter(CDog[0], CDog[1], s=50, marker="x", color="k", label="C-DOG")
    axes[0, 1].set_xlabel("Easting (m)")
    axes[0, 1].set_ylabel("Northing (m)")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].axis("equal")

    axes[0, 2].scatter(
        CDog[0], CDog[1], s=50, marker="x", color="k", zorder=3, label="C-DOG"
    )
    axes[0, 2].scatter(
        guess[0],
        guess[1],
        s=50,
        marker="o",
        color="r",
        zorder=4,
        label="Final Estimate",
    )
    axes[0, 2].scatter(
        initial_guess[0],
        initial_guess[1],
        s=50,
        marker="o",
        color="g",
        zorder=1,
        label="Initial Guess",
    )
    axes[0, 2].scatter(
        estimate_arr[:, 0],
        estimate_arr[:, 1],
        s=50,
        marker="o",
        color="b",
        zorder=2,
        label="Estimate Iterations",
    )
    axes[0, 2].set_xlim(CDog[0] - (3.1 * position_std), CDog[0] + (3.1 * position_std))
    axes[0, 2].set_ylim(CDog[1] - (3.1 * position_std), CDog[1] + (3.1 * position_std))
    for i in range(1, 4):
        ell = Ellipse(
            xy=(CDog[0], CDog[1]),
            width=position_std * i * 2,
            height=position_std * i * 2,
            angle=0,
            color="k",
        )
        ell.set_facecolor("none")
        axes[0, 2].add_artist(ell)
    axes[0, 2].legend(loc="upper right")

    axes[1, 0].plot(sound_velocity, depth, color="b")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(min(sound_velocity), max(sound_velocity))
    axes[1, 0].set_ylabel("Depth")
    axes[1, 0].set_xlabel("Sound Velocity (m/s)")

    # Acoustic vs GNSS plot
    GPS_Coord_Num = list(range(len(GPS_Coordinates)))

    axes[1, 1].scatter(
        GPS_Coord_Num,
        times_known,
        s=5,
        label="Observed Travel Times",
        alpha=0.6,
        marker="o",
        color="b",
        zorder=2,
    )
    axes[1, 1].scatter(
        GPS_Coord_Num,
        times_calc,
        s=10,
        label="Modelled Travel Times",
        alpha=1,
        marker="x",
        color="r",
        zorder=1,
    )
    axes[1, 1].axvline(zoom_idx, color="k", linestyle="--")
    axes[1, 1].axvline(zoom_idx + 100, color="k", linestyle="--")
    axes[1, 1].set_ylabel("Travel Time (s)")
    axes[1, 1].legend(loc="upper right")

    axes[1, 2].scatter(
        GPS_Coord_Num[zoom_idx : zoom_idx + zoom_length],
        times_known[zoom_idx : zoom_idx + zoom_length],
        s=5,
        label="Observed Travel Times",
        alpha=0.6,
        marker="o",
        color="b",
        zorder=2,
    )
    axes[1, 2].scatter(
        GPS_Coord_Num[zoom_idx : zoom_idx + zoom_length],
        times_calc[zoom_idx : zoom_idx + zoom_length],
        s=10,
        label="Modelled Travel Times",
        alpha=1,
        marker="x",
        color="r",
        zorder=1,
    )

    axes[1, 2].legend(loc="upper right")

    # Histogram and normal distributions
    n, bins, patches = axes[2, 0].hist(
        difference_data * 1000,
        orientation="horizontal",
        bins=30,
        alpha=0.5,
        density=True,
    )
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    axes[2, 0].set_xlim([n.min(), n.max()])
    axes[2, 0].set_ylim([mu - 3 * std, mu + 3 * std])
    p = norm.pdf(x, mu, std)
    point1, point2 = norm.pdf(np.array([-std, std]), mu, std)
    axes[2, 0].plot(p, x, "k", linewidth=2, label="Normal Distribution of Differences")
    axes[2, 0].scatter([point1, point2], [-std, std], s=10, color="r", zorder=1)

    # add horizontal lines for the noise and uncertainty
    axes[2, 0].axhline(-std, color="r", label="Observed Noise")
    axes[2, 0].axhline(std, color="r")
    axes[2, 0].text(-0.2, std * 1.2, "$\\sigma_p$", va="center", color="r")

    if position_noise != 0:
        axes[2, 0].axhline(
            -position_noise / 1515 * 1000, color="g", label="Input Position Noise"
        )
        axes[2, 0].axhline(position_noise / 1515 * 1000, color="g")
        axes[2, 0].text(
            -0.2,
            position_noise / 1515 * 1000 * 0.5,
            "$\\sigma_x$",
            va="center",
            color="g",
        )

    if time_noise != 0:
        axes[2, 0].axhline(-time_noise * 1000, color="y", label="Input Time Noise")
        axes[2, 0].axhline(time_noise * 1000, color="y")
        axes[2, 0].text(-0.2, time_noise * 1000, "$\\sigma_t$", va="center", color="y")

    # invert axis and plot
    axes[2, 0].set_ylabel(f"Difference (ms) \n Std: {np.round(std, 3)} ms")
    axes[2, 0].set_xlabel("Normalized Frequency")
    axes[2, 0].invert_xaxis()
    # axes[2, 0].axis('off')

    # Difference plot
    axes[2, 1].scatter(GPS_Coord_Num, difference_data * 1000, s=1)
    axes[2, 1].axvline(zoom_idx, color="k", linestyle="--")
    axes[2, 1].axvline(zoom_idx + 100, color="k", linestyle="--")
    axes[2, 1].set_xlabel("Time(ms)")
    axes[2, 1].set_ylim([mu - 3 * std, mu + 3 * std])

    axes[2, 2].scatter(
        GPS_Coord_Num[zoom_idx : zoom_idx + zoom_length],
        difference_data[zoom_idx : zoom_idx + zoom_length] * 1000,
        s=1,
    )
    axes[2, 2].set_xlabel("Time(ms)")
    axes[2, 2].set_ylim([mu - 3 * std, mu + 3 * std])

    # if sim == 1:
    #     plt.savefig('../../Figs/init_sim_noise_ray_tracing.png')
    # elif sim == 2:
    #     plt.savefig('../../Figs/final_sim_noise_ray_tracing.png')
    # else:
    #     plt.savefig('../../Figs/Noise_ray_tracing.png')
    plt.show()
    return axes
