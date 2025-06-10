import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def leverHist(transponder_coordinates_Actual, transponder_coordinates_Found):
    # Plot histograms of coordinate differences between found transponder and actual transponder
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Coordinate differences between calculated transponder and actual transponder, 2cm noise in each GPS",
        y=0.92,
    )
    xmin, xmax = -15, 15
    colors = ["blue", "red", "green"]
    axis = ["x", "y", "z"]
    mu_arr, std_arr, n_arr = [0] * 3, [0] * 3, [0] * 3
    for i in range(3):
        axs[i].set_xlim(xmin, xmax)
        axs[i].set_ylim(0, 275)
        n, bins, patches = axs[i].hist(
            (transponder_coordinates_Found[:, i] - transponder_coordinates_Actual[:, i])
            * 100,
            bins=30,
            color=colors[i],
            alpha=0.7,
            density=True,
        )
        n_arr[i] = n.max()
        mu_arr[i], std_arr[i] = norm.fit(
            (transponder_coordinates_Found[:, i] - transponder_coordinates_Actual[:, i])
            * 100
        )
    for i in range(3):
        xmin, xmax = -3 * max(std_arr), 3 * max(std_arr)
        x = np.linspace(xmin, xmax, 100)
        axs[i].set_xlim(xmin, xmax)
        axs[i].set_ylim(0, max(n_arr))
        p = norm.pdf(x, mu_arr[i], std_arr[i])
        p_noise = norm.pdf(x, 0, 2)  # noise of GPS in centimeters
        axs[i].plot(x, p, "k", linewidth=2, label="Normal Distribution of Differences")
        axs[i].plot(x, p_noise, color="y", linewidth=2, label="GPS Noise Distribution")
        if i == 2:
            axs[i].legend(loc="upper right", fontsize="8")
        axs[i].set_xlabel(f"{axis[i]}-difference std={round(std_arr[i], 4)} (cm)")
    plt.show()
    return axs
