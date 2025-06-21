import numpy as np
import matplotlib.pyplot as plt

from Modular_Synthetic import modular_synthetic

"""Do both mapping to RMSE and mapping to average distance from real CDOG"""


def combined_dependence():
    """Map position and timing noise to resulting RMSE."""
    space_axis = np.linspace(1 * 10**-2, 5 * 10**-2, 25)
    time_axis = np.linspace(2 * 10**-6, 5 * 10**-5, 25)

    [X, Y] = np.meshgrid(space_axis, time_axis)

    Z_std = np.zeros((len(Y[:, 0]), len(X[0])))
    Z_error = np.zeros((len(Y[:, 0]), len(X[0])))

    for i in range(len(X[0])):
        print(i)
        for j in range(len(Y[:, 0])):
            position_noise = X[0, i]
            time_noise = Y[j, 0]

            (
                inversion_result,
                CDOG_data,
                CDOG_full,
                GPS_data,
                GPS_full,
                CDOG_clock,
                GPS_clock,
                CDOG,
                transponder_coordinates,
                GPS_Coordinates,
                offset,
                lever,
            ) = modular_synthetic(
                time_noise,
                position_noise,
                0,
                0,
                esv1="global_table_esv",
                esv2="global_table_esv_perturbed",
                generate_type=1,
                inversion_type=1,
                plot=False,
            )
            diff_data = CDOG_full - GPS_full
            std_diff = np.std(diff_data)

            Z_std[j, i] = std_diff
            Z_error[j, i] = np.linalg.norm(inversion_result[:3] - CDOG) * 100

    # Plot expected uncertainty contours
    Z_stdexp = np.sqrt(0.00103**2 * np.square(X) + np.square(Y))

    CS = plt.contour(
        X * 100,
        Y * 1000000,
        Z_stdexp * 1515 * 100,
        colors="k",
        levels=np.arange(0, 14, 2),
    )
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.contourf(X * 100, Y * 1000000, Z_std * 1515 * 100, levels=np.arange(0, 14, 1))
    cbar = plt.colorbar()
    cbar.set_label("RMSE of residuals (cm)", rotation=270, labelpad=15)
    plt.xlabel("GPS Position Noise (cm)")
    plt.ylabel("C-DOG Time Noise ($\mu$s)")
    plt.title("Combined dependence of GPS position and C-DOG time noise")
    plt.show()


if __name__ == "__main__":
    combined_dependence()
