import numpy as np
import matplotlib.pyplot as plt

from Inversion_Workflow.Synthetic.Modular_Synthetic import (
    modular_synthetic,
)
from data import gps_output_path

"""Do both mapping to RMSE and mapping to average distance from real CDOG"""


def combined_dependence(generate=True):
    """Map position and timing noise to resulting RMSE and CDOG error."""
    data_path = gps_output_path("Plot_Data/Combined_Dependence.npz")

    if generate:
        space_axis = np.linspace(1 * 10**-2, 5 * 10**-2, 5)
        time_axis = np.linspace(2 * 10**-6, 5 * 10**-5, 5)

        X, Y = np.meshgrid(space_axis, time_axis)

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

        # Save everything needed for easy re-plotting
        np.savez(
            data_path,
            # raw axes and grids
            space_axis=space_axis,             # m
            time_axis=time_axis,               # s
            X=X,                               # m
            Y=Y,                               # s

            # outputs
            Z_std=Z_std,                       # s
            Z_error=Z_error,                   # cm
            Z_stdexp=Z_stdexp,                 # s

            # convenient plot-scaled versions
            X_cm=X * 100,
            Y_us=Y * 1_000_000,
            Z_std_cm=Z_std * 1515 * 100,
            Z_stdexp_cm=Z_stdexp * 1515 * 100,

            # metadata
            residual_scale_mps=1515,
            position_noise_units="m",
            time_noise_units="s",
            Z_std_units="s",
            Z_error_units="cm",
        )
        print(f"Saved plotting data to: {data_path}")

    else:
        npzfile = np.load(data_path)
        X = npzfile["X"]
        Y = npzfile["Y"]
        Z_std = npzfile["Z_std"]
        Z_error = npzfile["Z_error"]
        Z_stdexp = npzfile["Z_stdexp"]

    CS = plt.contour(
        X * 100,
        Y * 1000000,
        Z_stdexp * 1515 * 100,
        colors="k",
        levels=np.arange(0, 14, 2),
    )
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.contourf(
        X * 100,
        Y * 1000000,
        Z_std * 1515 * 100,
        levels=np.arange(0, 14, 1),
    )
    cbar = plt.colorbar()
    cbar.set_label("$c\sigma_\epsilon$ (cm)", rotation=270, labelpad=15)
    plt.xlabel("$\sigma_d$ (cm)")
    plt.ylabel("$\sigma_t$ ($\mu s$)")
    plt.show()


if __name__ == "__main__":
    combined_dependence(generate=True)

    npzfile = np.load(gps_output_path("Plot_Data/Combined_Dependence.npz"))
    print("Saved arrays:")
    for key in npzfile.files:
        print(f"{key}: {npzfile[key].shape}")