import numpy as np
import matplotlib.pyplot as plt

from Inversion_Workflow.Synthetic.Modular_Synthetic import (
    modular_synthetic,
)
from data import gps_output_path

"""Do both mapping to RMSE and mapping to average distance from real CDOG
Requires that offset is enforced as integer ambiguity was not working
at time of running
"""


def combined_dependence(generate=True):
    """Map position and timing noise to resulting RMSE and CDOG error."""
    data_path = gps_output_path("Plot_Data/Combined_Dependence.npz")

    sound_speed = 1515.0  # m/s

    if generate:
        space_axis = np.linspace(0, 5 * 10**-2, 25)  # m
        time_axis = np.linspace(0, 5 * 10**-5, 25)  # s

        X, Y = np.meshgrid(space_axis, time_axis)

        Z_std = np.zeros_like(X)  # residual std in seconds
        Z_error = np.zeros_like(X)  # CDOG position error in cm

        max_attempts = 10
        threshold_factor = 2.2
        min_threshold_cm = 2.0  # prevents zero-noise case from requiring exact zero

        for i in range(X.shape[1]):
            print(i)

            for j in range(X.shape[0]):
                position_noise = X[j, i]
                time_noise = Y[j, i]

                expected_std_cm = np.sqrt(
                    (position_noise * 100) ** 2 + (time_noise * sound_speed * 100) ** 2
                )

                threshold_cm = max(
                    threshold_factor * expected_std_cm,
                    min_threshold_cm,
                )

                for attempt in range(max_attempts):
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
                    std_diff = np.std(diff_data)  # seconds
                    std_diff_cm = std_diff * sound_speed * 100

                    if std_diff_cm <= threshold_cm:
                        break

                    print(
                        f"Retrying i={i}, j={j}, "
                        f"attempt={attempt + 1}, "
                        f"std={std_diff_cm:.2f} cm, "
                        f"threshold={threshold_cm:.2f} cm"
                    )

                if std_diff_cm > threshold_cm:
                    print(
                        f"Warning: accepted i={i}, j={j} after "
                        f"{max_attempts} attempts with "
                        f"std={std_diff_cm:.2f} cm, "
                        f"threshold={threshold_cm:.2f} cm"
                    )

                Z_std[j, i] = std_diff
                Z_error[j, i] = np.linalg.norm(inversion_result[:3] - CDOG) * 100

        # Expected uncertainty in seconds
        Z_stdexp = np.sqrt((X / sound_speed) ** 2 + Y**2)

        # Expected uncertainty in cm
        Z_stdexp_cm = Z_stdexp * sound_speed * 100

        np.savez(
            data_path,
            space_axis=space_axis,
            time_axis=time_axis,
            X=X,
            Y=Y,
            Z_std=Z_std,
            Z_error=Z_error,
            Z_stdexp=Z_stdexp,
            X_cm=X * 100,
            Y_us=Y * 1_000_000,
            Z_std_cm=Z_std * sound_speed * 100,
            Z_stdexp_cm=Z_stdexp_cm,
            residual_scale_mps=sound_speed,
            position_noise_units="m",
            time_noise_units="s",
            Z_std_units="s",
            Z_error_units="cm",
            Z_stdexp_units="s",
            Z_stdexp_cm_units="cm",
            threshold_factor=threshold_factor,
            min_threshold_cm=min_threshold_cm,
            max_attempts=max_attempts,
        )

        print(f"Saved plotting data to: {data_path}")

    else:
        npzfile = np.load(data_path)

        X = npzfile["X"]
        Y = npzfile["Y"]
        Z_std = npzfile["Z_std"]

        # Recompute expected uncertainty to make sure it uses current formula
        Z_stdexp = np.sqrt((X / sound_speed) ** 2 + Y**2)

    Z_plot_cm = Z_std * sound_speed * 100
    Z_stdexp_cm = Z_stdexp * sound_speed * 100

    CS = plt.contour(
        X * 100,
        Y * 1_000_000,
        Z_stdexp_cm,
        colors="k",
        levels=np.arange(0, 14, 2),
    )
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    CF = plt.contourf(
        X * 100,
        Y * 1_000_000,
        Z_plot_cm,
        levels=np.arange(0, 14, 1),
        extend="max",
    )

    cbar = plt.colorbar(CF)
    cbar.set_label("$c\\sigma_\\epsilon$ (cm)", rotation=270, labelpad=15)

    plt.xlabel("$\\sigma_d$ (cm)")
    plt.ylabel("$\\sigma_t$ ($\\mu s$)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    combined_dependence(generate=True)

    npzfile = np.load(gps_output_path("Plot_Data/Combined_Dependence.npz"))

    print("Saved arrays:")
    for key in npzfile.files:
        print(f"{key}: {npzfile[key].shape}")
