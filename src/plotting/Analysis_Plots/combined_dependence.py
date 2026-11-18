import numpy as np
import matplotlib.pyplot as plt

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Synthetic.Modular_Synthetic import (
    BERMUDA_GPS1_TO_OTHERS,
    TRUE_LEVER,
    modular_synthetic,
)
from acoustics.synthetic_esv_tables import ESV_GENERATE, ESV_MODERATE
from bermuda_data import gps_output_path

"""Do both mapping to RMSE and mapping to average distance from real CDOG.
The clock offset is recovered from the data, not given.
"""


def position_noise_gain(GPS_Coordinates, gps1_to_others, lever, CDOG):
    """RMS over epochs of the range error per unit receiver noise sigma_d.

    The transponder is ``centroid + R (lever - receiver centroid)`` with R fitted
    to four receivers that each carry independent noise of sd sigma_d. To first
    order the fit error is a centroid shift (variance sigma_d^2 / 4 per axis) plus
    a rotation error with covariance sigma_d^2 inv(sum_k S_k), where
    ``S_k = |r_k|^2 I - r_k r_k^T`` and r_k is receiver k from the centroid. With
    ``a`` the centroid-to-transponder vector and u the unit DOG-to-transponder
    vector, ``g^2 = 1/4 + (a x u)^T inv(sum S) (a x u)``, averaged over epochs.

    ``GPS_Coordinates`` is ``(N, 4, 3)`` ECEF metres without added noise.
    """
    centroid = GPS_Coordinates.mean(axis=1)
    r = GPS_Coordinates - centroid[:, None, :]
    transponder = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    a = transponder - centroid
    u = transponder - CDOG
    u = u / np.linalg.norm(u, axis=1)[:, None]

    summed_S = (r**2).sum(axis=(1, 2))[:, None, None] * np.eye(3) - np.einsum(
        "nki,nkj->nij", r, r
    )
    w = np.cross(a, u)
    rotation_term = np.einsum(
        "ni,ni->n", w, np.linalg.solve(summed_S, w[..., None])[..., 0]
    )
    return np.sqrt(np.mean(0.25 + rotation_term))


def combined_dependence(
    generate=True, num_realizations=5, inversion_esv=ESV_MODERATE, solve_g=False
):
    """Map position and timing noise to resulting RMSE and CDOG error.

    Each grid point averages ``num_realizations`` runs; the per-run values are saved
    as ``Z_std_all`` and ``Z_error_all``. ESV_GENERATE as ``inversion_esv`` is the
    identical-table control; results are saved per table name.

    ``solve_g`` scales the expected contours' sigma_d by the gain g from
    ``position_noise_gain`` (receiver noise to range error) instead of 1. g is
    computed from the noise-free Bermuda geometry (DOG 3, 50x downsample).
    """
    data_path = gps_output_path(f"Plot_Data/Combined_Dependence_{inversion_esv}.npz")

    sound_speed = 1515.0  # m/s

    g = 1.0
    if solve_g:
        # Zero noise: the returned GPS_Coordinates are the noise-free receivers.
        out = modular_synthetic(0, 0, 0, 0, generate_type=1, plot=False)
        g = position_noise_gain(out[9], BERMUDA_GPS1_TO_OTHERS, TRUE_LEVER[1], out[7])
        print(f"Receiver-noise gain g = {g:.3f}")

    if generate:
        space_axis = np.linspace(0, 5 * 10**-2, 25)  # m
        time_axis = np.linspace(0, 5 * 10**-5, 25)  # s

        X, Y = np.meshgrid(space_axis, time_axis)

        Z_std_all = np.zeros((num_realizations, *X.shape))  # residual std in seconds
        Z_error_all = np.zeros((num_realizations, *X.shape))  # CDOG error in cm

        for i in range(X.shape[1]):
            print(i)

            for j in range(X.shape[0]):
                for k in range(num_realizations):
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
                        Y[j, i],
                        X[j, i],
                        0,
                        0,
                        esv1=ESV_GENERATE,
                        esv2=inversion_esv,
                        generate_type=1,
                        inversion_type=1,
                        plot=False,
                    )

                    Z_std_all[k, j, i] = np.std(CDOG_full - GPS_full)
                    Z_error_all[k, j, i] = (
                        np.linalg.norm(inversion_result[:3] - CDOG) * 100
                    )

        Z_std = Z_std_all.mean(axis=0)
        Z_error = Z_error_all.mean(axis=0)

        # Expected uncertainty in seconds
        Z_stdexp = np.sqrt((g * X / sound_speed) ** 2 + Y**2)

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
            g=g,
            Z_std_all=Z_std_all,
            Z_error_all=Z_error_all,
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
        )

        print(f"Saved plotting data to: {data_path}")

    else:
        npzfile = np.load(data_path)

        X = npzfile["X"]
        Y = npzfile["Y"]
        Z_std = npzfile["Z_std"]

        # Recompute expected uncertainty to make sure it uses current formula
        Z_stdexp = np.sqrt((g * X / sound_speed) ** 2 + Y**2)

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
    generate = True
    num_realizations = 5
    solve_g = True  # contours use the computed receiver-noise gain g
    # Moderate SSP perturbation, plus the identical-table control.
    for inversion_esv in (ESV_MODERATE, ESV_GENERATE):
        combined_dependence(
            generate=generate,
            num_realizations=num_realizations,
            inversion_esv=inversion_esv,
            solve_g=solve_g,
        )
