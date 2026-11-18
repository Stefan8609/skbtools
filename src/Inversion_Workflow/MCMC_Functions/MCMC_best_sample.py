import json

import numpy as np
import matplotlib.pyplot as plt

from plotting.Plot_Modular import (
    time_series_plot,
    range_residual,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_split,
    calculateTimesRayTracing_windows,
)

from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from Inversion_Workflow.MCMC_Functions.MCMC_diagnostics import pooled_run
from bermuda_data import gps_output_path


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 20,  # change this freely
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)


def load_min_logpost_params(npz_path, loglike=False):
    """
    Load an MCMC chain saved with np.savez and return the parameter set
    corresponding to the maximum logpost.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file (e.g. gps_output_path("mcmc_chain_good.npz")).

    Returns
    -------
    dict
        A dictionary with keys 'lever', 'gps1_grid', 'CDOG_aug',
        'esv_bias', 'time_bias', 'logpost', each holding the value
        (or array) at the minimum-logpost index.
    """
    data = np.load(npz_path)
    logpost = data["logpost"]
    loglikelihood = data["loglike"]
    if loglike:
        idx_min = np.argmax(loglikelihood)
    else:
        idx_min = np.argmax(logpost)

    return {
        "lever": data["lever"][idx_min],
        "CDOG_aug": data["CDOG_aug"][idx_min],
        "esv_bias": data["esv_bias"][idx_min],
        "time_bias": data["time_bias"][idx_min],
        "logpost": logpost[idx_min],
        "loglike": loglikelihood[idx_min],
    }


def split_samples(path, n_splits):
    """
    Split the MCMC chain into multiple parts for parallel processing.

    Parameters
    ----------
    path : str
        Path to the .npz file containing the MCMC chain.
    n_splits : int, optional
        Number of splits to create (default is 5).
    """
    dir = gps_output_path(path)
    esv_biases = np.zeros((n_splits, 3))
    for i in range(n_splits):
        # Load the MCMC chain
        data = np.load(f"{dir}/split_{i}.npz")
        logpost = data["logpost"]
        idx_min = np.argmax(logpost)

        # print(f"Best parameters for split {i} \n",
        #       f"Lever guess: {data['lever'][idx_min]}\n",
        #       f"GPS1 grid guess: {data['gps1_grid'][idx_min]}\n",
        #       f"CDOG augment: {data['CDOG_aug'][idx_min]}\n",
        #       f"ESV bias: {data['esv_bias'][idx_min]}\n",
        #       f"Time bias: {data['time_bias'][idx_min]}\n",)
        esv_biases[i, :] = data["esv_bias"][idx_min].T

        print(f"ESV bias for split {i}: {data['esv_bias'][idx_min].T}")
    print(esv_biases[:, 0], esv_biases[:, 1], esv_biases[:, 2])
    print(esv_biases)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    splits = np.arange(n_splits)
    for dog in range(3):
        ax = axes[dog]
        ax.plot(splits, esv_biases[:, dog], marker="o", linestyle="-")
        ax.set_title(f"DOG {dog + 1}")
        ax.set_xlabel("Split index")
        if dog == 0:
            ax.set_ylabel("ESV bias (m)")
        ax.set_xticks(splits)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("ESV Bias Across Splits for Each DOG")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_best_sample(
    npz_path,
    GPS_Coordinates,
    GPS_data,
    CDOG_guess,
    CDOG_all_data,
    offsets,
    dz_array,
    angle_array,
    esv_matrix,
    CDOG_num=3,
    timestamp=None,
    loglike=False,
    chain_name=None,
):
    """
    Plot the best sample from an MCMC chain based on minimum logpost.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing the MCMC chain.
    downsample : int, optional
        Step used when plotting chain values (default is 1).
    """
    CDOG_to_index = {1: 0, 3: 1, 4: 2}
    CDOG_index = CDOG_to_index[CDOG_num]

    best = load_min_logpost_params(npz_path, loglike=loglike)

    lever_guess = best["lever"]
    CDOG_augments = best["CDOG_aug"]
    esv_bias = best["esv_bias"]
    time_bias = best["time_bias"]
    logpost = best["logpost"]
    loglikelihood = best["loglike"]

    print("\n")
    print(f"Best sample by {'loglikelihood' if loglike else 'logpost'}:")
    print("==============================")
    print("Best parameters:")
    print("Lever guess:", lever_guess)
    print("CDOG augment:", CDOG_augments)
    print("ESV bias:", esv_bias)
    print("Time bias:", time_bias)
    print("Log posterior:", logpost)
    print("Log likelihood:", loglikelihood)

    gps1_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.393414, -4.223503, 0.029415],
            [-12.095685, -0.945685, 0.004397],
            [-8.686741, 5.169188, -0.024993],
        ]
    )

    split_esv = (esv_bias.ndim == 2) and (esv_bias.shape[1] > 1)
    trans_coords = findTransponder(GPS_Coordinates, gps1_grid, lever_guess)

    inv_guess = CDOG_guess + CDOG_augments[CDOG_index]
    CDOG_data = CDOG_all_data[CDOG_index]

    if split_esv:
        times_guess, esv = calculateTimesRayTracing_split(
            inv_guess,
            trans_coords,
            esv_bias[CDOG_index],
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            inv_guess,
            trans_coords,
            esv_bias[CDOG_index],
            dz_array,
            angle_array,
            esv_matrix,
        )

    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        offsets[CDOG_index],
        0.6,
        CDOG_data,
        GPS_data - time_bias[CDOG_index],
        times_guess,
        trans_coords,
        esv,
    )

    # Plotting
    time_series_plot(
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        position_noise=0,
        time_noise=0,
        block=True,
        save=True,
        path=f"Figs/MCMC/{chain_name}",
        zoom_start=35000,
        DOG_num=CDOG_num,
    )
    range_residual(
        transponder_coordinates_full,
        esv_full,
        inv_guess,
        CDOG_full,
        GPS_full,
        GPS_clock,
        save=True,
        path=f"Figs/MCMC/{chain_name}",
        DOG_num=CDOG_num,
    )


def plot_run_best_sample(run_name, DOG_num, loglike=False):
    """Fig. 9 for a run of the reworked sampler: residuals at the best draw.

    Uses the sampler's own fixed pairs and forward model (``setup_bermuda`` and
    ``residuals_cm``), so the residuals are exactly those the likelihood saw,
    including any clock drift. ``GPS_full`` is the modelled travel time,
    ``CDOG_full`` the observed one (modelled plus residual), and the clocks are
    the GPS epoch plus the travel time, as ``two_pointer_index`` returns them.
    """
    chain = pooled_run(run_name)
    config = json.loads(str(chain["config_json"]))
    model, problem = ms.setup_bermuda(config)
    if chain["fixed_phi"].size:
        model["fixed_phi"] = chain["fixed_phi"]
    best = np.argmax(chain["loglike"] if loglike else chain["logpost"])
    theta = chain["theta_samples"][best]
    j = config["dogs"].index(DOG_num)
    pairs = problem["pair_slices"][j]

    params = ms.unpack(theta, model)
    trans = (problem["R_stacked"] @ params["lever"]).reshape(-1, 3) + problem["d"]
    DOG_position = problem["CDOG_reference"] + params["CDOG_aug"][j]
    if "epoch_window" in problem:
        times, esv = calculateTimesRayTracing_windows(
            DOG_position,
            trans,
            params["esv_bias"][j],
            problem["epoch_window"],
            problem["dz_array"],
            problem["angle_array"],
            problem["esv_matrix"],
        )
    elif model["num_splits"] > 1:
        times, esv = calculateTimesRayTracing_split(
            DOG_position,
            trans,
            params["esv_bias"][j],
            problem["dz_array"],
            problem["angle_array"],
            problem["esv_matrix"],
        )
    else:
        times, esv = calculateTimesRayTracing_Bias_Real(
            DOG_position,
            trans,
            params["esv_bias"][j, 0],
            problem["dz_array"],
            problem["angle_array"],
            problem["esv_matrix"],
        )
    residual_s = ms.residuals_cm(theta, model, problem)[pairs] / ms.RANGE_CM_PER_S
    gps_index = problem["pair_gps_index"][pairs]
    gps_time = problem["pair_t_from_mid"][pairs] + problem["t_mid"][j]

    GPS_full = times[gps_index]
    GPS_clock = gps_time + GPS_full
    CDOG_full = GPS_full + residual_s
    CDOG_clock = GPS_clock + residual_s

    sigma, phi = ms.noise_parameters(theta, model)
    print(
        f"Run {run_name}, DOG {DOG_num}: best draw {best}, "
        f"c*sigma {sigma[j]:.2f} cm, phi {phi[j]:.3f}, "
        f"residual rms {np.sqrt(np.mean((residual_s * ms.RANGE_CM_PER_S) ** 2)):.2f} cm"
    )
    path = f"Figs/MCMC/{('loglike_' if loglike else 'logpost_') + run_name}"
    time_series_plot(
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        position_noise=0,
        time_noise=0,
        block=True,
        save=True,
        path=path,
        zoom_start=30000,
        DOG_num=DOG_num,
    )
    range_residual(
        trans[gps_index],
        esv[gps_index],
        DOG_position,
        CDOG_full,
        GPS_full,
        GPS_clock,
        save=True,
        path=path,
        DOG_num=DOG_num,
    )


if __name__ == "__main__":
    # A run of the reworked sampler (Output/MCMC_Chains/<RUN_NAME>), all chains
    # pooled. plot_best_sample above still reads old single-file chains.
    RUN_NAME = "ar1_common30_drift"
    DOG_NUM = 1
    LOGLIKE = False

    plot_run_best_sample(RUN_NAME, DOG_NUM, loglike=LOGLIKE)
