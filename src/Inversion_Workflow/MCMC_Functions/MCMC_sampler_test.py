"""MCMC utilities for synthetic data generation."""

import numpy as np
import scipy.io as sio
from numba import njit
from numba.typed import List
import math

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_split,
)
from data import gps_data_path, gps_output_path


@njit(cache=True, fastmath=True)
def compute_log_likelihood(
    lever_guess,
    gps1_grid_guess,
    CDOG_augments,
    esv_bias,
    time_bias,
    GPS_Coordinates,
    GPS_data,
    CDOG_reference,
    CDOG_all_data,
    offsets,
    dz_array,
    angle_array,
    esv_matrix,
    sigma_cm=20.0,  # <-- NEW: noise scale in cm
):
    """Compute Gaussian log-likelihood using SSE of residuals (in cm)."""

    # Only treat as "split ESV" if there is more than 1 split column
    split_esv = (esv_bias.ndim == 2) and (esv_bias.shape[1] > 1)

    # Compute transponder positions
    trans_coords = findTransponder(GPS_Coordinates, gps1_grid_guess, lever_guess)

    total_sse = 0.0
    total_n = 0

    inv_sigma2 = 1.0 / (sigma_cm * sigma_cm)

    for j in range(3):
        inv_guess = CDOG_reference + CDOG_augments[j]
        CDOG_data = CDOG_all_data[j]

        if split_esv:
            times_guess, esv = calculateTimesRayTracing_split(
                inv_guess,
                trans_coords,
                esv_bias[j],
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            # If esv_bias is (3,1), pass scalar per DOG
            eb = esv_bias[j, 0] if esv_bias.ndim == 2 else esv_bias[j]
            times_guess, esv = calculateTimesRayTracing_Bias_Real(
                inv_guess,
                trans_coords,
                eb,
                dz_array,
                angle_array,
                esv_matrix,
            )

        gps_t = GPS_data - time_bias[j]

        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offsets[j],
            0.4,
            CDOG_data,
            gps_t,
            times_guess,
            trans_coords,
            esv,
        )

        # residual in cm
        dt = CDOG_full - GPS_full
        resid = dt * 1515.0 * 100.0

        # SSE with NaN guard
        for r in resid:
            if not math.isnan(r):
                total_sse += r * r
                total_n += 1

    if total_n == 0:
        return -1e30, 0  # extremely bad likelihood if nothing paired

    # Gaussian loglike (up to additive constant)
    return (
        -0.5 * total_sse * inv_sigma2 - 0.5 * total_n * math.log(sigma_cm * sigma_cm),
        total_n,
    )


@njit(cache=True, fastmath=True)
def compute_log_prior(
    lever_guess,
    gps1_grid_guess,
    CDOG_augments,
    esv_bias,
    time_bias,
    initial_lever_base,
    initial_gps_grid,
    initial_CDOG_augments,
    initial_esv_bias,
    initial_time_bias,
    lever_scale,
    gps_grid_scale,
    CDOG_aug_scale,
    esv_bias_scale,
    time_bias_scale,
):
    lp = 0.0
    lp += -0.5 * np.sum(((lever_guess - initial_lever_base) / lever_scale) ** 2)
    lp += -0.5 * np.sum(((gps1_grid_guess - initial_gps_grid) / gps_grid_scale) ** 2)
    lp += -0.5 * np.sum(((CDOG_augments - initial_CDOG_augments) / CDOG_aug_scale) ** 2)
    lp += -0.5 * np.sum(((esv_bias - initial_esv_bias) / esv_bias_scale) ** 2)
    lp += -0.5 * np.sum(((time_bias - initial_time_bias) / time_bias_scale) ** 2)
    return lp


@njit(cache=True, fastmath=True)
def mcmc_sampler(
    n_iters,
    burn_in,
    initial_lever_base,
    initial_gps_grid,
    initial_CDOG_augments,
    initial_esv_bias,
    initial_time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    GPS_Coordinates,
    GPS_data,
    CDOG_reference,
    CDOG_all_data,
    offsets,
    proposal_lever=None,
    proposal_gps_grid=0.0,
    proposal_CDOG_aug=0.1,
    proposal_esv_bias=0.01,
    proposal_time_bias=0.000005,
    prior_lever=None,
    prior_gps_grid=0.1,
    prior_CDOG_aug=25.0,
    prior_esv_bias=1.0,
    prior_time_bias=0.5,
):
    """Run a simple Metropolis-Hastings sampler over several parameters.

    Parameters
    ----------
    n_iters : int
        Number of MCMC iterations to perform.
    initial_lever_base : ndarray
        Starting lever arm vector.
    initial_gps_grid : ndarray
        Initial GPS grid offsets.
    initial_CDOG_augments : ndarray
        Initial CDOG augment vectors for each DOG.
    initial_esv_bias, initial_time_bias : ndarray
        Starting bias values for each DOG.
    dz_array, angle_array, esv_matrix : ndarray
        ESV lookup table used during the likelihood evaluation.
    GPS_Coordinates, GPS_data : ndarray
        Observed GPS positions and times.
    CDOG_reference : ndarray
        Base DOG position.
    CDOG_all_data : list
        Raw DOG arrival times for each unit.
    offsets : ndarray
        Offset guesses for each DOG.

    Returns
    -------
    dict
        Dictionary containing arrays for all sampled parameters.
    """

    if proposal_lever is None:
        proposal_lever = np.array([0.01, 0.01, 0.05])

    if prior_lever is None:
        prior_lever = np.array([0.5, 0.5, 1.0])

    # default proposal stds
    proposal_scales = {
        "lever": proposal_lever,
        "gps_grid": proposal_gps_grid,
        "CDOG_aug": proposal_CDOG_aug,
        "esv_bias": proposal_esv_bias,
        "time_bias": proposal_time_bias,
    }

    # default prior stds
    prior_scales = {
        "lever": prior_lever,
        "gps_grid": prior_gps_grid,
        "CDOG_aug": prior_CDOG_aug,
        "esv_bias": prior_esv_bias,
        "time_bias": prior_time_bias,
    }

    # reshape initial_esv_bias into a (3, num_splits) array for Numba compatibility
    if initial_esv_bias.ndim == 1:
        num_splits = 1
        ebias_curr = initial_esv_bias.reshape(3, 1)
    else:
        num_splits = initial_esv_bias.shape[1]
        ebias_curr = initial_esv_bias

    # initialize chains (always 3D for ebias_chain)
    lever_chain = np.zeros((n_iters, 3))
    gps_chain = np.zeros((n_iters, 4, 3))
    cdog_aug_chain = np.zeros((n_iters, 3, 3))
    ebias_chain = np.zeros((n_iters, 3, num_splits))
    tbias_chain = np.zeros((n_iters, 3))
    loglike_chain = np.zeros(n_iters)
    logpost_chain = np.zeros(n_iters)

    # set initial state
    lever_curr = initial_lever_base.copy()
    gps_curr = initial_gps_grid.copy()
    cdog_aug_curr = initial_CDOG_augments.copy()
    tbias_curr = initial_time_bias.copy()

    # compute initial likelihood & prior
    ll_curr, total_n = compute_log_likelihood(
        lever_curr,
        gps_curr,
        cdog_aug_curr,
        ebias_curr,
        tbias_curr,
        GPS_Coordinates,
        GPS_data,
        CDOG_reference,
        CDOG_all_data,
        offsets,
        dz_array,
        angle_array,
        esv_matrix,
    )
    lpr_curr = compute_log_prior(
        lever_curr,
        gps_curr,
        cdog_aug_curr,
        ebias_curr,
        tbias_curr,
        initial_lever_base,
        initial_gps_grid,
        initial_CDOG_augments,
        initial_esv_bias,
        initial_time_bias,
        prior_scales["lever"],
        prior_scales["gps_grid"],
        prior_scales["CDOG_aug"],
        prior_scales["esv_bias"],
        prior_scales["time_bias"],
    )
    lpo_curr = ll_curr + lpr_curr

    acc_lever = 0
    acc_aug = 0
    acc_eb = 0
    acc_tb = 0

    for it in range(n_iters):
        # -------------------------
        # 1) LEVER block
        # -------------------------
        lever_prop = lever_curr.copy()
        for i in range(3):
            lever_prop[i] = lever_curr[i] + np.random.normal(
                0.0, proposal_scales["lever"][i]
            )

        ll_prop, total_n = compute_log_likelihood(
            lever_prop,
            gps_curr,
            cdog_aug_curr,
            ebias_curr,
            tbias_curr,
            GPS_Coordinates,
            GPS_data,
            CDOG_reference,
            CDOG_all_data,
            offsets,
            dz_array,
            angle_array,
            esv_matrix,
        )
        lpr_prop = compute_log_prior(
            lever_prop,
            gps_curr,
            cdog_aug_curr,
            ebias_curr,
            tbias_curr,
            initial_lever_base,
            initial_gps_grid,
            initial_CDOG_augments,
            initial_esv_bias,
            initial_time_bias,
            prior_scales["lever"],
            prior_scales["gps_grid"],
            prior_scales["CDOG_aug"],
            prior_scales["esv_bias"],
            prior_scales["time_bias"],
        )
        lpo_prop = ll_prop + lpr_prop

        delta = lpo_prop - lpo_curr
        if delta >= 0.0 or np.log(np.random.rand()) < delta:
            lever_curr = lever_prop
            ll_curr = ll_prop
            lpr_curr = lpr_prop
            lpo_curr = lpo_prop
            acc_lever += 1

        # -------------------------
        # 2) CDOG AUG block
        # -------------------------
        cdog_aug_prop = cdog_aug_curr + np.random.normal(
            0.0, proposal_scales["CDOG_aug"], cdog_aug_curr.shape
        )

        ll_prop, total_n = compute_log_likelihood(
            lever_curr,
            gps_curr,
            cdog_aug_prop,
            ebias_curr,
            tbias_curr,
            GPS_Coordinates,
            GPS_data,
            CDOG_reference,
            CDOG_all_data,
            offsets,
            dz_array,
            angle_array,
            esv_matrix,
        )
        lpr_prop = compute_log_prior(
            lever_curr,
            gps_curr,
            cdog_aug_prop,
            ebias_curr,
            tbias_curr,
            initial_lever_base,
            initial_gps_grid,
            initial_CDOG_augments,
            initial_esv_bias,
            initial_time_bias,
            prior_scales["lever"],
            prior_scales["gps_grid"],
            prior_scales["CDOG_aug"],
            prior_scales["esv_bias"],
            prior_scales["time_bias"],
        )
        lpo_prop = ll_prop + lpr_prop

        delta = lpo_prop - lpo_curr
        if delta >= 0.0 or np.log(np.random.rand()) < delta:
            cdog_aug_curr = cdog_aug_prop
            ll_curr = ll_prop
            lpr_curr = lpr_prop
            lpo_curr = lpo_prop
            acc_aug += 1

        # -------------------------
        # 3) ESV BIAS block
        # -------------------------
        ebias_prop = ebias_curr + np.random.normal(
            0.0, proposal_scales["esv_bias"], ebias_curr.shape
        )

        ll_prop, total_n = compute_log_likelihood(
            lever_curr,
            gps_curr,
            cdog_aug_curr,
            ebias_prop,
            tbias_curr,
            GPS_Coordinates,
            GPS_data,
            CDOG_reference,
            CDOG_all_data,
            offsets,
            dz_array,
            angle_array,
            esv_matrix,
        )
        lpr_prop = compute_log_prior(
            lever_curr,
            gps_curr,
            cdog_aug_curr,
            ebias_prop,
            tbias_curr,
            initial_lever_base,
            initial_gps_grid,
            initial_CDOG_augments,
            initial_esv_bias,
            initial_time_bias,
            prior_scales["lever"],
            prior_scales["gps_grid"],
            prior_scales["CDOG_aug"],
            prior_scales["esv_bias"],
            prior_scales["time_bias"],
        )
        lpo_prop = ll_prop + lpr_prop

        delta = lpo_prop - lpo_curr
        if delta >= 0.0 or np.log(np.random.rand()) < delta:
            ebias_curr = ebias_prop
            ll_curr = ll_prop
            lpr_curr = lpr_prop
            lpo_curr = lpo_prop
            acc_eb += 1

        # -------------------------
        # 4) TIME BIAS block
        # -------------------------
        tbias_prop = tbias_curr + np.random.normal(
            0.0, proposal_scales["time_bias"], tbias_curr.shape
        )

        ll_prop, total_n = compute_log_likelihood(
            lever_curr,
            gps_curr,
            cdog_aug_curr,
            ebias_curr,
            tbias_prop,
            GPS_Coordinates,
            GPS_data,
            CDOG_reference,
            CDOG_all_data,
            offsets,
            dz_array,
            angle_array,
            esv_matrix,
        )
        lpr_prop = compute_log_prior(
            lever_curr,
            gps_curr,
            cdog_aug_curr,
            ebias_curr,
            tbias_prop,
            initial_lever_base,
            initial_gps_grid,
            initial_CDOG_augments,
            initial_esv_bias,
            initial_time_bias,
            prior_scales["lever"],
            prior_scales["gps_grid"],
            prior_scales["CDOG_aug"],
            prior_scales["esv_bias"],
            prior_scales["time_bias"],
        )
        lpo_prop = ll_prop + lpr_prop

        delta = lpo_prop - lpo_curr
        if delta >= 0.0 or np.log(np.random.rand()) < delta:
            tbias_curr = tbias_prop
            ll_curr = ll_prop
            lpr_curr = lpr_prop
            lpo_curr = lpo_prop
            acc_tb += 1

        # -------------------------
        # record (unchanged)
        # -------------------------
        lever_chain[it, :] = lever_curr
        gps_chain[it, :, :] = gps_curr
        cdog_aug_chain[it, :, :] = cdog_aug_curr
        ebias_chain[it, :, :] = ebias_curr
        tbias_chain[it, :] = tbias_curr
        loglike_chain[it] = ll_curr
        logpost_chain[it] = lpo_curr

        if it % 100 == 0:
            print(
                "Iter",
                it,
                ": logpost =",
                float(int(lpo_curr * 100) / 100.0),
                "Pairs =",
                total_n,
            )
        if it % 1000 == 0 and it > 0:
            print(
                "acc (last 1000): lever",
                acc_lever / 1000,
                "aug",
                acc_aug / 1000,
                "esv",
                acc_eb / 1000,
                "tb",
                acc_tb / 1000,
            )
            acc_lever = acc_aug = acc_eb = acc_tb = 0

    if burn_in <= n_iters:
        # discard burn-in samples
        lever_chain = lever_chain[burn_in:]
        gps_chain = gps_chain[burn_in:]
        cdog_aug_chain = cdog_aug_chain[burn_in:]
        ebias_chain = ebias_chain[burn_in:]
        tbias_chain = tbias_chain[burn_in:]
        loglike_chain = loglike_chain[burn_in:]
        logpost_chain = logpost_chain[burn_in:]

    # #Print final acceptance rates
    # print("Final acceptance rates:")
    # print(" Lever: ", acc_lever / n_iters)
    # print(" CDOG Augment:", acc_aug / n_iters)
    # print(" ESV Bias: ", acc_eb / n_iters)
    # print(" Time Bias: ", acc_tb / n_iters)

    return (
        lever_chain,
        gps_chain,
        cdog_aug_chain,
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
    )


# Example usage:
if __name__ == "__main__":
    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 50
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"][::downsample]
    GPS_data = data["GPS_data"][::downsample]
    CDOG_reference = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    CDOG_all_data = []
    for i in (1, 3, 4):
        tmp = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{i}-camp.mat"))["tags"].astype(
            float
        )
        tmp[:, 1] /= 1e9
        CDOG_all_data.append(tmp)

    typed_CDOG_all_data = List()
    for arr in CDOG_all_data:
        typed_CDOG_all_data.append(arr)

    offsets = np.array([1866.016, 3175.017, 1939.0178])

    # initial parameters
    init_lever = np.array([-12.80, 9.50, -12.15])

    init_gps_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.393414, -4.223503, 0.029415],
            [-12.095685, -0.945685, 0.004397],
            [-8.686741, 5.169188, -0.024993],
        ]
    )
    init_aug = np.array(
        [
            [-396.91, 369.80, 774.24],
            [826.22, -112.94, -733.06],
            [236.20, -1306.98, -2189.99],
        ]
    )
    # init_ebias = np.array([-0.4775, -0.3199, 0.1122])
    values = np.array([-0.4078, -0.2667, -0.1230])
    # values = np.array([0.4775, 0.3199, -0.1122])
    n = 4  # number of splits for ESV bias
    init_ebias = np.tile(values.reshape(-1, 1), (1, n))
    init_tbias = np.array([0.0, 0.0, 0.0])

    # proposal scales
    proposal_lever = np.array([0.01, 0.01, 0.025])
    proposal_gps_grid = 0.0
    proposal_CDOG_aug = 0.015
    proposal_esv_bias = 0.002
    proposal_time_bias = 0.0

    # prior scales
    prior_lever = np.array([0.5, 0.5, 0.5])
    prior_gps_grid = 0.1
    prior_CDOG_aug = 0.5
    prior_esv_bias = 1.0
    prior_time_bias = 0.01

    (
        lever_chain,
        gps_chain,
        cdog_aug_chain,
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
    ) = mcmc_sampler(
        n_iters=50000,
        burn_in=1000,
        initial_lever_base=init_lever,
        initial_gps_grid=init_gps_grid,
        initial_CDOG_augments=init_aug,
        initial_esv_bias=init_ebias,
        initial_time_bias=init_tbias,
        dz_array=dz_array,
        angle_array=angle_array,
        esv_matrix=esv_matrix,
        GPS_Coordinates=GPS_Coordinates,
        GPS_data=GPS_data,
        CDOG_reference=CDOG_reference,
        CDOG_all_data=typed_CDOG_all_data,
        offsets=offsets,
        proposal_lever=proposal_lever,
        proposal_gps_grid=proposal_gps_grid,
        proposal_CDOG_aug=proposal_CDOG_aug,
        proposal_esv_bias=proposal_esv_bias,
        proposal_time_bias=proposal_time_bias,
        prior_lever=prior_lever,
        prior_gps_grid=prior_gps_grid,
        prior_CDOG_aug=prior_CDOG_aug,
        prior_esv_bias=prior_esv_bias,
        prior_time_bias=prior_time_bias,
    )

    np.savez(
        gps_output_path("mcmc_chain_test.npz"),
        # posterior chains
        lever=lever_chain,
        gps1_grid=gps_chain,
        CDOG_aug=cdog_aug_chain,
        esv_bias=ebias_chain,
        time_bias=tbias_chain,
        loglike=loglike_chain,
        logpost=logpost_chain,
        # initial values
        init_lever=init_lever,
        init_gps_grid=init_gps_grid,
        init_CDOG_aug=init_aug,
        init_esv_bias=init_ebias,
        init_time_bias=init_tbias,
        # priors
        prior_lever=prior_lever,
        prior_gps_grid=prior_gps_grid,
        prior_CDOG_aug=prior_CDOG_aug,
        prior_esv_bias=prior_esv_bias,
        prior_time_bias=prior_time_bias,
        # proposals
        proposal_lever=proposal_lever,
        proposal_gps_grid=proposal_gps_grid,
        proposal_CDOG_aug=proposal_CDOG_aug,
        proposal_esv_bias=proposal_esv_bias,
        proposal_time_bias=proposal_time_bias,
    )
