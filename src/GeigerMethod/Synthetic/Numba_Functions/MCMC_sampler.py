"""MCMC utilities for synthetic data generation."""

import numpy as np
import scipy.io as sio
from numba import njit
from numba.typed import List
import math

from GeigerMethod.Synthetic.Numba_Functions.Numba_time_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import two_pointer_index
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from GeigerMethod.Synthetic.Numba_Functions.ESV_bias_split import (
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
):
    """Compute the log likelihood for a given parameter set."""
    split_esv = False
    if esv_bias.ndim == 2:
        split_esv = True
    # first: compute transponder positions
    trans_coords = findTransponder(GPS_Coordinates, gps1_grid_guess, lever_guess)

    total_ssq = 0.0
    # loop each DOG
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
            times_guess, esv = calculateTimesRayTracing_Bias_Real(
                inv_guess,
                trans_coords,
                esv_bias[j],
                dz_array,
                angle_array,
                esv_matrix,
            )

            """Note doing GPS_data - time_bias to
            include time_bias in offset when calculating travel times"""
        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offsets[j],
            0.6,
            CDOG_data,
            GPS_data + time_bias[j],
            times_guess,
            trans_coords,
            esv,
            True,
        )
        resid = (CDOG_full - GPS_full) * 1515 * 100
        # manual nanmean of squared residuals to avoid temporary arrays
        sum_sq = 0.0
        count = 0
        for r in resid:
            if not math.isnan(r):
                sum_sq += r * r
                count += 1
        if count > 0:
            total_ssq += math.sqrt(sum_sq / count)
    # assume Gaussian errors with unit variance:
    return -0.5 * total_ssq


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
    ll_curr = compute_log_likelihood(
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

    for it in range(n_iters):
        # propose new state
        # — lever (elementwise) —
        lever_prop = lever_curr.copy()
        for i in range(3):
            lever_prop[i] = lever_curr[i] + np.random.normal(
                0.0, proposal_scales["lever"][i]
            )

        # — gps_grid & CDOG_aug (vector draws) —
        gps_prop = gps_curr + np.random.normal(
            0.0, proposal_scales["gps_grid"], gps_curr.shape
        )
        cdog_aug_prop = cdog_aug_curr + np.random.normal(
            0.0, proposal_scales["CDOG_aug"], cdog_aug_curr.shape
        )

        # — esv_bias (matching shape) —
        ebias_prop = ebias_curr + np.random.normal(
            0.0, proposal_scales["esv_bias"], ebias_curr.shape
        )

        # — time_bias —
        tbias_prop = tbias_curr + np.random.normal(
            0.0, proposal_scales["time_bias"], tbias_curr.shape
        )

        # evaluate
        ll_prop = compute_log_likelihood(
            lever_prop,
            gps_prop,
            cdog_aug_prop,
            ebias_prop,
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
            lever_prop,
            gps_prop,
            cdog_aug_prop,
            ebias_prop,
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

        # acceptance
        delta = lpo_prop - lpo_curr
        if delta >= 0 or np.log(np.random.rand()) < delta:
            lever_curr, gps_curr, cdog_aug_curr, ebias_curr, tbias_curr = (
                lever_prop,
                gps_prop,
                cdog_aug_prop,
                ebias_prop,
                tbias_prop,
            )
            ll_curr = ll_prop
            lpr_curr = lpr_prop
            lpo_curr = lpo_prop

        # record
        lever_chain[it, :] = lever_curr
        gps_chain[it, :, :] = gps_curr
        cdog_aug_chain[it, :, :] = cdog_aug_curr
        ebias_chain[it, :, :] = ebias_curr
        tbias_chain[it, :] = tbias_curr
        loglike_chain[it] = ll_curr
        logpost_chain[it] = lpo_curr

        if it % 100 == 0:
            print("Iter", it, ": logpost =", float(int(lpo_curr * 100) / 100.0))

    if burn_in <= n_iters:
        # discard burn-in samples
        lever_chain = lever_chain[burn_in:]
        gps_chain = gps_chain[burn_in:]
        cdog_aug_chain = cdog_aug_chain[burn_in:]
        ebias_chain = ebias_chain[burn_in:]
        tbias_chain = tbias_chain[burn_in:]
        loglike_chain = loglike_chain[burn_in:]
        logpost_chain = logpost_chain[burn_in:]

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
    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_normal.mat"))
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

    offsets = np.array([1866.0, 3175.0, 1939.0])

    # initial parameters
    init_lever = np.array([-13.12, 9.72, -15.9])

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
            [-397.63809, 371.47355, 773.26347],
            [825.31541, -110.93683, -734.15039],
            [236.27742, -1307.44426, -2189.59746],
        ]
    )
    # init_ebias = np.array([-0.4775, -0.3199, 0.1122])
    values = np.array([-0.4775, -0.3199, 0.1122])
    # values = np.array([0.4775, 0.3199, -0.1122])
    n = 7  # number of splits for ESV bias
    init_ebias = np.tile(values.reshape(-1, 1), (1, n))
    init_tbias = np.array([0.01518602, 0.015779, 0.018898])

    # proposal scales
    proposal_lever = np.array([0.01, 0.01, 0.05])
    proposal_gps_grid = 0.0
    proposal_CDOG_aug = 0.1
    proposal_esv_bias = 0.01
    proposal_time_bias = 0.000005

    # prior scales
    prior_lever = np.array([0.5, 0.5, 0.5])
    prior_gps_grid = 0.1
    prior_CDOG_aug = 0.5
    prior_esv_bias = 1.0
    prior_time_bias = 0.5

    (
        lever_chain,
        gps_chain,
        cdog_aug_chain,
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
    ) = mcmc_sampler(
        n_iters=1000,
        burn_in=500,
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

    # Save the chain to a .npz file
    inital_dict = {
        "lever": init_lever,
        "gps_grid": init_gps_grid,
        "CDOG_aug": init_aug,
        "esv_bias": init_ebias,
        "time_bias": init_tbias,
    }

    prior_dict = {
        "lever": prior_lever,
        "gps_grid": prior_gps_grid,
        "CDOG_aug": prior_CDOG_aug,
        "esv_bias": prior_esv_bias,
        "time_bias": prior_time_bias,
    }

    chain = {
        "lever": lever_chain,
        "gps1_grid": gps_chain,
        "CDOG_aug": cdog_aug_chain,
        "esv_bias": ebias_chain,
        "time_bias": tbias_chain,
        "loglike": loglike_chain,
        "logpost": logpost_chain,
        "initial": inital_dict,
        "prior": prior_dict,
    }
    np.savez(gps_output_path("mcmc_chain_test.npz"), **chain)
