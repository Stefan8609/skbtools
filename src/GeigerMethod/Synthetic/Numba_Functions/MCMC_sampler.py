"""MCMC utilities for synthetic data generation."""

import numpy as np
import scipy.io as sio

from GeigerMethod.Synthetic.Numba_Functions.Numba_time_bias import (
    calculateTimesRayTracing_Bias_Real,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import two_pointer_index
from GeigerMethod.Synthetic.Numba_Functions.Numba_Geiger import findTransponder
from GeigerMethod.Synthetic.Numba_Functions.ESV_bias_split import (
    calculateTimesRayTracing_split,
)
from data import gps_data_path


# @njit
def compute_log_likelihood(
    lever_guess,
    gps1_grid_guess,
    CDOG_augments,
    esv_bias,
    time_bias,
    GPS_Coordinates,
    GPS_data,
    CDOG_guess,
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
        inv_guess = CDOG_guess + CDOG_augments[j]
        CDOG_data = CDOG_all_data[j]
        try:
            # Calculate the times either with the split ESV bias or the regular ESV bias
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
        except Exception as e:
            # if inversion fails, give very low likelihood
            print("Error: ", e)
            return -np.inf
        resid = (CDOG_full - GPS_full) * 1515 * 100
        total_ssq += np.sqrt(np.nansum(resid**2) / len(resid))
    # assume Gaussian errors with unit variance:
    return -0.5 * total_ssq


# @njit
def mcmc_sampler(
    n_iters,
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
    CDOG_guess,
    CDOG_all_data,
    offsets,
    proposal_scales=None,
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
    CDOG_guess : ndarray
        Base DOG position.
    CDOG_all_data : list
        Raw DOG arrival times for each unit.
    offsets : ndarray
        Offset guesses for each DOG.
    proposal_scales : dict, optional
        Dictionary of proposal standard deviations.

    Returns
    -------
    dict
        Dictionary containing arrays for all sampled parameters.
    """
    # default proposal stds
    if proposal_scales is None:
        proposal_scales = {
            "lever": np.array([0.01, 0.01, 0.1]),
            "gps_grid": 0.0,
            "CDOG_aug": 0.1,
            "esv_bias": 0.001,
            "time_bias": 0.000001,
        }

    split_esv = False
    if initial_esv_bias.ndim == 2:
        split_esv = True
        num_splits = initial_esv_bias.shape[1]

    # initialize chain
    lever_chain = np.zeros((n_iters, 3))
    gps_chain = np.zeros((n_iters, 4, 3))
    cdog_aug_chain = np.zeros((n_iters, 3, 3))
    if split_esv:
        ebias_chain = np.zeros((n_iters, 3, num_splits))
    else:
        ebias_chain = np.zeros((n_iters, 3))
    tbias_chain = np.zeros((n_iters, 3))
    logpost_chain = np.zeros(n_iters)

    # set initial state
    lever_curr = initial_lever_base.copy()
    gps_curr = initial_gps_grid.copy()
    cdog_aug_curr = initial_CDOG_augments.copy()
    ebias_curr = initial_esv_bias.copy()
    tbias_curr = initial_time_bias.copy()

    ll_curr = compute_log_likelihood(
        lever_curr,
        gps_curr,
        cdog_aug_curr,
        ebias_curr,
        tbias_curr,
        GPS_Coordinates,
        GPS_data,
        CDOG_guess,
        CDOG_all_data,
        offsets,
        dz_array,
        angle_array,
        esv_matrix,
    )

    for it in range(n_iters):
        # propose new state
        lever_prop = lever_curr + np.random.normal(0, proposal_scales["lever"], 3)
        gps_prop = gps_curr + np.random.normal(
            0, proposal_scales["gps_grid"], gps_curr.shape
        )
        cdog_aug_prop = cdog_aug_curr + np.random.normal(
            0, proposal_scales["CDOG_aug"], cdog_aug_curr.shape
        )
        ebias_prop = ebias_curr + np.random.normal(
            0, proposal_scales["esv_bias"], ebias_curr.shape
        )
        tbias_prop = tbias_curr + np.random.normal(0, proposal_scales["time_bias"], 3)

        ll_prop = compute_log_likelihood(
            lever_prop,
            gps_prop,
            cdog_aug_prop,
            ebias_prop,
            tbias_prop,
            GPS_Coordinates,
            GPS_data,
            CDOG_guess,
            CDOG_all_data,
            offsets,
            dz_array,
            angle_array,
            esv_matrix,
        )

        delta = ll_prop - ll_curr

        # accept‐reject
        if delta >= 0 or np.log(np.random.rand()) < delta:
            lever_curr, gps_curr, cdog_aug_curr, ebias_curr, tbias_curr = (
                lever_prop,
                gps_prop,
                cdog_aug_prop,
                ebias_prop,
                tbias_prop,
            )
            ll_curr = ll_prop

        # record
        lever_chain[it] = lever_curr
        gps_chain[it] = gps_curr
        cdog_aug_chain[it] = cdog_aug_curr
        ebias_chain[it] = ebias_curr
        tbias_chain[it] = tbias_curr
        logpost_chain[it] = ll_curr

        if it % 100 == 0:
            ll_rounded = float(int(ll_curr * 100) / 100.0)  # round to 2 decimal places
            print("Iter", it, ": logpost =", ll_rounded)

    return {
        "lever": lever_chain,
        "gps1_grid": gps_chain,
        "CDOG_aug": cdog_aug_chain,
        "esv_bias": ebias_chain,
        "time_bias": tbias_chain,
        "logpost": logpost_chain,
    }


# Example usage:
if __name__ == "__main__":
    # — load your data once —

    esv = sio.loadmat(gps_data_path("global_table_esv_normal.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 50
    data = np.load(gps_data_path("Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"][::downsample]
    GPS_data = data["GPS_data"][::downsample]
    CDOG_guess = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    CDOG_all_data = []
    for i in (1, 3, 4):
        tmp = sio.loadmat(gps_data_path(f"DOG{i}-camp.mat"))["tags"].astype(float)
        tmp[:, 1] /= 1e9
        CDOG_all_data.append(tmp)

    offsets = np.array([1866.0, 3175.0, 1939.0])

    # initial parameters
    init_lever = np.array([-12.4659, 9.6021, -13.2993])
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
    init_ebias = np.array(
        [
            [-0.4775, -0.4775, -0.4775, -0.4775],
            [-0.3199, -0.3199, -0.3199, -0.3199],
            [0.1122, 0.1122, 0.1122, 0.1122],
        ]
    )

    init_tbias = np.array([0.01518602, 0.015779, 0.018898])

    chain = mcmc_sampler(
        n_iters=20000,
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
        CDOG_guess=CDOG_guess,
        CDOG_all_data=CDOG_all_data,
        offsets=offsets,
    )

    """Saving and plotting the chain"""
    np.savez(
        "mcmc_chain.npz",
        lever=chain["lever"],
        gps1_grid=chain["gps1_grid"],
        CDOG_aug=chain["CDOG_aug"],
        esv_bias=chain["esv_bias"],
        time_bias=chain["time_bias"],
        logpost=chain["logpost"],
    )
