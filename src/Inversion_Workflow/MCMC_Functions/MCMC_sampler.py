"""MCMC utilities for synthetic data generation."""

import math

import numpy as np
import scipy.io as sio
from numba import njit
from numba.typed import List

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_split,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from data import gps_data_path, gps_output_path


@njit(cache=True)
def find_fixed_pair_indices(
    offset,
    threshold,
    CDOG_data,
    GPS_data,
    GPS_travel_times,
):
    """Find the CDOG and GPS row indices once using the initial model."""

    max_pairs = min(CDOG_data.shape[0], GPS_data.shape[0])
    cdog_indices = np.empty(max_pairs, dtype=np.int64)
    gps_indices = np.empty(max_pairs, dtype=np.int64)

    cdog_pointer = 0
    gps_pointer = 0
    count = 0

    while cdog_pointer < CDOG_data.shape[0] and gps_pointer < GPS_data.shape[0]:
        cdog_time = CDOG_data[cdog_pointer, 0] + CDOG_data[cdog_pointer, 1] - offset
        gps_time = GPS_data[gps_pointer] + GPS_travel_times[gps_pointer]

        if abs(gps_time - cdog_time) < threshold:
            cdog_indices[count] = cdog_pointer
            gps_indices[count] = gps_pointer

            cdog_pointer += 1
            gps_pointer += 1
            count += 1

        elif gps_time < cdog_time:
            gps_pointer += 1

        else:
            cdog_pointer += 1

    return cdog_indices[:count], gps_indices[:count]


@njit(cache=True, fastmath=True)
def compute_log_likelihood(
    trans_coords,
    CDOG_augments,
    esv_bias,
    time_bias,
    CDOG_reference,
    gps_pair_indices,
    pair_time_base,
    pair_counts,
    dz_array,
    angle_array,
    esv_matrix,
    sigma_cm=20.0,
):
    """Compute Gaussian log-likelihood using fixed CDOG-GPS pairs."""

    split_esv = esv_bias.shape[1] > 1

    total_sse = 0.0
    total_n = 0

    inv_sigma2 = 1.0 / (sigma_cm * sigma_cm)
    log_sigma2 = math.log(sigma_cm * sigma_cm)
    residual_scale = 1515.0 * 100.0

    for j in range(CDOG_augments.shape[0]):
        inv_guess = CDOG_reference + CDOG_augments[j]

        if split_esv:
            times_guess, _ = calculateTimesRayTracing_split(
                inv_guess,
                trans_coords,
                esv_bias[j],
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times_guess, _ = calculateTimesRayTracing_Bias_Real(
                inv_guess,
                trans_coords,
                esv_bias[j, 0],
                dz_array,
                angle_array,
                esv_matrix,
            )

        for i in range(pair_counts[j]):
            gps_i = gps_pair_indices[j, i]

            residual = (
                pair_time_base[j, i] + time_bias[j] - times_guess[gps_i]
            ) * residual_scale

            if not math.isnan(residual):
                total_sse += residual * residual
                total_n += 1

    if total_n == 0:
        return -1e30, 0

    return (
        -0.5 * total_sse * inv_sigma2 - 0.5 * total_n * log_sigma2,
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
    """Run a Metropolis-within-Gibbs sampler over the model parameters."""

    if proposal_lever is None:
        proposal_lever = np.array([0.01, 0.01, 0.05])

    if prior_lever is None:
        prior_lever = np.array([0.5, 0.5, 1.0])

    # Always use a two-dimensional ESV array and matching prior center.
    if initial_esv_bias.ndim == 1:
        ebias_center = initial_esv_bias.reshape(
            initial_esv_bias.shape[0],
            1,
        ).copy()
    else:
        ebias_center = initial_esv_bias.copy()

    num_dogs = initial_CDOG_augments.shape[0]
    num_splits = ebias_center.shape[1]

    lever_chain = np.zeros((n_iters, initial_lever_base.shape[0]))
    cdog_aug_chain = np.zeros(
        (
            n_iters,
            initial_CDOG_augments.shape[0],
            initial_CDOG_augments.shape[1],
        )
    )
    ebias_chain = np.zeros((n_iters, num_dogs, num_splits))
    tbias_chain = np.zeros((n_iters, initial_time_bias.shape[0]))
    loglike_chain = np.zeros(n_iters)
    logpost_chain = np.zeros(n_iters)

    lever_curr = initial_lever_base.copy()
    gps_curr = initial_gps_grid.copy()
    cdog_aug_curr = initial_CDOG_augments.copy()
    ebias_curr = ebias_center.copy()
    tbias_curr = initial_time_bias.copy()

    # These only change when the lever or GPS grid changes.
    trans_curr = findTransponder(
        GPS_Coordinates,
        gps_curr,
        lever_curr,
    )

    # ---------------------------------------------------------
    # Determine the fixed CDOG-GPS pairing once
    # ---------------------------------------------------------
    max_pairs = GPS_data.shape[0]

    cdog_pair_indices = np.zeros(
        (num_dogs, max_pairs),
        dtype=np.int64,
    )
    gps_pair_indices = np.zeros(
        (num_dogs, max_pairs),
        dtype=np.int64,
    )
    pair_time_base = np.zeros((num_dogs, max_pairs))
    pair_counts = np.zeros(
        num_dogs,
        dtype=np.int64,
    )

    split_esv = ebias_curr.shape[1] > 1

    for j in range(num_dogs):
        inv_guess = CDOG_reference + cdog_aug_curr[j]

        if split_esv:
            times_guess, _ = calculateTimesRayTracing_split(
                inv_guess,
                trans_curr,
                ebias_curr[j],
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times_guess, _ = calculateTimesRayTracing_Bias_Real(
                inv_guess,
                trans_curr,
                ebias_curr[j, 0],
                dz_array,
                angle_array,
                esv_matrix,
            )

        if tbias_curr[j] == 0.0:
            gps_t = GPS_data
        else:
            gps_t = GPS_data - tbias_curr[j]

        cdog_indices, gps_indices = find_fixed_pair_indices(
            offsets[j],
            0.4,
            CDOG_all_data[j],
            gps_t,
            times_guess,
        )

        pair_counts[j] = cdog_indices.shape[0]

        cdog_pair_indices[j, : pair_counts[j]] = cdog_indices
        gps_pair_indices[j, : pair_counts[j]] = gps_indices

        # Fixed observed portion of:
        #
        # CDOG_full - GPS_full
        # = CDOG_time - GPS_time + time_bias - travel_time
        #
        # The first two terms do not change during sampling.
        for i in range(pair_counts[j]):
            cdog_i = cdog_indices[i]
            gps_i = gps_indices[i]

            pair_time_base[j, i] = (
                CDOG_all_data[j][cdog_i, 0]
                + CDOG_all_data[j][cdog_i, 1]
                - offsets[j]
                - GPS_data[gps_i]
            )

    print("Fixed pairs per DOG:", pair_counts)

    ll_curr, n_curr = compute_log_likelihood(
        trans_curr,
        cdog_aug_curr,
        ebias_curr,
        tbias_curr,
        CDOG_reference,
        gps_pair_indices,
        pair_time_base,
        pair_counts,
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
        ebias_center,
        initial_time_bias,
        prior_lever,
        prior_gps_grid,
        prior_CDOG_aug,
        prior_esv_bias,
        prior_time_bias,
    )

    lpo_curr = ll_curr + lpr_curr

    acc_lever = 0
    acc_aug = 0
    acc_eb = 0
    acc_tb = 0

    lever_active = (
        proposal_lever[0] > 0.0 or proposal_lever[1] > 0.0 or proposal_lever[2] > 0.0
    )

    for it in range(n_iters):
        # -------------------------
        # 1) LEVER block
        # -------------------------
        if lever_active:
            lever_prop = lever_curr.copy()

            for i in range(lever_curr.shape[0]):
                lever_prop[i] += np.random.normal(
                    0.0,
                    proposal_lever[i],
                )

            # Only the lever block requires new transponder coordinates.
            trans_prop = findTransponder(
                GPS_Coordinates,
                gps_curr,
                lever_prop,
            )

            ll_prop, n_prop = compute_log_likelihood(
                trans_prop,
                cdog_aug_curr,
                ebias_curr,
                tbias_curr,
                CDOG_reference,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
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
                ebias_center,
                initial_time_bias,
                prior_lever,
                prior_gps_grid,
                prior_CDOG_aug,
                prior_esv_bias,
                prior_time_bias,
            )

            lpo_prop = ll_prop + lpr_prop
            delta = lpo_prop - lpo_curr

            if delta >= 0.0 or np.log(np.random.rand()) < delta:
                lever_curr = lever_prop
                trans_curr = trans_prop

                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop

                acc_lever += 1

        # -------------------------
        # 2) CDOG AUG block
        # -------------------------
        if proposal_CDOG_aug > 0.0:
            cdog_aug_prop = cdog_aug_curr + np.random.normal(
                0.0,
                proposal_CDOG_aug,
                cdog_aug_curr.shape,
            )

            ll_prop, n_prop = compute_log_likelihood(
                trans_curr,
                cdog_aug_prop,
                ebias_curr,
                tbias_curr,
                CDOG_reference,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
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
                ebias_center,
                initial_time_bias,
                prior_lever,
                prior_gps_grid,
                prior_CDOG_aug,
                prior_esv_bias,
                prior_time_bias,
            )

            lpo_prop = ll_prop + lpr_prop
            delta = lpo_prop - lpo_curr

            if delta >= 0.0 or np.log(np.random.rand()) < delta:
                cdog_aug_curr = cdog_aug_prop

                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop

                acc_aug += 1

        # -------------------------
        # 3) ESV BIAS block
        # -------------------------
        if proposal_esv_bias > 0.0:
            ebias_prop = ebias_curr + np.random.normal(
                0.0,
                proposal_esv_bias,
                ebias_curr.shape,
            )

            ll_prop, n_prop = compute_log_likelihood(
                trans_curr,
                cdog_aug_curr,
                ebias_prop,
                tbias_curr,
                CDOG_reference,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
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
                ebias_center,
                initial_time_bias,
                prior_lever,
                prior_gps_grid,
                prior_CDOG_aug,
                prior_esv_bias,
                prior_time_bias,
            )

            lpo_prop = ll_prop + lpr_prop
            delta = lpo_prop - lpo_curr

            if delta >= 0.0 or np.log(np.random.rand()) < delta:
                ebias_curr = ebias_prop

                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop

                acc_eb += 1

        # -------------------------
        # 4) TIME BIAS block
        # -------------------------
        # Skip the entire likelihood evaluation when fixed.
        if proposal_time_bias > 0.0:
            tbias_prop = tbias_curr + np.random.normal(
                0.0,
                proposal_time_bias,
                tbias_curr.shape,
            )

            ll_prop, n_prop = compute_log_likelihood(
                trans_curr,
                cdog_aug_curr,
                ebias_curr,
                tbias_prop,
                CDOG_reference,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
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
                ebias_center,
                initial_time_bias,
                prior_lever,
                prior_gps_grid,
                prior_CDOG_aug,
                prior_esv_bias,
                prior_time_bias,
            )

            lpo_prop = ll_prop + lpr_prop
            delta = lpo_prop - lpo_curr

            if delta >= 0.0 or np.log(np.random.rand()) < delta:
                tbias_curr = tbias_prop

                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop

                acc_tb += 1

        lever_chain[it] = lever_curr
        cdog_aug_chain[it] = cdog_aug_curr
        ebias_chain[it] = ebias_curr
        tbias_chain[it] = tbias_curr
        loglike_chain[it] = ll_curr
        logpost_chain[it] = lpo_curr

        if it % 100 == 0:
            print(
                "Iter",
                it,
                ": logpost =",
                float(int(lpo_curr * 100) / 100.0),
                "Pairs =",
                n_curr,
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

            acc_lever = 0
            acc_aug = 0
            acc_eb = 0
            acc_tb = 0

    if burn_in <= n_iters:
        lever_chain = lever_chain[burn_in:]
        cdog_aug_chain = cdog_aug_chain[burn_in:]
        ebias_chain = ebias_chain[burn_in:]
        tbias_chain = tbias_chain[burn_in:]
        loglike_chain = loglike_chain[burn_in:]
        logpost_chain = logpost_chain[burn_in:]

    return (
        lever_chain,
        cdog_aug_chain,
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
    )


if __name__ == "__main__":
    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))
    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    downsample = 1

    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    leg1 = (GPS_data / 3600 >= 9.0) & (GPS_data / 3600 <= 11)
    leg2 = (GPS_data / 3600 >= 12.4) & (GPS_data / 3600 <= 15)
    leg_mask = leg1 | leg2

    GPS_Coordinates = GPS_Coordinates[leg_mask]
    GPS_data = GPS_data[leg_mask]

    GPS_Coordinates = GPS_Coordinates[::downsample]
    GPS_data = GPS_data[::downsample]

    CDOG_reference = np.array(
        [
            1976671.618715,
            -5069622.53769779,
            3306330.69611698,
        ]
    )

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

    offsets = np.array(
        [
            1866.016,
            3175.017,
            1939.0178,
        ]
    )

    # Initial parameters
    init_lever = np.array([-13.0, 9.10, -12.75])

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

    values = np.array([-0.4078, -0.2667, -0.1230])

    n = 4
    init_ebias = np.tile(
        values.reshape(-1, 1),
        (1, n),
    )

    init_tbias = np.array([0.0, 0.0, 0.0])

    # Proposal scales
    proposal_lever = np.array([0.0032, 0.0032, 0.008])
    proposal_gps_grid = 0.0
    proposal_CDOG_aug = 0.003
    proposal_esv_bias = 0.0006
    proposal_time_bias = 0.000002

    # Prior scales
    prior_lever = np.array([0.5, 0.5, 0.5])
    prior_gps_grid = 0.1
    prior_CDOG_aug = 0.5
    prior_esv_bias = 1.0
    prior_time_bias = 0.01

    (
        lever_chain,
        cdog_aug_chain,
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
    ) = mcmc_sampler(
        n_iters=5000,
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
        # Posterior chains
        lever=lever_chain,
        CDOG_aug=cdog_aug_chain,
        esv_bias=ebias_chain,
        time_bias=tbias_chain,
        loglike=loglike_chain,
        logpost=logpost_chain,
        # Initial values
        init_lever=init_lever,
        init_gps_grid=init_gps_grid,
        init_CDOG_aug=init_aug,
        init_esv_bias=init_ebias,
        init_time_bias=init_tbias,
        # Priors
        prior_lever=prior_lever,
        prior_gps_grid=prior_gps_grid,
        prior_CDOG_aug=prior_CDOG_aug,
        prior_esv_bias=prior_esv_bias,
        prior_time_bias=prior_time_bias,
        # Proposals
        proposal_lever=proposal_lever,
        proposal_gps_grid=proposal_gps_grid,
        proposal_CDOG_aug=proposal_CDOG_aug,
        proposal_esv_bias=proposal_esv_bias,
        proposal_time_bias=proposal_time_bias,
    )
