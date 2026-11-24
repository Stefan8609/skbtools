"""Joint before/after MCMC search for the Puerto Rico survey."""

import math

import numpy as np
from numba import njit
from numba.typed import List

from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_split,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Inversion.Numba_xAline import pair_indices
from Inversion_Workflow.PuertoRico.Crossover_Mask import crossover_leg_mask
from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from puertorico_data import gps_data_path, gps_output_path


PAIR_THRESHOLD_SECONDS = 0.05
PAIR_OUTLIER_THRESHOLD_SECONDS = 0.002
PAIR_OUTLIER_HALF_WINDOW = 10
PAIR_OUTLIER_MAD_MULTIPLIER = 10.0
PAIR_OUTLIER_SEGMENT_GAP_SECONDS = 60.0


@njit(cache=True)
def filter_pair_outliers(
    offset,
    CDOG_data,
    GPS_data,
    GPS_travel_times,
    cdog_indices,
    gps_indices,
):
    """Remove isolated timing residuals using a local median and MAD."""
    pair_count = cdog_indices.shape[0]
    residuals = np.empty(pair_count)
    keep = np.ones(pair_count, dtype=np.bool_)

    for pair in range(pair_count):
        cdog_index = cdog_indices[pair]
        gps_index = gps_indices[pair]
        residuals[pair] = (
            CDOG_data[cdog_index, 0]
            + CDOG_data[cdog_index, 1]
            - offset
            - GPS_data[gps_index]
            - GPS_travel_times[gps_index]
        )

    segment_start = 0
    while segment_start < pair_count:
        segment_end = segment_start + 1
        while segment_end < pair_count:
            previous_gps = gps_indices[segment_end - 1]
            current_gps = gps_indices[segment_end]
            if (
                GPS_data[current_gps] - GPS_data[previous_gps]
                > PAIR_OUTLIER_SEGMENT_GAP_SECONDS
            ):
                break
            segment_end += 1

        for pair in range(segment_start, segment_end):
            lower = max(segment_start, pair - PAIR_OUTLIER_HALF_WINDOW)
            upper = min(segment_end, pair + PAIR_OUTLIER_HALF_WINDOW + 1)
            local_residuals = residuals[lower:upper]
            local_median = np.median(local_residuals)
            local_mad = np.median(np.abs(local_residuals - local_median))
            local_threshold = max(
                PAIR_OUTLIER_THRESHOLD_SECONDS,
                PAIR_OUTLIER_MAD_MULTIPLIER * 1.4826 * local_mad,
            )
            keep[pair] = abs(residuals[pair] - local_median) <= local_threshold

        segment_start = segment_end

    return cdog_indices[keep], gps_indices[keep]


@njit(cache=True)
def precompute_transponder_transforms(GPS_Coordinates, gps1_to_others):
    """Calculate the rigid-body transforms used by every lever proposal."""
    xs, ys, zs = gps1_to_others.T
    xyzs_init = np.vstack((xs, ys, zs))
    rotations = np.empty((GPS_Coordinates.shape[0], 3, 3))
    displacements = np.empty((GPS_Coordinates.shape[0], 3))

    for epoch in range(GPS_Coordinates.shape[0]):
        new_xs, new_ys, new_zs = GPS_Coordinates[epoch].T
        xyzs_final = np.vstack((new_xs, new_ys, new_zs))
        rotations[epoch], displacements[epoch] = findRotationAndDisplacement(
            xyzs_init,
            xyzs_final,
        )

    return rotations, displacements


@njit(cache=True, fastmath=True)
def calculate_travel_times(
    trans_coords,
    CDOG_augment,
    esv_bias,
    CDOG_reference,
    survey_bounds,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Calculate travel times and ESV values for both surveys."""
    split_esv = esv_bias.shape[1] > 1
    travel_times = np.empty(trans_coords.shape[0])
    esv_values = np.empty(trans_coords.shape[0])

    for survey in range(survey_bounds.shape[0]):
        start = survey_bounds[survey, 0]
        end = survey_bounds[survey, 1]
        survey_trans_coords = trans_coords[start:end]
        inv_guess = CDOG_reference + CDOG_augment

        if split_esv:
            times_guess, survey_esv = calculateTimesRayTracing_split(
                inv_guess,
                survey_trans_coords,
                esv_bias[survey],
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times_guess, survey_esv = calculateTimesRayTracing_Bias_Real(
                inv_guess,
                survey_trans_coords,
                esv_bias[survey, 0],
                dz_array,
                angle_array,
                esv_matrix,
            )

        travel_times[start:end] = times_guess
        esv_values[start:end] = survey_esv

    return travel_times, esv_values


@njit(cache=True, fastmath=True)
def compute_log_likelihood(
    travel_times,
    time_bias,
    survey_bounds,
    gps_pair_indices,
    pair_time_base,
    pair_counts,
    sigma_cm=20.0,
):
    """Compute the joint likelihood using fixed CDOG-GPS pairs."""
    total_sse = 0.0
    total_n = 0
    inv_sigma2 = 1.0 / (sigma_cm * sigma_cm)
    log_sigma2 = math.log(sigma_cm * sigma_cm)
    residual_scale = NOMINAL_SOUND_SPEED_MPS * 100.0

    for survey in range(survey_bounds.shape[0]):
        start = survey_bounds[survey, 0]

        for pair in range(pair_counts[survey]):
            gps_index = gps_pair_indices[survey, pair]
            residual = (
                pair_time_base[survey, pair]
                + time_bias[survey]
                - travel_times[start + gps_index]
            ) * residual_scale

            if not math.isnan(residual):
                total_sse += residual * residual
                total_n += 1

    if total_n == 0:
        return -1e30, 0

    return -0.5 * total_sse * inv_sigma2 - 0.5 * total_n * log_sigma2, total_n


@njit(cache=True, fastmath=True)
def compute_log_prior(
    lever_guess,
    gps1_grid_guess,
    CDOG_augment,
    esv_bias,
    time_bias,
    initial_lever_base,
    initial_gps_grid,
    initial_CDOG_augment,
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
    lp += -0.5 * np.sum(((CDOG_augment - initial_CDOG_augment) / CDOG_aug_scale) ** 2)
    lp += -0.5 * np.sum(((esv_bias - initial_esv_bias) / esv_bias_scale) ** 2)
    lp += -0.5 * np.sum(((time_bias - initial_time_bias) / time_bias_scale) ** 2)
    return lp


@njit(cache=True, fastmath=True)
def mcmc_sampler(
    n_iters,
    burn_in,
    initial_lever_base,
    initial_gps_grid,
    initial_CDOG_augment,
    initial_esv_bias,
    initial_time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    GPS_Coordinates,
    GPS_data,
    survey_bounds,
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
    """Run a joint Metropolis-within-Gibbs sampler."""
    if proposal_lever is None:
        proposal_lever = np.array([0.01, 0.01, 0.05])

    if prior_lever is None:
        prior_lever = np.array([0.5, 0.5, 1.0])

    if initial_esv_bias.ndim == 1:
        ebias_center = initial_esv_bias.reshape(initial_esv_bias.shape[0], 1).copy()
    else:
        ebias_center = initial_esv_bias.copy()

    num_surveys = initial_time_bias.shape[0]
    num_splits = ebias_center.shape[1]

    lever_chain = np.zeros((n_iters, initial_lever_base.shape[0]))
    cdog_aug_chain = np.zeros((n_iters, 3))
    ebias_chain = np.zeros((n_iters, num_surveys, num_splits))
    tbias_chain = np.zeros((n_iters, num_surveys))
    loglike_chain = np.zeros(n_iters)
    logpost_chain = np.zeros(n_iters)

    lever_curr = initial_lever_base.copy()
    gps_curr = initial_gps_grid.copy()
    cdog_aug_curr = initial_CDOG_augment.copy()
    ebias_curr = ebias_center.copy()
    tbias_curr = initial_time_bias.copy()

    rotations, displacements = precompute_transponder_transforms(
        GPS_Coordinates,
        gps_curr,
    )
    trans_curr = np.empty((GPS_Coordinates.shape[0], 3))
    for epoch in range(GPS_Coordinates.shape[0]):
        trans_curr[epoch] = rotations[epoch] @ lever_curr + displacements[epoch]

    travel_times_curr, esv_values_curr = calculate_travel_times(
        trans_curr,
        cdog_aug_curr,
        ebias_curr,
        CDOG_reference,
        survey_bounds,
        dz_array,
        angle_array,
        esv_matrix,
    )

    max_pairs = GPS_data.shape[0]
    gps_pair_indices = np.zeros((num_surveys, max_pairs), dtype=np.int64)
    pair_time_base = np.zeros((num_surveys, max_pairs))
    pair_counts = np.zeros(num_surveys, dtype=np.int64)
    excluded_pair_counts = np.zeros(num_surveys, dtype=np.int64)

    for survey in range(num_surveys):
        start = survey_bounds[survey, 0]
        end = survey_bounds[survey, 1]
        survey_GPS_data = GPS_data[start:end]
        survey_travel_times = travel_times_curr[start:end]

        cdog_indices, gps_indices = pair_indices(
            offsets[survey],
            PAIR_THRESHOLD_SECONDS,
            CDOG_all_data[survey],
            survey_GPS_data - tbias_curr[survey],
            survey_travel_times,
        )
        unfiltered_pair_count = cdog_indices.shape[0]
        cdog_indices, gps_indices = filter_pair_outliers(
            offsets[survey],
            CDOG_all_data[survey],
            survey_GPS_data - tbias_curr[survey],
            survey_travel_times,
            cdog_indices,
            gps_indices,
        )
        excluded_pair_counts[survey] = unfiltered_pair_count - cdog_indices.shape[0]

        pair_counts[survey] = cdog_indices.shape[0]
        gps_pair_indices[survey, : pair_counts[survey]] = gps_indices

        for pair in range(pair_counts[survey]):
            cdog_index = cdog_indices[pair]
            gps_index = gps_indices[pair]
            pair_time_base[survey, pair] = (
                CDOG_all_data[survey][cdog_index, 0]
                + CDOG_all_data[survey][cdog_index, 1]
                - offsets[survey]
                - survey_GPS_data[gps_index]
            )

    print("Fixed pairs per survey:", pair_counts)
    print("Excluded outlier pairs per survey:", excluded_pair_counts)

    ll_curr, n_curr = compute_log_likelihood(
        travel_times_curr,
        tbias_curr,
        survey_bounds,
        gps_pair_indices,
        pair_time_base,
        pair_counts,
    )

    lpr_curr = compute_log_prior(
        lever_curr,
        gps_curr,
        cdog_aug_curr,
        ebias_curr,
        tbias_curr,
        initial_lever_base,
        initial_gps_grid,
        initial_CDOG_augment,
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

    for iteration in range(n_iters):
        if lever_active:
            lever_prop = lever_curr.copy()

            for component in range(lever_curr.shape[0]):
                lever_prop[component] += np.random.normal(
                    0.0,
                    proposal_lever[component],
                )

            trans_prop = np.empty_like(trans_curr)
            for epoch in range(GPS_Coordinates.shape[0]):
                trans_prop[epoch] = rotations[epoch] @ lever_prop + displacements[epoch]

            travel_times_prop, esv_values_prop = calculate_travel_times(
                trans_prop,
                cdog_aug_curr,
                ebias_curr,
                CDOG_reference,
                survey_bounds,
                dz_array,
                angle_array,
                esv_matrix,
            )

            ll_prop, n_prop = compute_log_likelihood(
                travel_times_prop,
                tbias_curr,
                survey_bounds,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
            )

            lpr_prop = compute_log_prior(
                lever_prop,
                gps_curr,
                cdog_aug_curr,
                ebias_curr,
                tbias_curr,
                initial_lever_base,
                initial_gps_grid,
                initial_CDOG_augment,
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
                travel_times_curr = travel_times_prop
                esv_values_curr = esv_values_prop
                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop
                acc_lever += 1

        if proposal_CDOG_aug > 0.0:
            cdog_aug_prop = cdog_aug_curr + np.random.normal(
                0.0,
                proposal_CDOG_aug,
                3,
            )

            travel_times_prop, esv_values_prop = calculate_travel_times(
                trans_curr,
                cdog_aug_prop,
                ebias_curr,
                CDOG_reference,
                survey_bounds,
                dz_array,
                angle_array,
                esv_matrix,
            )

            ll_prop, n_prop = compute_log_likelihood(
                travel_times_prop,
                tbias_curr,
                survey_bounds,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
            )

            lpr_prop = compute_log_prior(
                lever_curr,
                gps_curr,
                cdog_aug_prop,
                ebias_curr,
                tbias_curr,
                initial_lever_base,
                initial_gps_grid,
                initial_CDOG_augment,
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
                travel_times_curr = travel_times_prop
                esv_values_curr = esv_values_prop
                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop
                acc_aug += 1

        if proposal_esv_bias > 0.0:
            ebias_prop = ebias_curr + np.random.normal(
                0.0,
                proposal_esv_bias,
                ebias_curr.shape,
            )

            travel_times_prop = travel_times_curr.copy()
            esv_values_prop = esv_values_curr.copy()

            for survey in range(num_surveys):
                start = survey_bounds[survey, 0]
                end = survey_bounds[survey, 1]
                survey_size = end - start
                base = survey_size // num_splits
                remainder = survey_size % num_splits
                block_start = start

                for split in range(num_splits):
                    block_size = base + (1 if split < remainder else 0)
                    block_end = block_start + block_size
                    esv_change = ebias_prop[survey, split] - ebias_curr[survey, split]

                    for epoch in range(block_start, block_end):
                        esv_values_prop[epoch] += esv_change
                        # Range is unchanged: range = travel time * ESV.
                        travel_times_prop[epoch] = (
                            travel_times_curr[epoch]
                            * esv_values_curr[epoch]
                            / esv_values_prop[epoch]
                        )

                    block_start = block_end

            ll_prop, n_prop = compute_log_likelihood(
                travel_times_prop,
                tbias_curr,
                survey_bounds,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
            )

            lpr_prop = compute_log_prior(
                lever_curr,
                gps_curr,
                cdog_aug_curr,
                ebias_prop,
                tbias_curr,
                initial_lever_base,
                initial_gps_grid,
                initial_CDOG_augment,
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
                travel_times_curr = travel_times_prop
                esv_values_curr = esv_values_prop
                ll_curr = ll_prop
                lpr_curr = lpr_prop
                lpo_curr = lpo_prop
                n_curr = n_prop
                acc_eb += 1

        if proposal_time_bias > 0.0:
            tbias_prop = tbias_curr + np.random.normal(
                0.0,
                proposal_time_bias,
                tbias_curr.shape,
            )

            ll_prop, n_prop = compute_log_likelihood(
                travel_times_curr,
                tbias_prop,
                survey_bounds,
                gps_pair_indices,
                pair_time_base,
                pair_counts,
            )

            lpr_prop = compute_log_prior(
                lever_curr,
                gps_curr,
                cdog_aug_curr,
                ebias_curr,
                tbias_prop,
                initial_lever_base,
                initial_gps_grid,
                initial_CDOG_augment,
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

        lever_chain[iteration] = lever_curr
        cdog_aug_chain[iteration] = cdog_aug_curr
        ebias_chain[iteration] = ebias_curr
        tbias_chain[iteration] = tbias_curr
        loglike_chain[iteration] = ll_curr
        logpost_chain[iteration] = lpo_curr

        if iteration % 100 == 0:
            print(
                "Iter",
                iteration,
                ": logpost =",
                float(int(lpo_curr * 100) / 100.0),
                "Pairs =",
                n_curr,
            )

        if iteration % 1000 == 0 and iteration > 0:
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
    trajectory_mode = "cross"  # "cross" or "full"
    n_esv_splits = 4
    downsample = 1
    n_iters = 100000
    burn_in = 1000

    dz_array, angle_array, esv_matrix = load_esv_table(
        gps_data_path("ESV_Tables/PuertoRico_table_transducer_60_DOG_2950_3050.mat")
    )

    survey_names = ("before", "after")
    GPS_coordinate_blocks = []
    GPS_time_blocks = []
    CDOG_all_data = List()
    survey_bounds = np.zeros((2, 2), dtype=np.int64)
    end = 0

    for survey, survey_name in enumerate(survey_names):
        data = np.load(
            gps_data_path(
                f"GPS_Data/Processed_GPS_Receivers_DOG_1_{survey_name}_clock_loss.npz"
            )
        )
        GPS_Coordinates = data["GPS_Coordinates"]
        GPS_data = data["GPS_data"]
        GPS_time_bounds = data["GPS_time_bounds"]
        crossover_leg_GPS_bounds = data["crossover_leg_GPS_bounds"]
        CDOG_data = data["CDOG_data"]

        if survey == 0:
            CDOG_reference = data["CDOG_guess"]
            gps1_to_others = data["gps1_to_others"]

        time_mask = (GPS_data >= GPS_time_bounds[0]) & (GPS_data <= GPS_time_bounds[1])

        trajectory_mask = time_mask
        if trajectory_mode == "cross":
            trajectory_mask &= crossover_leg_mask(
                GPS_data,
                crossover_leg_GPS_bounds,
            )

        GPS_Coordinates = GPS_Coordinates[trajectory_mask][::downsample]
        GPS_data = GPS_data[trajectory_mask][::downsample]

        start = end
        end += len(GPS_data)
        survey_bounds[survey] = (start, end)
        GPS_coordinate_blocks.append(GPS_Coordinates)
        GPS_time_blocks.append(GPS_data)
        CDOG_all_data.append(CDOG_data)

    GPS_Coordinates = np.concatenate(GPS_coordinate_blocks)
    GPS_data = np.concatenate(GPS_time_blocks)
    offsets = np.array([-19.32504165, -7458.62104948])

    init_lever = np.array([-0.948182, 0.633095, 0.3])
    init_aug = np.zeros(3)
    initial_esv_values = np.array([-0.12007630, -0.05052049])
    init_ebias = np.tile(initial_esv_values[:, None], (1, n_esv_splits))
    init_tbias = np.array([0.0, 0.0])

    proposal_lever = np.array([0.0032, 0.0032, 0.008])
    proposal_gps_grid = 0.0
    proposal_CDOG_aug = 0.005
    proposal_esv_bias = 0.001
    proposal_time_bias = 0.000002

    prior_lever = np.array([0.5, 0.5, 1.0])
    prior_gps_grid = 0.1
    prior_CDOG_aug = 2.0
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
        n_iters=n_iters,
        burn_in=burn_in,
        initial_lever_base=init_lever,
        initial_gps_grid=gps1_to_others,
        initial_CDOG_augment=init_aug,
        initial_esv_bias=init_ebias,
        initial_time_bias=init_tbias,
        dz_array=dz_array,
        angle_array=angle_array,
        esv_matrix=esv_matrix,
        GPS_Coordinates=GPS_Coordinates,
        GPS_data=GPS_data,
        survey_bounds=survey_bounds,
        CDOG_reference=CDOG_reference,
        CDOG_all_data=CDOG_all_data,
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
        gps_output_path(f"mcmc_chain_puertorico_{trajectory_mode}.npz"),
        lever=lever_chain,
        CDOG_aug=cdog_aug_chain[:, None, :],
        esv_bias=ebias_chain,
        time_bias=tbias_chain,
        loglike=loglike_chain,
        logpost=logpost_chain,
        init_lever=init_lever,
        init_gps_grid=gps1_to_others,
        init_CDOG_aug=init_aug[None, :],
        init_esv_bias=init_ebias,
        init_time_bias=init_tbias,
        prior_lever=prior_lever,
        prior_gps_grid=prior_gps_grid,
        prior_CDOG_aug=prior_CDOG_aug,
        prior_esv_bias=prior_esv_bias,
        prior_time_bias=prior_time_bias,
        proposal_lever=proposal_lever,
        proposal_gps_grid=proposal_gps_grid,
        proposal_CDOG_aug=proposal_CDOG_aug,
        proposal_esv_bias=proposal_esv_bias,
        proposal_time_bias=proposal_time_bias,
        pair_outlier_threshold_seconds=PAIR_OUTLIER_THRESHOLD_SECONDS,
        pair_outlier_half_window=PAIR_OUTLIER_HALF_WINDOW,
        pair_outlier_mad_multiplier=PAIR_OUTLIER_MAD_MULTIPLIER,
        pair_outlier_segment_gap_seconds=PAIR_OUTLIER_SEGMENT_GAP_SECONDS,
    )
