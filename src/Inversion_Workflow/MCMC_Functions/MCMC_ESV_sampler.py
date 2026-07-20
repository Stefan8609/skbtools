"""MCMC sampler for four fixed-geometry ESV splits and timing bias."""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numba import njit
from numba.typed import List

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import find_esv
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from data import gps_data_path, gps_output_path
from geometry.ECEF_Geodetic import ECEF_Geodetic


def find_x_split_edges(
    transponder_coordinates,
    GPS_data,
    DOG1_coordinates,
):
    """Find four continuous X-arm blocks from two time-separated crossings."""

    # The largest time gap separates the two traversals across the X.
    gap = np.argmax(np.diff(GPS_data)) + 1

    if gap < 3 or transponder_coordinates.shape[0] - gap < 3:
        raise ValueError("Could not separate the trajectory into two X crossings.")

    # Project the local trajectory onto its best-fitting horizontal plane.
    track_center = np.mean(transponder_coordinates, axis=0)
    centered_track = transponder_coordinates - track_center

    _, _, vh = np.linalg.svd(
        centered_track,
        full_matrices=False,
    )

    plane = vh[:2]

    # Keep DOG1 as the origin in the projected coordinate system.
    xy = (transponder_coordinates - DOG1_coordinates) @ plane.T

    # Fit a line through the first crossing.
    center1 = np.mean(xy[:gap], axis=0)

    _, _, vh1 = np.linalg.svd(
        xy[:gap] - center1,
        full_matrices=False,
    )

    direction1 = vh1[0]

    # Fit a line through the second crossing.
    center2 = np.mean(xy[gap:], axis=0)

    _, _, vh2 = np.linalg.svd(
        xy[gap:] - center2,
        full_matrices=False,
    )

    direction2 = vh2[0]

    # Find the intersection of the two fitted lines.
    matrix = np.column_stack(
        (
            direction1,
            -direction2,
        )
    )

    if abs(np.linalg.det(matrix)) > 1e-10:
        parameters = np.linalg.solve(
            matrix,
            center2 - center1,
        )

        x_center = center1 + parameters[0] * direction1

    else:
        # Fall back to DOG1 if the two fitted lines are nearly parallel.
        x_center = np.zeros(2)

    # Find the closest sampled point to the X center on each traversal.
    cut1 = np.argmin(
        np.sum(
            (xy[:gap] - x_center) ** 2,
            axis=1,
        )
    )

    cut2 = gap + np.argmin(
        np.sum(
            (xy[gap:] - x_center) ** 2,
            axis=1,
        )
    )

    # Ensure all four splits contain points.
    cut1 = np.clip(
        cut1,
        1,
        gap - 2,
    )

    cut2 = np.clip(
        cut2,
        gap + 1,
        transponder_coordinates.shape[0] - 2,
    )

    split_edges = np.array(
        [
            0,
            cut1 + 1,
            gap,
            cut2 + 1,
            transponder_coordinates.shape[0],
        ],
        dtype=np.int64,
    )

    x_center_ecef = DOG1_coordinates + x_center[0] * plane[0] + x_center[1] * plane[1]

    return split_edges, x_center_ecef


@njit(cache=True)
def find_fixed_pair_indices(
    offset,
    threshold,
    CDOG_data,
    GPS_data,
    GPS_travel_times,
):
    """Find the CDOG and GPS row indices once using the initial model."""

    max_pairs = min(
        CDOG_data.shape[0],
        GPS_data.shape[0],
    )

    cdog_indices = np.empty(
        max_pairs,
        dtype=np.int64,
    )

    gps_indices = np.empty(
        max_pairs,
        dtype=np.int64,
    )

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

    return (
        cdog_indices[:count],
        gps_indices[:count],
    )


@njit(cache=True, fastmath=True)
def precompute_fixed_geometry(
    transponder_coordinates,
    CDOG_coordinates,
    split_edges,
    dz_array,
    angle_array,
    esv_matrix,
):
    """Precompute fixed ranges, table ESVs, and X-arm membership."""

    num_dogs = CDOG_coordinates.shape[0]
    num_points = transponder_coordinates.shape[0]
    num_splits = split_edges.shape[0] - 1

    distances = np.zeros(
        (
            num_dogs,
            num_points,
        )
    )

    base_esv = np.zeros(
        (
            num_dogs,
            num_points,
        )
    )

    split_ids = np.zeros(
        num_points,
        dtype=np.int64,
    )

    for split in range(num_splits):
        for i in range(
            split_edges[split],
            split_edges[split + 1],
        ):
            split_ids[i] = split

    transponder_depth = ECEF_Geodetic(transponder_coordinates)[2]

    for dog in range(num_dogs):
        guess = CDOG_coordinates[dog]
        dog_depth = ECEF_Geodetic(guess[np.newaxis, :])[2][0]

        difference = transponder_coordinates - guess

        distances[dog] = np.sqrt(
            np.sum(
                difference * difference,
                axis=1,
            )
        )

        dz = transponder_depth - dog_depth
        beta = np.arcsin(dz / distances[dog]) * 180.0 / np.pi

        base_esv[dog] = find_esv(
            beta,
            dz,
            dz_array,
            angle_array,
            esv_matrix,
        )

    return (
        distances,
        base_esv,
        split_ids,
    )


@njit(cache=True)
def compute_group_log_likelihood(
    esv_bias,
    time_bias,
    pair_time_base,
    pair_distance,
    pair_esv,
    pair_count,
    sigma_cm,
):
    """Compute one DOG/split likelihood from fixed geometry and pairs."""

    if pair_count == 0:
        return 0.0

    total_sse = 0.0
    residual_scale = 1515.0 * 100.0

    for i in range(pair_count):
        predicted_esv = pair_esv[i] + esv_bias

        if predicted_esv <= 0.0 or not math.isfinite(predicted_esv):
            return -1e30

        travel_time = pair_distance[i] / predicted_esv

        residual = (pair_time_base[i] + time_bias - travel_time) * residual_scale

        if not math.isfinite(residual):
            return -1e30

        total_sse += residual * residual

    sigma2 = sigma_cm * sigma_cm

    return -0.5 * total_sse / sigma2 - 0.5 * pair_count * math.log(sigma2)


@njit(cache=True)
def mcmc_sampler(
    n_iters,
    burn_in,
    initial_esv_bias,
    initial_time_bias,
    transponder_coordinates,
    GPS_data,
    CDOG_coordinates,
    CDOG_all_data,
    offsets,
    split_edges,
    dz_array,
    angle_array,
    esv_matrix,
    proposal_esv_bias=0.0006,
    proposal_time_bias=0.000002,
    prior_esv_bias=1.0,
    prior_time_bias=0.01,
    sigma_cm=20.0,
    pair_threshold=0.4,
):
    """Sample four ESV biases and one timing bias for each selected DOG."""

    num_dogs = initial_esv_bias.shape[0]
    num_splits = initial_esv_bias.shape[1]
    num_points = GPS_data.shape[0]

    if num_splits != 4 or split_edges.shape[0] != 5:
        raise ValueError("This sampler requires exactly four ESV splits.")

    if initial_time_bias.shape[0] != num_dogs:
        raise ValueError("Initial timing bias must have one value per DOG.")

    if burn_in < 0 or burn_in >= n_iters:
        raise ValueError("burn_in must be smaller than n_iters.")

    # Force floating-point states even when integer initial values are supplied.
    ebias_center = initial_esv_bias.astype(np.float64)
    ebias_curr = ebias_center.copy()

    tbias_center = initial_time_bias.astype(np.float64)
    tbias_curr = tbias_center.copy()

    # ---------------------------------------------------------
    # Calculate all fixed geometry and base ESV values once
    # ---------------------------------------------------------
    (
        distances,
        base_esv,
        split_ids,
    ) = precompute_fixed_geometry(
        transponder_coordinates,
        CDOG_coordinates,
        split_edges,
        dz_array,
        angle_array,
        esv_matrix,
    )

    pair_time_base = np.zeros(
        (
            num_dogs,
            num_splits,
            num_points,
        )
    )

    pair_distance = np.zeros(
        (
            num_dogs,
            num_splits,
            num_points,
        )
    )

    pair_esv = np.zeros(
        (
            num_dogs,
            num_splits,
            num_points,
        )
    )

    pair_counts = np.zeros(
        (
            num_dogs,
            num_splits,
        ),
        dtype=np.int64,
    )

    initial_travel_times = np.zeros(num_points)

    # ---------------------------------------------------------
    # Determine fixed CDOG-GPS pairs once using initial values
    # ---------------------------------------------------------
    for dog in range(num_dogs):
        for i in range(num_points):
            split = split_ids[i]

            initial_travel_times[i] = distances[dog, i] / (
                base_esv[dog, i] + ebias_curr[dog, split]
            )

        gps_pair_time = GPS_data - tbias_curr[dog]

        (
            cdog_indices,
            gps_indices,
        ) = find_fixed_pair_indices(
            offsets[dog],
            pair_threshold,
            CDOG_all_data[dog],
            gps_pair_time,
            initial_travel_times,
        )

        for i in range(cdog_indices.shape[0]):
            cdog_i = cdog_indices[i]
            gps_i = gps_indices[i]
            split = split_ids[gps_i]
            pair_i = pair_counts[dog, split]

            # Timing bias is added later in the likelihood.
            pair_time_base[dog, split, pair_i] = (
                CDOG_all_data[dog][cdog_i, 0]
                + CDOG_all_data[dog][cdog_i, 1]
                - offsets[dog]
                - GPS_data[gps_i]
            )

            pair_distance[dog, split, pair_i] = distances[dog, gps_i]
            pair_esv[dog, split, pair_i] = base_esv[dog, gps_i]
            pair_counts[dog, split] += 1

    print("Fixed pairs by DOG row and X split:")
    print(pair_counts)

    for dog in range(num_dogs):
        for split in range(num_splits):
            if pair_counts[dog, split] == 0:
                raise ValueError("A selected DOG/split group contains no fixed pairs.")

    # ---------------------------------------------------------
    # Initial likelihood and priors
    # ---------------------------------------------------------
    group_loglike = np.zeros(
        (
            num_dogs,
            num_splits,
        )
    )

    esv_logprior = np.zeros(
        (
            num_dogs,
            num_splits,
        )
    )

    time_logprior = np.zeros(num_dogs)

    for dog in range(num_dogs):
        for split in range(num_splits):
            group_loglike[dog, split] = compute_group_log_likelihood(
                ebias_curr[dog, split],
                tbias_curr[dog],
                pair_time_base[dog, split],
                pair_distance[dog, split],
                pair_esv[dog, split],
                pair_counts[dog, split],
                sigma_cm,
            )

            esv_logprior[dog, split] = (
                -0.5
                * ((ebias_curr[dog, split] - ebias_center[dog, split]) / prior_esv_bias)
                ** 2
            )

        time_logprior[dog] = (
            -0.5 * ((tbias_curr[dog] - tbias_center[dog]) / prior_time_bias) ** 2
        )

    ll_curr = np.sum(group_loglike)
    lpr_curr = np.sum(esv_logprior) + np.sum(time_logprior)
    lpo_curr = ll_curr + lpr_curr

    ebias_chain = np.zeros(
        (
            n_iters,
            num_dogs,
            num_splits,
        )
    )

    tbias_chain = np.zeros(
        (
            n_iters,
            num_dogs,
        )
    )

    loglike_chain = np.zeros(n_iters)
    logpost_chain = np.zeros(n_iters)

    acceptance_esv = np.zeros(
        (
            num_dogs,
            num_splits,
        ),
        dtype=np.int64,
    )

    acceptance_time = np.zeros(
        num_dogs,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # MCMC
    # ---------------------------------------------------------
    for it in range(n_iters):
        for dog in range(num_dogs):
            # ---------------------------------------------
            # Four independent ESV-bias updates
            # ---------------------------------------------
            for split in range(num_splits):
                ebias_prop = ebias_curr[dog, split] + np.random.normal(
                    0.0,
                    proposal_esv_bias,
                )

                ll_prop = compute_group_log_likelihood(
                    ebias_prop,
                    tbias_curr[dog],
                    pair_time_base[dog, split],
                    pair_distance[dog, split],
                    pair_esv[dog, split],
                    pair_counts[dog, split],
                    sigma_cm,
                )

                lpr_prop = (
                    -0.5
                    * ((ebias_prop - ebias_center[dog, split]) / prior_esv_bias) ** 2
                )

                delta = (
                    ll_prop
                    + lpr_prop
                    - group_loglike[dog, split]
                    - esv_logprior[dog, split]
                )

                if delta >= 0.0 or np.log(np.random.rand()) < delta:
                    ll_curr += ll_prop - group_loglike[dog, split]
                    lpr_curr += lpr_prop - esv_logprior[dog, split]

                    ebias_curr[dog, split] = ebias_prop
                    group_loglike[dog, split] = ll_prop
                    esv_logprior[dog, split] = lpr_prop

                    acceptance_esv[dog, split] += 1

            # ---------------------------------------------
            # One timing-bias update for this DOG
            # ---------------------------------------------
            if proposal_time_bias > 0.0:
                tbias_prop = tbias_curr[dog] + np.random.normal(
                    0.0,
                    proposal_time_bias,
                )

                dog_loglike_prop = np.zeros(num_splits)

                for split in range(num_splits):
                    dog_loglike_prop[split] = compute_group_log_likelihood(
                        ebias_curr[dog, split],
                        tbias_prop,
                        pair_time_base[dog, split],
                        pair_distance[dog, split],
                        pair_esv[dog, split],
                        pair_counts[dog, split],
                        sigma_cm,
                    )

                dog_ll_curr = np.sum(group_loglike[dog])
                dog_ll_prop = np.sum(dog_loglike_prop)

                lpr_prop = (
                    -0.5 * ((tbias_prop - tbias_center[dog]) / prior_time_bias) ** 2
                )

                delta = dog_ll_prop + lpr_prop - dog_ll_curr - time_logprior[dog]

                if delta >= 0.0 or np.log(np.random.rand()) < delta:
                    ll_curr += dog_ll_prop - dog_ll_curr
                    lpr_curr += lpr_prop - time_logprior[dog]

                    tbias_curr[dog] = tbias_prop
                    group_loglike[dog] = dog_loglike_prop
                    time_logprior[dog] = lpr_prop

                    acceptance_time[dog] += 1

        lpo_curr = ll_curr + lpr_curr

        ebias_chain[it] = ebias_curr
        tbias_chain[it] = tbias_curr
        loglike_chain[it] = ll_curr
        logpost_chain[it] = lpo_curr

        if (it + 1) % 100 == 0:
            print(
                "Iter",
                it + 1,
                ": logpost =",
                float(int(lpo_curr * 100) / 100.0),
                "Pairs =",
                np.sum(pair_counts),
            )

        if (it + 1) % 1000 == 0:
            print("ESV acc (last 1000), rows are DOGs and columns are splits 1-4:")
            print(acceptance_esv / 1000.0)
            print("Timing acc (last 1000), one value per DOG:")
            print(acceptance_time / 1000.0)

            acceptance_esv[:, :] = 0
            acceptance_time[:] = 0

    ebias_chain = ebias_chain[burn_in:]
    tbias_chain = tbias_chain[burn_in:]
    loglike_chain = loglike_chain[burn_in:]
    logpost_chain = logpost_chain[burn_in:]

    return (
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
        pair_counts,
    )


if __name__ == "__main__":
    # Change this to any subset of [1, 3, 4].
    # Examples: [1], [3, 4], or [1, 3, 4].
    SEARCH_DOGS = np.array([1, 3, 4])

    CHAIN_FILE = gps_output_path("mcmc_esv_time_chain.npz")
    DOWNSAMPLE = 1
    SAVE = False

    esv = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"))

    dz_array = esv["distance"].flatten()
    angle_array = esv["angle"].flatten()
    esv_matrix = esv["matrice"]

    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))

    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    leg1 = (GPS_data / 3600 >= 9.0) & (GPS_data / 3600 <= 11.0)
    leg2 = (GPS_data / 3600 >= 12.4) & (GPS_data / 3600 <= 15.0)
    leg_mask = leg1 | leg2

    GPS_Coordinates = GPS_Coordinates[leg_mask][::DOWNSAMPLE]
    GPS_data = GPS_data[leg_mask][::DOWNSAMPLE]

    CDOG_reference = np.array(
        [
            1976671.618715,
            -5069622.53769779,
            3306330.69611698,
        ]
    )

    ALL_DOGS = np.array([1, 3, 4])

    if SEARCH_DOGS.size == 0:
        raise ValueError("SEARCH_DOGS must contain at least one DOG.")

    if np.unique(SEARCH_DOGS).shape[0] != SEARCH_DOGS.shape[0]:
        raise ValueError("SEARCH_DOGS cannot contain duplicate DOG numbers.")

    if not np.all(np.isin(SEARCH_DOGS, ALL_DOGS)):
        raise ValueError("SEARCH_DOGS can only contain DOG 1, 3, or 4.")

    selected = np.array(
        [np.where(ALL_DOGS == dog)[0][0] for dog in SEARCH_DOGS],
        dtype=int,
    )

    offsets_all = np.array(
        [
            1866.016,
            3175.017,
            1939.0178,
        ]
    )

    fixed_lever = np.array(
        [
            -13.16771776,
            9.08917827,
            -12.94647562,
        ]
    )

    fixed_gps_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.393414, -4.223503, 0.029415],
            [-12.095685, -0.945685, 0.004397],
            [-8.686741, 5.169188, -0.024993],
        ]
    )

    fixed_CDOG_aug_all = np.array(
        [
            [-396.8083375, 370.23261256, 774.14404697],
            [826.18512886, -111.85558415, -733.73615228],
            [235.77476616, -1305.38300727, -2190.89445398],
        ]
    )

    all_CDOG_coordinates = CDOG_reference + fixed_CDOG_aug_all

    # These coordinates remain fixed for the complete sampler.
    transponder_coordinates = findTransponder(
        GPS_Coordinates,
        fixed_gps_grid,
        fixed_lever,
    )

    # Always use the fixed DOG1 location to define the X center,
    # even when DOG1 is not one of the sampled DOGs.
    (
        split_edges,
        x_center,
    ) = find_x_split_edges(
        transponder_coordinates,
        GPS_data,
        all_CDOG_coordinates[0],
    )

    print("Searched DOGs:", SEARCH_DOGS)
    print("Optimized X split edges:", split_edges)
    print(
        "Estimated X-center distance from fixed DOG1 (m):",
        np.linalg.norm(x_center - all_CDOG_coordinates[0]),
    )

    for split in range(4):
        start = split_edges[split]
        end = split_edges[split + 1]

        print(
            "Split",
            split + 1,
            ": indices",
            start,
            "to",
            end - 1,
            ", hours",
            GPS_data[start] / 3600.0,
            "to",
            GPS_data[end - 1] / 3600.0,
        )

    DOG_order = SEARCH_DOGS.copy()
    offsets = offsets_all[selected]
    fixed_CDOG_aug = fixed_CDOG_aug_all[selected]
    CDOG_coordinates = all_CDOG_coordinates[selected]

    CDOG_all_data = []

    for dog in DOG_order:
        tmp = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{dog}-camp.mat"))["tags"].astype(
            float
        )

        tmp[:, 1] /= 1e9
        CDOG_all_data.append(tmp)

    typed_CDOG_all_data = List()

    for arr in CDOG_all_data:
        typed_CDOG_all_data.append(arr)

    # ---------------------------------------------------------
    # Initial values, proposals, and priors
    # ---------------------------------------------------------
    initial_esv_values_all = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    initial_esv_bias = np.tile(
        initial_esv_values_all[selected].reshape(-1, 1),
        (1, 4),
    )

    initial_time_bias_all = np.array(
        [
            -0.00057115,
            0.00049583,
            0.00152436,
        ],
        dtype=np.float64,
    )

    initial_time_bias = initial_time_bias_all[selected]

    proposal_esv_bias = 0.01
    proposal_time_bias = 0.001

    prior_esv_bias = 5.0
    prior_time_bias = 1.0

    (
        ebias_chain,
        tbias_chain,
        loglike_chain,
        logpost_chain,
        pair_counts,
    ) = mcmc_sampler(
        n_iters=1000000,
        burn_in=100000,
        initial_esv_bias=initial_esv_bias,
        initial_time_bias=initial_time_bias,
        transponder_coordinates=transponder_coordinates,
        GPS_data=GPS_data,
        CDOG_coordinates=CDOG_coordinates,
        CDOG_all_data=typed_CDOG_all_data,
        offsets=offsets,
        split_edges=split_edges,
        dz_array=dz_array,
        angle_array=angle_array,
        esv_matrix=esv_matrix,
        proposal_esv_bias=proposal_esv_bias,
        proposal_time_bias=proposal_time_bias,
        prior_esv_bias=prior_esv_bias,
        prior_time_bias=prior_time_bias,
        sigma_cm=20.0,
        pair_threshold=0.4,
    )

    np.savez(
        CHAIN_FILE,
        # Posterior chains
        esv_bias=ebias_chain,
        time_bias=tbias_chain,
        loglike=loglike_chain,
        logpost=logpost_chain,
        # Initial values
        initial_esv_bias=initial_esv_bias,
        initial_time_bias=initial_time_bias,
        # Proposals and priors
        proposal_esv_bias=proposal_esv_bias,
        proposal_time_bias=proposal_time_bias,
        prior_esv_bias=prior_esv_bias,
        prior_time_bias=prior_time_bias,
        # X splitting information
        split_edges=split_edges,
        split_time_hours=GPS_data[split_edges[:-1]] / 3600.0,
        x_center=x_center,
        # Fixed pairing information
        pair_counts=pair_counts,
        DOG_order=DOG_order,
        # Fixed model values
        fixed_lever=fixed_lever,
        fixed_gps_grid=fixed_gps_grid,
        fixed_CDOG_aug=fixed_CDOG_aug,
        CDOG_coordinates=CDOG_coordinates,
        transponder_coordinates=transponder_coordinates,
        GPS_data=GPS_data,
        offsets=offsets,
        downsample=DOWNSAMPLE,
    )

    # ---------------------------------------------------------
    # Load results for plotting
    # ---------------------------------------------------------
    chain = np.load(CHAIN_FILE)

    esv_bias_chain = chain["esv_bias"]
    time_bias_chain = chain["time_bias"]
    initial_esv_bias = chain["initial_esv_bias"]
    initial_time_bias = chain["initial_time_bias"]

    split_edges = chain["split_edges"].astype(int)
    x_center = chain["x_center"]
    DOG_order = chain["DOG_order"].astype(int)

    CDOG_coordinates = chain["CDOG_coordinates"]
    transponder_coordinates = chain["transponder_coordinates"]

    trans_lat, trans_lon, trans_height = ECEF_Geodetic(transponder_coordinates)

    CDOG_lat, CDOG_lon, CDOG_height = ECEF_Geodetic(CDOG_coordinates)

    center_lat, center_lon, center_height = ECEF_Geodetic(x_center[np.newaxis, :])

    split_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    dog_colors = {
        1: "red",
        3: "green",
        4: "blue",
    }

    # ---------------------------------------------------------
    # 1. Trajectory and selected DOG locations
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    for split in range(4):
        start = split_edges[split]
        end = split_edges[split + 1]

        ax.plot(
            trans_lon[start:end],
            trans_lat[start:end],
            ".",
            markersize=3,
            color=split_colors[split],
            label=f"Split {split + 1}",
        )

    for i, dog in enumerate(DOG_order):
        ax.scatter(
            CDOG_lon[i],
            CDOG_lat[i],
            marker="x",
            s=90,
            linewidths=2.5,
            color=dog_colors[dog],
            label=f"DOG {dog}",
            zorder=5,
        )

    ax.scatter(
        center_lon[0],
        center_lat[0],
        marker="+",
        s=130,
        linewidths=2,
        color="black",
        label="Estimated X center",
        zorder=5,
    )

    ax.set_title("Trajectory ESV Splits and Selected DOG Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend()

    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(trans_lat))))

    fig.tight_layout()

    if SAVE:
        fig.savefig(
            "trajectory_esv_splits.png",
            dpi=300,
            bbox_inches="tight",
        )

    # ---------------------------------------------------------
    # 2. ESV and timing-bias chains
    # ---------------------------------------------------------
    samples = np.arange(esv_bias_chain.shape[0])

    fig, axes = plt.subplots(
        len(DOG_order),
        2,
        figsize=(13, 3.5 * len(DOG_order)),
        sharex="col",
    )

    if len(DOG_order) == 1:
        axes = axes[np.newaxis, :]

    for dog_i, dog in enumerate(DOG_order):
        esv_ax = axes[dog_i, 0]
        time_ax = axes[dog_i, 1]

        for split in range(4):
            esv_ax.plot(
                samples,
                esv_bias_chain[:, dog_i, split],
                linewidth=0.8,
                color=split_colors[split],
                label=f"Split {split + 1}",
            )

            esv_ax.axhline(
                initial_esv_bias[dog_i, split],
                linewidth=0.8,
                linestyle="--",
                color=split_colors[split],
                alpha=0.7,
            )

        time_ax.plot(
            samples,
            time_bias_chain[:, dog_i] * 1000.0,
            linewidth=0.8,
            color="black",
        )

        time_ax.axhline(
            initial_time_bias[dog_i] * 1000.0,
            linewidth=0.8,
            linestyle="--",
            color="black",
            alpha=0.7,
        )

        esv_ax.set_ylabel(f"DOG {dog}\nESV bias (m/s)")
        time_ax.set_ylabel(f"DOG {dog}\nTiming bias (ms)")

        esv_ax.grid(True)
        time_ax.grid(True)

    axes[0, 0].set_title("ESV Bias by MCMC Sample")
    axes[0, 1].set_title("Timing Bias by MCMC Sample")
    axes[0, 0].legend(ncol=4, loc="upper right")

    axes[-1, 0].set_xlabel("Posterior Sample")
    axes[-1, 1].set_xlabel("Posterior Sample")

    fig.tight_layout()

    if SAVE:
        fig.savefig(
            "esv_time_bias_chains.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
