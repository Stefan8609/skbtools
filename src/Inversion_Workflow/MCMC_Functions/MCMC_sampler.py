"""Bayesian sampler for the joint Bermuda model (DOGs 1, 3, 4).

Parameters, in the order of the sampled vector ``theta``:

- ``lever`` (3): GPS1 to transducer lever arm, vessel body frame, metres.
- ``CDOG_aug`` (n_dogs, 3): DOG positions as ECEF offsets from ``CDOG_reference``, m.
- ``esv_bias`` (n_dogs, n_splits): dc, additive ESV bias, m/s, one per consecutive
  block of GPS epochs (blocks as in ``np.array_split`` over all epochs, so every
  DOG shares the block edges). Default ``esv_model`` "splits".
- ``esv_model`` "common_windows" replaces ``esv_bias`` by ``esv_offset`` (n_dogs)
  and ``esv_common`` (n_windows - 1): dc of DOG j in window k is
  ``esv_offset[j] + common[k]``, with one common series shared by all DOGs on
  fixed GPS-time windows. The common series has zero mean over the windows (its
  last window is minus the sum of the others), so ``esv_offset`` is each DOG's
  mean dc. Windows are ``esv_window_s`` long from the first to the last paired
  GPS time; a window with fewer than ``esv_window_min_pairs`` pairs (all DOGs)
  is merged into a neighbour, so every window is constrained by data. The edges
  are stored in the configuration (``esv_window_edges_s``) and saved with the
  chain. No smoothness prior: each window has the same Gaussian prior as dc.
- ``time_bias`` (n_dogs): internal clock correction, s (see sign convention below).
- ``drift_rate`` (n_dogs, optional): clock drift, s/s, Eq. 6 convention.
- ``log_sigma`` (n_dogs, modes "iid", "ar1", "ar1_white"): log of the residual sd
  in cm; in "ar1_white" the sd of the slowly varying AR(1) component.
- ``phi`` (n_dogs, modes "ar1" and "ar1_white" unless fixed): lag-one
  autocorrelation of the AR(1) component.
- ``log_sigma_white`` (n_dogs, mode "ar1_white"): log of the white-noise sd in cm.
- ``atanh_phi`` (n_dogs, mode "ar1_white" unless phi is fixed): phi is sampled as
  atanh(phi). Its posterior sits close to 1, where log sigma depends on phi
  along a curve that a random walk follows poorly; in atanh(phi) that relation
  is nearly linear. The prior on phi itself stays uniform (Jacobian term).

Time-bias sign convention. The code models the observed travel time of a pair as
``T_obs = (t_DOG - offset) - t_GPS + time_bias``, so ``time_bias`` is added to the
observation, whereas Eq. 6 subtracts the clock offset dt. The two are related by
``dt = offset - time_bias`` (the same relation used for Table 1). The sampler keeps
the internal convention; every output reports ``dt`` in the Eq. 6 convention. With
drift, ``dt(t) = dt0 + drift_rate * (t - t_mid)``, so internally the observation
gets ``time_bias - drift_rate * (t - t_mid)``.

Fixed pairing. Each DOG arrival is paired with a GPS epoch once, by the alignment
two-pointer search at the start model with threshold ``pair_threshold_s``, and the
pairs then stay fixed for every chain. Re-pairing inside the sampler would change
the number of data with the parameters and make the likelihood discontinuous. The
pairing is insensitive to the parameters: the 0.4 s window is about 600 m of range,
while the posterior spans centimetres. Pairs whose start-model residual fails the
6xMAD rule of the inversion (``residual_inliers``) are dropped and recorded.

Residuals are in centimetres of range at a nominal 1515 m/s (the c*sigma convention
of Table 1). Log-likelihoods omit the constant ``-n/2 log(2 pi)``.

The file follows the flow of a run: configuration and run variants, the fixed-pair
problem, the parameter layout and priors, forward model and residuals, likelihood,
the start point (Laplace approximation), adaptive Metropolis, save and resume, and
the entry points.
"""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np
import scipy.io as sio
from numba import njit

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_heights,
    calculateTimesRayTracing_split,
    calculateTimesRayTracing_windows,
)
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import (
    vessel_rotations as find_vessel_rotations,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import residual_inliers
from geometry.ECEF_Geodetic import ECEF_Geodetic
from bermuda_data import (
    CDOG_GUESS_AUGMENTS,
    CDOG_REFERENCE,
    GPS1_TO_OTHERS,
    LEG_WINDOWS_H,
    PROJECT_ROOT,
    gps_data_path,
    gps_output_path,
    leg_mask,
)

RANGE_CM_PER_S = NOMINAL_SOUND_SPEED_MPS * 100.0
LIKELIHOOD_MODES = ("iid_fixed", "iid", "ar1", "ar1_white")

# Lever-arm prior, vessel body frame, metres from GPS1. The horizontal center is
# the moon-pool center from the R/V Atlantic Explorer General Arrangement drawings
# (Rev 02-Jun-2014), digitized in plotting/Atlantic_Explorer/ and transformed to
# the body frame with GPS1_OFFSET = (39.5, 2.2, 14.2) m (measured from the
# schematic) and ROTATION_DEG = 29.5 (GNSS grid orientation: x-axis perpendicular
# to and bisecting the line between the two forward receivers). Reproduced by
# plotting/Atlantic_Explorer/moon_pool_lever.py. The 0.5 m horizontal sd covers the
# ~1 m moon-pool opening. The vertical center assumes the transducer sits near the
# keel, about 14 m below GPS1; that depth is not measured, hence the 2 m sd.
PRIOR_LEVER_CENTER_M = [-13.1, 9.7, -14.0]
PRIOR_LEVER_SD_M = [0.5, 0.5, 2.0]

# Everything that defines a run. Kept JSON-serializable: it is saved with every
# chain, and a resumed chain must have the same configuration.
CONFIG = {
    "run_name": "bermuda_ar1",
    # "iid_fixed": old likelihood (iid, sigma fixed at fixed_sigma_cm);
    # "iid": iid with a sampled sigma per DOG; "ar1": AR(1) residuals per DOG.
    "likelihood": "ar1",
    "fixed_sigma_cm": 20.0,
    # ar1: fix phi at its estimate from the start-mode residuals instead of sampling.
    "fix_phi": False,
    # ar1: consecutive pairs further apart than this start a new AR(1) segment.
    "ar1_gap_threshold_s": 30.0,
    # Linear clock drift per DOG.
    "clock_drift": False,
    # GPS-time window [start, end] (s) of the epochs used; None uses all. For the
    # half-split repeatability runs.
    "gps_time_window_s": None,
    # Alternating GPS-time blocks [block_s, parity]: keep the epochs whose block
    # index floor((t - t_first) / block_s) has this parity (0 or 1); None uses all.
    # Short blocks give two subsets with nearly the same azimuth coverage, for the
    # geometry-balanced split; applied after gps_time_window_s.
    "gps_time_blocks": None,
    # Survey legs [[start_h, end_h], ...] in GPS hours (bermuda_data.leg_mask): keep
    # the epochs inside any of them; None uses all. Applied after the two above.
    "gps_leg_windows_h": None,
    # Data (paths inside Bermuda_Data). The GPS epochs of the DOG 1 file are used
    # for all DOGs, as in the inversion of the joint model.
    "dogs": [1, 3, 4],
    "esv_table": "ESV_Tables/global_table_esv_extended.mat",
    "gps_file": "GPS_Data/Processed_GPS_Receivers_DOG_1.npz",
    "num_esv_splits": 4,
    "CDOG_reference": CDOG_REFERENCE.tolist(),
    # Receiver grid (body frame, m from GPS1) as in the inversion drivers.
    "gps1_to_others": GPS1_TO_OTHERS.tolist(),
    # Alignment: integer plus sub-integer offsets (s) and the pairing window.
    # Gauss-Newton offset search of Bermuda_Data_time_bias.py on the PRIDE_final
    # GNSS (2026-11-21). Submitted-GNSS values: 1866.016, 3175.017, 1939.0178.
    "offsets_s": [1866.016727, 3175.016802, 1939.016538],
    "pair_threshold_s": 0.4,
    # Pairs with start-model residuals beyond this many MADs are dropped (None: keep).
    "outlier_mad_factor": 6.0,
    # Start model: the pairing model and the start of the mode search. It does not
    # enter the prior. Per-DOG Gauss-Newton solutions on the PRIDE_final GNSS
    # (2026-11-21); the lever arm is the mean of the three per-DOG lever arms. The
    # submitted-GNSS start (inputs of the Fig. 7 chain) was lever
    # [-12.8, 9.5, -12.15], dc [-0.4078, -0.2667, -0.1230] m/s.
    "start_lever": [-12.965, 9.471, -14.547],
    "start_CDOG_aug": [
        [-397.535, 371.119, 773.408],
        [825.428, -111.056, -734.226],
        [235.188, -1304.573, -2191.449],
    ],
    "start_esv_bias": [-0.1633, -0.1478, -0.1858],
    # Priors (independent Gaussians unless noted).
    "prior_lever_center_m": PRIOR_LEVER_CENTER_M,
    "prior_lever_sd_m": PRIOR_LEVER_SD_M,
    # DOG positions: the Gauss-Newton starting guesses of Bermuda_Data_time_bias.py.
    # Replace with deployment positions when they are available.
    "prior_CDOG_aug_center_m": [CDOG_GUESS_AUGMENTS[dog].tolist() for dog in (1, 3, 4)],
    "prior_CDOG_aug_sd_m": 5.0,
    "prior_esv_bias_center_m_s": 0.0,
    "prior_esv_bias_sd_m_s": 2.0,
    # dt is centered on the alignment offset, i.e. time_bias on zero.
    "prior_dt_sd_s": 0.01,
    "prior_drift_rate_sd": 1e-6,
    "prior_log_sigma_center": float(np.log(20.0)),
    "prior_log_sigma_sd": 1.0,
    # Uniform prior on phi.
    "prior_phi_bounds": [-0.999, 0.999],
    # Adaptive Metropolis.
    "n_burn": 20000,
    "n_samples": 200000,
    "thin": 10,
    "first_adapt_window": 1000,
    # Final part of burn-in in which only the proposal scale adapts.
    "scale_only_fraction": 0.2,
    "target_acceptance": 0.234,
    "overdispersion": 3.0,
    "seed": 20261116,
    "checkpoint_every": 5000,
    "log_every": 1000,
}

# AR(1) + white noise: residuals are a slowly varying AR(1) component plus
# independent ping-to-ping noise. The white-noise sd gets the same weak prior as
# sigma. phi may come closer to 1 because the slow component decorrelates over
# minutes (hundreds of pairs). phi near 1 mixes slowly, so burn-in is longer.
AR1_WHITE = {
    "likelihood": "ar1_white",
    "prior_log_sigma_white_center": float(np.log(20.0)),
    "prior_log_sigma_white_sd": 1.0,
    "prior_phi_bounds": [-0.9999, 0.9999],
    "n_burn": 40000,
}
# 16 ESV splits per DOG roughly double the dimension: burn in and sample longer.
# With white noise as well, the 20k-iteration smoke runs had not converged.
LONG_RUN = {"n_burn": 40000, "n_samples": 400000}
WHITE_LONG_RUN = {"n_burn": 80000, "n_samples": 600000}
# Sensitivity model: one dc(t) series shared by all DOGs on fixed
# GPS-time windows plus a constant dc per DOG. The minimum of 300 pairs (all DOGs
# together) is about 10 min of one DOG; a typical 30-min window holds 1000-4000.
COMMON_WINDOWS = {
    "esv_model": "common_windows",
    "num_esv_splits": 1,
    "esv_window_s": 1800.0,
    "esv_window_min_pairs": 300,
    "esv_window_edges_s": None,
}

RUN_VARIANTS = {
    "iid_fixed": {"likelihood": "iid_fixed"},
    "iid": {"likelihood": "iid"},
    "ar1": {"likelihood": "ar1"},
    "ar1_drift": {"likelihood": "ar1", "clock_drift": True},
    # ESV-split test with AR(1) noise (ar1 itself has 4 splits).
    "ar1_splits1": {"likelihood": "ar1", "num_esv_splits": 1},
    "ar1_splits8": {"likelihood": "ar1", "num_esv_splits": 8},
    "ar1_splits16": {"likelihood": "ar1", "num_esv_splits": 16, **LONG_RUN},
    # AR(1) + white noise, alone (4 splits) and with the ESV-split test.
    "ar1_white": AR1_WHITE,
    "ar1_white_splits1": {**AR1_WHITE, "num_esv_splits": 1},
    "ar1_white_splits8": {**AR1_WHITE, "num_esv_splits": 8},
    "ar1_white_splits16": {**AR1_WHITE, "num_esv_splits": 16, **WHITE_LONG_RUN},
    # Linear clock drift under AR(1) + white noise, whose slow component could
    # absorb a drift. No 16-split run: DOG 4 has two 16-split blocks with no pairs.
    "ar1_white_drift": {**AR1_WHITE, "clock_drift": True},
    "ar1_white_drift_splits1": {**AR1_WHITE, "clock_drift": True, "num_esv_splits": 1},
    "ar1_white_drift_splits8": {**AR1_WHITE, "clock_drift": True, "num_esv_splits": 8},
    # Common dc(t) on 30-min (about 28 windows) and 1-h windows, and with drift.
    "ar1_common30": {"likelihood": "ar1", **COMMON_WINDOWS, **LONG_RUN},
    "ar1_common30_drift": {
        "likelihood": "ar1",
        **COMMON_WINDOWS,
        "clock_drift": True,
        **LONG_RUN,
    },
    "ar1_common60": {
        "likelihood": "ar1",
        **COMMON_WINDOWS,
        "esv_window_s": 3600.0,
        **LONG_RUN,
    },
    # Window-length convergence: 60 -> 30 min moved DOG heights by up to 84 cm.
    "ar1_common15": {
        "likelihood": "ar1",
        **COMMON_WINDOWS,
        "esv_window_s": 900.0,
        **LONG_RUN,
    },
    # Common dc(t) with the AR(1) + white likelihood: does the shared series take
    # up the slow component that AR(1) + white alone assigns to each DOG?
    "ar1_white_common30": {**AR1_WHITE, **COMMON_WINDOWS, **WHITE_LONG_RUN},
}
# Repeatability. Halves of the survey split at the median GPS epoch of the
# DOG 1 file (47135 s), each with the 30-min common dc(t), and each DOG alone (its
# own lever arm) with the 4-split model of "ar1".
HALF_SPLIT_GPS_S = 47135.0
RUN_VARIANTS["ar1_common30_half1"] = {
    **RUN_VARIANTS["ar1_common30"],
    "gps_time_window_s": [0.0, HALF_SPLIT_GPS_S],
}
RUN_VARIANTS["ar1_common30_half2"] = {
    **RUN_VARIANTS["ar1_common30"],
    "gps_time_window_s": [HALF_SPLIT_GPS_S, np.inf],
}
# The time halves see the DOGs from different azimuths (half 2 has almost no E/SE
# sector for DOGs 3 and 4), and each half also re-estimates the lever arm.
# (a) Lever arm pinned (1 mm prior sd) at the posterior mean of ar1_common30: it is
# a property of the vessel, so the halves differ only through the DOG, clock and
# sound-speed parameters.
# Posterior mean of ar1_common30 on the PRIDE_final GNSS (cluster chains, 2026-11-21).
# The submitted-GNSS value was [-12.987376, 9.271818, -14.125968].
FINAL_LEVER_M = [-12.963208, 9.369362, -14.637982]
PINNED_LEVER = {"prior_lever_center_m": FINAL_LEVER_M, "prior_lever_sd_m": [0.001] * 3}
for _half in (1, 2):
    RUN_VARIANTS[f"ar1_common30_half{_half}_pinlever"] = {
        **RUN_VARIANTS[f"ar1_common30_half{_half}"],
        **PINNED_LEVER,
    }
# (b) Geometry-balanced split: alternating 15-min blocks. The two subsets cover the
# same azimuths (total-variation distance of the 30-degree azimuth histograms
# 0.05-0.08 per DOG, vs 0.65 for the time halves) and every 30-min dc(t) window
# keeps data from both. They share the slow ocean state, so their difference tests
# the noise model and fast errors, not slow time-varying error.
ALT_BLOCK_S = 900.0
for _parity, _name in ((0, "a"), (1, "b")):
    RUN_VARIANTS[f"ar1_common30_alt15{_name}"] = {
        **RUN_VARIANTS["ar1_common30"],
        "gps_time_blocks": [ALT_BLOCK_S, _parity],
    }


def select_dogs(dogs):
    """Overrides that keep these DOGs of CONFIG, in this order (per-DOG entries)."""
    index = [CONFIG["dogs"].index(dog) for dog in dogs]
    overrides = {"dogs": list(dogs)}
    for key in (
        "offsets_s",
        "start_CDOG_aug",
        "start_esv_bias",
        "prior_CDOG_aug_center_m",
    ):
        overrides[key] = [CONFIG[key][j] for j in index]
    return overrides


# The two survey legs of bermuda_data.LEG_WINDOWS_H together, a cross over the
# array, under the final model with the lever arm free; compared with the full
# trajectory, jointly and per DOG (below). Never one leg alone: on a straight
# line, rotating a DOG about the line leaves every range unchanged, so a single
# leg leaves its across-track position to ~1 m (0.5-1.6 m linearized sd).
RUN_VARIANTS["ar1_common30_legs"] = {
    **RUN_VARIANTS["ar1_common30"],
    "gps_leg_windows_h": [list(window) for window in LEG_WINDOWS_H],
}

for _dog in (1, 3, 4):
    RUN_VARIANTS[f"ar1_dog{_dog}"] = {"likelihood": "ar1", **select_dogs([_dog])}
    # Per-DOG lever arm under the final model (30-min windows, one DOG, so the
    # "common" series is that DOG's own dc(t); sparse windows merge as usual).
    # With the lever pinned at the joint value, a DOG's change from the joint run
    # comes from its own dc(t), offset and dt alone; on the legs, pinning also
    # removes the lever-arm/horizontal degeneracy of two headings.
    for _suffix in ("", "_legs"):
        RUN_VARIANTS[f"ar1_common30{_suffix}_dog{_dog}"] = {
            **RUN_VARIANTS[f"ar1_common30{_suffix}"],
            **select_dogs([_dog]),
        }
        RUN_VARIANTS[f"ar1_common30{_suffix}_dog{_dog}_pinlever"] = {
            **RUN_VARIANTS[f"ar1_common30{_suffix}_dog{_dog}"],
            **PINNED_LEVER,
        }


def run_config(variant, overrides=None):
    """Return CONFIG with a run variant and any overrides applied.

    The run is named after the variant unless the overrides set ``run_name``;
    outputs go to ``Output/MCMC_Chains/<run_name>/``.
    """
    if variant not in RUN_VARIANTS:
        raise ValueError(
            f"Unknown run variant {variant!r}; expected one of {list(RUN_VARIANTS)}"
        )
    return {
        **CONFIG,
        **RUN_VARIANTS[variant],
        "run_name": variant,
        **(overrides or {}),
    }


@njit(cache=True)
def find_fixed_pair_indices(
    offset,
    threshold,
    CDOG_data,
    GPS_data,
    GPS_travel_times,
):
    """CDOG and GPS row indices of the pairs at the start model (Eqs. A19-A20).

    A DOG arrival and a GPS epoch pair when the predicted arrival is within
    ``threshold`` s; the first such match in time order is taken.
    """
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


def time_block_mask(gps_time, block_s, parity):
    """Epochs in alternating GPS-time blocks of length block_s with this parity."""
    gps_time = np.asarray(gps_time, dtype=float)
    if block_s <= 0 or parity not in (0, 1):
        raise ValueError(f"need block_s > 0 and parity 0 or 1; got {block_s}, {parity}")
    block = np.floor((gps_time - gps_time.min()) / block_s).astype(np.int64)
    return block % 2 == parity


def vessel_rotations(GPS_Coordinates, gps1_to_others):
    """Per-epoch body-to-ECEF rotation R (N, 3, 3) and displacement d (N, 3).

    ``Find_Transponder.vessel_rotations`` for a receiver grid given as a list, as
    stored in CONFIG. With the grid fixed, R and d do not depend on the sampled
    parameters and are computed once.
    """
    return find_vessel_rotations(
        np.asarray(GPS_Coordinates, dtype=float),
        np.asarray(gps1_to_others, dtype=float),
    )


def dog_travel_times(
    DOG_position,
    trans_coords,
    esv_bias_row,
    dz_array,
    angle_array,
    esv_matrix,
    epoch_window=None,
):
    """Modelled travel times (s) from the DOG to every transducer epoch.

    With ``epoch_window`` (window index per epoch, "common_windows" model) epoch i
    gets ``esv_bias_row[epoch_window[i]]``. Otherwise uses the split forward model
    for more than one ESV block, as in ``compute_log_likelihood``.
    """
    if epoch_window is not None:
        times, _ = calculateTimesRayTracing_windows(
            DOG_position,
            trans_coords,
            esv_bias_row,
            epoch_window,
            dz_array,
            angle_array,
            esv_matrix,
        )
        return times
    if esv_bias_row.shape[0] > 1:
        times, _ = calculateTimesRayTracing_split(
            DOG_position, trans_coords, esv_bias_row, dz_array, angle_array, esv_matrix
        )
    else:
        times, _ = calculateTimesRayTracing_Bias_Real(
            DOG_position,
            trans_coords,
            esv_bias_row[0],
            dz_array,
            angle_array,
            esv_matrix,
        )
    return times


def split_index(epochs, n_epochs, num_splits):
    """Split of each epoch index when n_epochs are cut as ``np.array_split`` does.

    The same blocks as ``calculateTimesRayTracing_split``, so a subset of epochs
    keeps the dc of the block it came from.
    """
    sizes = [len(block) for block in np.array_split(np.arange(n_epochs), num_splits)]
    starts = np.cumsum([0] + sizes[:-1])
    return np.searchsorted(starts, epochs, side="right") - 1


def window_index(times, edges):
    """Window of each time for GPS-time edges (s); the last window is closed.

    Times before the first edge or after the last fall in the first or last
    window (only unpaired epochs do, and they do not enter the likelihood).
    """
    edges = np.asarray(edges, dtype=float)
    inside = np.searchsorted(edges, times, side="right") - 1
    return np.clip(inside, 0, edges.size - 2).astype(np.int64)


def common_window_edges(pair_gps_time, window_s, min_pairs):
    """GPS-time edges (s) of the shared dc windows; every window holds data.

    Windows of ``window_s`` seconds run from the first to the last paired GPS
    time (all DOGs); the last one ends at the final pair and may be shorter. A
    window with fewer than ``min_pairs`` pairs (all DOGs together) is merged into
    the previous window (the first into the next) until none is left, so no
    window value rests on the prior alone. Returns the edges and the number of
    merges.
    """
    if window_s <= 0:
        raise ValueError(f"esv_window_s must be positive; got {window_s}")
    start, end = float(np.min(pair_gps_time)), float(np.max(pair_gps_time))
    n_windows = max(1, int(np.ceil((end - start) / window_s)))
    edges = list(start + window_s * np.arange(n_windows))
    edges.append(end)
    n_merged = 0
    while len(edges) > 2:
        counts = np.bincount(
            window_index(pair_gps_time, edges), minlength=len(edges) - 1
        )
        sparse = np.flatnonzero(counts < min_pairs)
        if sparse.size == 0:
            break
        k = int(sparse[0])
        del edges[1 if k == 0 else k]
        n_merged += 1
    return edges, n_merged


def load_bermuda_inputs(config):
    """Read the ESV table, GPS epochs and DOG arrivals named in the configuration."""
    paths = {
        "esv_table": gps_data_path(config["esv_table"]),
        "gps_file": gps_data_path(config["gps_file"]),
    }
    for dog in config["dogs"]:
        paths[f"DOG{dog}"] = gps_data_path(f"CDOG_Data/DOG{dog}-camp.mat")

    dz_array, angle_array, esv_matrix = load_esv_table(paths["esv_table"])
    gps = np.load(paths["gps_file"])
    CDOG_all_data = []
    for dog in config["dogs"]:
        tags = sio.loadmat(paths[f"DOG{dog}"])["tags"].astype(float)
        tags[:, 1] /= 1e9  # fractional part is stored in nanoseconds
        CDOG_all_data.append(tags)

    file_hashes = {}
    for name, path in paths.items():
        file_hashes[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(path),
        }
    return {
        "GPS_Coordinates": gps["GPS_Coordinates"],
        "GPS_data": gps["GPS_data"],
        "CDOG_all_data": CDOG_all_data,
        "dz_array": np.ascontiguousarray(dz_array),
        "angle_array": np.ascontiguousarray(angle_array),
        "esv_matrix": np.ascontiguousarray(esv_matrix),
        "input_files": file_hashes,
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_problem(
    config,
    GPS_Coordinates,
    GPS_data,
    CDOG_all_data,
    dz_array,
    angle_array,
    esv_matrix,
    input_files=None,
):
    """Fix the CDOG-GPS pairs at the start model and collect the sampler data.

    GPS_Coordinates is (N, 4, 3) ECEF metres, GPS_data (N,) GPS seconds, and each
    CDOG_all_data[j] is (M_j, 2) with integer and fractional DOG-clock seconds.
    """
    num_dogs = len(config["dogs"])
    num_splits = config["num_esv_splits"]
    GPS_Coordinates = np.asarray(GPS_Coordinates, dtype=float)
    GPS_data = np.asarray(GPS_data, dtype=float)
    n_epochs = GPS_data.shape[0]
    if GPS_Coordinates.shape != (n_epochs, 4, 3):
        raise ValueError(
            f"GPS_Coordinates must be ({n_epochs}, 4, 3); got {GPS_Coordinates.shape}"
        )
    if len(CDOG_all_data) != num_dogs or len(config["offsets_s"]) != num_dogs:
        raise ValueError(
            f"Expected {num_dogs} DOG arrays and offsets for dogs {config['dogs']}; "
            f"got {len(CDOG_all_data)} and {len(config['offsets_s'])}"
        )
    for j, tags in enumerate(CDOG_all_data):
        if tags.ndim != 2 or tags.shape[1] != 2:
            raise ValueError(f"CDOG data {j} must be (M, 2); got {tags.shape}")

    R, d = vessel_rotations(GPS_Coordinates, config["gps1_to_others"])
    CDOG_reference = np.array(config["CDOG_reference"], dtype=float)
    offsets = np.array(config["offsets_s"], dtype=float)
    start_trans = R @ np.array(config["start_lever"], dtype=float) + d
    start_aug = np.array(config["start_CDOG_aug"], dtype=float)

    gps_pair_indices = np.zeros((num_dogs, n_epochs), dtype=np.int64)
    pair_time_base_2d = np.zeros((num_dogs, n_epochs))
    pair_counts = np.zeros(num_dogs, dtype=np.int64)
    dropped_pairs = []
    for j in range(num_dogs):
        tags = np.ascontiguousarray(CDOG_all_data[j], dtype=float)
        esv_start = np.full(num_splits, float(config["start_esv_bias"][j]))
        times = dog_travel_times(
            CDOG_reference + start_aug[j],
            start_trans,
            esv_start,
            dz_array,
            angle_array,
            esv_matrix,
        )
        cdog_idx, gps_idx = find_fixed_pair_indices(
            offsets[j], config["pair_threshold_s"], tags, GPS_data, times
        )
        base = tags[cdog_idx, 0] + tags[cdog_idx, 1] - offsets[j] - GPS_data[gps_idx]

        if config["outlier_mad_factor"] is not None:
            start_residual_cm = (base - times[gps_idx]) * RANGE_CM_PER_S
            keep = residual_inliers(
                start_residual_cm, float(config["outlier_mad_factor"])
            )
            for k in np.flatnonzero(~keep):
                dropped_pairs.append(
                    {
                        "dog": config["dogs"][j],
                        "cdog_index": int(cdog_idx[k]),
                        "gps_index": int(gps_idx[k]),
                        "gps_time_s": float(GPS_data[gps_idx[k]]),
                        "start_residual_cm": float(start_residual_cm[k]),
                    }
                )
            cdog_idx, gps_idx, base = cdog_idx[keep], gps_idx[keep], base[keep]

        pair_counts[j] = gps_idx.shape[0]
        gps_pair_indices[j, : pair_counts[j]] = gps_idx
        pair_time_base_2d[j, : pair_counts[j]] = base

    pair_dog = np.repeat(np.arange(num_dogs), pair_counts)
    pair_gps_index = np.concatenate(
        [gps_pair_indices[j, : pair_counts[j]] for j in range(num_dogs)]
    )
    pair_time_base = np.concatenate(
        [pair_time_base_2d[j, : pair_counts[j]] for j in range(num_dogs)]
    )
    pair_gps_time = GPS_data[pair_gps_index]
    bounds = np.concatenate([[0], np.cumsum(pair_counts)])
    pair_slices = [slice(bounds[j], bounds[j + 1]) for j in range(num_dogs)]

    t_mid = np.empty(num_dogs)
    segment_start = np.zeros(pair_dog.shape[0], dtype=bool)
    for j, sel in enumerate(pair_slices):
        if pair_counts[j] < 2:
            raise ValueError(f"DOG {config['dogs'][j]} has {pair_counts[j]} pairs")
        times_j = pair_gps_time[sel]
        lag = np.diff(times_j)
        if np.any(lag <= 0):
            raise ValueError(
                f"Paired GPS times of DOG {config['dogs'][j]} not increasing"
            )
        t_mid[j] = 0.5 * (times_j[0] + times_j[-1])
        starts = np.concatenate([[True], lag > config["ar1_gap_threshold_s"]])
        segment_start[sel] = starts

    windows = {}
    if config.get("esv_model", "splits") == "common_windows":
        edges = config["esv_window_edges_s"]
        if edges is None:
            edges, n_merged = common_window_edges(
                pair_gps_time,
                float(config["esv_window_s"]),
                int(config["esv_window_min_pairs"]),
            )
            print(f"dc windows: {len(edges) - 1}, {n_merged} merged", flush=True)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("esv_window_edges_s must be increasing")
        pair_window = window_index(pair_gps_time, edges)
        counts = np.bincount(pair_window, minlength=len(edges) - 1)
        if np.any(counts == 0):
            raise ValueError(
                f"dc windows {np.flatnonzero(counts == 0).tolist()} hold no pairs; "
                "a window without data would rest on the prior alone"
            )
        windows = {
            "esv_window_edges_s": [float(edge) for edge in edges],
            "epoch_window": window_index(GPS_data, edges),
            "pair_window": pair_window,
        }

    print(
        "Fixed pairs per DOG:",
        pair_counts.tolist(),
        "| dropped outliers:",
        len(dropped_pairs),
        "| AR(1) segments:",
        int(segment_start.sum()),
        flush=True,
    )
    return {
        "R_stacked": np.ascontiguousarray(R.reshape(n_epochs * 3, 3)),
        "d": d,
        "CDOG_reference": CDOG_reference,
        "offsets": offsets,
        "dz_array": dz_array,
        "angle_array": angle_array,
        "esv_matrix": esv_matrix,
        "n_epochs": n_epochs,
        "gps_pair_indices": gps_pair_indices,
        "pair_time_base_2d": pair_time_base_2d,
        "pair_counts": pair_counts,
        "pair_dog": pair_dog,
        "pair_gps_index": pair_gps_index,
        "pair_time_base": pair_time_base,
        "pair_t_from_mid": pair_gps_time - t_mid[pair_dog],
        "pair_slices": pair_slices,
        "segment_start": segment_start,
        "t_mid": t_mid,
        "dropped_pairs": dropped_pairs,
        "input_files": input_files or {},
        **windows,
    }


def num_esv_windows(config):
    """Number of dc windows of a "common_windows" configuration."""
    edges = config["esv_window_edges_s"]
    if edges is None:
        raise ValueError(
            "esv_window_edges_s is not set: build the problem first (setup_bermuda "
            "stores the edges in the configuration)"
        )
    return len(edges) - 1


def common_series_matrix(n_windows):
    """(n_windows, n_windows - 1) map from esv_common to the zero-mean series."""
    return np.vstack([np.eye(n_windows - 1), -np.ones((1, n_windows - 1))])


def parameter_layout(config):
    """Slices of each parameter block in theta, and a name for every entry."""
    dogs = config["dogs"]
    blocks = [
        ("lever", [f"lever_{axis}" for axis in "xyz"]),
        ("CDOG_aug", [f"CDOG_aug_DOG{dog}_{axis}" for dog in dogs for axis in "xyz"]),
    ]
    if config.get("esv_model", "splits") == "common_windows":
        blocks += [
            ("esv_offset", [f"esv_offset_DOG{dog}" for dog in dogs]),
            (
                "esv_common",
                [f"esv_common_w{k}" for k in range(num_esv_windows(config) - 1)],
            ),
        ]
    else:
        blocks.append(
            (
                "esv_bias",
                [
                    f"esv_bias_DOG{dog}_split{s}"
                    for dog in dogs
                    for s in range(config["num_esv_splits"])
                ],
            )
        )
    blocks.append(("time_bias", [f"time_bias_internal_DOG{dog}" for dog in dogs]))
    if config["clock_drift"]:
        blocks.append(("drift_rate", [f"drift_rate_DOG{dog}" for dog in dogs]))
    if config["likelihood"] != "iid_fixed":
        blocks.append(("log_sigma", [f"log_sigma_cm_DOG{dog}" for dog in dogs]))
    if config["likelihood"] == "ar1" and not config["fix_phi"]:
        blocks.append(("phi", [f"phi_DOG{dog}" for dog in dogs]))
    if config["likelihood"] == "ar1_white":
        if not config["fix_phi"]:
            blocks.append(("atanh_phi", [f"atanh_phi_DOG{dog}" for dog in dogs]))
        blocks.append(
            ("log_sigma_white", [f"log_sigma_white_cm_DOG{dog}" for dog in dogs])
        )

    layout = {}
    names = []
    for block, entries in blocks:
        layout[block] = slice(len(names), len(names) + len(entries))
        names += entries
    return layout, names


def build_model(config):
    """Parameter layout, prior centers and sds, and the start vector.

    Every parameter except phi has an independent Gaussian prior; phi is uniform
    on ``prior_phi_bounds``.
    """
    if config["likelihood"] not in LIKELIHOOD_MODES:
        raise ValueError(
            f"likelihood must be one of {LIKELIHOOD_MODES}; "
            f"got {config['likelihood']!r}"
        )
    if config["likelihood"] == "iid_fixed" and config["clock_drift"]:
        raise ValueError("iid_fixed reproduces the old likelihood, which has no drift")
    common_windows = config.get("esv_model", "splits") == "common_windows"
    if config.get("esv_model", "splits") not in ("splits", "common_windows"):
        raise ValueError(
            f"esv_model must be 'splits' or 'common_windows'; "
            f"got {config['esv_model']!r}"
        )
    if common_windows and (
        config["likelihood"] == "iid_fixed" or config["num_esv_splits"] != 1
    ):
        raise ValueError(
            "common_windows needs a sampled-noise likelihood and num_esv_splits = 1"
        )

    num_dogs = len(config["dogs"])
    # Columns of the per-DOG dc matrix: splits, or windows of the common series.
    num_splits = num_esv_windows(config) if common_windows else config["num_esv_splits"]
    layout, names = parameter_layout(config)
    n_params = len(names)

    center = np.full(n_params, np.nan)
    sd = np.full(n_params, np.nan)
    center[layout["lever"]] = config["prior_lever_center_m"]
    sd[layout["lever"]] = config["prior_lever_sd_m"]
    center[layout["CDOG_aug"]] = np.ravel(config["prior_CDOG_aug_center_m"])
    sd[layout["CDOG_aug"]] = config["prior_CDOG_aug_sd_m"]
    if common_windows:
        center[layout["esv_offset"]] = config["prior_esv_bias_center_m_s"]
        sd[layout["esv_offset"]] = config["prior_esv_bias_sd_m_s"]
        center[layout["esv_common"]] = 0.0
        sd[layout["esv_common"]] = config["prior_esv_bias_sd_m_s"]
    else:
        center[layout["esv_bias"]] = config["prior_esv_bias_center_m_s"]
        sd[layout["esv_bias"]] = config["prior_esv_bias_sd_m_s"]
    # dt = offset - time_bias is centered on the alignment offset.
    center[layout["time_bias"]] = 0.0
    sd[layout["time_bias"]] = config["prior_dt_sd_s"]
    if "drift_rate" in layout:
        center[layout["drift_rate"]] = 0.0
        sd[layout["drift_rate"]] = config["prior_drift_rate_sd"]
    if "log_sigma" in layout:
        center[layout["log_sigma"]] = config["prior_log_sigma_center"]
        sd[layout["log_sigma"]] = config["prior_log_sigma_sd"]
    if "log_sigma_white" in layout:
        center[layout["log_sigma_white"]] = config["prior_log_sigma_white_center"]
        sd[layout["log_sigma_white"]] = config["prior_log_sigma_white_sd"]
    if not np.all(sd[np.isfinite(sd)] > 0):
        raise ValueError("Prior sds must be positive")

    start = np.zeros(n_params)
    start[layout["lever"]] = config["start_lever"]
    start[layout["CDOG_aug"]] = np.ravel(config["start_CDOG_aug"])
    if common_windows:
        start[layout["esv_offset"]] = config["start_esv_bias"]
    else:
        start[layout["esv_bias"]] = np.repeat(config["start_esv_bias"], num_splits)
    if "log_sigma" in layout:
        start[layout["log_sigma"]] = config["prior_log_sigma_center"]
    if "log_sigma_white" in layout:
        start[layout["log_sigma_white"]] = config["prior_log_sigma_white_center"]

    # Finite-difference steps for the residual Jacobian of the geometric block.
    fd_step = np.zeros(n_params)
    fd_step[layout["lever"]] = 1e-3
    fd_step[layout["CDOG_aug"]] = 1e-3
    for block in ("esv_bias", "esv_offset", "esv_common"):
        if block in layout:
            fd_step[layout[block]] = 1e-3
    fd_step[layout["time_bias"]] = 1e-6
    geometric_end = layout["time_bias"].stop
    if "drift_rate" in layout:
        fd_step[layout["drift_rate"]] = 1e-10
        geometric_end = layout["drift_rate"].stop

    return {
        "config": config,
        "num_dogs": num_dogs,
        "num_splits": num_splits,
        "layout": layout,
        "names": names,
        "prior_center": center,
        "prior_sd": sd,
        "gaussian": np.isfinite(sd),
        "start_theta": start,
        "fd_step": fd_step,
        "geometric_end": geometric_end,
        "fixed_phi": None,
    }


def esv_bias_matrix(theta, model):
    """dc (m/s) per DOG and split or window, (..., n_dogs, num_splits).

    theta is (n_params,) or (n_draws, n_params). In the "common_windows" model dc
    of DOG j in window k is ``esv_offset[j] + common[k]``.
    """
    layout = model["layout"]
    shape = theta.shape[:-1] + (model["num_dogs"], model["num_splits"])
    if "esv_bias" in layout:
        return theta[..., layout["esv_bias"]].reshape(shape)
    common = (
        theta[..., layout["esv_common"]] @ common_series_matrix(model["num_splits"]).T
    )
    return theta[..., layout["esv_offset"], None] + common[..., None, :]


def unpack(theta, model):
    """Split theta into named arrays; absent blocks are returned as zeros."""
    layout = model["layout"]
    num_dogs = model["num_dogs"]
    params = {
        "lever": theta[layout["lever"]],
        "CDOG_aug": theta[layout["CDOG_aug"]].reshape(num_dogs, 3),
        "esv_bias": esv_bias_matrix(theta, model),
        "time_bias": theta[layout["time_bias"]],
    }
    for block in ("drift_rate", "log_sigma", "phi", "log_sigma_white"):
        params[block] = theta[layout[block]] if block in layout else np.zeros(num_dogs)
    if "atanh_phi" in layout:
        params["phi"] = np.tanh(theta[layout["atanh_phi"]])
    return params


def residuals_cm(theta, model, problem):
    """Range residuals (cm), observed minus modelled, for all pairs in DOG order."""
    epoch_window = problem.get("epoch_window")
    if ("esv_offset" in model["layout"]) != (epoch_window is not None):
        raise ValueError("Model and problem disagree on the esv_model")
    params = unpack(theta, model)
    trans = (problem["R_stacked"] @ params["lever"]).reshape(-1, 3) + problem["d"]
    # Travel times only at the paired epochs (the model is per epoch). With several
    # dc blocks, each pair takes the block of its epoch, and the transducer heights
    # are computed once for all DOGs.
    pair_gps_index = problem["pair_gps_index"]
    if epoch_window is not None:
        pair_block = problem["pair_window"]
    elif model["num_splits"] > 1:
        pair_block = split_index(
            pair_gps_index, problem["n_epochs"], model["num_splits"]
        )
    else:
        pair_block = None
    if pair_block is not None:
        heights = ECEF_Geodetic(trans)[2]
    residuals = np.empty(problem["pair_dog"].shape[0])
    for j, sel in enumerate(problem["pair_slices"]):
        DOG_position = problem["CDOG_reference"] + params["CDOG_aug"][j]
        epochs = pair_gps_index[sel]
        if pair_block is None:
            times, _ = calculateTimesRayTracing_Bias_Real(
                DOG_position,
                trans[epochs],
                params["esv_bias"][j, 0],
                problem["dz_array"],
                problem["angle_array"],
                problem["esv_matrix"],
            )
        else:
            times, _ = calculateTimesRayTracing_heights(
                DOG_position,
                trans[epochs],
                heights[epochs],
                params["esv_bias"][j],
                pair_block[sel],
                problem["dz_array"],
                problem["angle_array"],
                problem["esv_matrix"],
            )
        residuals[sel] = (
            problem["pair_time_base"][sel]
            + params["time_bias"][j]
            - params["drift_rate"][j] * problem["pair_t_from_mid"][sel]
            - times
        ) * RANGE_CM_PER_S
    return residuals


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
    """iid Gaussian log-likelihood with sd ``sigma_cm``, the "iid_fixed" mode.

    The original likelihood, kept unchanged. Returns ``(value, n_pairs)``.
    """
    split_esv = esv_bias.shape[1] > 1

    total_sse = 0.0
    total_n = 0

    inv_sigma2 = 1.0 / (sigma_cm * sigma_cm)
    log_sigma2 = math.log(sigma_cm * sigma_cm)
    residual_scale = RANGE_CM_PER_S

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


def whiten_ar1(x, segment_start, pair_dog, sigma_cm, phi):
    """Map AR(1) residuals, or Jacobian columns, to unit-variance innovations.

    x is (n,) or (n, k), ordered in time within each DOG. Within a segment,
    ``e_t = phi * e_(t-1) + eta_t`` with marginal sd sigma, so eta has sd
    ``sigma * sqrt(1 - phi**2)``; the first pair of a segment has sd sigma. Returns
    the whitened array and the innovation sds (n,), in cm.
    """
    rho = np.where(segment_start, 0.0, phi[pair_dog])
    innovation_sd = sigma_cm[pair_dog] * np.sqrt(1.0 - rho * rho)
    previous = np.zeros_like(x)
    previous[1:] = x[:-1]
    if x.ndim == 2:
        return (x - rho[:, None] * previous) / innovation_sd[:, None], innovation_sd
    return (x - rho * previous) / innovation_sd, innovation_sd


def ar1_log_likelihood(residuals, segment_start, pair_dog, sigma_cm, phi):
    """Gaussian AR(1) log-likelihood per DOG; phi = 0 gives the iid likelihood."""
    whitened, innovation_sd = whiten_ar1(
        residuals, segment_start, pair_dog, sigma_cm, phi
    )
    return -0.5 * np.dot(whitened, whitened) - np.sum(np.log(innovation_sd))


@njit(cache=True)
def kalman_whiten(x, segment_start, pair_dog, sigma_cm, phi, sigma_white_cm):
    """Standardized innovations of an AR(1) plus white-noise model, per column.

    Each residual is ``e_t = a_t + w_t``: ``a_t`` is AR(1) with lag-one
    correlation phi and marginal sd sigma, restarting at each segment start, and
    ``w_t`` is white noise with sd sigma_white. A scalar Kalman filter predicts
    each residual from the previous ones; its gain does not depend on the data,
    so the same linear map whitens residuals and Jacobian columns alike.

    x is (n, k), time-ordered within each DOG. Returns the innovations divided by
    their sd, (n, k), and the innovation variances (n,), in cm^2. With
    sigma_white = 0 this is whiten_ar1; with phi = 0 it is iid with variance
    sigma^2 + sigma_white^2.
    """
    n, k = x.shape
    whitened = np.empty((n, k))
    innovation_var = np.empty(n)
    state_mean = np.zeros(k)
    state_var = 0.0
    for t in range(n):
        j = pair_dog[t]
        slow_var = sigma_cm[j] * sigma_cm[j]
        if segment_start[t]:
            state_mean[:] = 0.0
            state_var = slow_var
        else:
            state_mean *= phi[j]
            state_var = phi[j] * phi[j] * state_var + slow_var * (1.0 - phi[j] * phi[j])
        total_var = state_var + sigma_white_cm[j] * sigma_white_cm[j]
        gain = state_var / total_var
        scale = np.sqrt(total_var)
        for c in range(k):
            innovation = x[t, c] - state_mean[c]
            whitened[t, c] = innovation / scale
            state_mean[c] += gain * innovation
        state_var -= gain * state_var
        innovation_var[t] = total_var
    return whitened, innovation_var


def whiten_residuals(x, problem, sigma_cm, phi, sigma_white_cm=None):
    """Whiten residuals (n,) or Jacobian columns (n, k) under the noise model.

    AR(1) (``whiten_ar1``) when sigma_white_cm is None, otherwise AR(1) plus white
    noise (``kalman_whiten``). Returns the whitened array and the innovation sds.
    """
    segment_start, pair_dog = problem["segment_start"], problem["pair_dog"]
    if sigma_white_cm is None:
        return whiten_ar1(x, segment_start, pair_dog, sigma_cm, phi)
    columns = np.ascontiguousarray(x.reshape(x.shape[0], -1))
    whitened, innovation_var = kalman_whiten(
        columns, segment_start, pair_dog, sigma_cm, phi, sigma_white_cm
    )
    return whitened.reshape(x.shape), np.sqrt(innovation_var)


def white_noise_sd(theta, model):
    """White-noise sd (cm) per DOG in mode "ar1_white", otherwise None."""
    if "log_sigma_white" not in model["layout"]:
        return None
    return np.exp(theta[model["layout"]["log_sigma_white"]])


def noise_parameters(theta, model):
    """Residual sd (cm) and phi per DOG used by the likelihood at theta."""
    params = unpack(theta, model)
    config = model["config"]
    if config["likelihood"] == "iid_fixed":
        sigma = np.full(model["num_dogs"], float(config["fixed_sigma_cm"]))
    else:
        sigma = np.exp(params["log_sigma"])
    if config["likelihood"] not in ("ar1", "ar1_white"):
        phi = np.zeros(model["num_dogs"])
    elif config["fix_phi"]:
        phi = model["fixed_phi"]
    else:
        phi = params["phi"]
    return sigma, phi


def log_prior(theta, model):
    """Log prior density up to a constant; -inf when phi is out of bounds."""
    layout = model["layout"]
    low, high = model["config"]["prior_phi_bounds"]
    jacobian = 0.0
    if "phi" in layout:
        phi = theta[layout["phi"]]
    elif "atanh_phi" in layout:
        phi = np.tanh(theta[layout["atanh_phi"]])
        # Uniform prior on phi, expressed in atanh(phi): d phi / d atanh = 1 - phi^2.
        jacobian = np.sum(np.log1p(-phi * phi))
    else:
        phi = None
    if phi is not None and (np.any(phi <= low) or np.any(phi >= high)):
        return -np.inf
    g = model["gaussian"]
    z = (theta[g] - model["prior_center"][g]) / model["prior_sd"][g]
    return -0.5 * np.dot(z, z) + jacobian


def log_likelihood(theta, model, problem):
    """Log-likelihood in the configured mode, up to the constant -n/2 log(2 pi)."""
    if model["config"]["likelihood"] == "iid_fixed":
        params = unpack(theta, model)
        trans = (problem["R_stacked"] @ params["lever"]).reshape(-1, 3) + problem["d"]
        value, _ = compute_log_likelihood(
            trans,
            params["CDOG_aug"],
            params["esv_bias"],
            params["time_bias"],
            problem["CDOG_reference"],
            problem["gps_pair_indices"],
            problem["pair_time_base_2d"],
            problem["pair_counts"],
            problem["dz_array"],
            problem["angle_array"],
            problem["esv_matrix"],
            float(model["config"]["fixed_sigma_cm"]),
        )
        return value
    sigma, phi = noise_parameters(theta, model)
    sigma_white = white_noise_sd(theta, model)
    if sigma_white is not None:
        whitened, innovation_sd = whiten_residuals(
            residuals_cm(theta, model, problem), problem, sigma, phi, sigma_white
        )
        return -0.5 * np.dot(whitened, whitened) - np.sum(np.log(innovation_sd))
    return ar1_log_likelihood(
        residuals_cm(theta, model, problem),
        problem["segment_start"],
        problem["pair_dog"],
        sigma,
        phi,
    )


def log_posterior(theta, model, problem):
    """Return (log posterior, log-likelihood); both -inf outside the support."""
    lp = log_prior(theta, model)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    ll = log_likelihood(theta, model, problem)
    if not np.isfinite(ll):
        return -np.inf, -np.inf
    return lp + ll, ll


def residual_noise_estimates(residuals, problem):
    """Per-DOG RMS (cm) and lag-one autocorrelation within AR(1) segments."""
    num_dogs = len(problem["pair_slices"])
    sigma = np.empty(num_dogs)
    phi = np.empty(num_dogs)
    follows = ~problem["segment_start"]
    for j, sel in enumerate(problem["pair_slices"]):
        r = residuals[sel]
        later = np.flatnonzero(follows[sel])
        sigma[j] = np.sqrt(np.mean(r * r))
        phi[j] = np.sum(r[later] * r[later - 1]) / np.sum(r[later - 1] ** 2)
    return sigma, phi


def ar1_white_noise_estimates(residuals, problem, phi_high):
    """Per-DOG slow sd (cm), phi and white-noise sd (cm) from residual moments.

    Within segments, the AR(1) plus white-noise model has autocovariances
    c0 = s^2 + w^2, c1 = phi s^2 and c2 = phi^2 s^2 at lags of 0, 1 and 2 pairs,
    so phi = c2 / c1, s^2 = c1 / phi and w^2 = c0 - s^2. The values only start
    the mode search; they are kept inside the prior support.
    """
    num_dogs = len(problem["pair_slices"])
    sigma, phi, sigma_white = np.empty(num_dogs), np.empty(num_dogs), np.empty(num_dogs)
    start = problem["segment_start"]
    for j, sel in enumerate(problem["pair_slices"]):
        r = residuals[sel]
        lag1 = np.flatnonzero(~start[sel])
        lag2 = lag1[(lag1 >= 2) & ~start[sel][np.maximum(lag1 - 1, 0)]]
        c0 = np.mean(r * r)
        c1 = np.mean(r[lag1] * r[lag1 - 1])
        c2 = np.mean(r[lag2] * r[lag2 - 2])
        phi[j] = np.clip(c2 / c1, 0.0, phi_high - 1e-6) if c1 > 0 else 0.0
        slow_var = min(c1 / phi[j], 0.99 * c0) if phi[j] > 0 else 0.5 * c0
        sigma[j] = np.sqrt(slow_var)
        sigma_white[j] = np.sqrt(max(c0 - slow_var, 0.01 * c0))
    return sigma, phi, sigma_white


def residual_jacobian(theta, model, problem):
    """Central finite-difference Jacobian of residuals_cm, (n_pairs, n_geometric).

    Columns are the geometric parameters (lever, DOG positions, dc, time bias,
    drift), in cm of range per unit of each parameter.
    """
    n_geo = model["geometric_end"]
    jacobian = np.empty((problem["pair_dog"].shape[0], n_geo))
    for k in range(n_geo):
        step = model["fd_step"][k]
        plus = theta.copy()
        minus = theta.copy()
        plus[k] += step
        minus[k] -= step
        jacobian[:, k] = (
            residuals_cm(plus, model, problem) - residuals_cm(minus, model, problem)
        ) / (2 * step)
    return jacobian


def laplace_approximation(model, problem, max_iterations=8):
    """Posterior mode estimate and Gaussian approximation, for starting the chains.

    The geometric parameters (lever, DOG positions, dc, time bias, drift) follow
    Gauss-Newton steps on the AR(1)-whitened residuals with a finite-difference
    Jacobian and the Gaussian priors; sigma and phi are set to their residual
    estimates at each step. Returns the mode estimate, its covariance (geometric
    block from the inverse Gauss-Newton Hessian; sigma and phi from their
    asymptotic variances), and the residual estimates of sigma (cm) and phi.
    """
    config = model["config"]
    mode = config["likelihood"]
    geo = slice(0, model["geometric_end"])
    precision = 1.0 / model["prior_sd"][geo] ** 2
    center = model["prior_center"][geo]
    low, high = config["prior_phi_bounds"]
    theta = model["start_theta"].copy()

    def noise_for_whitening(residuals):
        if mode == "ar1_white":
            return ar1_white_noise_estimates(residuals, problem, high)
        sigma, phi = residual_noise_estimates(residuals, problem)
        if mode == "iid_fixed":
            sigma = np.full(model["num_dogs"], float(config["fixed_sigma_cm"]))
        if mode != "ar1":
            phi = np.zeros(model["num_dogs"])
        return sigma, np.clip(phi, low + 1e-6, high - 1e-6), None

    def objective(theta_trial, noise):
        residuals = residuals_cm(theta_trial, model, problem)
        w, _ = whiten_residuals(residuals, problem, *noise)
        z = theta_trial[geo] - center
        return 0.5 * np.dot(w, w) + 0.5 * np.dot(precision * z, z)

    def whitened_system(theta_point):
        residuals = residuals_cm(theta_point, model, problem)
        noise = noise_for_whitening(residuals)
        jacobian = residual_jacobian(theta_point, model, problem)
        w, _ = whiten_residuals(residuals, problem, *noise)
        J_w, _ = whiten_residuals(jacobian, problem, *noise)
        hessian = J_w.T @ J_w + np.diag(precision)
        gradient = J_w.T @ w + precision * (theta_point[geo] - center)
        return hessian, gradient, noise

    for iteration in range(max_iterations):
        hessian, gradient, noise = whitened_system(theta)
        step = -np.linalg.solve(hessian, gradient)
        current = objective(theta, noise)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = theta.copy()
            trial[geo] += fraction * step
            if objective(trial, noise) < current:
                theta = trial
                accepted = True
                break
        newton_decrement = np.sqrt(abs(step @ hessian @ step))
        print(
            f"Mode search iteration {iteration}: objective {current:.2f}, "
            f"Newton decrement {newton_decrement:.3g}",
            flush=True,
        )
        if not accepted or newton_decrement < 1e-3:
            break

    hessian, _, (sigma, phi, sigma_white) = whitened_system(theta)
    n_params = theta.shape[0]
    cov = np.zeros((n_params, n_params))
    # Inverted in unit-diagonal form: dt (s) beside positions (m) puts the raw
    # condition number past double precision when a parameter combination is held
    # mainly by its prior (a single-circle track), and the raw inverse then fails
    # to be positive definite.
    scale = 1.0 / np.sqrt(np.diag(hessian))
    cov[geo, geo] = scale[:, None] * np.linalg.inv(hessian * np.outer(scale, scale))
    cov[geo, geo] *= scale[None, :]
    pair_counts = problem["pair_counts"].astype(float)
    layout = model["layout"]
    if "log_sigma" in layout:
        theta[layout["log_sigma"]] = np.log(sigma)
        n_effective = pair_counts * (1.0 - phi) / (1.0 + phi)
        cov[layout["log_sigma"], layout["log_sigma"]] = np.diag(1.0 / (2 * n_effective))
    if "phi" in layout:
        theta[layout["phi"]] = phi
        cov[layout["phi"], layout["phi"]] = np.diag((1.0 - phi**2) / pair_counts)
    if "atanh_phi" in layout:
        theta[layout["atanh_phi"]] = np.arctanh(phi)
        # var(phi) ~ (1 - phi^2) / n, and d atanh(phi) / d phi = 1 / (1 - phi^2).
        cov[layout["atanh_phi"], layout["atanh_phi"]] = np.diag(
            1.0 / (pair_counts * (1.0 - phi**2))
        )
    if "log_sigma_white" in layout:
        theta[layout["log_sigma_white"]] = np.log(sigma_white)
        cov[layout["log_sigma_white"], layout["log_sigma_white"]] = np.diag(
            1.0 / (2 * pair_counts)
        )
    return theta, cov, sigma, phi


def initial_state(
    chain_index, model, problem, n_kept_total, n_burn_kept_total, n_windows
):
    """Seed the chain, find the mode, and draw an overdispersed start point."""
    config = model["config"]
    seed_sequence = np.random.SeedSequence([config["seed"], chain_index])
    rng = np.random.Generator(np.random.PCG64(seed_sequence))

    laplace_theta, laplace_cov, sigma_hat, phi_hat = laplace_approximation(
        model, problem
    )
    if config["likelihood"] in ("ar1", "ar1_white") and config["fix_phi"]:
        model["fixed_phi"] = phi_hat
    print(
        "Mode estimate: sigma_cm",
        np.round(sigma_hat, 2).tolist(),
        "phi",
        np.round(phi_hat, 4).tolist(),
        flush=True,
    )

    # Overdispersed start: the mode plus `overdispersion` times a draw from the
    # Gaussian approximation. Redraw if the prior support (phi bounds) is left.
    cholesky = np.linalg.cholesky(laplace_cov)
    d = laplace_theta.shape[0]
    for _ in range(100):
        theta = laplace_theta + config[
            "overdispersion"
        ] * cholesky @ rng.standard_normal(d)
        logpost, loglike = log_posterior(theta, model, problem)
        if np.isfinite(logpost):
            break
    else:
        raise RuntimeError("No start point with a finite posterior in 100 draws")

    commit, dirty = git_revision()
    metadata = {
        "chain_index": int(chain_index),
        "seed": [int(config["seed"]), int(chain_index)],
        "rng": "numpy PCG64 from SeedSequence(seed)",
        "likelihood": config["likelihood"],
        "clock_drift": config["clock_drift"],
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "param_names": model["names"],
        "laplace_sigma_cm": sigma_hat.tolist(),
        "laplace_phi": phi_hat.tolist(),
        "dropped_pairs": problem["dropped_pairs"],
        "input_files": problem["input_files"],
        "time_bias_convention": "dt (Eq. 6) = offset - time_bias; outputs report dt",
        "runs": [],
    }
    return {
        "iteration": 0,
        "theta": theta,
        "logpost": logpost,
        "loglike": loglike,
        "log_lambda": 0.0,
        "proposal_cov": laplace_cov.copy(),
        "window_index": 0,
        "window_n": 0,
        "window_mean": np.zeros(d),
        "window_m2": np.zeros((d, d)),
        "window_accepted": 0,
        "window_acceptance": np.full(n_windows, np.nan),
        "n_accepted_sampling": 0,
        "rng": rng,
        "theta_samples": np.full((n_kept_total, d), np.nan),
        "loglike_samples": np.full(n_kept_total, np.nan),
        "logpost_samples": np.full(n_kept_total, np.nan),
        "burnin_theta": np.full((n_burn_kept_total, d), np.nan),
        "burnin_logpost": np.full(n_burn_kept_total, np.nan),
        "n_kept": 0,
        "n_burn_kept": 0,
        "start_theta": theta.copy(),
        "laplace_theta": laplace_theta,
        "laplace_cov": laplace_cov,
        "metadata": metadata,
    }


def adaptation_window_ends(n_burn, first_window):
    """End iterations of the burn-in adaptation windows (doubling lengths).

    A remainder shorter than twice the next window is folded into the last one.
    """
    ends = []
    start = 0
    size = first_window
    while start < n_burn:
        end = start + size
        if n_burn - end < 2 * size:
            end = n_burn
        ends.append(end)
        start = end
        size *= 2
    return ends


def run_chain(chain_index, model, problem, out_path, stop_at=None):
    """Run or resume one adaptive Metropolis chain, checkpointing to out_path.

    Burn-in (``n_burn`` iterations): the proposal is Gaussian with covariance
    ``lambda**2 * 2.38**2 / d * C``. C starts at the Laplace covariance and is
    re-estimated from the chain at the end of each adaptation window (windows
    double in length), in the style of Haario et al. (2001). The last
    ``scale_only_fraction`` of burn-in keeps C fixed. log lambda follows a
    Robbins-Monro update toward ``target_acceptance``, restarted after every
    change of C. Both are frozen after burn-in, so the saved samples come from a
    fixed-kernel Metropolis chain.
    ``stop_at`` (an iteration count) ends this call early with a checkpoint, as a
    job time limit would; calling again resumes.
    """
    config = model["config"]
    n_burn = config["n_burn"]
    n_total = n_burn + config["n_samples"]
    thin = config["thin"]
    target = config["target_acceptance"]
    n_scale_only = int(round(config["scale_only_fraction"] * n_burn))
    if not 0 < n_scale_only < n_burn:
        raise ValueError(
            f"scale_only_fraction * n_burn must leave both phases non-empty; got "
            f"{n_scale_only} of {n_burn}"
        )
    # Covariance windows, then the scale-only phase ending at n_burn.
    window_ends = adaptation_window_ends(
        n_burn - n_scale_only, config["first_adapt_window"]
    ) + [n_burn]
    n_kept_total = -(-config["n_samples"] // thin)
    n_burn_kept_total = -(-n_burn // thin)
    out_path = Path(out_path)

    if out_path.exists():
        check_same_inputs(out_path, problem)
        state = load_state(out_path, model, n_kept_total, n_burn_kept_total)
        print(
            f"[chain {chain_index}] resuming at iteration {state['iteration']}",
            flush=True,
        )
    else:
        state = initial_state(
            chain_index,
            model,
            problem,
            n_kept_total,
            n_burn_kept_total,
            len(window_ends),
        )
    state["n_burn"] = n_burn
    state["n_total"] = n_total
    if state["iteration"] >= n_total:
        print(f"[chain {chain_index}] already complete: {out_path}", flush=True)
        return state
    state["metadata"]["runs"].append(_run_record(state["iteration"]))

    rng = state["rng"]
    theta = state["theta"]
    logpost = state["logpost"]
    loglike = state["loglike"]
    d = theta.shape[0]
    base_scale = 2.38 / np.sqrt(d)
    cholesky = np.linalg.cholesky(state["proposal_cov"])
    regularizer = 1e-3 * np.diag(np.diag(state["laplace_cov"]))
    last_stop = n_total if stop_at is None else min(stop_at, n_total)
    accepted_since_log = 0
    clock = time.time()

    for it in range(state["iteration"], last_stop):
        z = rng.standard_normal(d)
        log_u = np.log(rng.random())
        proposal = theta + np.exp(state["log_lambda"]) * base_scale * (cholesky @ z)
        logpost_proposal, loglike_proposal = log_posterior(proposal, model, problem)
        log_alpha = logpost_proposal - logpost
        accepted = log_u < log_alpha
        if accepted:
            theta, logpost, loglike = proposal, logpost_proposal, loglike_proposal
            accepted_since_log += 1

        if it < n_burn:
            window = state["window_index"]
            previous_end = window_ends[window - 1] if window > 0 else 0
            state["log_lambda"] += (it - previous_end + 1) ** -0.6 * (
                np.exp(min(0.0, log_alpha)) - target
            )
            state["window_accepted"] += int(accepted)
            state["window_n"] += 1
            covariance_window = window < len(window_ends) - 1
            if covariance_window:
                delta = theta - state["window_mean"]
                state["window_mean"] = state["window_mean"] + delta / state["window_n"]
                state["window_m2"] = state["window_m2"] + np.outer(
                    delta, theta - state["window_mean"]
                )
            if it % thin == 0:
                state["burnin_theta"][state["n_burn_kept"]] = theta
                state["burnin_logpost"][state["n_burn_kept"]] = logpost
                state["n_burn_kept"] += 1
            if it + 1 == window_ends[window]:
                n = state["window_n"]
                if covariance_window:
                    sample_cov = state["window_m2"] / (n - 1)
                    state["proposal_cov"] = (n * sample_cov + 5.0 * regularizer) / (
                        n + 5.0
                    )
                    cholesky = np.linalg.cholesky(state["proposal_cov"])
                state["window_acceptance"][window] = state["window_accepted"] / n
                print(
                    f"[chain {chain_index}] adaptation window {window} "
                    f"({'covariance' if covariance_window else 'scale only'}) "
                    f"ends at {it + 1}: acceptance {state['window_accepted'] / n:.3f}, "
                    f"lambda {np.exp(state['log_lambda']):.3f}",
                    flush=True,
                )
                state["window_index"] += 1
                state["window_n"] = 0
                state["window_mean"] = np.zeros(d)
                state["window_m2"] = np.zeros((d, d))
                state["window_accepted"] = 0
        else:
            state["n_accepted_sampling"] += int(accepted)
            if (it - n_burn) % thin == 0:
                k = state["n_kept"]
                state["theta_samples"][k] = theta
                state["loglike_samples"][k] = loglike
                state["logpost_samples"][k] = logpost
                state["n_kept"] += 1

        done = it + 1
        if done % config["log_every"] == 0:
            elapsed = time.time() - clock
            sigma, _ = noise_parameters(theta, model)
            print(
                f"[chain {chain_index}] {done}/{n_total} "
                f"{'burn-in' if done <= n_burn else 'sampling'} "
                f"acceptance {accepted_since_log / config['log_every']:.3f} "
                f"logpost {logpost:.1f} sigma_cm {np.round(sigma, 2).tolist()} "
                f"({1000 * elapsed / config['log_every']:.1f} s per 1000)",
                flush=True,
            )
            accepted_since_log = 0
            clock = time.time()
        if done % config["checkpoint_every"] == 0 or done == last_stop:
            state.update(iteration=done, theta=theta, logpost=logpost, loglike=loglike)
            save_chain(out_path, state, model, problem)

    return state


def git_revision():
    """Commit hash of the repository and whether tracked files are modified."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def _run_record(start_iteration):
    return {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "start_iteration": int(start_iteration),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
    }


def derived_outputs(theta_samples, model, problem):
    """Named parameter arrays for saved samples, with dt in the Eq. 6 convention."""
    n = theta_samples.shape[0]
    layout = model["layout"]
    num_dogs = model["num_dogs"]
    out = {
        "lever": theta_samples[:, layout["lever"]],
        "CDOG_aug": theta_samples[:, layout["CDOG_aug"]].reshape(n, num_dogs, 3),
        "esv_bias": esv_bias_matrix(theta_samples, model),
        # Eq. 6 clock offset: dt = offset - time_bias.
        "dt": problem["offsets"] - theta_samples[:, layout["time_bias"]],
        "drift_rate": (
            theta_samples[:, layout["drift_rate"]]
            if "drift_rate" in layout
            else np.zeros((n, num_dogs))
        ),
    }
    sigma = np.empty((n, num_dogs))
    phi = np.empty((n, num_dogs))
    for i in range(n):
        sigma[i], phi[i] = noise_parameters(theta_samples[i], model)
    out["sigma_cm"] = sigma
    out["phi"] = phi
    if "esv_common" in layout:
        # Common dc series (m/s) per window, zero mean over the windows.
        out["esv_common"] = theta_samples[:, layout["esv_common"]] @ (
            common_series_matrix(model["num_splits"]).T
        )
    if "log_sigma_white" in layout:
        out["sigma_white_cm"] = np.exp(theta_samples[:, layout["log_sigma_white"]])
    return out


def save_chain(path, state, model, problem):
    """Write the chain and its sampler state atomically (a kill cannot corrupt it)."""
    n_kept = state["n_kept"]
    n_burn_kept = state["n_burn_kept"]
    theta_samples = state["theta_samples"][:n_kept]
    arrays = {
        # Samples after burn-in (thinned). The named arrays of derived_outputs
        # (three quarters of the file in the common-window runs) are not stored;
        # load_chain rebuilds them exactly from theta_samples.
        "loglike": state["loglike_samples"][:n_kept],
        "logpost": state["logpost_samples"][:n_kept],
        "theta_samples": theta_samples,
        "param_names": np.array(model["names"]),
        "burnin_theta": state["burnin_theta"][:n_burn_kept],
        "burnin_logpost": state["burnin_logpost"][:n_burn_kept],
        "window_acceptance": state["window_acceptance"],
        "acceptance_rate_sampling": np.array(
            state["n_accepted_sampling"] / max(1, state["iteration"] - state["n_burn"])
            if state["iteration"] > state["n_burn"]
            else np.nan
        ),
        "n_done": np.array(state["iteration"]),
        "n_total": np.array(state["n_total"]),
        # Provenance.
        "config_json": np.array(json.dumps(model["config"], sort_keys=True)),
        "metadata_json": np.array(json.dumps(state["metadata"], sort_keys=True)),
        "start_theta": state["start_theta"],
        "laplace_theta": state["laplace_theta"],
        "laplace_cov": state["laplace_cov"],
        "fixed_phi": (
            np.array(model["fixed_phi"])
            if model["fixed_phi"] is not None
            else np.array([])
        ),
        "prior_center": model["prior_center"],
        "prior_sd": model["prior_sd"],
        "CDOG_reference": problem["CDOG_reference"],
        "offsets_s": problem["offsets"],
        "t_mid_s": problem["t_mid"],
        "pair_counts": problem["pair_counts"],
        # Sampler state for resuming.
        "state_theta": state["theta"],
        "state_logpost": np.array(state["logpost"]),
        "state_loglike": np.array(state["loglike"]),
        "state_log_lambda": np.array(state["log_lambda"]),
        "state_proposal_cov": state["proposal_cov"],
        "state_window_index": np.array(state["window_index"]),
        "state_window_n": np.array(state["window_n"]),
        "state_window_mean": state["window_mean"],
        "state_window_m2": state["window_m2"],
        "state_window_accepted": np.array(state["window_accepted"]),
        "state_n_accepted_sampling": np.array(state["n_accepted_sampling"]),
        "state_rng": np.array(json.dumps(state["rng"].bit_generator.state)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def load_chain(path):
    """Arrays of a chain file plus the named arrays of derived_outputs.

    The named arrays are rebuilt from theta_samples and the saved configuration.
    Files written before they were dropped from save_chain are returned as stored.
    """
    with np.load(path) as saved:
        chain = {key: saved[key] for key in saved.files}
    if "lever" not in chain:
        model = build_model(json.loads(str(chain["config_json"])))
        if chain["fixed_phi"].size:
            model["fixed_phi"] = chain["fixed_phi"]
        problem = {"offsets": chain["offsets_s"]}
        chain.update(derived_outputs(chain["theta_samples"], model, problem))
    return chain


def load_state(path, model, n_kept_total, n_burn_kept_total):
    """Restore a checkpoint written by save_chain for the same configuration."""
    with np.load(path) as saved:
        saved_config = json.loads(str(saved["config_json"]))
        if saved_config != json.loads(json.dumps(model["config"], sort_keys=True)):
            raise ValueError(
                f"{path} was written with a different configuration; use another "
                "run_name or remove the file to start over"
            )
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = json.loads(str(saved["state_rng"]))
        d = saved["state_theta"].shape[0]
        theta_samples = np.full((n_kept_total, d), np.nan)
        loglike_samples = np.full(n_kept_total, np.nan)
        logpost_samples = np.full(n_kept_total, np.nan)
        burnin_theta = np.full((n_burn_kept_total, d), np.nan)
        burnin_logpost = np.full(n_burn_kept_total, np.nan)
        n_kept = saved["theta_samples"].shape[0]
        n_burn_kept = saved["burnin_theta"].shape[0]
        theta_samples[:n_kept] = saved["theta_samples"]
        loglike_samples[:n_kept] = saved["loglike"]
        logpost_samples[:n_kept] = saved["logpost"]
        burnin_theta[:n_burn_kept] = saved["burnin_theta"]
        burnin_logpost[:n_burn_kept] = saved["burnin_logpost"]
        if saved["fixed_phi"].size:
            model["fixed_phi"] = saved["fixed_phi"]
        return {
            "iteration": int(saved["n_done"]),
            "theta": saved["state_theta"],
            "logpost": float(saved["state_logpost"]),
            "loglike": float(saved["state_loglike"]),
            "log_lambda": float(saved["state_log_lambda"]),
            "proposal_cov": saved["state_proposal_cov"],
            "window_index": int(saved["state_window_index"]),
            "window_n": int(saved["state_window_n"]),
            "window_mean": saved["state_window_mean"],
            "window_m2": saved["state_window_m2"],
            "window_accepted": int(saved["state_window_accepted"]),
            "window_acceptance": saved["window_acceptance"],
            "n_accepted_sampling": int(saved["state_n_accepted_sampling"]),
            "rng": rng,
            "theta_samples": theta_samples,
            "loglike_samples": loglike_samples,
            "logpost_samples": logpost_samples,
            "burnin_theta": burnin_theta,
            "burnin_logpost": burnin_logpost,
            "n_kept": n_kept,
            "n_burn_kept": n_burn_kept,
            "start_theta": saved["start_theta"],
            "laplace_theta": saved["laplace_theta"],
            "laplace_cov": saved["laplace_cov"],
            "metadata": json.loads(str(saved["metadata_json"])),
        }


def check_same_inputs(path, problem):
    """Refuse to resume a chain written from different input files (e.g. ESV table).

    The configuration check of load_state compares file paths only, so a chain
    sampled on an older ESV table would otherwise count as finished.
    """
    with np.load(path) as saved:
        saved_files = json.loads(str(saved["metadata_json"])).get("input_files", {})
    current = problem.get("input_files") or {}
    changed = sorted(
        name
        for name in current
        if name in saved_files
        and saved_files[name]["sha256"] != current[name]["sha256"]
    )
    if changed:
        raise ValueError(
            f"{path} was written from different input files ({', '.join(changed)}); "
            "move the old chains aside or use another run_name"
        )


def select_epochs(config, GPS_data, GPS_Coordinates):
    """GPS epochs of the run: ``gps_time_window_s``, ``gps_time_blocks`` and
    ``gps_leg_windows_h`` of the configuration, in that order. Returns
    ``(GPS_data, GPS_Coordinates)`` with time order kept."""
    keep = np.ones(len(GPS_data), dtype=bool)
    window = config.get("gps_time_window_s")
    if window is not None:
        if len(window) != 2 or not window[0] < window[1]:
            raise ValueError(f"gps_time_window_s must be [start, end]; got {window}")
        keep &= (GPS_data >= window[0]) & (GPS_data < window[1])
    blocks = config.get("gps_time_blocks")
    if blocks is not None:
        # Blocks count from the first epoch left by the time window.
        in_blocks = time_block_mask(GPS_data[keep], blocks[0], int(blocks[1]))
        keep[np.flatnonzero(keep)[~in_blocks]] = False
    legs = config.get("gps_leg_windows_h")
    if legs is not None:
        if any(len(leg) != 2 or not leg[0] < leg[1] for leg in legs):
            raise ValueError(
                f"gps_leg_windows_h must be [[start_h, end_h], ...]; got {legs}"
            )
        keep &= leg_mask(GPS_data, legs)
    if not keep.any():
        raise ValueError("no GPS epochs left after the time selection of the run")
    return GPS_data[keep], GPS_Coordinates[keep]


def setup_bermuda(config):
    """Load the Bermuda data and build the sampler model and fixed pairs."""
    inputs = load_bermuda_inputs(config)
    GPS_data, GPS_Coordinates = select_epochs(
        config, inputs["GPS_data"], inputs["GPS_Coordinates"]
    )
    problem = build_problem(
        config,
        GPS_Coordinates,
        GPS_data,
        inputs["CDOG_all_data"],
        inputs["dz_array"],
        inputs["angle_array"],
        inputs["esv_matrix"],
        input_files=inputs["input_files"],
    )
    if "esv_window_edges_s" in problem:
        # The windows depend on the pairs; keep them in the saved configuration.
        config = {**config, "esv_window_edges_s": problem["esv_window_edges_s"]}
    return build_model(config), problem


def chain_output_path(config, chain_index):
    """``Output/MCMC_Chains/<run_name>/chain_<NN>.npz``."""
    return gps_output_path(
        "MCMC_Chains", config["run_name"], f"chain_{chain_index:02d}.npz"
    )


def run_bermuda_chain(chain_index, config):
    """Run (or resume) chain `chain_index` of the configured Bermuda run."""
    model, problem = setup_bermuda(config)
    return run_chain(
        chain_index, model, problem, chain_output_path(config, chain_index)
    )


def warm_up_numba(config):
    """Compile and cache the Numba kernels once, before parallel chains start."""
    # The splits-model kernels, whatever the esv_model of the run.
    config = {**config, "esv_model": "splits"}
    model, problem = setup_bermuda({**config, "likelihood": "ar1", "clock_drift": True})
    log_posterior(model["start_theta"], model, problem)
    white = build_model({**config, **AR1_WHITE})
    log_posterior(white["start_theta"], white, problem)
    sigma = np.ones(white["num_dogs"])
    whiten_residuals(
        np.ones((problem["pair_dog"].shape[0], 2)), problem, sigma, 0.5 * sigma, sigma
    )
    fixed = build_model({**config, "likelihood": "iid_fixed", "clock_drift": False})
    log_posterior(fixed["start_theta"], fixed, problem)
    common, common_problem = setup_bermuda(
        {**config, "likelihood": "ar1", **COMMON_WINDOWS}
    )
    log_posterior(common["start_theta"], common, common_problem)
    print(
        "Numba kernels compiled; NUMBA_NUM_THREADS =",
        numba.config.NUMBA_NUM_THREADS,
        flush=True,
    )


if __name__ == "__main__":
    # One chain per process. The Slurm array index sets the chain's seed, start
    # point and output file; MCMC_RUN picks a variant from RUN_VARIANTS.
    run_name = os.environ.get("MCMC_RUN", "ar1")
    chain_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    run_bermuda_chain(chain_index, run_config(run_name))
