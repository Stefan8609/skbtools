"""Ensembles of realistic synthetic surveys, inverted with the paper's Gauss-Newton.

Each realization generates DOGs 1, 3 and 4 from one seed (one ocean, one ship, three
clocks; see ``realistic_synthetic``) and inverts each DOG with the paper's model
(position, time bias, one constant dc, optionally a clock drift):

- ``"lever_fixed"``: lever arm at its true value (an externally measured lever),
  ``initial_bias_geiger`` on every ``DOWNSAMPLE``-th epoch, then
  ``final_bias_geiger`` on all epochs.
- ``"lever_free"``: ``gauss_newton_offset_search`` (integer second and lever arm,
  started ``LEVER_START_SD_M`` from the truth) on every ``DOWNSAMPLE``-th epoch, then
  ``final_bias_geiger`` on all epochs at that lever arm.

Per realization and DOG it records the position error (ENU, cm), the formal sd of the
paper's least-squares model (white residuals, and inflated for AR(1) residuals), the
clock and dc estimates, and residual statistics. One npz per realization goes to
``OUTPUT_DIR/<experiment>/`` and one CSV row per realization and DOG to
``OUTPUT_DIR/<experiment>.csv``. Every realization is reproducible from (seed, knobs).

Experiments are defined in ``experiment_cells``; set ``EXPERIMENT`` and run the file.
"""

import csv
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio

from acoustics.synthetic_esv_tables import ESV_GENERATE
from bermuda_data import PROJECT_ROOT, gps_data_path
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    gauss_newton_offset_search,
    initial_bias_geiger,
)
from Inversion_Workflow.Synthetic.Realistic_Synthetic import (
    GPS1_TO_OTHERS,
    M2_PERIOD_H,
    enu_rotation,
    realistic_synthetic,
)

# -------------------------
# Config
# -------------------------
EXPERIMENT = "sanity"
N_REALIZATIONS = 10
FIRST_SEED = 0
N_WORKERS = 10
DOGS = (1, 3, 4)
# Generation and inversion use the same ESV table, so every error comes from a knob.
GENERATION_TABLE = ESV_GENERATE
INVERSION_TABLE = ESV_GENERATE
DOWNSAMPLE = 50
# The lever-free offset search needs more epochs: at 50 its integer-second
# confidence test (delta chi^2 >= 50, white-noise formula) fails for about a quarter
# of the "random" preset DOGs; at 10 it resolves them.
OFFSET_SEARCH_DOWNSAMPLE = 10
# A fit that keeps fewer pairs than this has diverged; it is recorded as failed.
MIN_PAIRS = 100
START_POSITION_SD_M = 20.0
LEVER_START_SD_M = 0.3
OUTPUT_DIR = PROJECT_ROOT / "revision/analysis/realistic_synthetic"
RANGE_SOUND_SPEED_MPS = 1515.0

# Realistic amplitudes. Basis in the comments; "guess" values are stated in the
# results document as assumptions. Directions of the current and the gradient are
# drawn per realization (``with_random_directions``).
CURRENT_SPEED_MPS = 0.03  # depth-mean current; guess (mesoscale + barotropic)
DC_GRADIENT_MPS_PER_KM = 0.005  # guess: thermocline slope 0.5 m/km x 10 mm/s per m
COMMON = {
    "current_m2_mps": 0.02,  # barotropic M2 tidal current; guess
    "offset_s": 1991.01236648,
    "drift_rate_s_per_s": 1e-9,  # R3 M1: CSAC frequency offsets of 1e-10 to 1e-9
    "transmit_delay_s": 30e-6,  # R3 M1: "tens of microseconds"
    "gnss_common_sd_m": (0.02, 0.02, 0.05),  # kinematic PPP; guess
    "gnss_common_tau_s": 1800.0,
    "timing_noise_s": 20e-6,  # paper, receive side
    "dropouts": "real_pattern",
}
PRESETS = {
    # dc(t) shaped like our own estimate. That estimate already contains vessel-side
    # GNSS error (K7), so the relative GNSS error is only iid antenna noise here.
    "observed": {
        **COMMON,
        "dc_series": "observed",
        "position_noise_m": 0.02,
    },
    # dc(t) as a random walk with the observed increment size (0.044 m/s per
    # sqrt(hour) at 30-min windows) plus a 10 mm/s M2 sinusoid (the M2 level of the
    # observed series), a lateral gradient, and the real antenna misfit.
    "random": {
        **COMMON,
        "dc_series": "random_walk",
        "dc_random_walk_mps_per_sqrt_h": 0.044,
        "dc_sinusoid_mps": 0.010,
        "dc_sinusoid_period_h": M2_PERIOD_H,
        "gnss_relative": "real_misfit_shifted",
    },
}
# Knob groups for the error budget: switched on alone, or off from the full preset.
BUDGET_GROUPS = {
    "dc_series": ("dc_series", "dc_random_walk_mps_per_sqrt_h"),
    "dc_sinusoid": ("dc_sinusoid_mps",),
    "dc_gradient": ("dc_gradient_mps_per_km",),
    "current_steady": ("current_mps",),
    "current_m2": ("current_m2_mps",),
    "clock_drift": ("drift_rate_s_per_s",),
    "transmit_delay": ("transmit_delay_s",),
    "gnss_common": ("gnss_common_sd_m",),
    "gnss_relative": ("gnss_relative", "position_noise_m"),
    "timing_noise": ("timing_noise_s",),
    "dropouts": ("dropouts",),
}


def with_random_directions(knobs, seed):
    """Point the steady current and the dc gradient in directions drawn from seed."""
    rng = np.random.default_rng([seed, 1000])
    current_dir, gradient_dir = rng.uniform(0.0, 2 * np.pi, 2)
    knobs = dict(knobs)
    if knobs.pop("random_current", False):
        knobs["current_mps"] = tuple(
            CURRENT_SPEED_MPS * np.array([np.sin(current_dir), np.cos(current_dir)])
        )
    if knobs.pop("random_gradient", False):
        knobs["dc_gradient_mps_per_km"] = tuple(
            DC_GRADIENT_MPS_PER_KM
            * np.array([np.sin(gradient_dir), np.cos(gradient_dir)])
        )
    return knobs


def full_preset(name):
    """Preset knobs plus random-direction current and gradient."""
    knobs = {**PRESETS[name], "random_current": True}
    if name == "random":
        knobs["random_gradient"] = True
    return knobs


def preset_group_knobs(preset, group):
    """Knobs of one budget group taken from the full preset (direction flags kept)."""
    full = full_preset(preset)
    knobs = {key: full[key] for key in BUDGET_GROUPS[group] if key in full}
    if group == "current_steady":
        knobs = {"random_current": True}
    if group == "dc_gradient":
        knobs = {"random_gradient": True} if preset == "random" else {}
    return knobs


def experiment_cells(experiment):
    """List of (cell name, knobs, inversion settings) for an experiment."""
    lever_fixed = {"lever": "lever_fixed", "clock_drift": False}
    cells = []
    if experiment == "isolation":
        # One knob at a time, fixed directions, so the size and the parameter it
        # leaks into can be read directly (Phase 2 check).
        single = {
            "dc_observed": {"dc_series": "observed"},
            "dc_random_walk": {
                "dc_series": "random_walk",
                "dc_random_walk_mps_per_sqrt_h": 0.044,
            },
            "dc_sinusoid_10mmps": {"dc_sinusoid_mps": 0.01},
            "dc_gradient_5mmps_per_km_E": {"dc_gradient_mps_per_km": (0.005, 0.0)},
            "current_3cmps_E": {"current_mps": (0.03, 0.0)},
            "current_3cmps_N": {"current_mps": (0.0, 0.03)},
            "current_m2_2cmps": {"current_m2_mps": 0.02},
            "drift_1ppb": {"drift_rate_s_per_s": 1e-9},
            "transmit_30us": {"transmit_delay_s": 30e-6},
            "gnss_common": {"gnss_common_sd_m": (0.02, 0.02, 0.05)},
            "gnss_real_misfit": {"gnss_relative": "real_misfit"},
            "gnss_iid_2cm": {"position_noise_m": 0.02},
            "timing_20us": {"timing_noise_s": 20e-6},
            "dropouts_real": {"dropouts": "real_pattern"},
        }
        for name, knobs in single.items():
            cells.append((name, knobs, lever_fixed))
        drift = {"lever": "lever_fixed", "clock_drift": True}
        cells.append(("drift_1ppb_drift_on", single["drift_1ppb"], drift))
        cells.append(("dc_observed_drift_on", single["dc_observed"], drift))
        cells.append(("none", {}, lever_fixed))
    elif experiment in ("sanity", "coverage"):
        for preset in PRESETS:
            for lever in ("lever_fixed", "lever_free"):
                for drift in (False, True):
                    if experiment == "sanity" and drift:
                        continue
                    name = f"{preset}_{lever}" + ("_drift" if drift else "")
                    settings = {"lever": lever, "clock_drift": drift}
                    cells.append((name, full_preset(preset), settings))
    elif experiment == "budget":
        for preset in PRESETS:
            cells.append((f"{preset}_all", full_preset(preset), lever_fixed))
            for group in BUDGET_GROUPS:
                alone = preset_group_knobs(preset, group)
                if not alone:
                    continue
                cells.append((f"{preset}_only_{group}", alone, lever_fixed))
                without = full_preset(preset)
                for key in BUDGET_GROUPS[group]:
                    without.pop(key, None)
                if group == "current_steady":
                    without.pop("random_current")
                if group == "dc_gradient":
                    without.pop("random_gradient")
                cells.append((f"{preset}_without_{group}", without, lever_fixed))
    elif experiment == "geometry":
        for track in ("bermuda", "circle", "grid"):
            for lost in (False, True):
                for lever in ("lever_fixed", "lever_free"):
                    knobs = {**full_preset("random"), "track": track}
                    knobs["lost_quadrant"] = lost
                    name = f"{track}{'_lost' if lost else ''}_{lever}"
                    cells.append((name, knobs, {"lever": lever, "clock_drift": False}))
    elif experiment == "halfsplit":
        # Each half of the survey inverted alone (lever free), as the real-data
        # half-split (R4 5): is the half-to-half spread a fair measure of the error?
        for preset in PRESETS:
            for half in (1, 2):
                settings = {"lever": "lever_free", "clock_drift": False, "half": half}
                cells.append((f"{preset}_half{half}", full_preset(preset), settings))
    elif experiment == "currents":
        for speed in (0.0, 0.02, 0.05, 0.10):
            for direction_deg in (0, 90, 180, 270):
                if speed == 0 and direction_deg:
                    continue
                angle = np.radians(direction_deg)
                current = (speed * np.sin(angle), speed * np.cos(angle))
                name = f"current_{speed * 100:.0f}cmps_{direction_deg:03d}deg"
                cells.append((name, {"current_mps": current}, lever_fixed))
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    return cells


def load_table(name):
    """ESV table arrays (dz, angle, matrix) from Bermuda_Data/ESV_Tables."""
    table = sio.loadmat(gps_data_path(f"ESV_Tables/{name}.mat"))
    return (
        np.ascontiguousarray(table["distance"].flatten()),
        np.ascontiguousarray(table["angle"].flatten()),
        np.ascontiguousarray(table["matrice"]),
    )


def invert(CDOG_data, GPS_Coordinates, GPS_data, start, lever_start, settings, tables):
    """Paper's Gauss-Newton inversion. Returns (estimate, offset, lever, pairs).

    ``estimate`` = [x, y, z, time_bias, esv_bias(, drift_rate)]; the Eq. 6 clock
    offset is ``offset - time_bias``. ``pairs`` is the output of
    ``final_bias_geiger`` after the estimate.
    """
    drift = settings["clock_drift"]
    ds = slice(None, None, DOWNSAMPLE)
    if settings["lever"] == "lever_fixed":
        lever = lever_start
        transponder = findTransponder(GPS_Coordinates[ds], GPS1_TO_OTHERS, lever)
        estimate, offset = initial_bias_geiger(
            start,
            CDOG_data,
            GPS_data[ds],
            transponder,
            *tables,
            real_data=True,
            clock_drift=drift,
        )
    else:
        ds = slice(None, None, OFFSET_SEARCH_DOWNSAMPLE)
        lever, offset, estimate, _ = gauss_newton_offset_search(
            CDOG_data,
            GPS_data[ds],
            GPS_Coordinates[ds],
            GPS1_TO_OTHERS,
            start,
            lever_start,
            *tables,
            real_data=True,
            clock_drift=drift,
        )
    transponder = findTransponder(GPS_Coordinates, GPS1_TO_OTHERS, lever)
    result = final_bias_geiger(
        estimate[:3],
        CDOG_data,
        GPS_data,
        transponder,
        offset,
        estimate[4],
        estimate[3],
        *tables,
        real_data=True,
        clock_drift=drift,
        drift_rate=estimate[5] if drift else 0.0,
    )
    return result[0], offset, lever, result[1:], transponder


def residual_summary(estimate, pairs, transponder, GPS_data, clock_drift):
    """Residual statistics and the formal ENU sd of the paper's model, cm.

    Formal covariance sigma^2 (J^T J)^-1 with columns E, N, U, dt, dc (and drift);
    the AR(1) version multiplies it by (1 + phi) / (1 - phi), phi the lag-1
    autocorrelation of residuals of pairs 1 s apart.
    """
    CDOG_full, model_times, _, GPS_clock = pairs
    residual = model_times - CDOG_full
    emission = np.round(GPS_clock - model_times)
    epoch = np.clip(np.searchsorted(GPS_data, emission), 0, len(GPS_data) - 1)
    to_enu = enu_rotation()
    line = (transponder[epoch] - estimate[:3]) @ to_enu.T
    slant = np.linalg.norm(line, axis=1)
    c_eff = slant / model_times
    columns = [-line / slant[:, None] / c_eff[:, None], np.ones((len(slant), 1))]
    columns.append((-model_times / c_eff)[:, None])
    if clock_drift:
        t = GPS_data[epoch]
        columns.append(((t - 0.5 * (t.min() + t.max())) / 1e4)[:, None])
    J = np.hstack(columns)
    sigma = residual.std()
    # pinv: on a circle centred near a DOG the slant range barely changes, so the
    # dt and dc columns are collinear; the condition number records it.
    scale = np.linalg.norm(J, axis=0)
    normal = (J / scale).T @ (J / scale)
    cov = sigma**2 * np.linalg.pinv(normal) / np.outer(scale, scale)
    consecutive = np.diff(GPS_data[epoch]) == 1
    phi = np.corrcoef(residual[:-1][consecutive], residual[1:][consecutive])[0, 1]
    sd_white = np.sqrt(np.diag(cov)[:3]) * 100
    elevation = np.degrees(np.arcsin(line[:, 2] / slant))
    hours = (GPS_data[epoch] - GPS_data[0]) / 3600.0
    residual_cm = residual * RANGE_SOUND_SPEED_MPS * 100
    time_bins = np.arange(0.0, 15.0, 0.25)
    slant_bins = np.arange(5000.0, 13500.0, 500.0)
    return {
        "condition_number": float(np.linalg.cond(normal)),
        "residual_rms_cm": float(residual_cm.std()),
        "residual_phi_1s": float(phi),
        "n_pairs": int(len(residual)),
        "sd_white_cm": sd_white,
        "sd_ar1_cm": sd_white * np.sqrt((1 + phi) / (1 - phi)),
        "elevation_min_deg": float(elevation.min()),
        "residual_by_time_cm": binned_mean(hours, residual_cm, time_bins),
        "residual_by_slant_cm": binned_mean(slant, residual_cm, slant_bins),
    }


def binned_mean(x, y, edges):
    """Mean of y in bins of x (nan where empty)."""
    index = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        if np.any(index == b):
            out[b] = y[index == b].mean()
    return out


def run_realization(task):
    """Generate one survey (three DOGs) and invert each DOG; return CSV rows."""
    experiment, cell, knobs, settings, seed = task
    tables_gen = load_table(GENERATION_TABLE)
    tables_inv = load_table(INVERSION_TABLE)
    knobs = with_random_directions(knobs, seed)
    to_enu = enu_rotation()
    rows, arrays = [], {}
    for DOG_num in DOGS:
        CDOG_data, CDOG, GPS_Coordinates, GPS_data, _, truth = realistic_synthetic(
            DOG_num, knobs, seed, *tables_gen
        )
        rng = np.random.default_rng([seed, DOG_num, 2000])
        start = CDOG + rng.normal(0.0, START_POSITION_SD_M, 3)
        lever_start = truth["lever"].copy()
        if settings["lever"] == "lever_free":
            lever_start += rng.normal(0.0, LEVER_START_SD_M, 3)
        row = {"experiment": experiment, "cell": cell, "seed": seed, "DOG": DOG_num}
        if settings.get("half"):
            # Split at the median GPS epoch; the DOG still hears every ping.
            first = GPS_data < np.median(GPS_data)
            half = first if settings["half"] == 1 else ~first
            GPS_Coordinates, GPS_data = GPS_Coordinates[half], GPS_data[half]
        try:
            estimate, offset, lever, pairs, transponder = invert(
                CDOG_data, GPS_Coordinates, GPS_data, start, lever_start, settings,
                tables_inv,
            )  # fmt: skip
            if len(pairs[0]) < MIN_PAIRS:
                raise ValueError(f"{len(pairs[0])} pairs after the final fit")
            summary = residual_summary(
                estimate, pairs, transponder, GPS_data, settings["clock_drift"]
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            row["failed"] = str(error)[:80]
            rows.append(row)
            continue
        error_enu_cm = to_enu @ (estimate[:3] - CDOG) * 100
        row.update(
            {
                "failed": "",
                "E_err_cm": error_enu_cm[0],
                "N_err_cm": error_enu_cm[1],
                "U_err_cm": error_enu_cm[2],
                "E_sd_white_cm": summary["sd_white_cm"][0],
                "N_sd_white_cm": summary["sd_white_cm"][1],
                "U_sd_white_cm": summary["sd_white_cm"][2],
                "E_sd_ar1_cm": summary["sd_ar1_cm"][0],
                "N_sd_ar1_cm": summary["sd_ar1_cm"][1],
                "U_sd_ar1_cm": summary["sd_ar1_cm"][2],
                "dt_err_us": (offset - estimate[3] - truth["dt_s"]) * 1e6,
                "wrong_second": int(np.round(offset - estimate[3] - truth["dt_s"])),
                "dc_mps": estimate[4],
                "drift_ppb": estimate[5] * 1e9 if settings["clock_drift"] else np.nan,
                "lever_x_err_cm": (lever[0] - truth["lever"][0]) * 100,
                "lever_y_err_cm": (lever[1] - truth["lever"][1]) * 100,
                "lever_z_err_cm": (lever[2] - truth["lever"][2]) * 100,
                "residual_rms_cm": summary["residual_rms_cm"],
                "residual_phi_1s": summary["residual_phi_1s"],
                "n_pairs": summary["n_pairs"],
                "elevation_min_deg": summary["elevation_min_deg"],
                "condition_number": summary["condition_number"],
            }
        )
        for name, delay in truth["delay_s"].items():
            row[f"delay_rms_us_{name}"] = float(np.sqrt(np.mean(delay**2)) * 1e6)
        rows.append(row)
        arrays[f"DOG{DOG_num}_residual_by_time_cm"] = summary["residual_by_time_cm"]
        arrays[f"DOG{DOG_num}_residual_by_slant_cm"] = summary["residual_by_slant_cm"]
        arrays[f"DOG{DOG_num}_dc_mps"] = truth["dc_mps"][::60]
    out = OUTPUT_DIR / experiment
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{cell}_seed{seed:04d}.npz",
        seed=seed,
        knobs=repr(knobs),
        settings=repr(settings),
        git_commit=GIT_COMMIT,
        **arrays,
    )
    return rows


def git_commit():
    """Short commit of the repository (with "-dirty" if files are modified)."""
    run = {"cwd": PROJECT_ROOT, "capture_output": True, "text": True}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **run).stdout
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], **run).stdout
    return commit.strip() + ("-dirty" if dirty.strip() else "")


GIT_COMMIT = git_commit()


def run_experiment(experiment, n_realizations, first_seed=0, n_workers=N_WORKERS):
    """Run every cell of an experiment over seeds; write the CSV; return the rows."""
    tasks = [
        (experiment, cell, knobs, settings, seed)
        for cell, knobs, settings in experiment_cells(experiment)
        for seed in range(first_seed, first_seed + n_realizations)
    ]
    with Pool(n_workers) as pool:
        rows = [row for result in pool.imap(run_realization, tasks) for row in result]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    suffix = f"_seed{first_seed:04d}" if first_seed else ""
    path = OUTPUT_DIR / f"{experiment}{suffix}.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {path}")
    return rows


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    # slurm/realistic_synthetic_array.sh sets these; locally the config above is used.
    experiment = os.environ.get("RS_EXPERIMENT", EXPERIMENT)
    n_realizations = int(os.environ.get("RS_SEEDS_PER_TASK", N_REALIZATIONS))
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    first_seed = FIRST_SEED + task * n_realizations
    n_workers = int(os.environ.get("RS_WORKERS", N_WORKERS))
    run_experiment(experiment, n_realizations, first_seed, n_workers)
