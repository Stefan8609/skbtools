"""Ensembles of realistic synthetic surveys, inverted with the paper's Gauss-Newton.

Each realization generates DOGs 1, 3 and 4 from one seed (one ocean, one ship, three
clocks; see ``realistic_synthetic``) and inverts each DOG with the paper's model
(position, time bias, one constant dc, optionally a clock drift), the lever arm
either fixed at its true value or free (see ``invert``).

Per realization and DOG it records the position error (ENU, cm), the formal sd of
the least-squares model (white residuals, and inflated for AR(1) residuals), the
clock and dc estimates, and residual statistics: one CSV row in
``OUTPUT_DIR/<experiment>.csv`` and one npz per realization in
``OUTPUT_DIR/<experiment>/``. Every realization is reproducible from (seed, knobs).

Experiments are defined in ``experiment_cells``; set ``EXPERIMENT`` and run the file.
``summary_table`` condenses an experiment CSV to one row per cell and DOG.
"""

import csv
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from acoustics.synthetic_esv_tables import ESV_GENERATE
from bermuda_data import PROJECT_ROOT, gps_data_path
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    NOMINAL_SOUND_SPEED_MPS,
)
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    gauss_newton_offset_search,
    initial_bias_geiger,
)
from Inversion_Workflow.MCMC_Functions.MCMC_sampler import ALT_BLOCK_S, time_block_mask
from Inversion_Workflow.Synthetic.Realistic_Synthetic import (
    GPS1_TO_OTHERS,
    enu_rotation,
    realistic_synthetic,
)

# Config
EXPERIMENT = "coverage"
AMPLITUDE_SET = "guess"  # key of AMPLITUDES: "guess" | "models_low" | "models_high"
N_REALIZATIONS = 10
FIRST_SEED = 0
N_WORKERS = 10
DOGS = (1, 3, 4)
# Generation and inversion use the same ESV table, so every error comes from a knob.
GENERATION_TABLE = ESV_GENERATE
INVERSION_TABLE = ESV_GENERATE
DOWNSAMPLE = 50
# The lever-free offset search needs more epochs: at 50 its integer-second test
# fails for about a quarter of the "random" preset DOGs.
OFFSET_SEARCH_DOWNSAMPLE = 10
# A fit that keeps fewer pairs than this has diverged; it is recorded as failed.
MIN_PAIRS = 100
START_POSITION_SD_M = 20.0
LEVER_START_SD_M = 0.3
OUTPUT_DIR = PROJECT_ROOT / "revision/analysis/realistic_synthetic"

# Amplitudes of the three sources the acoustic data cannot constrain. "guess" is
# the original assumption; "models_low" and "models_high" bracket ocean models at
# the site on 16 June 2020:
# - depth-mean current: GLORYS12 3.0 cm/s, HYCOM GOFS 3.1 11 cm/s;
# - ESV gradient of a vertical ray, 0-4 km: GLORYS12 2.6 mm/s/km, HYCOM 8 mm/s/km;
# - M2 tidal current: HAMTIDE11a maximum speed 0.9 cm/s (its ellipse is
#   0.78 x 0.46 cm/s; the knob is circular).
AMPLITUDES = {
    "guess": {
        "current_speed_mps": 0.03,
        "gradient_mps_per_km": 0.005,
        "current_m2_mps": 0.02,
    },
    "models_low": {
        "current_speed_mps": 0.030,
        "gradient_mps_per_km": 0.0026,
        "current_m2_mps": 0.009,
    },
    "models_high": {
        "current_speed_mps": 0.11,
        "gradient_mps_per_km": 0.008,
        "current_m2_mps": 0.009,
    },
}
COMMON = {
    "offset_s": 1991.01236648,
    "drift_rate_s_per_s": 1e-9,  # CSAC frequency offsets are 1e-10 to 1e-9
    "transmit_delay_s": 30e-6,  # tens of microseconds
    "gnss_common_sd_m": (0.02, 0.02, 0.05),  # kinematic PPP; guess
    "gnss_common_tau_s": 1800.0,
    "timing_noise_s": 20e-6,  # paper, receive side
    "dropouts": "real_pattern",
}
PRESETS = {
    # dc(t) shaped like our own estimate, with iid antenna noise.
    "observed": {
        **COMMON,
        "dc_series": "observed",
        "position_noise_m": 0.02,
    },
    # dc(t) as a random walk with the increment size and M2 level of the observed
    # series, plus a lateral gradient and the real antenna misfit.
    "random": {
        **COMMON,
        "dc_series": "random_walk",
        "dc_random_walk_mps_per_sqrt_h": 0.022,
        "dc_sinusoid_mps": 0.010,
        "gnss_relative": "real_misfit_shifted",
    },
}
# Knob groups for the error budget: switched on alone, or off from the full preset.
# "gradient_mps_per_km" and "current_speed_mps" are sizes that
# ``with_random_directions`` turns into the knobs of the same source.
BUDGET_GROUPS = {
    "dc_series": ("dc_series", "dc_random_walk_mps_per_sqrt_h"),
    "dc_sinusoid": ("dc_sinusoid_mps",),
    "dc_gradient": ("gradient_mps_per_km",),
    "current_steady": ("current_speed_mps",),
    "current_m2": ("current_m2_mps",),
    "clock_drift": ("drift_rate_s_per_s",),
    "transmit_delay": ("transmit_delay_s",),
    "gnss_common": ("gnss_common_sd_m",),
    "gnss_relative": ("gnss_relative", "position_noise_m"),
    "timing_noise": ("timing_noise_s",),
    "dropouts": ("dropouts",),
}


def with_random_directions(knobs, seed):
    """Point a "current_speed_mps" and a "gradient_mps_per_km" in directions drawn
    from seed, as the ``current_mps`` and ``dc_gradient_mps_per_km`` knobs."""
    rng = np.random.default_rng([seed, 1000])
    current_dir, gradient_dir = rng.uniform(0.0, 2 * np.pi, 2)
    knobs = dict(knobs)
    if "current_speed_mps" in knobs:
        east_north = np.array([np.sin(current_dir), np.cos(current_dir)])
        knobs["current_mps"] = tuple(knobs.pop("current_speed_mps") * east_north)
    if "gradient_mps_per_km" in knobs:
        east_north = np.array([np.sin(gradient_dir), np.cos(gradient_dir)])
        size = knobs.pop("gradient_mps_per_km")
        knobs["dc_gradient_mps_per_km"] = tuple(size * east_north)
    return knobs


def full_preset(name, amplitudes="guess"):
    """Preset knobs plus the AMPLITUDES set (the gradient only for "random")."""
    knobs = {**PRESETS[name], **AMPLITUDES[amplitudes]}
    if name != "random":
        del knobs["gradient_mps_per_km"]
    return knobs


def preset_group_knobs(preset, group, amplitudes="guess"):
    """Knobs of one budget group, taken from the full preset."""
    full = full_preset(preset, amplitudes)
    return {key: full[key] for key in BUDGET_GROUPS[group] if key in full}


def experiment_cells(experiment, amplitudes="guess"):
    """List of (cell name, knobs, inversion settings) for an experiment.

    ``amplitudes`` picks the AMPLITUDES set of the preset-based experiments; a set
    other than "guess" is appended to every cell name. The "currents" sweep
    ignores it.
    """
    lever_fixed = {"lever": "lever_fixed", "clock_drift": False}
    cells = []
    if experiment == "coverage":
        for preset in PRESETS:
            for lever in ("lever_fixed", "lever_free"):
                for drift in (False, True):
                    name = f"{preset}_{lever}" + ("_drift" if drift else "")
                    settings = {"lever": lever, "clock_drift": drift}
                    cells.append((name, full_preset(preset, amplitudes), settings))
    elif experiment == "budget":
        for preset in PRESETS:
            knobs = full_preset(preset, amplitudes)
            cells.append((f"{preset}_all", knobs, lever_fixed))
            for group in BUDGET_GROUPS:
                alone = preset_group_knobs(preset, group, amplitudes)
                if not alone:
                    continue
                cells.append((f"{preset}_only_{group}", alone, lever_fixed))
                without = {key: v for key, v in knobs.items() if key not in alone}
                cells.append((f"{preset}_without_{group}", without, lever_fixed))
                # Is a drift recovered alone, and does dc(t) alone produce one?
                if group in ("clock_drift", "dc_series"):
                    drift = {"lever": "lever_fixed", "clock_drift": True}
                    cells.append((f"{preset}_only_{group}_drift", alone, drift))
    elif experiment == "geometry":
        for track in ("bermuda", "hourglass"):
            for lever in ("lever_fixed", "lever_free"):
                knobs = {**full_preset("random", amplitudes), "track": track}
                settings = {"lever": lever, "clock_drift": False}
                cells.append((f"{track}_{lever}", knobs, settings))
    elif experiment == "halfsplit":
        # Each half of the survey inverted alone, as the real-data half-split: is
        # the half-to-half spread a fair measure of the error?
        for preset in PRESETS:
            for half in (1, 2):
                settings = {"lever": "lever_free", "clock_drift": False, "half": half}
                knobs = full_preset(preset, amplitudes)
                cells.append((f"{preset}_half{half}", knobs, settings))
            # Alternating 15-min blocks, as the real-data ar1_common30_alt15a/b runs.
            for parity, name in ((0, "a"), (1, "b")):
                settings = {
                    "lever": "lever_free",
                    "clock_drift": False,
                    "alt_block": (ALT_BLOCK_S, parity),
                }
                knobs = full_preset(preset, amplitudes)
                cells.append((f"{preset}_alt15{name}", knobs, settings))
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
    if amplitudes != "guess":
        cells = [(f"{name}_{amplitudes}", knobs, st) for name, knobs, st in cells]
    return cells


def load_table(name):
    """ESV table arrays (dz, angle, matrix) from Bermuda_Data/ESV_Tables."""
    dz_array, angle_array, esv_matrix = load_esv_table(
        gps_data_path(f"ESV_Tables/{name}.mat")
    )
    return dz_array, angle_array, np.ascontiguousarray(esv_matrix)


def invert(CDOG_data, GPS_Coordinates, GPS_data, start, lever_start, settings, tables):
    """Paper's Gauss-Newton inversion on every ``DOWNSAMPLE``-th epoch, then
    ``final_bias_geiger`` on all epochs.

    ``"lever_fixed"`` keeps ``lever_start`` (an externally measured lever);
    ``"lever_free"`` runs ``gauss_newton_offset_search`` for the integer second and
    the lever arm. Returns ``(estimate, offset, lever, pairs, transponder)``:
    ``estimate`` = [x, y, z, time_bias, esv_bias(, drift_rate)], the Eq. 6 clock
    offset being ``offset - time_bias``; ``pairs`` is the rest of the output of
    ``final_bias_geiger``.
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
    # pinv: where the slant range barely changes, the dt and dc columns are collinear.
    scale = np.linalg.norm(J, axis=0)
    normal = (J / scale).T @ (J / scale)
    cov = sigma**2 * np.linalg.pinv(normal) / np.outer(scale, scale)
    consecutive = np.diff(GPS_data[epoch]) == 1
    phi = np.corrcoef(residual[:-1][consecutive], residual[1:][consecutive])[0, 1]
    sd_white = np.sqrt(np.diag(cov)[:3]) * 100
    elevation = np.degrees(np.arcsin(line[:, 2] / slant))
    hours = (GPS_data[epoch] - GPS_data[0]) / 3600.0
    residual_cm = residual * NOMINAL_SOUND_SPEED_MPS * 100
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
        if settings.get("alt_block"):
            block_s, parity = settings["alt_block"]
            keep = time_block_mask(GPS_data, block_s, parity)
            GPS_Coordinates, GPS_data = GPS_Coordinates[keep], GPS_data[keep]
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


def merge_task_csvs(experiment, amplitudes="guess"):
    """Join the per-task CSVs of a slurm array run into ``<name>_all.csv``.

    Reads ``<name>.csv`` (task 0) and every ``<name>_seed*.csv`` in ``OUTPUT_DIR``,
    ``<name>`` being the experiment plus the amplitude label. Raises if a (cell,
    seed, DOG) appears twice. Returns the path of the merged file.
    """
    name = experiment + ("" if amplitudes == "guess" else f"_{amplitudes}")
    paths = [OUTPUT_DIR / f"{name}.csv"] + sorted(OUTPUT_DIR.glob(f"{name}_seed*.csv"))
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    repeated = data.duplicated(["cell", "seed", "DOG"])
    if repeated.any():
        raise ValueError(f"{repeated.sum()} repeated (cell, seed, DOG) rows in {name}")
    merged = OUTPUT_DIR / f"{name}_all.csv"
    data.to_csv(merged, index=False)
    print(f"{len(paths)} files, {len(data)} rows -> {merged}")
    return merged


def summary_table(csv_path):
    """Bias, rms and 95 % coverage per cell and DOG of an experiment CSV, in cm.

    Uses the fits that succeeded and found the right clock second (``n`` of them;
    ``n_excluded`` others). Horizontal bias is the length of the mean E, N error,
    vertical bias the mean U error; rms is about zero (the truth), so it includes
    the bias. Coverage is the fraction of components (E and N, or U) whose error is
    within 1.96 AR(1)-inflated formal sd. Writes ``<name>_summary.csv`` next to the
    input and returns the table.
    """
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)
    good = (data["failed"].fillna("") == "") & (data["wrong_second"] == 0)
    rows = []
    for (cell, DOG_num), group in data.groupby(["cell", "DOG"], sort=False):
        fits = group[good[group.index]]
        horizontal = fits[["E_err_cm", "N_err_cm"]].to_numpy()
        horizontal_sd = fits[["E_sd_ar1_cm", "N_sd_ar1_cm"]].to_numpy()
        vertical = fits["U_err_cm"].to_numpy()
        rows.append(
            {
                "cell": cell,
                "DOG": DOG_num,
                "n": len(fits),
                "n_excluded": len(group) - len(fits),
                "horizontal_bias_cm": np.linalg.norm(horizontal.mean(axis=0)),
                "vertical_bias_cm": vertical.mean(),
                "horizontal_rms_cm": np.sqrt(np.mean(np.sum(horizontal**2, axis=1))),
                "vertical_rms_cm": np.sqrt(np.mean(vertical**2)),
                "horizontal_coverage_95": np.mean(
                    np.abs(horizontal) <= 1.96 * horizontal_sd
                ),
                "vertical_coverage_95": np.mean(
                    np.abs(vertical) <= 1.96 * fits["U_sd_ar1_cm"].to_numpy()
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(csv_path.with_name(f"{csv_path.stem}_summary.csv"), index=False)
    return table


def git_commit():
    """Short commit of the repository (with "-dirty" if files are modified)."""
    run = {"cwd": PROJECT_ROOT, "capture_output": True, "text": True}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **run).stdout
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], **run).stdout
    return commit.strip() + ("-dirty" if dirty.strip() else "")


GIT_COMMIT = git_commit()


def run_experiment(
    experiment, n_realizations, first_seed=0, n_workers=N_WORKERS, amplitudes="guess"
):
    """Run every cell of an experiment over seeds; write the CSV; return the rows."""
    tasks = [
        (experiment, cell, knobs, settings, seed)
        for cell, knobs, settings in experiment_cells(experiment, amplitudes)
        for seed in range(first_seed, first_seed + n_realizations)
    ]
    with Pool(n_workers) as pool:
        rows = [row for result in pool.imap(run_realization, tasks) for row in result]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    label = "" if amplitudes == "guess" else f"_{amplitudes}"
    suffix = f"_seed{first_seed:04d}" if first_seed else ""
    path = OUTPUT_DIR / f"{experiment}{label}{suffix}.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {path}")
    return rows


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # slurm/realistic_synthetic_array.sh sets these; locally the config above is used.
    experiment = os.environ.get("RS_EXPERIMENT", EXPERIMENT)
    n_realizations = int(os.environ.get("RS_SEEDS_PER_TASK", N_REALIZATIONS))
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    first_seed = FIRST_SEED + task * n_realizations
    n_workers = int(os.environ.get("RS_WORKERS", N_WORKERS))
    amplitudes = os.environ.get("RS_AMPLITUDES", AMPLITUDE_SET)
    run_experiment(experiment, n_realizations, first_seed, n_workers, amplitudes)
