"""MCMC of the final model on realistic synthetic surveys: accuracy and coverage.

Each seed generates one survey of DOGs 1, 3 and 4 with ``realistic_synthetic`` (the
``PRESET`` of ``Realistic_Synthetic_Ensemble``) and samples it with the final real-data
model (``RUN_VARIANT`` of ``MCMC_sampler.RUN_VARIANTS``, joint lever arm, common dc(t)
windows plus per-DOG offsets, AR(1) noise) using the sampler's own priors, pairing and
outlier rule. Only what the real data supply from outside the sampler is replaced:

- ESV table: ``INVERSION_TABLE`` of the ensemble (the generation table).
- DOG prior centres and start: truth + N(0, ``PRIOR_CENTER_ERROR_SD_M``) per axis, as
  the real prior centres are Gauss-Newton guesses within metres of the solution.
- Alignment offsets: the true clock offset rounded to ``OFFSET_ROUND_S`` (the dt prior,
  10 ms sd, is centred there).
- GNSS: DOG 1's vessel positions for all DOGs. In the "random" preset the three DOGs
  share them exactly; in "observed" the iid antenna noise differs per DOG.

``run_synthetic_chain`` writes one chain per (seed, chain) to
``OUTPUT_DIR/<run>/seed<seed>_chain<chain>.npz`` (resumable, as on real data) and the
truth to ``seed<seed>_truth.npz``. ``summarize`` pools the chains of every seed and
writes ``OUTPUT_DIR/<run>.csv``: per seed and DOG, the ENU error of the posterior mean,
the posterior sd, whether the central 68 % and 95 % intervals hold the truth, the same
for the lever arm and dt, the dc error, and the max R-hat.
"""

import csv
import json
import os

import numpy as np

from Inversion_Workflow.MCMC_Functions.MCMC_diagnostics import (
    rank_normalized_split_rhat,
)
from Inversion_Workflow.MCMC_Functions.MCMC_sampler import (
    build_model,
    build_problem,
    esv_bias_matrix,
    load_chain,
    run_chain,
    run_config,
)
from Inversion_Workflow.Synthetic.Realistic_Synthetic import (
    GPS1_TO_OTHERS,
    SITE_ORIGIN_ECEF,
    enu_rotation,
    realistic_synthetic,
)
from Inversion_Workflow.Synthetic.Realistic_Synthetic_Ensemble import (
    GENERATION_TABLE,
    INVERSION_TABLE,
    OUTPUT_DIR,
    full_preset,
    load_table,
    with_random_directions,
)

# -------------------------
# Config
# -------------------------
PRESET = "random"
RUN_VARIANT = "ar1_common30"
N_SURVEYS = 20
N_CHAINS = 2
FIRST_SEED = 0
# Sampler overrides, e.g. {"n_burn": 2000, "n_samples": 2000} for a quick check.
OVERRIDES = {}
DOGS = (1, 3, 4)
PRIOR_CENTER_ERROR_SD_M = 1.0
OFFSET_ROUND_S = 1e-3


def run_name():
    return f"mcmc_{RUN_VARIANT}_{PRESET}"


def synthetic_problem(seed):
    """Sampler model, problem and truth for the survey of one seed."""
    knobs = with_random_directions(full_preset(PRESET), seed)
    tables_gen = load_table(GENERATION_TABLE)
    tables_inv = load_table(INVERSION_TABLE)
    surveys = [realistic_synthetic(dog, knobs, seed, *tables_gen) for dog in DOGS]
    _, _, GPS_Coordinates, GPS_data, _, _ = surveys[0]
    truths = [survey[5] for survey in surveys]

    true_aug = np.array([survey[1] - SITE_ORIGIN_ECEF for survey in surveys])
    rng = np.random.default_rng([seed, 3000])
    prior_aug = true_aug + rng.normal(0.0, PRIOR_CENTER_ERROR_SD_M, true_aug.shape)
    dt_true = np.array([truth["dt_s"] for truth in truths])
    config = run_config(
        RUN_VARIANT,
        {
            "run_name": run_name(),
            "dogs": list(DOGS),
            "esv_table": f"ESV_Tables/{INVERSION_TABLE}.mat",
            "gps_file": f"synthetic seed {seed}, preset {PRESET}",
            "CDOG_reference": SITE_ORIGIN_ECEF.tolist(),
            "gps1_to_others": GPS1_TO_OTHERS.tolist(),
            "offsets_s": (np.round(dt_true / OFFSET_ROUND_S) * OFFSET_ROUND_S).tolist(),
            "start_CDOG_aug": prior_aug.tolist(),
            "prior_CDOG_aug_center_m": prior_aug.tolist(),
            "start_esv_bias": [0.0] * len(DOGS),
            "seed": int(20261119 + 1000 * seed),
            **OVERRIDES,
        },
    )
    problem = build_problem(
        config,
        GPS_Coordinates,
        GPS_data,
        [survey[0] for survey in surveys],
        *tables_inv,
    )
    if "esv_window_edges_s" in problem:
        config = {**config, "esv_window_edges_s": problem["esv_window_edges_s"]}

    # dc truth per DOG and window: mean of the true dc(t) over that DOG's epochs.
    edges = np.asarray(problem.get("esv_window_edges_s", [-np.inf, np.inf]))
    true_dc = np.full((len(DOGS), len(edges) - 1), np.nan)
    for j, truth in enumerate(truths):
        t = truth["GPS_data_all"][truth["keep"]]
        dc = np.broadcast_to(truth["dc_mps"], truth["GPS_data_all"].shape)[
            truth["keep"]
        ]
        window = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, None)
        for k in range(len(edges) - 1):
            if np.any(window == k):
                true_dc[j, k] = dc[window == k].mean()
    truth = {
        "seed": seed,
        "knobs": repr(knobs),
        "CDOG_aug": true_aug,
        "lever": truths[0]["lever"],
        "dt_s": dt_true,
        "drift_rate_s_per_s": truths[0]["drift_rate_s_per_s"],
        "dc_window_mps": true_dc,
        "prior_CDOG_aug_center_m": prior_aug,
    }
    return build_model(config), problem, truth


def seed_paths(seed):
    folder = OUTPUT_DIR / run_name()
    return folder / f"seed{seed:04d}_truth.npz", [
        folder / f"seed{seed:04d}_chain{c:02d}.npz" for c in range(N_CHAINS)
    ]


def run_synthetic_chain(seed, chain_index):
    """Run (or resume) one chain on the survey of ``seed``; save the truth once."""
    model, problem, truth = synthetic_problem(seed)
    truth_path, chain_paths = seed_paths(seed)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    if not truth_path.exists():
        np.savez(truth_path, **truth)
    return run_chain(chain_index, model, problem, chain_paths[chain_index])


def interval_flags(draws, true_value):
    """Whether the central 68 % and 95 % intervals of ``draws`` hold the truth."""
    lo68, hi68, lo95, hi95 = np.percentile(draws, [16, 84, 2.5, 97.5], axis=0)
    return (
        (lo68 <= true_value) & (true_value <= hi68),
        (lo95 <= true_value) & (true_value <= hi95),
    )


def summarize(seeds):
    """One CSV row per seed and DOG from every finished seed; return the rows."""
    to_enu = enu_rotation()
    rows = []
    for seed in seeds:
        truth_path, chain_paths = seed_paths(seed)
        if not truth_path.exists() or not all(p.exists() for p in chain_paths):
            continue
        chains = [load_chain(path) for path in chain_paths]
        if any(int(c["n_done"]) < int(c["n_total"]) for c in chains):
            print(f"seed {seed}: chains unfinished, skipped")
            continue
        truth = np.load(truth_path)
        n = min(len(c["theta_samples"]) for c in chains)
        rhat = max(
            rank_normalized_split_rhat(
                np.stack([c["theta_samples"][:n, i] for c in chains])
            )
            for i in range(chains[0]["theta_samples"].shape[1])
        )
        model = build_model(json.loads(str(chains[0]["config_json"])))
        theta = np.concatenate([c["theta_samples"] for c in chains])
        aug = np.concatenate([c["CDOG_aug"] for c in chains])
        lever = np.concatenate([c["lever"] for c in chains])
        dt = np.concatenate([c["dt"] for c in chains])
        dc = esv_bias_matrix(theta, model)

        lever_in68, lever_in95 = interval_flags(lever, truth["lever"])
        for j, dog in enumerate(DOGS):
            enu = (aug[:, j] - truth["CDOG_aug"][j]) @ to_enu.T * 100
            in68, in95 = interval_flags(enu, 0.0)
            dt_us = (dt[:, j] - truth["dt_s"][j]) * 1e6
            dt_in68, dt_in95 = interval_flags(dt_us, 0.0)
            prior_enu = (
                to_enu @ (truth["prior_CDOG_aug_center_m"][j] - truth["CDOG_aug"][j])
            ) * 100
            row = {"seed": seed, "DOG": dog, "max_rhat": rhat, "n_draws": len(theta)}
            for a, axis in enumerate("ENU"):
                row[f"{axis}_err_cm"] = enu[:, a].mean()
                row[f"{axis}_sd_cm"] = enu[:, a].std()
                row[f"{axis}_in68"] = int(in68[a])
                row[f"{axis}_in95"] = int(in95[a])
                row[f"{axis}_prior_center_err_cm"] = prior_enu[a]
            for a, axis in enumerate("xyz"):
                row[f"lever_{axis}_err_cm"] = (
                    lever[:, a].mean() - truth["lever"][a]
                ) * 100
                row[f"lever_{axis}_sd_cm"] = lever[:, a].std() * 100
                row[f"lever_{axis}_in95"] = int(lever_in95[a])
            row["dt_err_us"] = dt_us.mean()
            row["dt_sd_us"] = dt_us.std()
            row["dt_in95"] = int(dt_in95)
            dc_err = dc[:, j].mean(axis=0) - truth["dc_window_mps"][j]
            row["dc_mean_err_mps"] = np.nanmean(dc_err)
            row["dc_window_rms_err_mps"] = np.sqrt(np.nanmean(dc_err**2))
            rows.append(row)
    if not rows:
        print("no finished seeds")
        return rows
    path = OUTPUT_DIR / f"{run_name()}.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for axis in "ENU":
        err = np.array([r[f"{axis}_err_cm"] for r in rows])
        sd = np.array([r[f"{axis}_sd_cm"] for r in rows])
        in95 = np.mean([r[f"{axis}_in95"] for r in rows])
        print(
            f"{axis}: rms error {np.sqrt(np.mean(err**2)):.1f} cm, median sd "
            f"{np.median(sd):.1f} cm, 95 % coverage {100 * in95:.0f} %"
        )
    print(f"{len(rows)} rows -> {path}")
    return rows


if __name__ == "__main__":
    # slurm/realistic_synthetic_mcmc_array.sh sets these. Array task i runs chain
    # i % N_CHAINS of seed FIRST_SEED + i // N_CHAINS; SYN_MCMC_SUMMARY=1 summarizes.
    seeds = range(FIRST_SEED, FIRST_SEED + N_SURVEYS)
    if os.environ.get("SYN_MCMC_SUMMARY") == "1":
        summarize(seeds)
    else:
        task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        run_synthetic_chain(FIRST_SEED + task // N_CHAINS, task % N_CHAINS)
