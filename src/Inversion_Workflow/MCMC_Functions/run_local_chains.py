"""Run a few Bermuda MCMC chains in parallel on one machine, before using Slurm.

Edit the settings below. Each run starts from a variant of RUN_VARIANTS in
MCMC_sampler.py; a setting left at None keeps that variant's value, and the
effective configuration is printed. Each chain is its own process, with Numba and
BLAS limited to one thread as on the cluster. Rerunning resumes every chain from
its last checkpoint.
"""

import os

# Runs
VARIANTS = ["ar1_common30"]  # one run per variant of RUN_VARIANTS (MCMC_sampler.py)
RUN_SUFFIX = "_local"  # output Output/MCMC_Chains/<variant><suffix>, apart from Slurm
CHAIN_INDICES = [0, 1]  # seed, start point and file of each chain
N_PROCESSES = 4  # chains run at the same time

# Model (None keeps the variant's value)
LIKELIHOOD = None  # "iid_fixed", "iid" or "ar1"; AR(1) + white: an ar1_white variant
CLOCK_DRIFT = None  # linear clock drift per DOG
DOGS = None  # e.g. [1, 3, 4]; one DOG gets its own lever arm
NUM_ESV_SPLITS = None  # "splits" models: dc blocks per DOG
ESV_WINDOW_S = None  # "common_windows" models: length of the dc windows, s

# Priors (None keeps the variant's value)
PRIOR_LEVER_CENTER_M = None  # body frame, m from GPS1
PRIOR_LEVER_SD_M = None
PRIOR_CDOG_AUG_SD_M = None  # DOG positions about the Gauss-Newton starting guesses
PRIOR_ESV_BIAS_SD_M_S = None
PRIOR_DT_SD_S = None

# Chain lengths, shorter than the cluster runs
N_BURN = 10000  # adaptive burn-in iterations
N_SAMPLES = 10000  # iterations kept after burn-in, before thinning
THIN = None  # keep every n-th iteration

for _variable in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from Inversion_Workflow.MCMC_Functions.MCMC_sampler import (  # noqa: E402
    chain_output_path,
    run_bermuda_chain,
    run_config,
    select_dogs,
    warm_up_numba,
)


def run_one(config, chain_index):
    """Run one chain; return its name, index and wall time in seconds."""
    start = time.time()
    run_bermuda_chain(chain_index, config)
    return config["run_name"], chain_index, time.time() - start


if __name__ == "__main__":
    settings = {
        "likelihood": LIKELIHOOD,
        "clock_drift": CLOCK_DRIFT,
        "num_esv_splits": NUM_ESV_SPLITS,
        "esv_window_s": ESV_WINDOW_S,
        "prior_lever_center_m": PRIOR_LEVER_CENTER_M,
        "prior_lever_sd_m": PRIOR_LEVER_SD_M,
        "prior_CDOG_aug_sd_m": PRIOR_CDOG_AUG_SD_M,
        "prior_esv_bias_sd_m_s": PRIOR_ESV_BIAS_SD_M_S,
        "prior_dt_sd_s": PRIOR_DT_SD_S,
        "n_burn": N_BURN,
        "n_samples": N_SAMPLES,
        "thin": THIN,
    }
    overrides = {key: value for key, value in settings.items() if value is not None}
    if DOGS is not None:
        overrides.update(select_dogs(DOGS))

    configs = [
        run_config(variant, {**overrides, "run_name": f"{variant}{RUN_SUFFIX}"})
        for variant in VARIANTS
    ]
    for config in configs:
        print(json.dumps(config, indent=1, sort_keys=True), flush=True)
    warm_up_numba(configs[0])
    jobs = [(config, index) for config in configs for index in CHAIN_INDICES]
    with Pool(N_PROCESSES) as pool:
        results = pool.starmap(run_one, jobs)

    print("\nrun              chain  s/1000 it  burn-in windows  sampling acceptance")
    for (config, index), (name, _, seconds) in zip(jobs, results, strict=False):
        with np.load(chain_output_path(config, index)) as chain:
            n_total = int(chain["n_total"])
            windows = np.round(chain["window_acceptance"], 2).tolist()
            acceptance = float(chain["acceptance_rate_sampling"])
        print(
            f"{name:16s} {index:5d}  {1000 * seconds / n_total:9.1f}  "
            f"{str(windows):15s}  {acceptance:.3f}"
        )
