"""Run a few Bermuda MCMC chains in parallel on one machine, before using Slurm.

Each chain is its own process, with Numba and BLAS limited to one thread as on
the cluster. Rerunning resumes every chain from its last checkpoint.
"""

import os

for _variable in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from Inversion_Workflow.MCMC_Functions.MCMC_sampler import (  # noqa: E402
    chain_output_path,
    run_bermuda_chain,
    run_config,
    warm_up_numba,
)


def run_one(config, chain_index):
    """Run one chain; return its name, index and wall time in seconds."""
    start = time.time()
    run_bermuda_chain(chain_index, config)
    return config["run_name"], chain_index, time.time() - start


if __name__ == "__main__":
    VARIANTS = ["iid", "ar1", "ar1_drift"]
    CHAIN_INDICES = [0, 1]
    N_PROCESSES = 4
    # Short runs, saved apart from cluster runs: Output/MCMC_Chains/<variant>_local.
    OVERRIDES = {"n_burn": 10000, "n_samples": 10000}

    configs = [
        run_config(variant, {**OVERRIDES, "run_name": f"{variant}_local"})
        for variant in VARIANTS
    ]
    warm_up_numba(configs[0])
    jobs = [(config, index) for config in configs for index in CHAIN_INDICES]
    with Pool(N_PROCESSES) as pool:
        results = pool.starmap(run_one, jobs)

    print("\nrun              chain  s/1000 it  burn-in windows  sampling acceptance")
    for (config, index), (name, _, seconds) in zip(jobs, results):
        with np.load(chain_output_path(config, index)) as chain:
            n_total = int(chain["n_total"])
            windows = np.round(chain["window_acceptance"], 2).tolist()
            acceptance = float(chain["acceptance_rate_sampling"])
        print(
            f"{name:16s} {index:5d}  {1000 * seconds / n_total:9.1f}  "
            f"{str(windows):15s}  {acceptance:.3f}"
        )
