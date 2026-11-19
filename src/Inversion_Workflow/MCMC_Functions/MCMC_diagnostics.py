"""Convergence diagnostics for a run of Bermuda MCMC chains.

Combines the chains of one run (Output/MCMC_Chains/<run_name>/chain_*.npz) and
reports, per parameter, the rank-normalized split R-hat and the bulk and tail
effective sample sizes of Vehtari et al. (2021, Bayesian Analysis 16, 667),
the acceptance rates, and trace and autocorrelation plots. Implemented in NumPy
and SciPy (arviz is not a dependency). Samples are the saved, thinned draws, so
ESS counts thinned draws.
"""

import csv
import json
import numpy as np
from scipy.stats import norm, rankdata

from Inversion_Workflow.MCMC_Functions.MCMC_sampler import load_chain
from bermuda_data import gps_output_path

RHAT_LIMIT = 1.01
ESS_MINIMUM = 400


def load_run(run_name):
    """Load every chain of a run; check they share one configuration and commit.

    Returns a list of dicts with the chain's arrays, sorted by chain index.
    """
    paths = sorted(gps_output_path("MCMC_Chains", run_name).glob("chain_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No chains in {gps_output_path('MCMC_Chains', run_name)}"
        )
    chains = []
    for path in paths:
        chain = load_chain(path)
        chain["path"] = path
        chain["metadata"] = json.loads(str(chain["metadata_json"]))
        chains.append(chain)

    configs = {str(chain["config_json"]) for chain in chains}
    commits = {chain["metadata"]["git_commit"] for chain in chains}
    if len(configs) != 1:
        raise ValueError(f"Chains of {run_name} were run with different configurations")
    if len(commits) != 1:
        print(f"WARNING: chains of {run_name} come from commits {sorted(commits)}")
    unfinished = [
        c["path"].name for c in chains if int(c["n_done"]) < int(c["n_total"])
    ]
    if unfinished:
        print(f"WARNING: unfinished chains (partial samples used): {unfinished}")
    return chains


def pooled_run(run_name):
    """All saved draws of a run pooled into one dict, for plotting.

    Uses the per-draw keys of the chain files (``lever``, ``CDOG_aug``,
    ``esv_bias``, ``dt`` in the Eq. 6 convention, ``drift_rate``, ``sigma_cm``,
    ``phi``, ``loglike``, ``logpost``, ``theta_samples``) concatenated over the
    chains, plus the run-level keys of the first chain. ``time_bias`` is added in
    the internal convention (offset - dt) for older plotting code.
    """
    chains = load_run(run_name)
    per_draw = (
        "lever",
        "CDOG_aug",
        "esv_bias",
        "dt",
        "drift_rate",
        "sigma_cm",
        "phi",
        "sigma_white_cm",
        "esv_common",
        "loglike",
        "logpost",
        "theta_samples",
    )
    pooled = {
        key: np.concatenate([chain[key] for chain in chains])
        for key in per_draw
        # sigma_white_cm exists only for "ar1_white" runs, esv_common only for
        # "common_windows" runs.
        if key in chains[0]
    }
    for key in (
        "param_names",
        "prior_center",
        "prior_sd",
        "offsets_s",
        "CDOG_reference",
        "config_json",
        "t_mid_s",
        "fixed_phi",
    ):
        pooled[key] = chains[0][key]
    pooled["time_bias"] = pooled["offsets_s"] - pooled["dt"]
    pooled["run_name"] = run_name
    return pooled


def stack_draws(chains, key="theta_samples"):
    """Draws as (n_chains, n_draws, ...) truncated to the shortest chain."""
    n_draws = min(chain[key].shape[0] for chain in chains)
    return np.stack([chain[key][:n_draws] for chain in chains])


def split_chains(draws):
    """Split each chain in half: (n_chains, n) -> (2 n_chains, n // 2)."""
    half = draws.shape[1] // 2
    return np.concatenate([draws[:, :half], draws[:, half : 2 * half]])


def rank_normalize(draws):
    """Normal scores of the pooled ranks (Blom offsets), same shape as draws."""
    ranks = rankdata(draws, method="average").reshape(draws.shape)
    return norm.ppf((ranks - 0.375) / (draws.size + 0.25))


def rhat(draws):
    """Potential scale reduction factor of (n_chains, n) draws, no splitting."""
    n = draws.shape[1]
    within = draws.var(axis=1, ddof=1).mean()
    between = n * draws.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def rank_normalized_split_rhat(draws):
    """Max of the rank-normalized split R-hat for location and for scale (folded)."""
    split = split_chains(draws)
    folded = np.abs(split - np.median(split))
    return max(rhat(rank_normalize(split)), rhat(rank_normalize(folded)))


def ess(draws):
    """Multi-chain effective sample size of (n_chains, n) draws.

    Autocorrelations are combined across chains as in Stan, and the sum is
    truncated with Geyer's initial monotone positive-pair sequence.
    """
    n_chains, n = draws.shape
    centered = draws - draws.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2 * n, axis=1)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    chain_variance = autocovariance[:, 0] * n / (n - 1)
    within = chain_variance.mean()
    variance_plus = within * (n - 1) / n
    if n_chains > 1:
        variance_plus += draws.mean(axis=1).var(ddof=1)
    if variance_plus <= 0:
        return np.nan
    rho = 1.0 - (within - autocovariance.mean(axis=0)) / variance_plus
    rho[0] = 1.0

    pair_sums = []
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair_sums.append(min(pair, pair_sums[-1]) if pair_sums else pair)
    tau = -1.0 + 2.0 * np.sum(pair_sums)
    return n_chains * n / max(tau, 1.0 / np.log10(n_chains * n))


def ess_bulk(draws):
    return ess(rank_normalize(split_chains(draws)))


def ess_tail(draws):
    """Min ESS of the indicators of the 5% and 95% quantiles."""
    split = split_chains(draws)
    low, high = np.quantile(split, [0.05, 0.95])
    return min(ess((split <= low).astype(float)), ess((split <= high).astype(float)))


def autocorrelation(x, max_lag):
    """Sample autocorrelation of a single chain for lags 0..max_lag."""
    x = x - x.mean()
    n = x.shape[0]
    spectrum = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1]
    return acf / acf[0]


def diagnose(chains):
    """Per-parameter diagnostics rows, including the log posterior."""
    names = [str(name) for name in chains[0]["param_names"]] + ["logpost"]
    theta = stack_draws(chains)
    logpost = stack_draws(chains, "logpost")[:, :, None]
    draws = np.concatenate([theta, logpost], axis=2)
    rows = []
    for k, name in enumerate(names):
        x = draws[:, :, k]
        rows.append(
            {
                "parameter": name,
                "mean": x.mean(),
                "sd": x.std(ddof=1),
                "rhat": rank_normalized_split_rhat(x),
                "ess_bulk": ess_bulk(x),
                "ess_tail": ess_tail(x),
            }
        )
    return rows


def plot_traces_and_autocorrelation(chains, out_path, max_lag=200, per_page=6):
    """Multi-page PDF: trace (all chains) and autocorrelation per parameter."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = [str(name) for name in chains[0]["param_names"]] + ["logpost"]
    theta = stack_draws(chains)
    draws = np.concatenate([theta, stack_draws(chains, "logpost")[:, :, None]], axis=2)
    thin = int(json.loads(str(chains[0]["config_json"]))["thin"])
    lags = np.arange(min(max_lag, draws.shape[1] - 1) + 1)

    with PdfPages(out_path) as pdf:
        for start in range(0, len(names), per_page):
            block = range(start, min(start + per_page, len(names)))
            fig, axes = plt.subplots(
                len(block),
                2,
                figsize=(11, 2.0 * len(block)),
                squeeze=False,
                gridspec_kw={"width_ratios": [3, 1]},
            )
            for row, k in enumerate(block):
                for c in range(draws.shape[0]):
                    axes[row, 0].plot(draws[c, :, k], lw=0.4, label=f"chain {c}")
                    axes[row, 1].plot(
                        lags * thin, autocorrelation(draws[c, :, k], lags[-1]), lw=0.8
                    )
                axes[row, 0].set_ylabel(names[k], fontsize=7)
                axes[row, 1].axhline(0.0, color="0.5", lw=0.5)
            axes[-1, 0].set_xlabel(f"saved draw (every {thin} iterations)")
            axes[-1, 1].set_xlabel("lag (iterations)")
            axes[0, 0].legend(fontsize=6, ncol=4, loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def write_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(run_name, make_plots=True):
    """Print and save the diagnostics of a run under <run>/analysis/."""
    chains = load_run(run_name)
    out_dir = gps_output_path("MCMC_Chains", run_name, "analysis")
    out_dir.mkdir(exist_ok=True)

    print(
        f"Run {run_name}: {len(chains)} chains, "
        f"git {chains[0]['metadata']['git_commit'][:7]}"
    )
    for chain in chains:
        print(
            f"  {chain['path'].name}: "
            f"{int(chain['n_done'])}/{int(chain['n_total'])} it, "
            f"{chain['theta_samples'].shape[0]} saved draws, burn-in windows "
            f"{np.round(chain['window_acceptance'], 3).tolist()}, sampling acceptance "
            f"{float(chain['acceptance_rate_sampling']):.3f}"
        )

    rows = diagnose(chains)
    write_csv(rows, out_dir / "diagnostics.csv")
    failing = [
        row
        for row in rows
        if row["rhat"] > RHAT_LIMIT
        or min(row["ess_bulk"], row["ess_tail"]) < ESS_MINIMUM
    ]
    worst_rhat = max(rows, key=lambda row: row["rhat"])
    worst_ess = min(rows, key=lambda row: min(row["ess_bulk"], row["ess_tail"]))
    print(
        f"  max R-hat {worst_rhat['rhat']:.3f} ({worst_rhat['parameter']}); "
        f"min ESS {min(worst_ess['ess_bulk'], worst_ess['ess_tail']):.0f} "
        f"({worst_ess['parameter']})"
    )
    print(
        f"  {len(failing)} of {len(rows)} parameters fail R-hat < {RHAT_LIMIT} or "
        f"ESS > {ESS_MINIMUM}: {[row['parameter'] for row in failing][:10]}"
    )
    if make_plots:
        plot_traces_and_autocorrelation(chains, out_dir / "trace_autocorrelation.pdf")
    print(f"  wrote {out_dir}")
    return rows


if __name__ == "__main__":
    RUN_NAMES = [
        "ar1",
        "ar1_common15",
        "ar1_white",
        "ar1_white_common30",
        "iid",
    ]

    for run_name in RUN_NAMES:
        run_diagnostics(run_name)
