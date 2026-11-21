"""MCMC diagnostics of the final run (supplementary figure FIG-9).

Message: the eight chains of the final run mix and agree after burn-in, and the
posterior widths match the linearised widths once the AR(1) weighting is used.

(a) Traces of E, N, U, l_z and δt of one DOG for all chains, burn-in included and
    shaded (adaptive Metropolis, discarded), as deviations from the pooled
    posterior mean. Chains 1-3 are coloured, the others grey.
(b) Posterior sd over the linearised (Gauss-Newton) sd at the MAP, with AR(1) and
    with i.i.d. residual weighting (linearized_sd.csv; R4 2a).
Printed for a paper table instead of drawn: rank-normalised split R̂ and bulk/tail
ESS (Vehtari et al. 2021, computed with MCMC_diagnostics) of twelve quantities and
the worst over all sampled parameters (diagnostics.csv); the acceptance rates of
each chain; posterior sd over prior sd (prior_sensitivity.csv; R4 2b, R3 M5).

Reviewer comments: R3 M5 (burn-in, acceptance, ESS, R̂, trace plots, σ fixed or
sampled), R4 2a, R4 2b. Inputs: chains and analysis/ CSVs of run ``ar1_common30``.
Run from the repo root with PYTHONPATH=src.
"""

import csv
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
RUN_DIR = fc.CHAINS_DIR / RUN
ANALYSIS_DIR = RUN_DIR / "analysis"
DIAGNOSTICS_CSV = ANALYSIS_DIR / "diagnostics.csv"
LINEARIZED_SD_CSV = ANALYSIS_DIR / "linearized_sd.csv"
PRIOR_SENSITIVITY_CSV = ANALYSIS_DIR / "prior_sensitivity.csv"
TARGET_DOG = 3
OUTPUT_NAME = "fig_FIG9_mcmc_diagnostics"
HEIGHT_MM = 112.0
# Chains 1-3 in muted, colour-blind-safe colours (Tol "muted"); the others grey.
HIGHLIGHT_COLORS = ("#332288", "#CC6677", "#DDCC77")
OTHER_CHAIN_COLOR = "0.75"
TRACE_EVERY = 4  # draw every 4th saved draw in (a), to limit over-plotting
BURN_IN_COLOR = "0.92"
N_DRAWN_TRACES = 5  # the first entries of TRACES are drawn; all are printed
# A trace axis is limited to the sampling-phase range (padded) when burn-in draws
# reach further than this many sampling-phase ranges; the clipped count is printed.
CLIP_FACTOR = 3.0
DATA_DOMINATED_RATIO = 0.1  # MCMC_posterior_summary thresholds
PRIOR_DOMINATED_RATIO = 0.5
# Traced quantities: (key, label, factor). Keys are reported_quantities names
# (deviation from the posterior mean), or "sigma", "phi", "logpost" (absolute).
TRACES = [
    (f"DOG{TARGET_DOG}_E_minus_Table1", "ΔE (cm)", 100.0),
    (f"DOG{TARGET_DOG}_N_minus_Table1", "ΔN (cm)", 100.0),
    (f"DOG{TARGET_DOG}_U_minus_Table1", "ΔU (cm)", 100.0),
    ("lever_z", f"{fc.LEVER_LABELS[2]} (cm)", 100.0),
    (f"DOG{TARGET_DOG}_dt_Eq6", f"{fc.DT_SYMBOL} (µs)", 1.0e6),
    ("lever_x", f"{fc.LEVER_LABELS[0]} (cm)", 100.0),
    ("lever_y", f"{fc.LEVER_LABELS[1]} (cm)", 100.0),
    (f"DOG{TARGET_DOG}_dc_mean", f"{fc.DC_SYMBOL} offset\n(mm/s)", 1000.0),
    (f"lever_z_minus_DOG{TARGET_DOG}_U", "$l_z$ − U (cm)", 100.0),
    ("sigma", "σ (cm)", 1.0),
    ("phi", "φ", 1.0),
    ("logpost", "log\nposterior", 1.0),
]
# Rows of panel (b) and of the printed prior table, top to bottom: (label, regex on
# linearized_sd names, regex on prior_sensitivity names). None: not available.
CATEGORIES = [
    ("Lever arm", r"lever_[xyz]$", r"lever_[xyz]$"),
    ("DOG E, N", r"DOG\d_[EN]_minus_Table1$", r"DOG\d_[EN]_minus_Table1$"),
    ("DOG U", r"DOG\d_U_minus_Table1$", r"DOG\d_U_minus_Table1$"),
    ("$l_z$ − U", r"lever_z_minus_DOG\d_U$", None),  # (d): from the draws
    (fc.DT_SYMBOL, r"DOG\d_dt_Eq6$", r"time_bias_internal_DOG\d$"),
    (f"{fc.DC_NAME},\nper-DOG offset", r"DOG\d_dc_mean$", r"esv_offset_DOG\d$"),
    (f"{fc.DC_NAME},\n30-min windows", r"common_dc_w\d+$", r"esv_common_w\d+$"),
    ("log σ", None, r"log_sigma_cm_DOG\d$"),
]
# -----------------------------------------------------------------------------------


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def trace_draws(chains, model, key, factor, posterior_mean):
    """(burn-in, sampling) arrays (n_chains, n) of one traced quantity."""
    from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
        reported_quantities,
    )

    names = [str(name) for name in chains[0]["param_names"]]
    out = []
    for theta_key, logpost_key in (
        ("burnin_theta", "burnin_logpost"),
        ("theta_samples", "logpost"),
    ):
        if key == "logpost":
            values = [chain[logpost_key] for chain in chains]
        elif key == "sigma":
            k = names.index(f"log_sigma_cm_DOG{TARGET_DOG}")
            values = [np.exp(chain[theta_key][:, k]) for chain in chains]
        elif key == "phi":
            k = names.index(f"phi_DOG{TARGET_DOG}")
            values = [chain[theta_key][:, k] for chain in chains]
        else:
            q_names, _, A, b = reported_quantities(
                model, chains[0]["offsets_s"], chains[0]["CDOG_reference"]
            )
            k = q_names.index(key)
            values = [
                (chain[theta_key] @ A[k] + b[k] - posterior_mean[key]) * factor
                for chain in chains
            ]
        out.append(np.array(values))
    return out


def linear_posterior_means(chains, model):
    """Pooled posterior mean of every reported quantity (dict name -> SI value)."""
    from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
        reported_quantities,
    )

    q_names, _, A, b = reported_quantities(
        model, chains[0]["offsets_s"], chains[0]["CDOG_reference"]
    )
    theta_mean = np.concatenate([c["theta_samples"] for c in chains]).mean(axis=0)
    return dict(zip(q_names, A @ theta_mean + b, strict=True))


def draw_traces(axes, traces, thin, n_burn):
    """(a): one row per traced quantity, all chains, burn-in shaded."""
    n_chains = traces[0][0].shape[0]
    n_highlight = len(HIGHLIGHT_COLORS)
    colors = list(HIGHLIGHT_COLORS) + [OTHER_CHAIN_COLOR] * (n_chains - n_highlight)
    clipped = {}
    for ax, (key, label, _), (burn, sample) in zip(axes, TRACES, traces, strict=False):
        ax.axvspan(0.0, n_burn / 1000.0, color=BURN_IN_COLOR, lw=0)
        x_burn = np.arange(1, burn.shape[1] + 1) * thin / 1000.0
        x_sample = (n_burn + np.arange(1, sample.shape[1] + 1) * thin) / 1000.0
        for c in list(range(n_highlight, n_chains)) + list(range(n_highlight)):
            x = np.concatenate([x_burn, x_sample])
            y = np.concatenate([burn[c], sample[c]])
            ax.plot(
                x[::TRACE_EVERY],
                y[::TRACE_EVERY],
                color=colors[c],
                lw=0.3,
                rasterized=True,
            )
        low, high = sample.min(), sample.max()
        if max(high - burn.min(), burn.max() - low) > CLIP_FACTOR * (high - low):
            pad = 0.15 * (high - low)
            ax.set_ylim(low - pad, high + pad)
            outside = (burn < low - pad) | (burn > high + pad)
            clipped[key] = (int(outside.sum()), float(burn.min()), float(burn.max()))
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True))
    axes[-1].set_xlabel("Iteration (thousands)")
    axes[-1].set_xlim(0.0, x_sample[-1])
    axes[0].text(
        n_burn / 2000.0,
        1.03,
        "burn-in",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="bottom",
    )
    handles = [Line2D([], [], color=colors[c], lw=1.2) for c in range(n_highlight)]
    handles.append(Line2D([], [], color=OTHER_CHAIN_COLOR, lw=1.2))
    labels = [f"chain {c + 1}" for c in range(n_highlight)]
    labels.append(f"chains {n_highlight + 1}–{n_chains}")
    axes[0].legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=n_highlight + 1,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    return clipped


def category_points(rows, name_key, value, pattern):
    """Values of the rows whose name matches ``pattern``."""
    return [
        value(row)
        for row in rows
        if pattern is not None and re.match(pattern, row[name_key])
    ]


def draw_sd_ratio(ax, linearized):
    """(b): posterior sd over linearised sd, AR(1) (black) and i.i.d. (grey open)."""
    rows = [category for category in CATEGORIES if category[1] is not None]
    for r, (_, pattern, _) in enumerate(rows):
        y = len(rows) - 1 - r
        for column, offset, style in (
            ("formal_iid_sd", 0.16, {"mfc": "white", "mec": "0.5"}),
            ("formal_model_sd", -0.16, {"mfc": "black", "mec": "black"}),
        ):
            ratios = category_points(
                linearized,
                "quantity",
                lambda row, column=column: (
                    float(row["posterior_sd"]) / float(row[column])
                ),
                pattern,
            )
            ax.plot(
                ratios,
                np.full(len(ratios), y + offset),
                ls="none",
                marker="o",
                ms=3.5,
                mew=0.8,
                **style,
            )
    ax.axvline(1.0, color=fc.CONTEXT, lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 5.0)
    ax.set_xticks([0.5, 1.0, 2.0, 4.0], labels=["0.5", "1", "2", "4"])
    ax.set_xticks([], minor=True)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Posterior sd / linearised sd")
    top = len(rows) - 0.1
    ax.text(1.0, top, "AR(1)", ha="center", va="bottom")
    ax.text(3.0, top, "i.i.d.", ha="center", va="bottom", color="0.4")


def print_prior_ratios(sensitivity, lzu_ratios):
    """Posterior sd / prior sd per category (was a panel; now for a paper table)."""
    for label, _, pattern in CATEGORIES:
        if pattern is None:
            ratios = list(lzu_ratios.values())
        else:
            ratios = category_points(
                sensitivity, "parameter", lambda row: float(row["ratio"]), pattern
            )
        name = label.replace("\n", " ")
        print(
            f"posterior sd / prior sd, {name}: {min(ratios):.4f} to {max(ratios):.4f}"
        )


if __name__ == "__main__":
    fc.require(
        RUN_DIR / "chain_00.npz",
        DIAGNOSTICS_CSV,
        LINEARIZED_SD_CSV,
        PRIOR_SENSITIVITY_CSV,
    )
    from Inversion_Workflow.MCMC_Functions import MCMC_diagnostics as md
    from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms

    paper_style.apply()
    chains = md.load_run(RUN)
    config = fc.run_config(chains[0])
    model = ms.build_model(config)
    means = linear_posterior_means(chains, model)
    traces = [
        trace_draws(chains, model, key, factor, means) for key, _, factor in TRACES
    ]

    rows_diag = []
    for _, sample in traces:
        rows_diag.append(
            {
                "rhat": md.rank_normalized_split_rhat(sample),
                "ess_bulk": md.ess_bulk(sample),
                "ess_tail": md.ess_tail(sample),
            }
        )
    diagnostics = read_csv(DIAGNOSTICS_CSV)
    worst = {
        "rhat": max(float(r["rhat"]) for r in diagnostics),
        "ess_bulk": min(float(r["ess_bulk"]) for r in diagnostics),
        "ess_tail": min(float(r["ess_tail"]) for r in diagnostics),
    }
    lever_z_csv = next(r for r in diagnostics if r["parameter"] == "lever_z")
    lever_z_row = [key for key, _, _ in TRACES].index("lever_z")
    print(
        f"check: lever_z R̂ here {rows_diag[lever_z_row]['rhat']:.5f}, "
        f"csv {float(lever_z_csv['rhat']):.5f}"
    )

    # Posterior / prior sd of l_z - U for each DOG (not in prior_sensitivity.csv).
    from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
        reported_quantities,
    )

    q_names, _, A, b = reported_quantities(
        model, chains[0]["offsets_s"], chains[0]["CDOG_reference"]
    )
    theta = np.concatenate([c["theta_samples"] for c in chains])
    lzu_ratios = {}
    for dog in config["dogs"]:
        k = q_names.index(f"lever_z_minus_DOG{dog}_U")
        _, prior_sd = fc.linear_gaussian_prior(
            A[k], chains[0]["prior_center"], chains[0]["prior_sd"], b[k]
        )
        lzu_ratios[dog] = (theta @ A[k]).std(ddof=1) / prior_sd

    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        ),
        layout="constrained",
    )
    grid = fig.add_gridspec(N_DRAWN_TRACES, 2, width_ratios=[2.6, 1.0])
    trace_axes = [fig.add_subplot(grid[0, 0])]
    trace_axes += [
        fig.add_subplot(grid[row, 0], sharex=trace_axes[0])
        for row in range(1, N_DRAWN_TRACES)
    ]
    for ax in trace_axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_ratio = fig.add_subplot(grid[:, 1])
    clipped = draw_traces(
        trace_axes, traces, int(config["thin"]), int(config["n_burn"])
    )
    draw_sd_ratio(ax_ratio, read_csv(LINEARIZED_SD_CSV))
    fc.despine(*trace_axes, ax_ratio)
    fig.align_ylabels(trace_axes)
    paper_style.panel_label(trace_axes[0], "a")
    paper_style.panel_label(ax_ratio, "b")
    print_prior_ratios(read_csv(PRIOR_SENSITIVITY_CSV), lzu_ratios)
    for c, chain in enumerate(chains):
        windows = chain["window_acceptance"]
        print(
            f"chain {c + 1}: acceptance sampling "
            f"{float(chain['acceptance_rate_sampling']):.3f}, adaptation "
            f"{windows.min():.2f}–{windows.max():.2f}"
        )

    n_draws = sum(c["theta_samples"].shape[0] for c in chains)
    print(
        f"{len(chains)} chains, burn-in {config['n_burn']} it, sampling "
        f"{config['n_samples']} it, thin {config['thin']}, {n_draws} pooled draws"
    )
    acceptance = [float(c["acceptance_rate_sampling"]) for c in chains]
    print(
        f"acceptance (sampling) {min(acceptance):.3f}–{max(acceptance):.3f}, "
        f"target {config['target_acceptance']}"
    )
    print(f"worst over all parameters: {worst}")
    for (key, _, _), row in zip(TRACES, rows_diag, strict=True):
        print(
            f"  {key:26s} R̂ {row['rhat']:.4f} ESS {row['ess_bulk']:.0f} / "
            f"{row['ess_tail']:.0f}"
        )
    for key, (count, low, high) in clipped.items():
        print(
            f"trace {key}: axis limited to the sampling range; {count} burn-in "
            f"draws outside (burn-in range {low:.1f} to {high:.1f})"
        )
    print(f"l_z − U posterior/prior sd: {lzu_ratios}")
    fc.save(fig, OUTPUT_NAME)
