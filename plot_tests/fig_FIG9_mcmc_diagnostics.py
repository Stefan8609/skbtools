"""MCMC diagnostics of the final run (supplementary figure FIG-9).

(a) Traces of the key parameters of one DOG for all chains, burn-in included and
    shaded (adaptive Metropolis, discarded). Linear quantities are deviations from
    the pooled posterior mean, as in FX-7; σ, φ and the log posterior are absolute.
(b) Table: rank-normalised split R̂ and bulk/tail ESS (Vehtari et al. 2021, computed
    with MCMC_diagnostics) of the traced quantities and the worst over all sampled
    parameters (diagnostics.csv); acceptance rates of each chain.
(c) Posterior sd over the linearised (Gauss-Newton) sd at the MAP, with i.i.d. and
    with AR(1) residual weighting (linearized_sd.csv; R4 2a).
(d) Posterior sd over prior sd (prior_sensitivity.csv; R4 2b, R3 M5 "prior centred
    on the solution").

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
HEIGHT_MM = 222.0
HIGHLIGHT_COLORS = ("#0072B2", "#D55E00", "#009E73")  # chains 1-3; others grey
OTHER_CHAIN_COLOR = "0.7"
TRACE_EVERY = 4  # draw every 4th saved draw in (a), to limit over-plotting
BURN_IN_COLOR = "0.90"
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
    ("lever_x", f"{fc.LEVER_LABELS[0]} (cm)", 100.0),
    ("lever_y", f"{fc.LEVER_LABELS[1]} (cm)", 100.0),
    ("lever_z", f"{fc.LEVER_LABELS[2]} (cm)", 100.0),
    (f"DOG{TARGET_DOG}_dc_mean", f"{fc.DC_SYMBOL} offset\n(mm/s)", 1000.0),
    (f"DOG{TARGET_DOG}_dt_Eq6", f"{fc.DT_SYMBOL} (µs)", 1.0e6),
    (f"lever_z_minus_DOG{TARGET_DOG}_U", "$l_z$ − U (cm)", 100.0),
    ("sigma", "σ (cm)", 1.0),
    ("phi", "φ", 1.0),
    ("logpost", "log\nposterior", 1.0),
]
TABLE_LABELS = [
    "ΔE", "ΔN", "ΔU", fc.LEVER_LABELS[0], fc.LEVER_LABELS[1], fc.LEVER_LABELS[2],
    f"{fc.DC_SYMBOL}", fc.DT_SYMBOL, "$l_z$−U", "σ", "φ", "log p",
]  # fmt: skip
# Rows of panels (c) and (d), top to bottom: (label, regex on linearized_sd names,
# regex on prior_sensitivity names). None: not available for that panel.
CATEGORIES = [
    ("Lever arm", r"lever_[xyz]$", r"lever_[xyz]$"),
    ("DOG E, N", r"DOG\d_[EN]_minus_Table1$", r"DOG\d_[EN]_minus_Table1$"),
    ("DOG U", r"DOG\d_U_minus_Table1$", r"DOG\d_U_minus_Table1$"),
    ("$l_z$ − U", r"lever_z_minus_DOG\d_U$", None),  # (d): from the draws
    (fc.DT_SYMBOL, r"DOG\d_dt_Eq6$", r"time_bias_internal_DOG\d$"),
    (f"{fc.DC_SYMBOL} offset", r"DOG\d_dc_mean$", r"esv_offset_DOG\d$"),
    (f"{fc.DC_SYMBOL}(t) windows", r"common_dc_w\d+$", r"esv_common_w\d+$"),
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


def draw_traces(subfig, traces, thin, n_burn):
    axes = subfig.subplots(6, 2, sharex=True)
    n_chains = traces[0][0].shape[0]
    n_highlight = len(HIGHLIGHT_COLORS)
    colors = list(HIGHLIGHT_COLORS) + [OTHER_CHAIN_COLOR] * (n_chains - n_highlight)
    clipped = {}
    for index, ((key, label, _), (burn, sample)) in enumerate(
        zip(TRACES, traces, strict=True)
    ):
        ax = axes[index % 6, index // 6]
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
                alpha=0.9 if c < n_highlight else 0.6,
                rasterized=True,
            )
        low, high = sample.min(), sample.max()
        if max(high - burn.min(), burn.max() - low) > CLIP_FACTOR * (high - low):
            pad = 0.15 * (high - low)
            ax.set_ylim(low - pad, high + pad)
            outside = (burn < low - pad) | (burn > high + pad)
            clipped[key] = (int(outside.sum()), float(burn.min()), float(burn.max()))
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(length=2, pad=1.5)
    for ax in axes[-1]:
        ax.set_xlabel("Iteration (thousands)")
        ax.set_xlim(0.0, x_sample[-1])
    subfig.align_ylabels(axes[:, 0])
    subfig.align_ylabels(axes[:, 1])
    paper_style.panel_label(axes[0, 0], "a")
    handles = [Line2D([], [], color=colors[c], lw=1.2) for c in range(n_highlight)]
    handles.append(Line2D([], [], color=OTHER_CHAIN_COLOR, lw=1.2))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BURN_IN_COLOR))
    labels = [f"{c + 1}" for c in range(n_highlight)]
    labels += [f"{n_highlight + 1}–{n_chains}", "burn-in (discarded)"]
    subfig.legend(
        handles,
        labels,
        loc="outside upper right",
        ncol=n_highlight + 2,
        title=f"Chain (DOG {TARGET_DOG}; linear quantities from the posterior mean)",
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    return clipped


def draw_table(ax, rows_diag, worst, chains):
    """Text table: R̂ and ESS of the traced quantities, then acceptance per chain."""
    ax.set_axis_off()
    labels = TABLE_LABELS + ["worst"]
    n_cols = len(labels) + 1
    col_w = 1.0 / n_cols
    y_rows = [0.93, 0.80, 0.67, 0.54]

    def put(col, y, text, **kw):
        ax.text(
            (col + 0.5) * col_w,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            **kw,
        )

    for c, text in enumerate(labels, start=1):
        put(c, y_rows[0], text)
    for r, (name, key, fmt) in enumerate(
        [
            ("R̂", "rhat", "{:.3f}"),
            ("ESS bulk", "ess_bulk", "{:.0f}"),
            ("ESS tail", "ess_tail", "{:.0f}"),
        ],
        start=1,
    ):
        put(0, y_rows[r], name)
        for c, row in enumerate(rows_diag + [worst], start=1):
            put(c, y_rows[r], fmt.format(row[key]))
    ax.plot([0, 1], [0.865, 0.865], transform=ax.transAxes, color="black", lw=0.5)

    # Acceptance per chain.
    n_chains = len(chains)
    label_w = 0.26  # width of the row-label column (axes fraction)
    acc_w = (1.0 - label_w) / n_chains
    y_acc = [0.34, 0.21, 0.08]

    ax.text(0.0, y_acc[0], "Chain", transform=ax.transAxes, va="center")
    ax.text(0.0, y_acc[1], "Acceptance, sampling", transform=ax.transAxes, va="center")
    ax.text(
        0.0, y_acc[2], "Acceptance, adaptation", transform=ax.transAxes, va="center"
    )
    for c, chain in enumerate(chains):
        x = label_w + (c + 0.5) * acc_w
        windows = chain["window_acceptance"]
        ax.text(
            x, y_acc[0], f"{c + 1}", transform=ax.transAxes, ha="center", va="center"
        )
        ax.text(
            x,
            y_acc[1],
            f"{float(chain['acceptance_rate_sampling']):.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.text(
            x,
            y_acc[2],
            f"{windows.min():.2f}–{windows.max():.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.plot([0, 1], [0.275, 0.275], transform=ax.transAxes, color="black", lw=0.5)
    paper_style.panel_label(ax, "b")


def category_points(rows, name_key, value, pattern):
    """(values, DOG or None) of the rows whose name matches ``pattern``."""
    out = []
    for row in rows:
        name = row[name_key]
        if pattern is not None and re.match(pattern, name):
            dog = re.search(r"DOG(\d)", name)
            out.append((value(row), int(dog.group(1)) if dog else None))
    return out


def draw_points(ax, y, points, filled=True, offset=0.0):
    for value, dog in points:
        style = paper_style.DOG_STYLE.get(dog, {"color": "black", "marker": "o"})
        ax.plot(
            value,
            y + offset,
            ls="none",
            marker=style["marker"],
            ms=3.5,
            mec=style["color"],
            mfc=style["color"] if filled else "white",
            mew=0.8,
            alpha=0.9 if dog else 0.6,
        )


def draw_ratio_panels(ax_lin, ax_prior, linearized, sensitivity, lzu_ratios):
    n = len(CATEGORIES)
    for r, (_, lin_pattern, prior_pattern) in enumerate(CATEGORIES):
        y = n - 1 - r
        if lin_pattern is None:
            ax_lin.text(
                0.97,
                y,
                "n/a (noise parameter)",
                transform=ax_lin.get_yaxis_transform(),
                ha="right",
                va="center",
                color="0.2",
            )
        else:
            iid = category_points(
                linearized,
                "quantity",
                lambda row: float(row["posterior_sd"]) / float(row["formal_iid_sd"]),
                lin_pattern,
            )
            model = category_points(
                linearized,
                "quantity",
                lambda row: float(row["posterior_sd"]) / float(row["formal_model_sd"]),
                lin_pattern,
            )
            draw_points(ax_lin, y, iid, filled=False, offset=0.18)
            draw_points(ax_lin, y, model, filled=True, offset=-0.18)
        if prior_pattern is None:
            points = [(ratio, dog) for dog, ratio in lzu_ratios.items()]
        else:
            points = category_points(
                sensitivity,
                "parameter",
                lambda row: float(row["ratio"]),
                prior_pattern,
            )
        draw_points(ax_prior, y, points)
    for ax in (ax_lin, ax_prior):
        ax.set_xscale("log")
        ax.set_ylim(-2.5, n - 0.4)  # empty band below log σ holds the key
        ax.set_yticks(range(n))
        ax.grid(axis="x", color="0.9", lw=0.5)
        ax.tick_params(length=2, pad=1.5)
    ax_lin.set_yticklabels([c[0] for c in CATEGORIES][::-1])
    ax_prior.set_yticklabels([])
    ax_lin.axvline(1.0, color="0.3", lw=0.6)
    ax_lin.set_xlim(0.3, 6.0)
    ax_lin.set_xticks([0.5, 1.0, 2.0, 5.0], labels=["0.5", "1", "2", "5"])
    ax_lin.set_xticks([], minor=True)
    ax_lin.set_xlabel("Posterior sd / linearised sd")
    ax_prior.axvline(DATA_DOMINATED_RATIO, color="0.3", lw=0.6, ls="--")
    ax_prior.axvline(PRIOR_DOMINATED_RATIO, color="0.3", lw=0.6, ls=":")
    ax_prior.set_xlim(5e-4, 1.0)
    ax_prior.set_xlabel("Posterior sd / prior sd")
    for ratio, text in (
        (DATA_DOMINATED_RATIO, "data-dominated"),
        (PRIOR_DOMINATED_RATIO, "prior-dominated"),
    ):
        ax_prior.text(
            ratio * 1.12,
            -2.4,
            f"{ratio:g}: {text}",
            rotation=90,
            ha="left",
            va="bottom",
            color="0.1",
        )
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=s["marker"],
            color=s["color"],
            ms=3.5,
            label=s["label"],
        )
        for s in paper_style.DOG_STYLE.values()
    ]
    handles += [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            color="black",
            alpha=0.6,
            ms=3.5,
            label="shared (lever arm, δc(t))",
        ),
    ]
    key = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            mfc="0.3",
            mec="0.3",
            ms=3.5,
            label="AR(1)",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            mfc="white",
            mec="0.3",
            ms=3.5,
            label="i.i.d.",
        ),
    ]
    ax_lin.legend(handles=key, loc="lower left", handletextpad=0.3, borderaxespad=0.3)
    ax_lin.figure.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.3,
    )
    paper_style.panel_label(ax_lin, "c")
    paper_style.panel_label(ax_prior, "d")


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
    print(
        f"check: lever_z R̂ here {rows_diag[5]['rhat']:.5f}, "
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
        figsize=(paper_style.DOUBLE_COLUMN_MM / 25.4, HEIGHT_MM / 25.4),
        layout="constrained",
    )
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[0.55, 0.17, 0.28])
    clipped = draw_traces(top, traces, int(config["thin"]), int(config["n_burn"]))
    draw_table(middle.subplots(), rows_diag, worst, chains)
    ax_lin, ax_prior = bottom.subplots(1, 2)
    draw_ratio_panels(
        ax_lin,
        ax_prior,
        read_csv(LINEARIZED_SD_CSV),
        read_csv(PRIOR_SENSITIVITY_CSV),
        lzu_ratios,
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
