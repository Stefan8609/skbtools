"""FIG-5: each DOG's E, N, U under each ESV-correction and noise model.

Message: the DOG positions move between ESV-correction and noise models by many
times the posterior sd of any one model, most of all in U.

Answers R3 M2 (posterior width is not accuracy), R4 5 (realistic uncertainty,
repeatability) and R1 3a/3c (how positions respond to a time-varying δc). Dots are
posterior means of E, N, U minus the submitted Table 1 position (local ENU at the
Table 1 position, cm) for eight MCMC runs; thick bars ±1 posterior sd, thin bars the
95 % interval. Alternative models are grey, the final model black. The light band,
centred on the final run, is as wide as the difference between the two half-survey
Gauss–Newton solutions (lever free, one δc; N5a): the real-data repeatability.
The spread of the means over the median posterior sd is printed for the caption.

Inputs: ``Output/MCMC_Chains/<run>/analysis/posterior_summary.csv`` and the first
chain of each run (ESV table hash), and ``revision/analysis/realistic_synthetic/
real_reference.csv``. Runs on the old ESV table are drawn with open markers.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pymap3d import ecef2geodetic

import _figure_common as fc
from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
    SUBMITTED_TABLE1_AUG_M,
    enu_rotation,
)
from Inversion_Workflow.Synthetic.Realistic_Synthetic import SITE_ORIGIN_ECEF
from Inversion_Workflow.Synthetic.Realistic_Synthetic import (
    enu_rotation as site_enu_rotation,
)
from plotting import paper_style

# --- configuration ------------------------------------------------------------------
NAME = "fig_FIG5_model_sensitivity"
# ESV-correction models, named on the y axis.
CATEGORIES = ("Constant", "4 blocks", "Common, 60 min", "Common, 30 min")
# (run in Output/MCMC_Chains, category index, noise and clock label, marker), drawn
# top to bottom. Cluster runs only (8 chains each), not the *_local test runs.
RUNS = (
    ("ar1_splits1", 0, "AR(1)", "o"),
    ("iid", 1, "iid", "D"),
    ("ar1", 1, "AR(1)", "o"),
    ("ar1_white", 1, "AR(1) + white", "s"),
    ("ar1_drift", 1, "AR(1) + drift", "v"),
    ("ar1_common60", 2, "AR(1)", "o"),
    (fc.FINAL_RUN, 3, "AR(1)", "o"),
    ("ar1_common30_drift", 3, "AR(1) + drift", "v"),
)
CURRENT_ESV_SHA = "d1369cae"  # ESV table on disk since the ray-tracing fix (H2)
REAL_REFERENCE_CSV = (
    fc.REPO / "revision/analysis/realistic_synthetic/real_reference.csv"
)
DOGS = (1, 3, 4)
COMPONENTS = ("E", "N", "U")
OLD_TABLE_MARK = "†"
BAND_COLOR = "0.92"
OTHER_COLOR = "0.55"  # alternative models
FINAL_COLOR = "black"
HEIGHT_MM = 135.0


def summary_path(run):
    return fc.CHAINS_DIR / run / "analysis" / "posterior_summary.csv"


def esv_sha(run):
    """First 8 hex digits of the ESV table hash the run's chains were sampled with."""
    chain = sorted((fc.CHAINS_DIR / run).glob("chain_*.npz"))[0]
    metadata = json.loads(str(np.load(chain)["metadata_json"]))
    return metadata["input_files"]["esv_table"]["sha256"][:8]


def load_runs():
    """Rows (run, dog, component, mean, sd, p2.5, p97.5 in cm) plus the run's table."""
    rows = []
    for run, category, noise, marker in RUNS:
        summary = pd.read_csv(summary_path(run)).set_index("quantity")
        old_table = esv_sha(run) != CURRENT_ESV_SHA
        for dog in DOGS:
            for component in COMPONENTS:
                values = summary.loc[f"DOG{dog}_{component}_minus_Table1"]
                rows.append(
                    {
                        "run": run,
                        "category": category,
                        "noise": noise,
                        "marker": marker,
                        "old_table": old_table,
                        "dog": dog,
                        "component": component,
                        **{k: 100 * values[k] for k in ("mean", "sd", "p2.5", "p97.5")},
                    }
                )
    return pd.DataFrame(rows)


def half_split_difference_cm():
    """Half 2 minus half 1 of the real-data GN solutions, ENU at Table 1 (cm).

    real_reference.csv holds E, N, U in the frame of ``SITE_ORIGIN_ECEF`` (equal to
    the chains' CDOG_reference); they are taken back to ECEF and rotated into the
    local frame at each DOG's Table 1 position, the frame of the MCMC summaries.
    """
    table = pd.read_csv(REAL_REFERENCE_CSV)
    to_site = site_enu_rotation()
    out = {}
    for dog in DOGS:
        origin = SITE_ORIGIN_ECEF + np.array(SUBMITTED_TABLE1_AUG_M[dog])
        rotation = enu_rotation(*ecef2geodetic(*origin)[:2])
        halves = []
        for case in ("half1_lever_free", "half2_lever_free"):
            row = table[(table["DOG"] == dog) & (table["case"] == case)].iloc[0]
            ecef = SITE_ORIGIN_ECEF + to_site.T @ row[["E_m", "N_m", "U_m"]].to_numpy(
                dtype=float
            )
            halves.append(100 * rotation @ (ecef - origin))
        out[dog] = dict(zip(COMPONENTS, halves[1] - halves[0], strict=True))
    return out


def run_label(run, category, noise, old_table):
    label = f"{CATEGORIES[category]}, {noise}"
    if run == fc.FINAL_RUN:
        label += " (final)"
    return label + (f" {OLD_TABLE_MARK}" if old_table else "")


def plot(rows, band):
    n_runs = len(RUNS)
    fig, axes = paper_style.figure(
        "double",
        HEIGHT_MM,
        nrows=len(DOGS),
        ncols=len(COMPONENTS),
        sharey=True,
        sharex="col",
    )
    y = -np.arange(n_runs, dtype=float)
    order = [run for run, *_ in RUNS]
    for i, dog in enumerate(DOGS):
        for j, component in enumerate(COMPONENTS):
            ax = axes[i, j]
            cell = rows[(rows["dog"] == dog) & (rows["component"] == component)]
            cell = cell.set_index("run").loc[list(order)]
            final = cell.loc[fc.FINAL_RUN, "mean"]
            half = abs(band[dog][component]) / 2
            ax.axvspan(final - half, final + half, color=BAND_COLOR, lw=0, zorder=0)
            for yy, (run, row) in zip(y, cell.iterrows(), strict=True):
                color = FINAL_COLOR if run == fc.FINAL_RUN else OTHER_COLOR
                ax.plot([row["p2.5"], row["p97.5"]], [yy, yy], color=color, lw=0.7)
                ax.plot(
                    [row["mean"] - row["sd"], row["mean"] + row["sd"]],
                    [yy, yy],
                    color=color,
                    lw=2.2,
                    solid_capstyle="butt",
                )
                ax.plot(
                    row["mean"],
                    yy,
                    marker="o",
                    markersize=4.0,
                    color=color,
                    markerfacecolor="white" if row["old_table"] else color,
                    markeredgewidth=0.8,
                    linestyle="none",
                    zorder=3,
                )
            ratio = fc.spread_over_sd(cell["mean"], cell["sd"])
            spread = cell["mean"].max() - cell["mean"].min()
            print(
                f"DOG {dog} {component}: spread of means {spread:.0f} cm = "
                f"{ratio:.0f} median posterior sd"
            )
            if i == len(DOGS) - 1:
                ax.set_xlabel(f"Δ{component} from Table 1 (cm)")
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 5, 10]))
        style = paper_style.DOG_STYLE[dog]
        axes[i, 0].set_title(style["label"], color=style["color"], loc="left")
    fc.despine(*axes.flat)
    labels = [
        run_label(run, category, noise, rows[rows["run"] == run]["old_table"].iloc[0])
        for run, category, noise, _ in RUNS
    ]
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].set_ylim(y[-1] - 0.7, y[0] + 0.7)
    handles = [
        Line2D(
            [],
            [],
            color=OTHER_COLOR,
            marker="o",
            lw=2.2,
            label="Posterior mean ± 1 sd; thin: 95 % interval",
        ),
        Line2D(
            [],
            [],
            color=OTHER_COLOR,
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label=f"{OLD_TABLE_MARK} old ESV table (provisional)",
        ),
        Patch(color=BAND_COLOR, label="Final ± ½ |half 2 − half 1| of the survey"),
    ]
    if not rows["old_table"].any():
        del handles[1]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return fig


if __name__ == "__main__":
    fc.require(REAL_REFERENCE_CSV, *[summary_path(run) for run, *_ in RUNS])
    rows = load_runs()
    band = half_split_difference_cm()
    for dog in DOGS:
        print(
            f"DOG {dog} half2 - half1 (cm):",
            {k: round(v, 1) for k, v in band[dog].items()},
        )
    print(
        rows.pivot_table(
            index="run", columns=["dog", "component"], values="mean"
        ).round(1)
    )
    paper_style.apply()  # after imports: some modules turn usetex on at import
    figure = plot(rows, band)
    fc.save(figure, NAME)
    plt.close(figure)
