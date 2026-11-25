"""Each DOG's position under each model, as dots with error bars.

For every run in ``RUNS`` with an ``analysis/posterior_summary.csv``, plots the
posterior mean E, N, U minus the submitted Table 1 position, with +-1 sd (thick) and
the 95 % interval (thin). Runs without chains are skipped. Run from the repository
root with PYTHONPATH=src.
"""

import csv

import matplotlib.pyplot as plt

from bermuda_data import gps_data_path
from plotting import paper_style

# (run name in Output/MCMC_Chains, label). Local runs are short and preliminary.
RUNS = [
    ("ar1_splits1", "Constant δc"),
    ("ar1", "4 δc blocks"),
    ("ar1_drift", "4 δc blocks + drift"),
    ("ar1_common60_local", "Common δc(t), 1 h*"),
    ("ar1_common30_local", "Common δc(t), 30 min*"),
    ("ar1_common30_drift_local", "Common δc(t), 30 min + drift*"),
    ("ar1_common30_half1", "Common δc(t), first half"),
    ("ar1_common30_half2", "Common δc(t), second half"),
    ("ar1_dog1", "Own lever arm, 4 blocks"),
    ("ar1_dog3", "Own lever arm, 4 blocks"),
    ("ar1_dog4", "Own lever arm, 4 blocks"),
]
DOGS = (1, 3, 4)
COMPONENTS = ("E", "N", "U")
FIGURE_NAME = "fig_model_sensitivity"


def load_summary(run):
    """Posterior summary rows of a run keyed by quantity, or None if absent."""
    path = gps_data_path("Output/MCMC_Chains", run, "analysis/posterior_summary.csv")
    if not path.exists():
        return None
    with open(path) as file:
        return {row["quantity"]: row for row in csv.DictReader(file)}


def collect(runs=RUNS, dogs=DOGS):
    """Rows (dog, label, component, mean, sd, p2.5, p97.5) in cm from Table 1."""
    rows = []
    for run, label in runs:
        summary = load_summary(run)
        if summary is None:
            print(f"skip {run}: no posterior_summary.csv")
            continue
        for dog in dogs:
            for component in COMPONENTS:
                key = f"DOG{dog}_{component}_minus_Table1"
                if key not in summary:
                    continue
                values = {
                    name: 100 * float(summary[key][name])
                    for name in ("mean", "sd", "p2.5", "p97.5")
                }
                rows.append({"dog": dog, "label": label, "component": component})
                rows[-1].update(values)
    return rows


def plot(rows):
    """Three panels (E, N, U); one line per DOG and model, grouped by DOG."""
    paper_style.apply()
    entries = list(dict.fromkeys((row["dog"], row["label"]) for row in rows))
    y_of = {}
    y, ticks, labels = 0.0, [], []
    for dog in DOGS:
        for entry in [e for e in entries if e[0] == dog]:
            y_of[entry] = y
            ticks.append(y)
            labels.append(entry[1])
            y -= 1.0
        y -= 0.8  # gap between DOGs
    height_mm = 22.0 + 4.2 * len(entries)
    fig, axes = paper_style.figure("double", height_mm, ncols=3, sharey=True)
    for ax, component in zip(axes, COMPONENTS, strict=True):
        for row in [r for r in rows if r["component"] == component]:
            style = paper_style.DOG_STYLE[row["dog"]]
            yy = y_of[(row["dog"], row["label"])]
            ax.plot([row["p2.5"], row["p97.5"]], [yy, yy], color=style["color"], lw=0.7)
            ax.plot(
                [row["mean"] - row["sd"], row["mean"] + row["sd"]],
                [yy, yy],
                color=style["color"],
                lw=2.0,
            )
            ax.plot(
                row["mean"],
                yy,
                marker=style["marker"],
                color=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
            )
        ax.axvline(0.0, color="0.6", lw=0.6, zorder=0)
        ax.set_xlabel(f"Δ{component} (cm)")
        ax.grid(axis="x", color="0.9", lw=0.5)
    axes[0].set_yticks(ticks, labels)
    for dog in DOGS:
        ys = [y_of[e] for e in entries if e[0] == dog]
        if ys:
            axes[-1].text(
                1.02,
                (max(ys) + min(ys)) / 2,
                paper_style.DOG_STYLE[dog]["label"],
                transform=axes[-1].get_yaxis_transform(),
                va="center",
                color=paper_style.DOG_STYLE[dog]["color"],
                fontweight="bold",
            )
    # Caption: dot = posterior mean, thick = +-1 sd, thin = 95 %, * = short chains.
    fig.supxlabel("Change from the submitted Table 1 position")
    for ax, letter in zip(axes, "abc", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


if __name__ == "__main__":
    figure = plot(collect())
    print(paper_style.save(figure, FIGURE_NAME))
    plt.close(figure)
