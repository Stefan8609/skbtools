"""Vertical and clock posteriors of the Bermuda MCMC under different noise models.

Compares runs with different noise models (iid, AR(1), AR(1) + white noise,
drift) and numbers of ESV splits, for the vertical lever arm and, per DOG, the up
coordinate (relative to the submitted Table 1 position), the Eq. 6 clock offset
dt (relative to the alignment offset) and the mean ESV bias. When these move
between models by more than their posterior widths, the widths understate the
model-dependent uncertainty.
"""

import json

import matplotlib.pyplot as plt

from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from Inversion_Workflow.MCMC_Functions.MCMC_diagnostics import pooled_run
from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
    reported_quantities,
)
from plotting.save import save_plot

# (quantity name template, axis label, scale to plot units, subtract the offset)
ROWS = [
    ("DOG{dog}_U_minus_Table1", "up - Table 1 (m)", 1.0, False),
    ("DOG{dog}_dt_Eq6", r"$\delta t$ - alignment offset ($\mu$s)", 1e6, True),
    ("DOG{dog}_dc_mean", r"mean $\delta c$ (m/s)", 1.0, False),
]


def run_quantities(run_name):
    """Reported quantities (dict of draws) of a run, named as in posterior_summary."""
    chain = pooled_run(run_name)
    config = json.loads(str(chain["config_json"]))
    model = ms.build_model(config)
    names, _, A, b = reported_quantities(
        model, chain["offsets_s"], chain["CDOG_reference"]
    )
    values = chain["theta_samples"] @ A.T + b
    return config["dogs"], chain["offsets_s"], dict(zip(names, values.T, strict=False))


def noise_model_comparison(runs, save=True, bins=60):
    """Overlay marginal posteriors of several runs.

    runs maps run names (folders in Output/MCMC_Chains) to legend labels. Returns
    the figure and a printed table of mean +- sd per run and quantity.
    """
    results = {name: run_quantities(name) for name in runs}
    dogs, offsets, _ = next(iter(results.values()))
    fig, axes = plt.subplots(len(ROWS) + 1, len(dogs), figsize=(4.2 * len(dogs), 12))

    def overlay(ax, key, scale, offset, label):
        ax.set_xlabel(label)
        for name, legend in runs.items():
            draws = results[name][2][key]
            draws = (draws - offset) * scale
            ax.hist(
                draws,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.4,
                label=f"{legend}: {draws.mean():.3g} $\\pm$ {draws.std(ddof=1):.2g}",
            )
        ax.legend(fontsize=7)

    overlay(axes[0, 0], "lever_z", 1.0, 0.0, "lever z (m)")
    axes[0, 0].set_title("Joint lever arm")
    for ax in axes[0, 1:]:
        ax.axis("off")
    for row, (template, label, scale, relative) in enumerate(ROWS, start=1):
        for j, dog in enumerate(dogs):
            offset = offsets[j] if relative else 0.0
            overlay(axes[row, j], template.format(dog=dog), scale, offset, label)
            axes[row, j].set_title(f"DOG {dog}")
    fig.tight_layout()

    print("quantity                     " + "  ".join(f"{runs[n]:>24s}" for n in runs))
    entries = [("lever_z", 1.0, 0.0)] + [
        (template.format(dog=dog), scale, offsets[j] if relative else 0.0)
        for template, _, scale, relative in ROWS
        for j, dog in enumerate(dogs)
    ]
    for key, scale, offset in entries:
        cells = []
        for name in runs:
            draws = (results[name][2][key] - offset) * scale
            cells.append(f"{draws.mean():12.4g} ± {draws.std(ddof=1):9.3g}")
        print(f"{key:28s} " + "  ".join(cells))

    if save:
        save_plot(fig, func_name="noise_model_comparison", subdir="Figs/MCMC")
    return fig


def model_summary_plot(runs, save=True):
    """Posterior mean +- sd of each quantity across runs, one panel per quantity.

    Readable for many runs (e.g. the noise-model x ESV-split grid). runs maps run
    names to labels, in plotting order.
    """
    results = {name: run_quantities(name) for name in runs}
    dogs, offsets, _ = next(iter(results.values()))
    panels = [("lever_z", "lever z (m)", 1.0, 0.0)] + [
        (
            template.format(dog=dog),
            f"DOG {dog}: {label}",
            scale,
            offsets[j] if relative else 0.0,
        )
        for template, label, scale, relative in ROWS
        for j, dog in enumerate(dogs)
    ]
    n_cols = len(dogs)
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 0.4 * len(runs) * n_rows + 2.0),
        squeeze=False,
    )
    positions = range(len(runs))
    for ax, (key, label, scale, offset) in zip(axes.ravel(), panels, strict=False):
        for y, name in zip(positions, runs, strict=False):
            draws = (results[name][2][key] - offset) * scale
            ax.errorbar(
                draws.mean(), y, xerr=draws.std(ddof=1), fmt="o", ms=4, capsize=3
            )
        ax.set_yticks(list(positions))
        ax.set_yticklabels(list(runs.values()), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(label, fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    if save:
        save_plot(fig, func_name="noise_model_summary", subdir="Figs/MCMC")
    return fig


if __name__ == "__main__":
    # Run folder -> label, in plotting order. Missing runs must be removed first.
    # Overlaid histograms for a few runs; the summary plot for the full grid of
    # noise models (iid, AR(1), AR(1) + white noise) and ESV splits per DOG.
    HISTOGRAM_RUNS = {
        "iid": "iid",
        "ar1": "AR(1)",
        "ar1_white": "AR(1) + white",
    }
    SUMMARY_RUNS = {
        "iid": "iid, 4 splits",
        "ar1_splits1": "AR(1), 1 split",
        "ar1": "AR(1), 4 splits",
        "ar1_splits8": "AR(1), 8 splits",
        "ar1_splits16": "AR(1), 16 splits",
        "ar1_drift": "AR(1) + drift, 4 splits",
        "ar1_white_splits1": "AR(1) + white, 1 split",
        "ar1_white": "AR(1) + white, 4 splits",
        "ar1_white_splits8": "AR(1) + white, 8 splits",
        "ar1_white_splits16": "AR(1) + white, 16 splits",
    }

    noise_model_comparison(HISTOGRAM_RUNS)
    model_summary_plot(SUMMARY_RUNS)
    plt.show()
