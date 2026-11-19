"""Corner plot of the key parameters for one DOG, with priors and correlations (Fig. 7).

Lower triangle: 2-D posterior histograms of the final MCMC run with the 68 % and
95 % highest-density regions. Diagonal: 1-D posterior marginals with the prior
density on the same density scale. Upper triangle: the posterior correlation
matrix, one coloured cell per pair. Every axis is the deviation from the posterior
mean in physical units, for one stated DOG (DOG 3, as in the submitted Fig. 7).

Quantities come from the linear map of ``reported_quantities`` (the same map as the
run's analysis tables): DOG E, N, U in local ENU, the lever arm (body frame), the
DOG's ESV-correction offset, its clock offset δt (Eq. 6) and l_z − U. The priors of
these linear combinations follow exactly from the sampler's independent Gaussian
priors (``fc.linear_gaussian_prior``).

Reviewer comments: R4 m2 and R1 4k (units, reference, target DOG, lever symbols),
R4 2b (prior on every marginal), R4 2c (signs of the δt correlations, printed by
this script), R3 #11 (no clipped labels, 8 pt, vector).
Inputs: the chains of run ``ar1_common30``. Run from the repo root with
PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
RUN_DIR = fc.CHAINS_DIR / RUN
TARGET_DOG = 3
OUTPUT_NAME = "fig_FX7_corner"
HEIGHT_MM = 206.0
MARGINS_MM = {"left": 18.5, "right": 15.0, "top": 15.0, "bottom": 20.0}
COLUMN_GAP_MM = 1.2
ROW_GAP_MM = 7.6  # room for the two-line sigma title above each diagonal panel
HALF_WIDTH_SD = 4.0  # axis half-range in posterior sd
N_BINS_2D = 36
N_BINS_1D = 50
SMOOTH_BINS = 1.0  # Gaussian smoothing (bins) of the 2-D histograms before contouring
PROBABILITIES = (0.68, 0.95)
CORRELATION_CMAP = "RdBu_r"
POSTERIOR_FILL = "0.80"
PRIOR_COLOR = fc.NEGATIVE_COLOR
# (name in reported_quantities, axis symbol, unit, factor from SI units)
QUANTITIES = [
    (f"DOG{TARGET_DOG}_E_minus_Table1", "ΔE", "cm", 100.0),
    (f"DOG{TARGET_DOG}_N_minus_Table1", "ΔN", "cm", 100.0),
    (f"DOG{TARGET_DOG}_U_minus_Table1", "ΔU", "cm", 100.0),
    ("lever_x", fc.LEVER_LABELS[0], "cm", 100.0),
    ("lever_y", fc.LEVER_LABELS[1], "cm", 100.0),
    ("lever_z", fc.LEVER_LABELS[2], "cm", 100.0),
    (f"DOG{TARGET_DOG}_dc_mean", f"{fc.DC_SYMBOL} offset", "mm/s", 1000.0),
    (f"DOG{TARGET_DOG}_dt_Eq6", fc.DT_SYMBOL, "µs", 1.0e6),
    (f"lever_z_minus_DOG{TARGET_DOG}_U", "$l_z$ − U", "cm", 100.0),
]
# -----------------------------------------------------------------------------------

NICE_STEPS = (1.0, 2.0, 2.5, 5.0)


def nice_tick(value):
    """Smallest 1, 2, 2.5 or 5 times a power of ten that is at least ``value``."""
    power = 10.0 ** np.floor(np.log10(value))
    return min(step * power for step in NICE_STEPS + (10.0,) if step * power >= value)


def key_draws_and_priors(pooled):
    """Draws (n, k) of the key quantities from their posterior mean, in plot units.

    Also returns the prior mean (from the posterior mean) and prior sd of each.
    """
    from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
    from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import (
        reported_quantities,
    )

    model = ms.build_model(fc.run_config(pooled))
    if list(model["names"]) != [str(name) for name in pooled["param_names"]]:
        raise ValueError("model parameters differ from the chain's param_names")
    names, _, A, b = reported_quantities(
        model, pooled["offsets_s"], pooled["CDOG_reference"]
    )
    theta = pooled["theta_samples"]
    draws, prior_mean, prior_sd = [], [], []
    for name, _, _, factor in QUANTITIES:
        k = names.index(name)
        values = theta @ A[k] + b[k]
        mean = values.mean()
        centre, sd = fc.linear_gaussian_prior(
            A[k], pooled["prior_center"], pooled["prior_sd"], b[k]
        )
        draws.append((values - mean) * factor)
        prior_mean.append((centre - mean) * factor)
        prior_sd.append(sd * factor)
    return np.column_stack(draws), np.array(prior_mean), np.array(prior_sd)


def label(i, two_lines=True):
    _, symbol, unit, _ = QUANTITIES[i]
    return f"{symbol}\n({unit})" if two_lines else f"{symbol} ({unit})"


def short_number(value):
    """Two significant figures for values below 10 (e.g. 2.0, 0.64), else integers."""
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}" if value >= 1 else f"{value:.2f}"


def prior_text(value, unit):
    """Prior sd with a larger unit for 10 ms and 2 m/s (1000 µs, 1000 mm/s)."""
    if unit in ("µs", "mm/s") and value >= 1000:
        return f"{value / 1000:g} {'ms' if unit == 'µs' else 'm/s'}"
    return f"{value:.0f} {unit}"


def draw_corner(fig, draws, prior_mean, prior_sd):
    n = draws.shape[1]
    sd = draws.std(axis=0, ddof=1)
    rho = np.corrcoef(draws, rowvar=False)
    ticks = [nice_tick(2.5 * s) for s in sd]
    half = [max(HALF_WIDTH_SD * s, 1.3 * t) for s, t in zip(sd, ticks, strict=True)]

    width_mm, height_mm = paper_style.DOUBLE_COLUMN_MM, HEIGHT_MM
    grid_w = width_mm - MARGINS_MM["left"] - MARGINS_MM["right"]
    grid_h = height_mm - MARGINS_MM["top"] - MARGINS_MM["bottom"]
    panel_w = (grid_w - (n - 1) * COLUMN_GAP_MM) / n
    panel_h = (grid_h - (n - 1) * ROW_GAP_MM) / n
    gs = fig.add_gridspec(
        n,
        n,
        left=MARGINS_MM["left"] / width_mm,
        right=1 - MARGINS_MM["right"] / width_mm,
        bottom=MARGINS_MM["bottom"] / height_mm,
        top=1 - MARGINS_MM["top"] / height_mm,
        wspace=COLUMN_GAP_MM / panel_w,
        hspace=ROW_GAP_MM / panel_h,
    )
    norm_rho = Normalize(-1.0, 1.0)
    cmap = plt.get_cmap(CORRELATION_CMAP)
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            if j > i:  # upper triangle: correlation matrix
                ax.set_facecolor(cmap(norm_rho(rho[i, j])))
                ax.text(
                    0.5,
                    0.5,
                    fc.signed(rho[i, j]),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="white" if abs(rho[i, j]) > 0.6 else "black",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            ax.set_xlim(-half[j], half[j])
            ax.set_xticks([-ticks[j], 0.0, ticks[j]])
            if i == j:
                draw_marginal(ax, draws[:, i], half[i], prior_mean[i], prior_sd[i])
                unit = QUANTITIES[i][2]
                ax.set_title(
                    f"σ {short_number(sd[i])} {unit}\n"
                    f"prior {prior_text(prior_sd[i], unit)}",
                    pad=1.5,
                )
            else:
                draw_joint(ax, draws[:, j], draws[:, i], half[j], half[i])
                ax.set_ylim(-half[i], half[i])
                ax.set_yticks([-ticks[i], 0.0, ticks[i]])
            if j == 0 and i > 0:
                ax.set_ylabel(label(i))
            elif i != j:
                ax.tick_params(labelleft=False)
            if i == n - 1:
                ax.set_xlabel(label(j))
                # Vertical tick labels: horizontal ones of neighbouring columns touch.
                ax.tick_params(axis="x", labelrotation=90)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(length=2, pad=1.5)
    return rho, sd


def draw_marginal(ax, x, half, prior_mean, prior_sd):
    edges = np.linspace(-half, half, N_BINS_1D + 1)
    density, _ = np.histogram(x, edges, density=True)
    # density=True normalises over the plotted range; rescale to all draws.
    density *= np.mean(np.abs(x) <= half)
    ax.stairs(density, edges, fill=True, color=POSTERIOR_FILL, lw=0)
    ax.stairs(density, edges, color="black", lw=0.6)
    grid = np.linspace(-half, half, 200)
    ax.plot(
        grid,
        norm.pdf(grid, prior_mean, prior_sd),
        color=PRIOR_COLOR,
        lw=1.2,
        ls=(0, (4, 1.5)),
    )
    # The baseline sits above the axis line so the nearly flat prior stays visible.
    ax.set_ylim(-0.1 * density.max(), 1.15 * density.max())
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def draw_joint(ax, x, y, half_x, half_y):
    edges_x = np.linspace(-half_x, half_x, N_BINS_2D + 1)
    edges_y = np.linspace(-half_y, half_y, N_BINS_2D + 1)
    counts, _, _ = np.histogram2d(x, y, [edges_x, edges_y])
    mesh = ax.pcolormesh(
        edges_x,
        edges_y,
        counts.T,
        cmap="Greys",
        vmin=0.0,
        vmax=counts.max() * 1.25,
        rasterized=True,
    )
    mesh.set_edgecolor("face")
    smooth = gaussian_filter(counts, SMOOTH_BINS)
    levels = fc.density_levels(smooth, PROBABILITIES)
    centres_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centres_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    ax.contour(
        centres_x,
        centres_y,
        smooth.T,
        levels=sorted(levels),
        colors="black",
        linewidths=[0.5, 0.9],
        linestyles=["--", "-"],
    )


def draw_legend_and_colourbar(fig, n):
    handles = [
        Patch(
            facecolor=POSTERIOR_FILL,
            edgecolor="black",
            lw=0.6,
            label="Posterior marginal",
        ),
        Line2D([], [], color="black", lw=0.9, label="68 % region"),
        Line2D([], [], color="black", lw=0.5, ls="--", label="95 % region"),
        Line2D(
            [],
            [],
            color=PRIOR_COLOR,
            lw=1.2,
            ls=(0, (4, 1.5)),
            label="Prior density (same scale)",
        ),
    ]
    width_mm = paper_style.DOUBLE_COLUMN_MM
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1 - MARGINS_MM["right"] / width_mm, 1.0),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.2,
        borderaxespad=0.3,
        title=f"DOG {TARGET_DOG}; every axis is the deviation from the posterior mean",
    )
    top = 1 - MARGINS_MM["top"] / HEIGHT_MM
    grid_h = (HEIGHT_MM - MARGINS_MM["top"] - MARGINS_MM["bottom"]) / HEIGHT_MM
    cax = fig.add_axes(
        [
            1 - (MARGINS_MM["right"] - 1.5) / width_mm,
            top - 0.42 * grid_h - 1.0 / HEIGHT_MM,
            2.5 / width_mm,
            0.42 * grid_h,
        ]
    )
    bar = fig.colorbar(ScalarMappable(Normalize(-1, 1), CORRELATION_CMAP), cax=cax)
    bar.set_ticks([-1.0, 0.0, 1.0], labels=["−1", "0", "1"])
    bar.ax.set_yticks([-0.5, 0.5], minor=True)
    bar.set_label("Posterior correlation ρ (upper triangle)")
    bar.ax.tick_params(length=2, pad=1.5)


def report_signs(pooled, draws, rho):
    """Print the R4 2c correlations (Eq. 6 δt) and their internal-convention signs."""
    names = [q[0] for q in QUANTITIES]
    i_dt = names.index(f"DOG{TARGET_DOG}_dt_Eq6")
    i_dc = names.index(f"DOG{TARGET_DOG}_dc_mean")
    i_lzu = names.index(f"lever_z_minus_DOG{TARGET_DOG}_U")
    dt_dc, dt_lzu = rho[i_dt, i_dc], rho[i_dt, i_lzu]
    # The internal time bias is offset - δt, so its correlations flip sign.
    internal = -draws[:, i_dt]
    print(
        f"DOG {TARGET_DOG}: corr(δt, δc offset) = {fc.signed(dt_dc, '+.3f')} "
        f"(R4 2c expects > 0: {'match' if dt_dc > 0 else 'MISMATCH'})"
    )
    print(
        f"DOG {TARGET_DOG}: corr(δt, l_z − U) = {fc.signed(dt_lzu, '+.3f')} "
        f"(R4 2c expects < 0: {'match' if dt_lzu < 0 else 'MISMATCH'})"
    )
    print(
        "Same with the internal time bias (offset − δt): "
        f"{fc.signed(fc.correlation(internal, draws[:, i_dc]), '+.3f')}, "
        f"{fc.signed(fc.correlation(internal, draws[:, i_lzu]), '+.3f')}"
    )


if __name__ == "__main__":
    fc.require(RUN_DIR / "chain_00.npz")
    paper_style.apply()
    # The grid is placed by hand (fixed gaps for the sigma titles); constrained
    # layout would otherwise be switched back on when the PDF is saved.
    plt.rcParams["figure.constrained_layout.use"] = False
    pooled = fc.pooled_run(RUN)
    draws, prior_mean, prior_sd = key_draws_and_priors(pooled)
    fig = plt.figure(
        figsize=(paper_style.DOUBLE_COLUMN_MM / 25.4, HEIGHT_MM / 25.4),
        layout="none",
    )
    rho, sd = draw_corner(fig, draws, prior_mean, prior_sd)
    draw_legend_and_colourbar(fig, len(QUANTITIES))
    fig.align_xlabels()
    fig.align_ylabels()
    for (name, _, unit, _), s, pm, ps in zip(
        QUANTITIES, sd, prior_mean, prior_sd, strict=True
    ):
        print(
            f"{name:26s} posterior sd {s:9.3f} {unit:5s} prior mean {pm:+10.1f} "
            f"sd {ps:9.1f}  ratio {s / ps:.4f}"
        )
    report_signs(pooled, draws, rho)
    fc.save(fig, OUTPUT_NAME)
