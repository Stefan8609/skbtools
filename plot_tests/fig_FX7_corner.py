"""Corner plot of the key parameters for one DOG, with priors and correlations (Fig. 7).

Message: the data, not the priors, determine every parameter, and the only strong
correlations tie U, the clock offset and the ESV-correction offset together.

Lower triangle: the 68 % and 95 % highest-density regions of the 2-D posteriors of
the final MCMC run. Diagonal: 1-D posterior marginals with the prior density on
the same density scale (thin grey line). Upper triangle: the posterior correlation
coefficients as plain numbers, lightly shaded where |ρ| ≥ 0.3. Every axis is the
deviation from the posterior mean in physical units, for one stated DOG (DOG 3, as
in the submitted Fig. 7).

Quantities come from the linear map of ``reported_quantities`` (the same map as the
run's analysis tables): DOG E, N, U in local ENU, the lever arm (body frame), the
DOG's ESV-correction offset and its clock offset δt (Eq. 6). l_z − U is not drawn;
its sd, prior and correlations are printed with the others for the caption or a
table. The priors of these linear combinations follow exactly from the sampler's
independent Gaussian priors (``fc.linear_gaussian_prior``).

Reviewer comments: R4 m2 and R1 4k (units, reference, target DOG, lever symbols),
R4 2b (prior on every marginal), R4 2c (signs of the δt correlations, printed by
this script), R3 #11 (no clipped labels, 8 pt, vector).
Inputs: the chains of run ``ar1_common30``. Run from the repo root with
PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
import numpy as np
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
HEIGHT_MM = 168.0
MARGINS_MM = {"left": 18.5, "right": 2.0, "top": 9.0, "bottom": 15.0}
COLUMN_GAP_MM = 1.5
ROW_GAP_MM = 1.5
HALF_WIDTH_SD = 4.0  # axis half-range in posterior sd
N_BINS_2D = 36
N_BINS_1D = 50
SMOOTH_BINS = 1.0  # Gaussian smoothing (bins) of the 2-D histograms before contouring
PROBABILITIES = (0.68, 0.95)
REGION_FILLS = ("0.84", "0.55")  # 95 % and 68 % regions
POSTERIOR_FILL = "0.84"
PRIOR_COLOR = "0.45"
STRONG_CORRELATION = 0.3  # |ρ| from which an upper-triangle cell is shaded
CORRELATION_SHADE = "0.90"
N_DRAWN = 8  # the first entries of QUANTITIES are drawn; all are printed
# (name in reported_quantities, axis symbol, unit, factor from SI units)
QUANTITIES = [
    (f"DOG{TARGET_DOG}_E_minus_Table1", "ΔE", "cm", 100.0),
    (f"DOG{TARGET_DOG}_N_minus_Table1", "ΔN", "cm", 100.0),
    (f"DOG{TARGET_DOG}_U_minus_Table1", "ΔU", "cm", 100.0),
    ("lever_x", fc.LEVER_LABELS[0], "cm", 100.0),
    ("lever_y", fc.LEVER_LABELS[1], "cm", 100.0),
    ("lever_z", fc.LEVER_LABELS[2], "cm", 100.0),
    (f"DOG{TARGET_DOG}_dc_mean", "ESV offset", "mm/s", 1000.0),
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


def draw_corner(fig, draws, prior_mean, prior_sd):
    """Corner of the first N_DRAWN quantities; returns all correlations and sds."""
    n = N_DRAWN
    sd = draws.std(axis=0, ddof=1)
    rho = np.corrcoef(draws, rowvar=False)
    ticks = [nice_tick(2.0 * s) for s in sd]
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
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            if j > i:  # upper triangle: correlation coefficients
                if abs(rho[i, j]) >= STRONG_CORRELATION:
                    ax.set_facecolor(CORRELATION_SHADE)
                ax.text(
                    0.5,
                    0.5,
                    fc.signed(rho[i, j]),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="black" if abs(rho[i, j]) >= STRONG_CORRELATION else "0.45",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            ax.set_xlim(-half[j], half[j])
            ax.set_xticks([-ticks[j], ticks[j]])
            if i == j:
                draw_marginal(ax, draws[:, i], half[i], prior_mean[i], prior_sd[i])
            else:
                draw_joint(ax, draws[:, j], draws[:, i], half[j], half[i])
                ax.set_ylim(-half[i], half[i])
                ax.set_yticks([-ticks[i], ticks[i]])
                fc.despine(ax)
            if j == 0 and i > 0:
                ax.set_ylabel(label(i))
            elif i != j:
                ax.tick_params(labelleft=False)
            if i == n - 1:
                ax.set_xlabel(label(j))
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(length=2, pad=1.5)
            ax.xaxis.set_major_formatter(lambda v, _: fc.signed(v, "g"))
            ax.yaxis.set_major_formatter(lambda v, _: fc.signed(v, "g"))
    return rho, sd


def draw_marginal(ax, x, half, prior_mean, prior_sd):
    edges = np.linspace(-half, half, N_BINS_1D + 1)
    density, _ = np.histogram(x, edges, density=True)
    # density=True normalises over the plotted range; rescale to all draws.
    density *= np.mean(np.abs(x) <= half)
    ax.stairs(density, edges, fill=True, color=POSTERIOR_FILL, lw=0)
    ax.stairs(density, edges, color="black", lw=0.6)
    ax.set_yticks([])
    grid = np.linspace(-half, half, 200)
    ax.plot(grid, norm.pdf(grid, prior_mean, prior_sd), color=PRIOR_COLOR, lw=0.7)
    # The baseline sits above the axis line so the nearly flat prior stays visible.
    ax.set_ylim(-0.1 * density.max(), 1.15 * density.max())
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def draw_joint(ax, x, y, half_x, half_y):
    """Filled 95 % and 68 % highest-density regions of a 2-D marginal."""
    edges_x = np.linspace(-half_x, half_x, N_BINS_2D + 1)
    edges_y = np.linspace(-half_y, half_y, N_BINS_2D + 1)
    counts, _, _ = np.histogram2d(x, y, [edges_x, edges_y])
    smooth = gaussian_filter(counts, SMOOTH_BINS)
    level_68, level_95 = fc.density_levels(smooth, PROBABILITIES)
    centres_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centres_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    ax.contourf(
        centres_x,
        centres_y,
        smooth.T,
        levels=[level_95, level_68, np.inf],
        colors=REGION_FILLS,
    )


def draw_legend(fig):
    handles = [
        Patch(facecolor=REGION_FILLS[1], lw=0, label="68 % region"),
        Patch(facecolor=REGION_FILLS[0], lw=0, label="95 % region"),
        Patch(
            facecolor=POSTERIOR_FILL,
            edgecolor="black",
            lw=0.6,
            label="Posterior marginal",
        ),
        Line2D([], [], color=PRIOR_COLOR, lw=0.7, label="Prior density"),
        Patch(
            facecolor=CORRELATION_SHADE,
            lw=0,
            label=f"Correlation ρ, |ρ| ≥ {STRONG_CORRELATION:g}",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.2,
    )


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
    draw_legend(fig)
    fig.align_xlabels()
    fig.align_ylabels()
    for (name, _, unit, _), s, pm, ps in zip(
        QUANTITIES, sd, prior_mean, prior_sd, strict=True
    ):
        print(
            f"{name:26s} posterior sd {s:9.3f} {unit:5s} prior mean {pm:+10.1f} "
            f"sd {ps:9.1f}  ratio {s / ps:.4f}"
        )
    names = [symbol.replace("$", "") for _, symbol, _, _ in QUANTITIES]
    print("posterior correlations (rows and columns: " + ", ".join(names) + ")")
    print(np.array2string(rho, precision=2, suppress_small=True, max_line_width=120))
    report_signs(pooled, draws, rho)
    fc.save(fig, OUTPUT_NAME)
