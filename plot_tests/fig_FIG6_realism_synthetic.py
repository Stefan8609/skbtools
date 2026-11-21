"""FIG-6: error budget and coverage of the realistic synthetics; residual realism.

Message: under realistic errors the time-varying ESV correction dominates the
position error, and the formal uncertainty of the paper's model covers the true
error far less often than its nominal level.

Answers R1 2g and R3 M8 (error budget), R3 M3 and R4 4 (the synthetic error model
must reproduce the real residuals) and R3 M2 (posterior width is not accuracy).
Synthetic surveys from ``Realistic_Synthetic`` (30 seeds × DOGs 1/3/4 per cell) and the
real data, all inverted with the paper's Gauss–Newton model (one δc, no drift, lever
arm fixed: at the truth for synthetics, at the ar1 posterior mean for the real data).

Main figure (``NAME``):
- (a) Error budget: each source alone, rms range residual and rms horizontal and
  vertical position error over seeds and DOGs, single sources sorted by horizontal
  error. Sources not simulated are hatched, with a bound from the repo documents
  where one exists, otherwise "not quantified".
- (b, c) |error| / formal sd, horizontal and vertical, for the white-noise and
  AR(1)-inflated sd against the |N(0, 1)| a correct sd would give. The 68 % and
  95 % coverage is printed for the caption.

Supplementary figure (``SUPPLEMENT_NAME``), are the synthetic residuals realistic:
- (a) Range residual in 15-min bins vs time: real data and the 5–95 % band of the
  synthetic surveys, both presets ("observed": δc(t) shaped like our own ar1_common30
  estimate, so not independent of the data; "random": random-walk δc(t)).
- (b) Residual autocorrelation vs lag: real data and 5 regenerated surveys per preset
  (seeds 0–4 of the coverage cells; the saved files keep only binned residuals, so
  these are recomputed, about 1 min, and cached in the output folder).
- (c, d) Residual rms and lag-1 (1 s) autocorrelation per DOG: real, 30 synthetic
  surveys per preset, and the submitted synthetics (App. B, Fig. B10).

Inputs: revision/analysis/realistic_synthetic/{budget,coverage,real_reference}.csv,
coverage/*.npz, real_reference.npz; Bermuda_Data GPS/DOG files and ESV tables for the
supplement's (b).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import _figure_common as fc
from bermuda_data import CDOG_GUESS_AUGMENTS, gps_data_path
from Inversion_Workflow.Synthetic import Realistic_Synthetic_Ensemble as ens
from Inversion_Workflow.Synthetic.Realistic_Synthetic import (
    SITE_ORIGIN_ECEF,
    realistic_synthetic,
)
from plotting import paper_style

# --- configuration ------------------------------------------------------------------
NAME = "fig_FIG6_realism_synthetic"
SUPPLEMENT_NAME = "fig_FIG6S_residual_realism"
RS_DIR = fc.REPO / "revision/analysis/realistic_synthetic"
BUDGET_CSV = RS_DIR / "budget.csv"
COVERAGE_CSV = RS_DIR / "coverage.csv"
COVERAGE_NPZ_DIR = RS_DIR / "coverage"
REAL_CSV = RS_DIR / "real_reference.csv"
REAL_NPZ = RS_DIR / "real_reference.npz"
DOGS = (1, 3, 4)
C_MPS = paper_style.RANGE_SOUND_SPEED_MPS
PRESETS = ("observed", "random")
# Supplement: the two presets as a darker and a lighter grey band, open and filled
# dots; the Bermuda data in the DOG colours.
PRESET_NAMES = {"observed": "observed-shape", "random": "random-walk"}
PRESET_BAND = {"observed": "0.55", "random": "0.85"}
# Coverage cells with the paper's model, lever fixed (panels a-d), and the cell of
# the z histograms (the preset closest to the real residuals, results doc §3).
LEVER_FIXED_CELL = "{preset}_lever_fixed"
Z_CELL = "observed_lever_fixed"
TIME_EDGES_H = np.arange(0.0, 15.0, 0.25)  # as Realistic_Synthetic_Ensemble bins
# Submitted paper, App. B.6 (Fig. B10, line 1098): synthetic range residual cσε.
SUBMITTED_SYNTHETIC_RESIDUAL_CM = 4.67

# (b): regenerated surveys and the real-data inversion of real_reference.py (same
# table, lever arm at the ar1 posterior mean, same starting positions).
ACF_SEEDS = range(5)
ACF_MAX_LAG_S = 6 * 3600
ACF_LAGS_S = np.unique(np.round(np.logspace(0, np.log10(ACF_MAX_LAG_S), 60))).astype(
    int
)
ACF_CACHE = fc.OUTPUT_DIR / "cache" / "FIG6_residual_acf.npz"
REAL_TABLE = "global_table_esv_extended"
REAL_LEVER_M = np.array([-12.959, 9.282, -14.157])

# (a): (label, budget cell or None, residual bound in cm for sources not simulated,
# its relation sign). Cells "only_<source>": that source alone, lever fixed. The
# first two rows are the totals; the single sources are sorted when drawn.
ESV_LOOKUP_RMS_S = 0.3e-6  # bilinear ESV lookup error, rms (REVISION_REFERENCE C3)
DC = fc.DC_NAME
TOTAL_ROWS = (
    (f"All sources, observed-shape {DC}", "observed_all"),
    (f"All sources, random-walk {DC}", "random_all"),
)
SOURCE_ROWS = (
    (f"{DC}, observed shape", "observed_only_dc_series"),
    (f"{DC}, random walk", "random_only_dc_series"),
    ("Lateral gradient, 5 mm/s/km", "random_only_dc_gradient"),
    ("Steady current, 3 cm/s", "observed_only_current_steady"),
    ("M2 tidal current, 2 cm/s", "observed_only_current_m2"),
    ("M2 sound speed, 10 mm/s", "random_only_dc_sinusoid"),
    ("GNSS common error, 2/2/5 cm", "observed_only_gnss_common"),
    ("GNSS antenna misfit, as observed", "random_only_gnss_relative"),
    ("GNSS antenna noise, 2 cm", "observed_only_gnss_relative"),
    ("Arrival timing noise, 20 µs", "observed_only_timing_noise"),
    ("Transmit delay, 30 µs", "observed_only_transmit_delay"),
    ("Clock drift, 1 ppb", "observed_only_clock_drift"),
)
# Not simulated: (label, residual bound in cm or None, relation sign).
NOT_SIMULATED_ROWS = (
    ("ESV table lookup", ESV_LOOKUP_RMS_S * C_MPS * 100, "≈"),  # C3
    ("Ray tracing, one static cast", 0.5, "≤"),  # S10
    ("Earth tide and loading", 0.1, "<"),  # S9a
    ("Multipath", None, ""),
)
TOTAL_COLOR, SOURCE_COLOR = "0.2", "0.6"
Z_EDGES = np.logspace(-2, 3, 26)
# (b, c): (column suffix, direct label, colour, line style).
SD_KINDS = (
    ("white", "white-noise sd", "0.5", "--"),
    ("ar1", "AR(1) sd", "black", "-"),
)
EXPECTED_COLOR = "0.88"
HEIGHT_MM = 158.0
SUPPLEMENT_HEIGHT_MM = 125.0


def ok_rows(path):
    table = pd.read_csv(path)
    return table[table["failed"].isna()]


def residual_series(
    DOG_num, CDOG_data, GPS_Coordinates, GPS_data, start, lever, tables
):
    """Emission time (s) and range residual (cm) of the paper's GN fit, time order."""
    settings = {"lever": "lever_fixed", "clock_drift": False}
    _, _, _, pairs, _ = ens.invert(
        CDOG_data, GPS_Coordinates, GPS_data, start, lever, settings, tables
    )
    CDOG_full, model_times, _, GPS_clock = pairs
    emission = np.round(GPS_clock - model_times)
    return emission, (model_times - CDOG_full) * C_MPS * 100


def residual_acfs():
    """ACF at ACF_LAGS_S of real and regenerated synthetic residuals (cached)."""
    if ACF_CACHE.exists():
        return dict(np.load(ACF_CACHE))
    out = {}
    tables = ens.load_table(REAL_TABLE)
    for dog in DOGS:
        data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{dog}.npz"))
        start = SITE_ORIGIN_ECEF + CDOG_GUESS_AUGMENTS[dog]
        t, r = residual_series(
            dog,
            data["CDOG_data"],
            data["GPS_Coordinates"],
            data["GPS_data"],
            start,
            REAL_LEVER_M.copy(),
            tables,
        )
        print(f"real DOG {dog}: residual rms {r.std():.2f} cm")
        out[f"real_DOG{dog}"] = fc.gapped_autocorrelation(t, r, ACF_MAX_LAG_S)[
            ACF_LAGS_S
        ]
    tables = ens.load_table(ens.GENERATION_TABLE)
    for preset in PRESETS:
        for seed in ACF_SEEDS:
            # Same knobs, seeds and starting points as Realistic_Synthetic_Ensemble.
            knobs = ens.with_random_directions(ens.full_preset(preset), seed)
            for dog in DOGS:
                CDOG_data, CDOG, GPS_Coordinates, GPS_data, _, truth = (
                    realistic_synthetic(dog, knobs, seed, *tables)
                )
                rng = np.random.default_rng([seed, dog, 2000])
                start = CDOG + rng.normal(0.0, ens.START_POSITION_SD_M, 3)
                t, r = residual_series(
                    dog,
                    CDOG_data,
                    GPS_Coordinates,
                    GPS_data,
                    start,
                    truth["lever"].copy(),
                    tables,
                )
                print(f"{preset} seed {seed} DOG {dog}: residual rms {r.std():.2f} cm")
                acf = fc.gapped_autocorrelation(t, r, ACF_MAX_LAG_S)[ACF_LAGS_S]
                out[f"{preset}_seed{seed}_DOG{dog}"] = acf
    ACF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ACF_CACHE, **out)
    return out


def binned_residual_bands(preset):
    """5th, 50th and 95th percentile over seeds of the binned residual, per DOG."""
    cell = LEVER_FIXED_CELL.format(preset=preset)
    paths = sorted(COVERAGE_NPZ_DIR.glob(f"{cell}_seed*.npz"))
    out = {}
    for dog in DOGS:
        stack = np.array([np.load(p)[f"DOG{dog}_residual_by_time_cm"] for p in paths])
        out[dog] = np.nanpercentile(stack, [5, 50, 95], axis=0)
    return out, len(paths)


def budget_table():
    """Per budget cell: rms residual, horizontal and vertical error (cm), and n."""
    budget = ok_rows(BUDGET_CSV)
    out = {}
    for label, cell in TOTAL_ROWS + SOURCE_ROWS:
        group = budget[budget["cell"] == cell]
        out[label] = {
            "residual": group["residual_rms_cm"].mean(),
            "horizontal": fc.horizontal_rms(group["E_err_cm"], group["N_err_cm"]),
            "vertical": float(np.sqrt(np.mean(group["U_err_cm"] ** 2))),
            "n": len(group),
        }
    return out


def plot_time_rows(subfig, bands, n_seeds, real_npz):
    axes = subfig.subplots(len(DOGS), 1, sharex=True)
    centre = 0.5 * (TIME_EDGES_H[1:] + TIME_EDGES_H[:-1])
    for ax, dog in zip(axes, DOGS, strict=True):
        for preset in PRESETS[::-1]:  # the observed band on top of the random one
            low, _, high = bands[preset][dog]
            ax.fill_between(
                centre, low, high, color=PRESET_BAND[preset], lw=0, step="mid"
            )
        style = paper_style.DOG_STYLE[dog]
        ax.step(
            centre,
            real_npz[f"lever_fixed_DOG{dog}_by_time"],
            where="mid",
            color=style["color"],
            lw=1.5,
        )
        ax.set_title(style["label"], loc="left", pad=2)
        ax.set_ylim(-95, 95)
        ax.set_yticks([-50, 0, 50])
    fc.despine(*axes)
    subfig.supylabel(
        "15-min mean range residual (cm)", fontsize=paper_style.FONT_SIZE_PT
    )
    axes[-1].set_xlabel(paper_style.TIME_LABEL)
    axes[-1].set_xlim(*paper_style.TIME_LIMITS_H)
    paper_style.panel_label(axes[0], "a")
    handles = [
        Patch(
            color=PRESET_BAND[p],
            label=f"Synthetic, {PRESET_NAMES[p]}: 5–95 % of {n_seeds} surveys",
        )
        for p in PRESETS
    ]
    subfig.legend(handles=handles, loc="outside upper right", ncols=2)
    return axes


def plot_acf(ax, acfs):
    lags_h = ACF_LAGS_S / 3600.0
    for preset in PRESETS[::-1]:
        curves = np.array(
            [acfs[f"{preset}_seed{s}_DOG{d}"] for s in ACF_SEEDS for d in DOGS]
        )
        ax.fill_between(
            lags_h,
            curves.min(axis=0),
            curves.max(axis=0),
            color=PRESET_BAND[preset],
            lw=0,
        )
    for dog in DOGS:
        style = paper_style.DOG_STYLE[dog]
        ax.plot(
            lags_h,
            acfs[f"real_DOG{dog}"],
            color=style["color"],
            lw=1.2,
        )
    ax.axhline(0.0, color=fc.CONTEXT, lw=0.5, zorder=0)
    ax.set_xscale("log")
    ticks_s = np.array([1, 60, 600, 3600, 6 * 3600])
    ax.set_xticks(ticks_s / 3600.0, ["1 s", "1 min", "10 min", "1 h", "6 h"])
    ax.minorticks_off()
    ax.set_xlim(lags_h[0], lags_h[-1])
    ax.set_ylim(-0.6, 1.02)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Residual autocorrelation")


def plot_strip(ax, values, real, ylabel):
    """Real value per DOG (DOG marker) beside the synthetic values of each preset."""
    for k, dog in enumerate(DOGS):
        style = paper_style.DOG_STYLE[dog]
        for m, preset in enumerate(PRESETS):
            v = values[preset][dog]
            x = k + (m - 0.5) * 0.3 + np.linspace(-0.06, 0.06, len(v))
            ax.plot(x, v, ".", color=PRESET_BAND[preset], markersize=3)
        ax.plot(
            k + 0.34,
            real[dog],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
            markersize=5,
        )
    ax.set_xticks(range(len(DOGS)), [paper_style.DOG_STYLE[d]["label"] for d in DOGS])
    ax.set_xlim(-0.6, len(DOGS) - 0.4)
    ax.set_ylabel(ylabel)


def plot_budget(subfig, budget):
    """(a): rows of sources, columns residual, horizontal and vertical error."""
    axes = subfig.subplots(1, 3, sharey=True)
    sources = sorted(SOURCE_ROWS, key=lambda row: -budget[row[0]]["horizontal"])
    labels = [row[0] for row in TOTAL_ROWS] + [row[0] for row in sources]
    labels += [row[0] for row in NOT_SIMULATED_ROWS]
    # A gap after the totals and before the sources that were not simulated.
    y = -np.arange(len(labels), dtype=float)
    y[len(TOTAL_ROWS) :] -= 0.6
    y[len(TOTAL_ROWS) + len(sources) :] -= 0.6
    position = dict(zip(labels, y, strict=True))
    columns = (
        ("residual", "Range residual, rms (cm)"),
        ("horizontal", "Horizontal error, rms (cm)"),
        ("vertical", "Vertical error, rms (cm)"),
    )
    for ax, (key, xlabel) in zip(axes, columns, strict=True):
        top = max(row[key] for row in budget.values())
        for label, _ in TOTAL_ROWS + SOURCE_ROWS:
            color = TOTAL_COLOR if (label, _) in TOTAL_ROWS else SOURCE_COLOR
            ax.barh(position[label], budget[label][key], 0.7, color=color, lw=0)
        for label, bound, sign in NOT_SIMULATED_ROWS:
            if key == "residual" and bound is not None:
                ax.barh(
                    position[label],
                    bound,
                    0.7,
                    facecolor="white",
                    edgecolor="0.3",
                    hatch="//////",
                    lw=0.5,
                )
                text = f"{sign} {bound:.1g} cm"
                ax.text(bound + 0.03 * top, position[label], text, va="center")
            else:
                ax.text(
                    0.03 * top,
                    position[label],
                    "not quantified",
                    va="center",
                    color="0.35",
                )
        ax.set_xlim(0, 1.05 * top)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.tick_params(axis="y", length=0)
    fc.despine(*axes)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(y[-1] - 0.6, y[0] + 0.6)
    paper_style.panel_label(axes[0], "a")
    return axes


def plot_z(ax, cell_rows, components, title):
    """|error| / formal sd of the estimates against the |N(0, 1)| of a correct sd."""
    ax.stairs(
        fc.abs_normal_bin_probability(Z_EDGES),
        Z_EDGES,
        fill=True,
        color=EXPECTED_COLOR,
        lw=0,
    )
    peaks = {}
    for kind, label, color, line_style in SD_KINDS:
        errors = np.concatenate([cell_rows[f"{a}_err_cm"] for a in components])
        sds = np.concatenate([cell_rows[f"{a}_sd_{kind}_cm"] for a in components])
        c68 = fc.coverage_fraction(errors, sds, 0.68)
        c95 = fc.coverage_fraction(errors, sds, 0.95)
        print(
            f"{title}, {label}: {100 * c68:.0f} % inside the 68 % interval, "
            f"{100 * c95:.0f} % inside the 95 % interval"
        )
        counts, _ = np.histogram(np.abs(errors / sds), Z_EDGES)
        fraction = counts / errors.size
        ax.stairs(fraction, Z_EDGES, color=color, lw=1.2, ls=line_style)
        top = np.argmax(fraction)
        peaks[label] = (np.sqrt(Z_EDGES[top] * Z_EDGES[top + 1]), fraction[top], color)
    ax.set_xscale("log")
    ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel(f"|true error| / formal sd, {title}")
    return peaks


def plot_coverage(subfig, coverage):
    """(b, c): horizontal and vertical |error| / formal sd, shared y axis."""
    ax_h, ax_v = subfig.subplots(1, 2, sharey=True)
    z_rows = coverage[coverage["cell"] == Z_CELL]
    for ax, components, title in ((ax_h, "EN", "horizontal"), (ax_v, "U", "vertical")):
        peaks = plot_z(ax, z_rows, components, title)
        # Labels above the two peaks, pointing away from each other.
        for label, (x, height, color) in peaks.items():
            ax.annotate(
                label,
                (x, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="left" if label.startswith("white") else "right",
                va="bottom",
                color=color,
            )
    ax_h.annotate(
        "if the sd were right",
        (0.75, 0.21),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="0.35",
    )
    ax_h.set_ylabel("Fraction of estimates")
    ax_h.set_ylim(0, 0.42)
    ax_h.set_yticks([0, 0.2, 0.4])
    fc.despine(ax_h, ax_v)
    paper_style.panel_label(ax_h, "b")
    paper_style.panel_label(ax_v, "c")


def plot(coverage, budget):
    """Main figure: error budget above the coverage histograms."""
    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.9, 1.0])
    plot_budget(top, budget)
    plot_coverage(bottom, coverage)
    return fig


def plot_supplement(bands, n_seeds, real_npz, real_csv, acfs, coverage):
    """Supplement: synthetic against real residuals in time, lag, rms and lag 1."""
    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            SUPPLEMENT_HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1.0])
    plot_time_rows(top, bands, n_seeds, real_npz)

    ax_acf, ax_rms, ax_phi = bottom.subplots(1, 3, width_ratios=[1.5, 1, 1])
    plot_acf(ax_acf, acfs)
    lever_fixed = real_csv[real_csv["case"] == "lever_fixed"].set_index("DOG")
    for ax, column, ylabel in (
        (ax_rms, "residual_rms_cm", "Residual rms (cm)"),
        (ax_phi, "residual_phi_1s", "Lag-1 (1 s) autocorrelation"),
    ):
        values = {
            p: {
                d: coverage[
                    (coverage["cell"] == LEVER_FIXED_CELL.format(preset=p))
                    & (coverage["DOG"] == d)
                ][column].to_numpy()
                for d in DOGS
            }
            for p in PRESETS
        }
        plot_strip(ax, values, lever_fixed[column], ylabel)
    ax_rms.axhline(SUBMITTED_SYNTHETIC_RESIDUAL_CM, color="black", lw=0.8, ls="--")
    ax_rms.set_ylim(0, None)
    fc.despine(ax_acf, ax_rms, ax_phi)
    for ax, letter in zip((ax_acf, ax_rms, ax_phi), "bcd", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


if __name__ == "__main__":
    fc.require(BUDGET_CSV, COVERAGE_CSV, REAL_CSV, REAL_NPZ, COVERAGE_NPZ_DIR)
    paper_style.apply()  # after imports: some modules turn usetex on at import
    acfs = residual_acfs()
    bands = {}
    for preset in PRESETS:
        bands[preset], n_seeds = binned_residual_bands(preset)
    budget = budget_table()
    for label, row in budget.items():
        print(f"{label:45s}", {k: round(v, 1) for k, v in row.items()})
    coverage = ok_rows(COVERAGE_CSV)
    figure = plot(coverage, budget)
    fc.save(figure, NAME)
    plt.close(figure)
    figure = plot_supplement(
        bands, n_seeds, np.load(REAL_NPZ), pd.read_csv(REAL_CSV), acfs, coverage
    )
    fc.save(figure, SUPPLEMENT_NAME)
    plt.close(figure)
