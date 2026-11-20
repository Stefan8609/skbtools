"""FIG-6: realistic synthetics against the Bermuda data, error budget and coverage.

Answers R1 2g and R3 M8 (error budget), R3 M3 and R4 4 (the synthetic error model
must reproduce the real residuals) and R3 M2 (posterior width is not accuracy).
Synthetic surveys from ``Realistic_Synthetic`` (30 seeds × DOGs 1/3/4 per cell) and the
real data, all inverted with the paper's Gauss–Newton model (one δc, no drift, lever
arm fixed: at the truth for synthetics, at the ar1 posterior mean for the real data).

- (a) Range residual in 15-min bins vs time: real data and the 5–95 % band of the
  synthetic surveys, both presets ("observed": δc(t) shaped like our own ar1_common30
  estimate, so not independent of the data; "random": random-walk δc(t)).
- (b) Residual autocorrelation vs lag: real data and 5 regenerated surveys per preset
  (seeds 0–4 of the coverage cells; the saved files keep only binned residuals, so
  these are recomputed, about 1 min, and cached in the output folder).
- (c, d) Residual rms and lag-1 (1 s) autocorrelation per DOG: real, 30 synthetic
  surveys per preset, and the submitted synthetics (App. B, Fig. B10).
- (e) Error budget: each source alone, range residual and rms E, N, U error over
  seeds and DOGs. Sources not simulated are hatched, with a bound from the repo
  documents where one exists, otherwise "n.q." (not quantified).
- (f, g) |z| = |error| / formal sd for the white-noise and AR(1)-inflated sd against
  |N(0, 1)|, with the 68 % and 95 % coverage.

Inputs: revision/analysis/realistic_synthetic/{budget,coverage,real_reference}.csv,
coverage/*.npz, real_reference.npz; Bermuda_Data GPS/DOG files and ESV tables for (b).
"""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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
RS_DIR = fc.REPO / "revision/analysis/realistic_synthetic"
BUDGET_CSV = RS_DIR / "budget.csv"
COVERAGE_CSV = RS_DIR / "coverage.csv"
COVERAGE_NPZ_DIR = RS_DIR / "coverage"
REAL_CSV = RS_DIR / "real_reference.csv"
REAL_NPZ = RS_DIR / "real_reference.npz"
DOGS = (1, 3, 4)
C_MPS = paper_style.RANGE_SOUND_SPEED_MPS
PRESETS = ("observed", "random")
PRESET_LABELS = {
    "observed": "Synthetic, observed δc(t)",
    "random": "Synthetic, random δc(t)",
}
PRESET_COLORS = {"observed": "#999999", "random": "#0072B2"}
BAND_ALPHA = {"observed": 0.75, "random": 0.3}
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

# (e): (label, budget cell or None, residual bound in cm for sources not simulated,
# note printed next to it). Cells "only_<source>": that source alone, lever fixed.
ESV_LOOKUP_RMS_S = 0.3e-6  # bilinear ESV lookup error, rms (REVISION_REFERENCE C3)
BUDGET_ROWS = (
    ("All sources, observed δc(t) preset", "observed_all", None, ""),
    ("All sources, random δc(t) preset", "random_all", None, ""),
    ("ESV: common δc(t), observed shape", "observed_only_dc_series", None, ""),
    ("ESV: random-walk δc(t), 0.044 m/s/√h", "random_only_dc_series", None, ""),
    ("ESV lookup (bilinear)", None, ESV_LOOKUP_RMS_S * C_MPS * 100, "C3"),
    ("Ray tracing, one static cast", None, 0.5, "≤; S10"),
    ("Lateral δc gradient, 5 mm/s/km", "random_only_dc_gradient", None, ""),
    ("Current, depth mean 3 cm/s", "observed_only_current_steady", None, ""),
    ("Tide: M2 current, 2 cm/s", "observed_only_current_m2", None, ""),
    ("Tide: M2 sound speed, 10 mm/s", "random_only_dc_sinusoid", None, ""),
    ("Tide: solid earth + loading (diff.)", None, 0.1, "<; S9a"),
    ("GNSS absolute: 2/2/5 cm, τ 30 min", "observed_only_gnss_common", None, ""),
    ("GNSS relative + attitude: real misfit", "random_only_gnss_relative", None, ""),
    ("GNSS relative: iid 2 cm per antenna", "observed_only_gnss_relative", None, ""),
    ("Receive side: timing noise 20 µs", "observed_only_timing_noise", None, ""),
    ("Transmit side: constant 30 µs", "observed_only_transmit_delay", None, ""),
    ("Clock drift, 1 ppb", "observed_only_clock_drift", None, ""),
    ("Multipath", None, None, ""),
)
Z_EDGES = np.logspace(-2, 3, 26)
# Dark edge so the DOG-coloured real lines stay visible over the bands.
EDGE = [path_effects.Stroke(linewidth=2.2, foreground="0.15"), path_effects.Normal()]
SD_KINDS = {"white": ("White-noise sd", "#D55E00"), "ar1": ("AR(1) sd", "#000000")}


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
    """Per budget row: residual (mean of per-DOG rms) and rms E, N, U error, cm."""
    budget = ok_rows(BUDGET_CSV)
    out = {}
    for label, cell, _, _ in BUDGET_ROWS:
        if cell is None:
            continue
        group = budget[budget["cell"] == cell]
        out[label] = {"residual": group["residual_rms_cm"].mean()}
        for a in "ENU":
            out[label][a] = float(np.sqrt(np.mean(group[f"{a}_err_cm"] ** 2)))
        out[label]["n"] = len(group)
    return out


def plot_time_rows(subfig, bands, n_seeds, real_npz):
    axes = subfig.subplots(len(DOGS), 1, sharex=True)
    centre = 0.5 * (TIME_EDGES_H[1:] + TIME_EDGES_H[:-1])
    for ax, dog in zip(axes, DOGS, strict=True):
        for preset in PRESETS[::-1]:  # the observed band on top of the random one
            low, _, high = bands[preset][dog]
            ax.fill_between(
                centre,
                low,
                high,
                color=PRESET_COLORS[preset],
                alpha=BAND_ALPHA[preset],
                lw=0,
                step="mid",
            )
        style = paper_style.DOG_STYLE[dog]
        ax.step(
            centre,
            real_npz[f"lever_fixed_DOG{dog}_by_time"],
            where="mid",
            color=style["color"],
            lw=1.3,
            path_effects=EDGE,
        )
        ax.axhline(0.0, color="0.3", lw=0.5, ls=":")
        ax.set_ylabel(f"{style['label']}\n(cm)", rotation=0, ha="right", va="center")
        ax.set_ylim(-95, 95)
        ax.set_yticks([-50, 0, 50])
    subfig.supylabel("15-min mean range residual", fontsize=paper_style.FONT_SIZE_PT)
    axes[-1].set_xlabel(paper_style.TIME_LABEL)
    axes[-1].set_xlim(*paper_style.TIME_LIMITS_H)
    paper_style.panel_label(axes[0], "a")
    handles = [Line2D([], [], color="0.2", lw=1.3, label="Bermuda data")]
    handles += [
        Patch(
            color=PRESET_COLORS[p],
            alpha=BAND_ALPHA[p],
            label=f"{p.capitalize()} δc(t): 5–95 % of {n_seeds} surveys",
        )
        for p in PRESETS
    ]
    axes[0].legend(
        handles=handles,
        loc="lower right",
        ncols=3,
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.1,
        handlelength=1.2,
        columnspacing=1.0,
    )
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
            color=PRESET_COLORS[preset],
            alpha=BAND_ALPHA[preset],
            lw=0,
        )
    for dog in DOGS:
        style = paper_style.DOG_STYLE[dog]
        ax.plot(
            lags_h,
            acfs[f"real_DOG{dog}"],
            color=style["color"],
            lw=1.1,
            marker=style["marker"],
            markevery=8,
            markersize=3,
            path_effects=EDGE,
        )
    ax.axhline(0.0, color="0.5", lw=0.5, ls=":")
    ax.set_xscale("log")
    ticks_s = np.array([1, 60, 600, 3600, 6 * 3600])
    ax.set_xticks(ticks_s / 3600.0, ["1 s", "1 min", "10 min", "1 h", "6 h"])
    ax.minorticks_off()
    ax.set_xlim(lags_h[0], lags_h[-1])
    ax.set_ylim(-0.6, 1.02)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Residual autocorrelation")
    n = len(ACF_SEEDS) * len(DOGS)
    ax.set_title(f"band: range of {n} synthetic series", loc="right")


def plot_strip(ax, values, real, ylabel):
    """Real value per DOG (DOG marker) beside the synthetic values of each preset."""
    for k, dog in enumerate(DOGS):
        style = paper_style.DOG_STYLE[dog]
        for m, preset in enumerate(PRESETS):
            v = values[preset][dog]
            x = k + (m - 0.5) * 0.3 + np.linspace(-0.06, 0.06, len(v))
            ax.plot(x, v, ".", color=PRESET_COLORS[preset], markersize=2.5)
        ax.plot(
            k + 0.32,
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
    axes = subfig.subplots(1, 4, sharey=True)
    y = -np.arange(len(BUDGET_ROWS), dtype=float)
    columns = (
        ("residual", "Residual rms (cm)"),
        ("E", "rms E error (cm)"),
        ("N", "rms N error (cm)"),
        ("U", "rms U error (cm)"),
    )
    for ax, (key, xlabel) in zip(axes, columns, strict=True):
        top = max(v[key] for v in budget.values())
        for yy, (label, cell, bound, note) in zip(y, BUDGET_ROWS, strict=True):
            if cell is not None:
                value = budget[label][key]
                preset = cell.split("_")[0]
                ax.barh(yy, value, 0.7, color=PRESET_COLORS[preset], lw=0)
                if "transmit" in cell and key == "residual":
                    ax.text(0.02 * top, yy, "absorbed by δt", va="center")
                continue
            if key == "residual" and bound is not None:
                ax.barh(
                    yy,
                    bound,
                    0.7,
                    facecolor="white",
                    edgecolor="0.3",
                    hatch="////",
                    lw=0.5,
                )
                prefix = note.split(";")[0] if ";" in note else ""
                source = note.split(";")[-1].strip()
                ax.text(
                    bound + 0.02 * top,
                    yy,
                    f"{prefix}{bound:.2g} cm ({source})",
                    va="center",
                    color="0.3",
                )
            else:
                ax.text(
                    0.02 * top, yy, "n.q.", va="center", color="0.3", style="italic"
                )
        ax.set_xlim(0, 1.08 * top)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="0.92", lw=0.5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [row[0] for row in BUDGET_ROWS])
    for tick in axes[0].get_yticklabels()[:2]:
        tick.set_fontweight("bold")
    axes[0].set_ylim(y[-1] - 0.6, y[0] + 0.6)
    paper_style.panel_label(axes[0], "e")
    handles = [
        Patch(
            facecolor="white",
            edgecolor="0.3",
            hatch="////",
            lw=0.5,
            label="not simulated: bound from docs",
        ),
        Line2D([], [], linestyle="none", label="n.q.: not quantified"),
    ]
    subfig.legend(handles=handles, loc="outside lower center", ncols=2)
    return axes


def plot_z(ax, cell_rows, components, title):
    for kind, (label, color) in SD_KINDS.items():
        z = np.concatenate(
            [
                cell_rows[f"{a}_err_cm"] / cell_rows[f"{a}_sd_{kind}_cm"]
                for a in components
            ]
        )
        errors = np.concatenate([cell_rows[f"{a}_err_cm"] for a in components])
        sds = np.concatenate([cell_rows[f"{a}_sd_{kind}_cm"] for a in components])
        c68 = fc.coverage_fraction(errors, sds, 0.68)
        c95 = fc.coverage_fraction(errors, sds, 0.95)
        counts, _ = np.histogram(np.abs(z), Z_EDGES)
        ax.stairs(
            counts / z.size,
            Z_EDGES,
            color=color,
            lw=1.2,
            label=f"{label}: {100 * c68:.0f} % in 68 %, {100 * c95:.0f} % in 95 %",
        )
    ax.stairs(
        fc.abs_normal_bin_probability(Z_EDGES),
        Z_EDGES,
        color="0.45",
        lw=0.9,
        ls="--",
        label="|N(0, 1)|: 68 % in 68 %, 95 % in 95 %",
    )
    for edge in (1.0, 1.96):
        ax.axvline(edge, color="0.6", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
    ax.set_xlabel("|z| = |true error| / formal sd")
    ax.set_ylabel("Fraction of estimates")
    ax.set_title(title, loc="left")
    ax.set_ylim(0, 1.75 * ax.get_ylim()[1])  # room for the legend above the data
    ax.legend(loc="upper left", borderaxespad=0.2, handlelength=1.5)


def plot(bands, n_seeds, real_npz, real_csv, acfs, coverage, budget):
    fig = plt.figure(figsize=(paper_style.DOUBLE_COLUMN_MM / 25.4, 225 / 25.4))
    top, row2, row3, row4 = fig.subfigures(4, 1, height_ratios=[1.2, 0.95, 1.75, 1.05])
    plot_time_rows(top, bands, n_seeds, real_npz)

    ax_acf, ax_rms, ax_phi = row2.subplots(1, 3, width_ratios=[1.6, 1, 1])
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
    ax_rms.text(
        len(DOGS) - 0.45,
        SUBMITTED_SYNTHETIC_RESIDUAL_CM + 1.0,
        "submitted (Fig. B10)",
        va="bottom",
        ha="right",
    )
    ax_rms.set_ylim(0, None)
    for ax, letter in zip((ax_acf, ax_rms, ax_phi), "bcd", strict=True):
        paper_style.panel_label(ax, letter)

    plot_budget(row3, budget)

    ax_h, ax_v = row4.subplots(1, 2)
    z_rows = coverage[coverage["cell"] == Z_CELL]
    plot_z(ax_h, z_rows, "EN", "Horizontal (E and N)")
    plot_z(ax_v, z_rows, "U", "Vertical (U)")
    paper_style.panel_label(ax_h, "f")
    paper_style.panel_label(ax_v, "g")
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
    figure = plot(
        bands,
        n_seeds,
        np.load(REAL_NPZ),
        pd.read_csv(REAL_CSV),
        acfs,
        ok_rows(COVERAGE_CSV),
        budget,
    )
    fc.save(figure, NAME)
    plt.close(figure)
