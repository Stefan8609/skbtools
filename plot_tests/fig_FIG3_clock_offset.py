"""FIG-3: how the DOG clock offset δt is self-calibrated from the 1 Hz pulse train.

Answers R3 #15 (why the fractional parts wrap when δt is wrong and why the circular
variance has a single global minimum), R2 3 (the circular-variance plus RMSE search
as a self-calibration, the one-way specific step) and R3 M1 (is the integer second
uniquely identified); absorbs Fig. A6.

What the data show, panel by panel:
(a) Fractional travel-time residuals (wrapped to ±0.5 s) against time for DOG 3: a
    wrong integer second pairs each arrival with another pulse, and the residuals
    then follow the range change over those seconds and wrap around the period; a
    fractional error only shifts them all equally.
(b) Circular variance 1 − R̄ against whole-second shifts of δt for each DOG: one
    global minimum. Stage 1 therefore fixes the integer second.
(c) Over ±2.5 s: the circular variance is flat within each second (blind to the
    fractional part) while the RMSE has a V-shaped tooth per second; stage 2 picks
    the fractional part from the RMSE inside the second chosen by stage 1.
(d) RMSE after re-fitting at δt + k for k = −5…5, all DOGs (R3 M1).
(e) DOG 3 range residual against line-of-sight speed for k = −1, 0, +1: a wrong
    second leaves a residual of k seconds times the radial speed.

Inputs: (a)-(c) the saved solution of Bermuda_Data_time_bias.py (current ESV
table, Gauss-Newton lever), Bermuda_Data/Output/Plot_Data/
integer_pick_metrics_DOG{1,3,4}.npz. (d)-(e) Integer_Ambiguity.py run for each DOG
(table from before the ray fix, Table 1 lever: the state of the submitted paper),
revision/analysis/integer_ambiguity/integer_ambiguity_DOG{n}.csv and
_mechanism.npz (written by run_integer_ambiguity_dogs.py there).
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

import _figure_common as fc
from plotting import paper_style

# --- configuration ---------------------------------------------------------------
NAME = "fig_FIG3_clock_offset"
DOGS = (1, 3, 4)
EXAMPLE_DOG = 3
PICK_METRICS = fc.REPO / "Bermuda_Data/Output/Plot_Data/integer_pick_metrics_DOG{}.npz"
INTEGER_DIR = fc.REPO / "revision/analysis/integer_ambiguity"
INTEGER_CSV = str(INTEGER_DIR / "integer_ambiguity_DOG{}.csv")
MECHANISM = str(INTEGER_DIR / "integer_ambiguity_DOG{}_mechanism.npz")
# (a): a wrong whole-second shift large enough that the residuals wrap, and a
# purely fractional error.
WRONG_INTEGER_S = 400
FRACTIONAL_ERROR_S = 0.3
GAP_S = 60.0  # lines in (a) break where pairs are further apart
# (b): whole-second shifts of δt; (c): fine scan around the solution.
WIDE_SHIFTS_S = np.arange(-300, 301, 1)
FINE_SHIFTS_S = np.round(np.arange(-2.5, 2.5 + 1e-9, 0.005), 6)
# (e): shifts shown and their colours/markers.
MECHANISM_SHIFTS = (-1, 0, 1)
SHIFT_STYLE = {
    -1: {"color": paper_style.CATEGORY_COLORS[0], "marker": "v"},
    0: {"color": "#555555", "marker": "o"},
    1: {"color": paper_style.CATEGORY_COLORS[1], "marker": "^"},
}
FIGURE_HEIGHT_MM = 150.0
# Axes rectangles (left, bottom, width, height) in mm; drawn without auto layout so
# that long titles and legends below the axes cannot squeeze the panels.
AXES_MM = {
    "a": (16.0, 101.0, 96.0, 40.0),
    "b": (128.0, 101.0, 43.0, 40.0),
    "c_cv": (16.0, 54.0, 42.0, 22.0),
    "c_rmse": (16.0, 30.0, 42.0, 24.0),
    "d": (74.0, 30.0, 40.0, 46.0),
    "e": (130.0, 30.0, 41.0, 46.0),
}

C = paper_style.RANGE_SOUND_SPEED_MPS


def load_solution(dog):
    """DOG arrivals, emission epochs, modelled travel times and δt of the solution."""
    data = np.load(str(PICK_METRICS).format(dog))
    dt_s = float(data["best_offset"] - data["time_bias"])  # Eq. 6 convention
    return data["CDOG_data"], data["GPS_data"], data["times_guess"], dt_s


def paired_residuals(dt_s, CDOG_data, GPS_data, travel_times):
    """Emission epoch (GPS s) and travel-time residual (s) of each pair at δt."""
    from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index

    n = GPS_data.size
    observed, arrival_model, modelled = two_pointer_index(
        dt_s, 0.5, CDOG_data, GPS_data, travel_times, np.zeros((n, 3)), np.zeros(n)
    )[1:4]
    return arrival_model - modelled, observed - modelled


def scientific(value):
    """Mathtext a×10ⁿ with one decimal."""
    exponent = int(np.floor(np.log10(value)))
    return rf"${value / 10**exponent:.1f}\times10^{{{exponent}}}$"


def read_integer_csv(dog):
    with open(INTEGER_CSV.format(dog)) as handle:
        rows = list(csv.DictReader(handle))
    k = np.array([int(r["k"]) for r in rows])
    fixed_us = np.array([float(r["rmse_us"]) for r in rows])
    refit_us = np.array([float(r["refit_rmse_us"]) for r in rows])
    return k, fixed_us, refit_us


def draw(solutions, wide, fine, integer, mechanism):
    paper_style.apply()
    # Fixed axes positions (AXES_MM); constrained layout would move them on save.
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            FIGURE_HEIGHT_MM / paper_style.MM_PER_INCH,
        ),
        layout="none",
    )

    def axes_mm(rect, **kwargs):
        """Axes at (left, bottom, width, height) in printed millimetres."""
        x0, y0, w, h = rect
        width, height = paper_style.DOUBLE_COLUMN_MM, FIGURE_HEIGHT_MM
        return fig.add_axes([x0 / width, y0 / height, w / width, h / height], **kwargs)

    ax_a = axes_mm(AXES_MM["a"])
    ax_b = axes_mm(AXES_MM["b"])
    ax_c_cv = axes_mm(AXES_MM["c_cv"])
    ax_c_rmse = axes_mm(AXES_MM["c_rmse"], sharex=ax_c_cv)
    ax_d = axes_mm(AXES_MM["d"])
    ax_e = axes_mm(AXES_MM["e"])

    # (a) fractional residuals for a wrong integer, a fractional error, the solution
    CDOG_data, GPS_data, travel_times, dt_hat = solutions[EXAMPLE_DOG]
    # The wrong-integer residuals wrap, so they are dots; the aligned ones are
    # drawn as lines (broken at data gaps) with distinct dash patterns.
    cases = [
        (WRONG_INTEGER_S, {"color": "#999999", "ls": "none", "marker": ".", "ms": 1.2}),
        (FRACTIONAL_ERROR_S, {"color": paper_style.CATEGORY_COLORS[0], "ls": "--"}),
        (0.0, {"color": paper_style.CATEGORY_COLORS[1], "ls": "-"}),
    ]
    for shift, style in cases:
        epoch, residual = paired_residuals(
            dt_hat + shift, CDOG_data, GPS_data, travel_times
        )
        wrapped = fc.wrap_to_period(residual)
        cv = fc.circular_variance(wrapped)
        hours = paper_style.hours_since_start(epoch)
        if style["ls"] != "none":
            gap = np.flatnonzero(np.diff(epoch) > GAP_S) + 1
            hours = np.insert(hours, gap, np.nan)
            wrapped = np.insert(wrapped, gap, np.nan)
        name = f"δt̂ {fc.signed(shift, '+g')} s" if shift else "δt̂ (solution)"
        artist = ax_a.plot(
            hours, wrapped, lw=1.2, label=f"{name}, 1 − R̄ = {scientific(cv)}", **style
        )[0]
        paper_style.rasterize_dense(artist, wrapped.size)
    ax_a.set_xlim(*paper_style.TIME_LIMITS_H)
    ax_a.set_ylim(-0.5, 0.5)
    ax_a.set_xlabel(paper_style.TIME_LABEL)
    ax_a.set_ylabel("Fractional residual (s)")
    ax_a.set_title(
        f"Stage 1 (DOG {EXAMPLE_DOG}): wrong integer second → fractional parts wrap",
        loc="left",
    )
    ax_a.legend(
        loc="center",
        bbox_to_anchor=(0.5, 85.5 / FIGURE_HEIGHT_MM),
        bbox_transform=fig.transFigure,
        ncol=3,
        markerscale=6,
        handlelength=2.2,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    # (b) circular variance against whole-second shifts, all DOGs
    for dog in DOGS:
        style = paper_style.DOG_STYLE[dog]
        ax_b.semilogy(WIDE_SHIFTS_S, wide[dog], color=style["color"], lw=0.8)
        best = WIDE_SHIFTS_S[np.argmin(wide[dog])]
        ax_b.semilogy(
            best,
            wide[dog].min(),
            style["marker"],
            color=style["color"],
            mec="black",
            mew=0.5,
            ms=5,
            ls="none",
            label=style["label"],
        )
    ax_b.set_xlim(WIDE_SHIFTS_S[0], WIDE_SHIFTS_S[-1])
    ax_b.set_xlabel(f"Whole-second shift of {fc.DT_SYMBOL} (s)")
    ax_b.set_ylabel("Circular variance 1 − R̄")
    ax_b.set_title("Stage 1: single minimum", loc="left")
    ax_b.set_xticks([-300, -150, 0, 150, 300])
    minima = {int(WIDE_SHIFTS_S[np.argmin(wide[dog])]) for dog in DOGS}
    ax_b.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        title="Min. at " + ", ".join(fc.signed(m, "d") for m in sorted(minima)) + " s",
        handletextpad=0.1,
        borderaxespad=0.2,
        alignment="left",
    )
    ax_b.set_ylim(1e-7, 5)

    # (c) fine scan: circular variance flat within a second, RMSE V-shaped
    cv, rmse_s = fine
    ax_c_cv.semilogy(
        FINE_SHIFTS_S, cv, color=paper_style.DOG_STYLE[EXAMPLE_DOG]["color"]
    )
    ax_c_rmse.semilogy(FINE_SHIFTS_S, rmse_s * 1e3, color="black")
    for ax in (ax_c_cv, ax_c_rmse):
        ax.axvspan(-0.5, 0.5, color="#DDDDDD", lw=0, zorder=0)
    ax_c_cv.set_ylim(1e-7, 1e-3)
    ax_c_rmse.set_ylim(0.05, 3000)
    ax_c_cv.set_yticks([1e-6, 1e-5, 1e-4])
    ax_c_rmse.set_yticks([0.1, 1, 10, 100])
    ax_c_cv.set_ylabel("1 − R̄")
    ax_c_rmse.set_ylabel("RMSE (ms)")
    ax_c_rmse.set_xlabel(f"{fc.DT_SYMBOL} − δt̂ (s)")
    ax_c_rmse.set_xlim(FINE_SHIFTS_S[0], FINE_SHIFTS_S[-1])
    plt.setp(ax_c_cv.get_xticklabels(), visible=False)
    ax_c_cv.set_title("Stage 2: RMSE → fraction", loc="left")
    ax_c_cv.text(
        0.97,
        0.93,
        f"DOG {EXAMPLE_DOG}",
        transform=ax_c_cv.transAxes,
        ha="right",
        va="top",
    )

    # (d) RMSE after re-fitting at δt + k, all DOGs
    for dog in DOGS:
        style = paper_style.DOG_STYLE[dog]
        k, fixed_us, refit_us = integer[dog]
        refit_cm = refit_us * 1e-6 * C * 100.0
        ax_d.semilogy(
            k,
            refit_cm,
            marker=style["marker"],
            color=style["color"],
            ms=3.5,
            label=f"{style['label']}: {refit_cm[k == 0][0]:.1f} cm at k = 0",
        )
    ax_d.set_xlabel(f"Integer shift k of {fc.DT_SYMBOL} (s)")
    ax_d.set_ylabel("RMSE after re-fit (cm)")
    ax_d.set_title("Check: integer second", loc="left")
    ax_d.set_xticks(np.arange(-4, 5, 2))
    ax_d.set_xlim(-5.5, 5.5)
    ax_d.set_ylim(10, 2000)
    ax_d.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        handletextpad=0.2,
        borderaxespad=0.0,
    )

    # (e) range residual against line-of-sight speed for k = -1, 0, +1
    for k in MECHANISM_SHIFTS:
        speed, residual_m = mechanism[k]
        slope, intercept, _ = fc.line_fit(speed, residual_m)
        style = SHIFT_STYLE[k]
        dots = ax_e.plot(
            speed,
            residual_m,
            style["marker"],
            ms=0.8,
            mew=0,
            color=style["color"],
            label=f"k = {fc.signed(k, '+d') if k else '0'}: slope "
            f"{fc.signed(slope, '+.2f')} s",
        )[0]
        paper_style.rasterize_dense(dots, speed.size)
        ends = np.array([np.nanmin(speed), np.nanmax(speed)])
        ax_e.plot(ends, slope * ends + intercept, color="black", lw=0.8)
    ax_e.set_xlabel("Line-of-sight speed (m/s)")
    ax_e.set_ylabel("Range residual (m)")
    ax_e.set_title("Why: residual ≈ k × speed", loc="left")
    ax_e.set_ylim(-3.3, 3.3)
    ax_e.text(
        0.5,
        0.04,
        f"DOG {EXAMPLE_DOG}",
        transform=ax_e.transAxes,
        ha="center",
        va="bottom",
    )
    for k, x, align in ((-1, 0.03, "left"), (1, 0.97, "right")):
        ax_e.text(
            x,
            0.96,
            f"k = {fc.signed(k, '+d')}",
            transform=ax_e.transAxes,
            ha=align,
            va="top",
            color=SHIFT_STYLE[k]["color"],
        )
    ax_e.legend(
        loc="upper center",
        markerscale=5,
        handletextpad=0.2,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, -0.24),
    )

    for ax, letter in zip((ax_a, ax_b, ax_c_cv, ax_d, ax_e), "abcde", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


def mechanism_data(dog):
    data = np.load(MECHANISM.format(dog))
    out = {}
    for k in MECHANISM_SHIFTS:
        keep = data[f"k{k}_inliers"]
        out[k] = (
            data[f"k{k}_radial_speed_mps"][keep],
            data[f"k{k}_residual_s"][keep] * C,
        )
    return out


if __name__ == "__main__":
    fc.require(
        *(str(PICK_METRICS).format(dog) for dog in DOGS),
        *(INTEGER_CSV.format(dog) for dog in DOGS),
        MECHANISM.format(EXAMPLE_DOG),
    )
    solutions = {dog: load_solution(dog) for dog in DOGS}
    wide = {}
    for dog in DOGS:
        CDOG_data, GPS_data, travel_times, dt_hat = solutions[dog]
        wide[dog] = fc.alignment_scan(
            dt_hat + WIDE_SHIFTS_S, CDOG_data, GPS_data, travel_times
        )[0]
        print(
            f"DOG {dog}: δt = {dt_hat:.6f} s; circular variance minimum at shift "
            f"{WIDE_SHIFTS_S[np.argmin(wide[dog])]} s; 1 − R̄ at 0, ±1: "
            f"{wide[dog][WIDE_SHIFTS_S == 0][0]:.2e}, "
            f"{wide[dog][WIDE_SHIFTS_S == -1][0]:.2e}, "
            f"{wide[dog][WIDE_SHIFTS_S == 1][0]:.2e}"
        )
    CDOG_data, GPS_data, travel_times, dt_hat = solutions[EXAMPLE_DOG]
    cv_fine, rmse_fine, _ = fc.alignment_scan(
        dt_hat + FINE_SHIFTS_S, CDOG_data, GPS_data, travel_times
    )
    inside = np.abs(FINE_SHIFTS_S) < 0.5
    print(
        f"DOG {EXAMPLE_DOG}: 1 − R̄ spread within ±0.5 s: "
        f"{np.ptp(cv_fine[inside]):.1e}; RMSE minimum at "
        f"{FINE_SHIFTS_S[np.argmin(rmse_fine)]:+.3f} s"
    )
    integer = {dog: read_integer_csv(dog) for dog in DOGS}
    for dog in DOGS:
        k, fixed_us, refit_us = integer[dog]
        ratio_fixed = fixed_us[k != 0] / fixed_us[k == 0]
        ratio_refit = refit_us[k != 0] / refit_us[k == 0]
        print(
            f"DOG {dog}: RMSE(k)/RMSE(0) fixed ≥ {ratio_fixed.min():.2f}, "
            f"re-fit ≥ {ratio_refit.min():.2f}"
        )
    mechanism = mechanism_data(EXAMPLE_DOG)
    fig = draw(solutions, wide, (cv_fine, rmse_fine), integer, mechanism)
    fc.save(fig, NAME)
