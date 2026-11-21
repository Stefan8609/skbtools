"""FIG-3: how the DOG clock offset δt is found from the 1 Hz pulse train.

Message: the clock offset is found in two steps, the integer second from the
circular variance and the fraction from the RMSE, and the integer is unambiguous
because a wrong second leaves a residual proportional to the line-of-sight speed.

The procedure as the code runs it (``align_offset`` in Numba_xAline.py:
``find_int_offset`` then ``find_subint_offset``), for DOGs 1, 3 and 4:
(a) Stage 1. Circular variance 1 − R̄ of the fractional residuals against
    whole-second shifts of δt: one minimum, which selects the integer second.
(b) Stage 2. RMSE of the paired residuals against δt within ±0.5 s of that second
    (shaded): its minimum selects the fraction. The neighbouring seconds (grey)
    have their own minima, five to six times higher.
(c) Why the integer is unambiguous: range residual against line-of-sight speed
    with δt shifted by k = −1, 0, +1 s. A wrong second pairs each arrival with
    the vessel position k seconds away, so the residual is k s times the speed.
The final model has no clock drift, so no drift panel. The phase-wrap
illustration of the earlier version is supplement material.

Answers R3 #15, R2 3 and R3 M1; absorbs Fig. A6.

Inputs: (a), (b) the saved solution of Bermuda_Data_time_bias.py (current ESV
table, Gauss-Newton lever), Bermuda_Data/Output/Plot_Data/
integer_pick_metrics_DOG{1,3,4}.npz. (c) Integer_Ambiguity.py run for each DOG
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
PICK_METRICS = fc.REPO / "Bermuda_Data/Output/Plot_Data/integer_pick_metrics_DOG{}.npz"
INTEGER_DIR = fc.REPO / "revision/analysis/integer_ambiguity"
INTEGER_CSV = str(INTEGER_DIR / "integer_ambiguity_DOG{}.csv")
MECHANISM = str(INTEGER_DIR / "integer_ambiguity_DOG{}_mechanism.npz")
# (a): whole-second shifts of δt about the solution.
WIDE_SHIFTS_S = np.arange(-300, 301, 1)
# (b): shifts over the chosen second and its two neighbours, dense near each
# whole second where the RMSE has its minima.
NEAR_MINIMUM_S = np.r_[0.0, np.geomspace(1e-5, 0.02, 25)]
FINE_SHIFTS_S = np.unique(
    np.round(
        np.r_[
            np.arange(-1.5, 1.5 + 1e-9, 0.01),
            [
                k + sign * near
                for k in (-1, 0, 1)
                for sign in (-1, 1)
                for near in NEAR_MINIMUM_S
            ],
        ],
        7,
    )
)
STAGE2_HALF_WINDOW_S = 0.5  # find_subint_offset searches this far from stage 1
# (c): shifts shown.
MECHANISM_SHIFTS = (-1, 0, 1)
FIGURE_HEIGHT_MM = 122.0
# Axes rectangles (left, bottom, width, height) in mm.
AXES_MM = {
    "a": (17.0, 74.0, 66.0, 42.0),
    "b": (104.0, 74.0, 66.0, 42.0),
    "c": [(17.0 + i * 52.5, 12.0, 48.0, 42.0) for i in range(3)],
}
WRONG_COLOR = fc.RAW_POINTS  # wrong seconds
# The three DOG curves of (a) and (b) coincide: nested line widths keep all visible.
NESTED_LINE_WIDTH_PT = {1: 3.2, 3: 1.9, 4: 0.8}

C = paper_style.RANGE_SOUND_SPEED_MPS


def load_solution(dog):
    """DOG arrivals, emission epochs, modelled travel times and δt of the solution."""
    data = np.load(str(PICK_METRICS).format(dog))
    dt_s = float(data["best_offset"] - data["time_bias"])  # Eq. 6 convention
    return data["CDOG_data"], data["GPS_data"], data["times_guess"], dt_s


def read_integer_csv(dog):
    with open(INTEGER_CSV.format(dog)) as handle:
        rows = list(csv.DictReader(handle))
    k = np.array([int(r["k"]) for r in rows])
    fixed_us = np.array([float(r["rmse_us"]) for r in rows])
    refit_us = np.array([float(r["refit_rmse_us"]) for r in rows])
    return k, fixed_us, refit_us


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


def draw(wide, fine, mechanism):
    paper_style.apply()
    # Fixed axes positions (AXES_MM); constrained layout would move them on save.
    plt.rcParams["figure.constrained_layout.use"] = False
    width_mm = paper_style.DOUBLE_COLUMN_MM
    fig = plt.figure(
        figsize=(
            width_mm / paper_style.MM_PER_INCH,
            FIGURE_HEIGHT_MM / paper_style.MM_PER_INCH,
        ),
        layout="none",
    )

    def axes_mm(rect, **kwargs):
        """Axes at (left, bottom, width, height) in printed millimetres."""
        x0, y0, w, h = rect
        return fig.add_axes(
            [x0 / width_mm, y0 / FIGURE_HEIGHT_MM, w / width_mm, h / FIGURE_HEIGHT_MM],
            **kwargs,
        )

    ax_a = axes_mm(AXES_MM["a"])
    ax_b = axes_mm(AXES_MM["b"])
    ax_c = [axes_mm(AXES_MM["c"][0])]
    ax_c += [axes_mm(rect, sharex=ax_c[0], sharey=ax_c[0]) for rect in AXES_MM["c"][1:]]
    fc.despine(ax_a, ax_b, *ax_c)

    # (a) stage 1: circular variance against whole-second shifts
    for dog in DOGS:
        style = paper_style.DOG_STYLE[dog]
        ax_a.semilogy(
            WIDE_SHIFTS_S,
            wide[dog],
            color=style["color"],
            lw=NESTED_LINE_WIDTH_PT[dog],
            label=style["label"],
        )
    ax_a.axvline(0.0, color=fc.ACCENT, lw=0.8, zorder=0)
    ax_a.set_xlim(WIDE_SHIFTS_S[0], WIDE_SHIFTS_S[-1])
    ax_a.set_ylim(1e-7, 3)
    ax_a.set_xticks([-300, -150, 0, 150, 300])
    ax_a.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax_a.minorticks_off()
    ax_a.set_xlabel(f"Whole-second shift of {fc.DT_SYMBOL} (s)")
    ax_a.set_ylabel("Circular variance 1 − R̄")
    ax_a.text(-10, 1.6e-7, "chosen second", color=fc.ACCENT, va="bottom", ha="right")
    ax_a.legend(loc="lower right", handlelength=1.6, borderaxespad=0.0)

    # (b) stage 2: RMSE against δt in the chosen second and its neighbours
    chosen = np.abs(FINE_SHIFTS_S) <= STAGE2_HALF_WINDOW_S
    ax_b.axvspan(
        -STAGE2_HALF_WINDOW_S, STAGE2_HALF_WINDOW_S, color="0.93", lw=0, zorder=0
    )
    for dog in DOGS:
        rmse_ms = fine[dog] * 1e3
        for side in (FINE_SHIFTS_S <= -STAGE2_HALF_WINDOW_S, FINE_SHIFTS_S >= 0.5):
            ax_b.semilogy(FINE_SHIFTS_S[side], rmse_ms[side], color=WRONG_COLOR, lw=1.0)
        ax_b.semilogy(
            FINE_SHIFTS_S[chosen],
            rmse_ms[chosen],
            color=paper_style.DOG_STYLE[dog]["color"],
            lw=NESTED_LINE_WIDTH_PT[dog],
        )
    ax_b.plot([0.0, 0.0], [0.05, 500.0], color=fc.ACCENT, lw=0.8, zorder=0)
    ax_b.set_xlim(-1.5, 1.5)
    ax_b.set_ylim(0.05, 2000)
    ax_b.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_b.set_xticklabels([fc.MINUS + "1", fc.MINUS + "0.5", "0", "0.5", "1"])
    ax_b.set_yticks([0.1, 1, 10, 100, 1000])
    ax_b.set_yticklabels(["0.1", "1", "10", "100", "1000"])
    ax_b.minorticks_off()
    ax_b.set_xlabel(f"Shift of {fc.DT_SYMBOL} (s)")
    ax_b.set_ylabel("RMSE of paired residuals (ms)")
    ax_b.text(0.04, 0.062, f"chosen {fc.DT_SYMBOL}", color=fc.ACCENT, ha="left")
    ax_b.text(0.0, 1500, "chosen second", ha="center", va="top")
    for k in (-1, 1):
        ax_b.text(k, 1500, "wrong second", ha="center", va="top", color="0.45")

    # (c) range residual against line-of-sight speed for k = -1, 0, +1
    for ax, dog in zip(ax_c, DOGS, strict=True):
        style = paper_style.DOG_STYLE[dog]
        for k in MECHANISM_SHIFTS:
            speed, residual_m = mechanism[dog][k]
            dots = ax.plot(
                speed,
                residual_m,
                ".",
                ms=1.5,
                mew=0,
                color=style["color"] if k == 0 else WRONG_COLOR,
                zorder=3 if k == 0 else 2,
            )[0]
            paper_style.rasterize_dense(dots, speed.size)
        ax.text(0.5, 1.0, style["label"], transform=ax.transAxes, ha="center", va="top")
        # Direct labels to the right of the three clouds.
        ax.text(2.5, 2.3, "k = +1", ha="left", va="center", color="0.35")
        ax.text(2.5, 0.0, "k = 0", ha="left", va="center")
        ax.text(2.5, -2.3, f"k = {fc.MINUS}1", ha="left", va="center", color="0.35")
        ax.set_xlabel("Line-of-sight speed (m/s)")
    ax_c[0].set_xlim(-2.6, 3.5)
    ax_c[0].set_ylim(-3.6, 3.6)
    ax_c[0].set_xticks([-2, -1, 0, 1, 2])
    ax_c[0].set_yticks([-3, 0, 3])
    ax_c[0].set_ylabel("Range residual (m)")
    for ax in ax_c[1:]:
        ax.tick_params(labelleft=False)

    for ax, letter in zip((ax_a, ax_b, ax_c[0]), "abc", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


if __name__ == "__main__":
    fc.require(
        *(str(PICK_METRICS).format(dog) for dog in DOGS),
        *(INTEGER_CSV.format(dog) for dog in DOGS),
        *(MECHANISM.format(dog) for dog in DOGS),
    )
    wide, fine = {}, {}
    for dog in DOGS:
        CDOG_data, GPS_data, travel_times, dt_hat = load_solution(dog)
        wide[dog] = fc.alignment_scan(
            dt_hat + WIDE_SHIFTS_S, CDOG_data, GPS_data, travel_times
        )[0]
        cv_fine, fine[dog], _ = fc.alignment_scan(
            dt_hat + FINE_SHIFTS_S, CDOG_data, GPS_data, travel_times
        )
        inside = np.abs(FINE_SHIFTS_S) < 0.45
        print(
            f"DOG {dog}: δt = {dt_hat:.6f} s; circular variance minimum at shift "
            f"{WIDE_SHIFTS_S[np.argmin(wide[dog])]} s; 1 − R̄ at 0, ±1: "
            f"{wide[dog][WIDE_SHIFTS_S == 0][0]:.2e}, "
            f"{wide[dog][WIDE_SHIFTS_S == -1][0]:.2e}, "
            f"{wide[dog][WIDE_SHIFTS_S == 1][0]:.2e}; spread within ±0.45 s "
            f"{np.ptp(cv_fine[inside]):.1e}; RMSE minimum "
            f"{fine[dog].min() * 1e3:.3f} ms at "
            f"{FINE_SHIFTS_S[np.argmin(fine[dog])]:+.5f} s; "
            f"RMSE at −1, +1 s: {fine[dog][FINE_SHIFTS_S == -1][0] * 1e3:.3f}, "
            f"{fine[dog][FINE_SHIFTS_S == 1][0] * 1e3:.3f} ms"
        )
    mechanism = {dog: mechanism_data(dog) for dog in DOGS}
    for dog in DOGS:
        k, fixed_us, refit_us = read_integer_csv(dog)
        slopes = [fc.line_fit(*mechanism[dog][shift])[0] for shift in MECHANISM_SHIFTS]
        print(
            f"DOG {dog}: RMSE(k)/RMSE(0) fixed ≥ "
            f"{(fixed_us[k != 0] / fixed_us[k == 0]).min():.2f}, re-fit ≥ "
            f"{(refit_us[k != 0] / refit_us[k == 0]).min():.2f}; residual/speed slopes "
            f"(k = −1, 0, +1): " + ", ".join(f"{slope:+.2f} s" for slope in slopes)
        )
    fig = draw(wide, fine, mechanism)
    fc.save(fig, NAME)
