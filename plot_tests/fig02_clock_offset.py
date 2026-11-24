"""Timing bias between the DOG clocks and GNSS time.

Question: do the data alone fix the timing bias of each DOG, including its integer
second?
Message: yes. The circular variance of the residuals has one minimum, in the
second-wide cell of the estimate (a). The RMSE has a narrow well at every whole
second, and the well at the estimate is by far the deepest (b). Every other whole
second fits several times worse even after the position is re-estimated (c),
because a timing bias wrong by 1 s leaves a large residual that changes sign as
the vessel passes over the DOG, while the residuals at the estimate stay flat (d).

Panels a and b evaluate Eqs. (A22) and (A23) on one grid of trial timing biases:
every SCAN_STEP_S from -SCAN_HALF_WIDTH_S to +SCAN_HALF_WIDTH_S about the estimate,
which includes every whole second. At each trial value the arrivals are paired
again with the emissions (Eq. A20). Nothing is interpolated or smoothed.

Inputs: Output/Plot_Data/integer_pick_metrics_DOG{n}.npz (pairing inputs at the
Gauss-Newton solution) and revision/analysis/integer_ambiguity/*.csv (re-fits at
wrong seconds).
"""

import csv

import numpy as np

import _figure_common as fc
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
import _style as st

NAME = "fig02_clock_offset"
PICK_METRICS = str(
    fc.REPO / "Bermuda_Data/Output/Plot_Data/integer_pick_metrics_DOG{}.npz"
)
INTEGER_DIR = fc.REPO / "revision/analysis/integer_ambiguity"
INTEGER_CSV = str(INTEGER_DIR / "integer_ambiguity_DOG{}.csv")
SCAN_HALF_WIDTH_S = 20.0
SCAN_STEP_S = 0.005
MECHANISM_DOG = 3  # panel d; the other DOGs look the same
WINDOW_H = (3.0, 5.7)  # part of the survey shown in panel d (one pass over the DOG)
PAIRING_GATE_S = 0.5  # as in fc.alignment_scan
SECOND_STYLE = {
    -1: {"color": "#0072B2", "label": "$\\delta t$ 1 s too small"},
    0: {"color": "black", "label": "Estimated $\\delta t$"},
    1: {"color": "#D55E00", "label": "$\\delta t$ 1 s too large"},
}
SHIFT_LABEL = "Shift of the timing bias $\\delta t$ from its estimate (s)"


def load_alignment_inputs(dog):
    """DOG arrivals, GNSS epochs, modelled travel times and the estimated bias (s)."""
    data = np.load(PICK_METRICS.format(dog))
    offset_s = float(data["best_offset"] - data["time_bias"])  # Eq. 6 convention
    return data["CDOG_data"], data["GPS_data"], data["times_guess"], offset_s


def refit_ratios(dog):
    """Integer shifts k and RMSE(k) / RMSE(0) after re-fitting the position at k."""
    with open(INTEGER_CSV.format(dog)) as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.array([int(row["k"]) for row in rows]),
        np.array([float(row["refit_ratio"]) for row in rows]),
    )


def paired_residuals(arrivals, epochs, travel_times, timing_bias_s):
    """Emission times (GPS s) and residuals (ms) of the pairing at one timing bias."""
    paired = two_pointer_index(
        timing_bias_s,
        PAIRING_GATE_S,
        np.ascontiguousarray(arrivals, dtype=float),
        np.ascontiguousarray(epochs, dtype=float),
        np.ascontiguousarray(travel_times, dtype=float),
        np.zeros((epochs.size, 3)),
        np.zeros(epochs.size),
    )
    return paired[2], (paired[1] - paired[3]) * 1e3


def main():
    st.apply()
    fig, axes = st.figure(st.DOUBLE_MM, 125.0, nrows=2, ncols=2)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    n_steps = int(round(SCAN_HALF_WIDTH_S / SCAN_STEP_S))
    shifts = np.arange(-n_steps, n_steps + 1) * SCAN_STEP_S
    whole = np.arange(-n_steps, n_steps + 1) % int(round(1.0 / SCAN_STEP_S)) == 0
    for dog in st.DOGS:
        style = st.DOG[dog]
        arrivals, epochs, travel_times, offset_s = load_alignment_inputs(dog)
        variance, rmse_s, _ = fc.alignment_scan(
            offset_s + shifts, arrivals, epochs, travel_times
        )
        rmse_ms = rmse_s * 1e3
        ax_a.plot(shifts, variance, color=style["color"], lw=0.9)
        ax_b.plot(shifts, rmse_ms, color=style["color"], lw=0.9)
        k, ratio = refit_ratios(dog)
        ax_c.plot(
            k,
            ratio,
            style["marker"] + "-",
            color=style["color"],
            lw=0.9,
            ms=4.5,
            label=style["label"],
        )
        best = np.argmin(rmse_ms)
        others = whole & (shifts != shifts[best])
        print(
            f"DOG {dog}: estimate {offset_s:.6f} s; v° minimum "
            f"{variance.min():.2e} (shift {shifts[np.argmin(variance)]:+.3f} s), "
            f"next whole seconds {variance[whole][n_steps // 200 - 1]:.2e}; R minimum "
            f"{rmse_ms[best]:.3f} ms at shift {shifts[best]:+.3f} s, lowest other "
            f"whole second {rmse_ms[others].min():.3f} ms; re-fit ratio at wrong "
            f"seconds ≥ {ratio[k != 0].min():.2f}"
        )

    for ax in (ax_a, ax_b):
        ax.set_yscale("log")
        ax.set_xlim(-SCAN_HALF_WIDTH_S, SCAN_HALF_WIDTH_S)
        ax.set_xlabel(SHIFT_LABEL)
    ax_a.set_ylabel("Circular variance $v^\\circ$")
    ax_b.set_ylabel("RMSE $R$ (ms)")

    ax_c.axhline(1.0, color="0.5", lw=0.6)
    ax_c.set_xticks(np.arange(-5, 6))
    ax_c.set_ylim(bottom=0.0)
    ax_c.set_xlabel("$\\delta t$ held this many seconds from its estimate (s)")
    ax_c.set_ylabel("Relative RMSE $R$ after re-fitting")
    ax_c.legend(loc="upper center")

    arrivals, epochs, travel_times, offset_s = load_alignment_inputs(MECHANISM_DOG)
    for k in (-1, 1, 0):
        clock_s, residual_ms = paired_residuals(
            arrivals, epochs, travel_times, offset_s + k
        )
        hours = st.hours(clock_s)
        inside = (hours >= WINDOW_H[0]) & (hours <= WINDOW_H[1])
        ax_d.scatter(
            hours[inside],
            residual_ms[inside],
            s=1.5,
            zorder=3,
            color=SECOND_STYLE[k]["color"],
            linewidths=0,
            rasterized=True,
        )
        ax_d.plot([], [], "o", ms=3.5, **SECOND_STYLE[k])
        print(
            f"DOG {MECHANISM_DOG}, δt shifted {k:+d} s: rms residual "
            f"{np.sqrt(np.mean(residual_ms**2)):.3f} ms"
        )
    ax_d.set_xlim(WINDOW_H)
    ax_d.set_ylim(-2.2, 3.0)
    ax_d.set_xlabel(st.TIME_LABEL)
    ax_d.set_ylabel(f"Residuals $\\epsilon$, DOG {MECHANISM_DOG} (ms)")
    ax_d.legend(
        loc="upper center", ncols=3, columnspacing=0.3, handletextpad=0.0, borderpad=0.3
    )

    for ax, letter in zip(axes.ravel(), "abcd", strict=True):
        st.panel_letter(ax, letter)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
