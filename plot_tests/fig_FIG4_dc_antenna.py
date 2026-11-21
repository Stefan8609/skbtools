"""FIG-4: the common ESV correction δc(t) beside the height of GNSS antenna 4.

Message: the ESV correction changes level where the height of GNSS antenna 4
steps, so it absorbs vessel-side GNSS error as well as sound speed.

(a) The common δc(t) of the final run, one value per window (posterior mean and 95 %
interval, mm/s). It has zero mean over the windows; each DOG's δc is its offset
plus this series. (b) Height of antenna 4 above where the rigid receiver grid
fitted to antennas 1-3 puts it (ship motion cancels): 1-s epochs and their 5-min
median. Vertical lines through both panels mark abrupt antenna-4 height steps
(R1 3c, R2 6, R3 M3). The per-DOG offsets are printed for the caption or a table;
the rigid-body misfit of the four-antenna fit is in FX-8. PROVISIONAL: the
antenna-4 GNSS QC (N3) is pending.

Window times: dc_windows.csv counts hours from the first window edge
(esv_window_edges_s[0] = 20116 s); the figure converts the run's edges (GPS_data
seconds) with paper_style.hours_since_start (origin 20056 s), a 60 s difference.

Inputs: Bermuda_Data/Output/MCMC_Chains/ar1_common30/{chain_*.npz,
analysis/dc_windows.csv, analysis/posterior_summary.csv} and the run's GNSS file.
"""

import csv

import numpy as np

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
CHAIN_DIR = fc.CHAINS_DIR / RUN
DC_WINDOWS_CSV = CHAIN_DIR / "analysis" / "dc_windows.csv"
SUMMARY_CSV = CHAIN_DIR / "analysis" / "posterior_summary.csv"
NAME = "fig_FIG4_dc_antenna"
DOGS = (1, 3, 4)
MEDIAN_MINUTES = 5.0  # median line of panel (b)
ANOMALY_LIMITS_M = (-1.2, 0.8)
HEIGHT_MM = 100.0
# -----------------------------------------------------------------------------------


def read_rows(path):
    """Rows of a CSV file as dicts of strings."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def common_dc(config):
    """Window edges (h since start) and common δc mean, 2.5 %, 97.5 % (mm/s)."""
    rows = read_rows(DC_WINDOWS_CSV)
    edges_s = np.asarray(config["esv_window_edges_s"], dtype=float)
    start_h = np.array([float(row["start_h"]) for row in rows])
    if len(rows) != edges_s.size - 1 or not np.allclose(
        edges_s[0] + 3600.0 * start_h, edges_s[:-1], atol=1.0
    ):
        raise ValueError(f"{DC_WINDOWS_CSV} windows do not match the run's edges")
    values = {
        key: 1000.0 * np.array([float(row[key]) for row in rows])
        for key in ("common_mean", "common_p2.5", "common_p97.5")
    }
    return paper_style.hours_since_start(edges_s), values


def dog_offsets():
    """Per-DOG ESV-correction offset: mean, 2.5 %, 97.5 % (mm/s), in DOGS order."""
    rows = {row["quantity"]: row for row in read_rows(SUMMARY_CSV)}
    out = []
    for dog in DOGS:
        row = rows[f"DOG{dog}_dc_mean"]
        if row["unit"] != "m/s":
            raise ValueError(f"DOG{dog}_dc_mean in {row['unit']}, expected m/s")
        out.append([1000.0 * float(row[key]) for key in ("mean", "p2.5", "p97.5")])
    return np.array(out)


def antenna_series(config):
    """GNSS epochs (h), antenna-4 height anomaly (m) and its steps (h, m)."""
    # Same anomaly and step detection as the FX-5 step marks.
    gps_s, anomaly, step_s, step_m = fc.antenna4_steps(config)
    hours = paper_style.hours_since_start(gps_s)
    return hours, anomaly, paper_style.hours_since_start(step_s), step_m


def draw(edges_h, common, hours, anomaly, steps_h):
    """Panels (a) and (b) on one time axis; returns the figure."""
    fig, (ax_dc, ax_an) = paper_style.figure(
        "double", HEIGHT_MM, nrows=2, sharex=True, height_ratios=(1.15, 1)
    )
    fc.despine(ax_dc, ax_an)

    ax_dc.stairs(
        common["common_p97.5"],
        edges_h,
        baseline=common["common_p2.5"],
        fill=True,
        color="0.78",
        lw=0,
    )
    ax_dc.stairs(common["common_mean"], edges_h, baseline=None, color="black", lw=1.0)
    ax_dc.axhline(0.0, color=fc.CONTEXT, lw=0.5, zorder=0)
    ax_dc.set_ylabel(f"{fc.DC_NAME} (mm/s)")

    edges = np.arange(
        paper_style.TIME_LIMITS_H[0],
        paper_style.TIME_LIMITS_H[1] + 1e-9,
        MEDIAN_MINUTES / 60.0,
    )
    points = ax_an.scatter(hours, anomaly, s=0.5, lw=0, color=fc.RAW_POINTS)
    paper_style.rasterize_dense(points, anomaly.size)
    centres, median, _ = fc.binned_percentiles(
        hours, anomaly, edges, (50.0,), min_count=20
    )
    ax_an.plot(centres, median[:, 0], color="black", lw=1.0)
    ax_an.axhline(0.0, color=fc.CONTEXT, lw=0.5, zorder=0)
    ax_an.set_ylim(*ANOMALY_LIMITS_M)
    ax_an.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax_an.set_ylabel("Antenna-4 height anomaly (m)")
    ax_an.set_xlabel(paper_style.TIME_LABEL)
    ax_an.set_xlim(*paper_style.TIME_LIMITS_H)
    ax_an.set_xticks(np.arange(0, 15, 2))

    # The antenna-4 height steps, through both panels.
    for ax in (ax_dc, ax_an):
        for step in steps_h:
            ax.axvline(step, color=fc.ACCENT, lw=0.6, zorder=1)
    ax_dc.annotate(
        "antenna-4 height steps",
        (steps_h[2], 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        color=fc.ACCENT,
        ha="left",
        va="top",
    )
    for ax, letter in zip((ax_dc, ax_an), "ab", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


if __name__ == "__main__":
    fc.require(CHAIN_DIR / "chain_00.npz", DC_WINDOWS_CSV, SUMMARY_CSV)
    paper_style.apply()
    pooled = fc.pooled_run(RUN)
    config = fc.run_config(pooled)
    edges_h, common = common_dc(config)
    offsets = dog_offsets()
    hours, anomaly, steps_h, steps_m = antenna_series(config)

    step_mean, step_sd = fc.window_steps(pooled["esv_common"])
    print("common δc steps at window edges (mm/s):")
    for edge, mean, sd in zip(edges_h[1:-1], step_mean, step_sd, strict=True):
        print(f"  {edge:6.3f} h  {1e3 * mean:+6.1f} ± {1e3 * sd:.1f}")
    after = edges_h[:-1] >= 10.5
    level_change = (
        common["common_mean"][after].mean() - common["common_mean"][~after].mean()
    )
    print(f"mean common δc after minus before 10.5 h: {level_change:+.1f} mm/s")
    print("antenna-4 steps and the window edge just before each:")
    paired_dc, paired_m = [], []
    for step_h, size_m in zip(steps_h, steps_m, strict=True):
        k = np.searchsorted(edges_h[1:-1], step_h) - 1
        lag = step_h - edges_h[1:-1][k] if k >= 0 else np.nan
        dc = 1e3 * step_mean[k] if k >= 0 else np.nan
        print(
            f"  {step_h:6.3f} h {size_m:+.2f} m; edge {lag:+.3f} h earlier, "
            f"δc step {dc:+.1f}"
        )
        if lag < 0.15:
            paired_dc.append(dc)
            paired_m.append(size_m)
    print(
        f"corr(δc step, antenna-4 step) over {len(paired_dc)} pairs: "
        f"{fc.correlation(paired_dc, paired_m):+.2f}"
    )
    mean_offset = offsets[:, 0].mean()
    for dog, (mean, low, high) in zip(DOGS, offsets, strict=True):
        print(
            f"DOG {dog} offset {mean:+.1f} mm/s (95 %: {low:+.1f} to {high:+.1f}); "
            f"mean of offsets {mean_offset:+.1f} inside: {low <= mean_offset <= high}"
        )
    fig = draw(edges_h, common, hours, anomaly, steps_h)
    fc.save(fig, NAME)
