"""FIG-4: the common ESV correction δc(t) beside antenna-4 GNSS consistency.

(a) The common δc(t) of the final run, one value per window (posterior mean and 95 %
interval, mm/s). It has zero mean over the windows; each DOG's δc is its offset
plus this series. (b) The per-DOG offsets (mean and 95 % interval). (c) Height of
antenna 4 above where the rigid receiver grid fitted to antennas 1-3 puts it (ship
motion cancels). (d) Rigid-body misfit of the four-antenna fit used to place the
transducer, as in revision/analysis/dc_blocks/antenna_consistency.py. Dashed lines
mark abrupt antenna-4 height steps, so a reader can see whether δc steps line up
with them (R1 3c, R2 6, R3 M3). PROVISIONAL: the antenna-4 GNSS QC (N3) is pending.

Window times: dc_windows.csv counts hours from the first window edge
(esv_window_edges_s[0] = 20116 s); the figure converts the run's edges (GPS_data
seconds) with paper_style.hours_since_start (origin 20056 s), a 60 s difference.

Inputs: Bermuda_Data/Output/MCMC_Chains/ar1_common30/{chain_*.npz,
analysis/dc_windows.csv, analysis/posterior_summary.csv} and the run's GNSS file.
"""

import csv

import numpy as np

import _figure_common as fc
from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
CHAIN_DIR = fc.CHAINS_DIR / RUN
DC_WINDOWS_CSV = CHAIN_DIR / "analysis" / "dc_windows.csv"
SUMMARY_CSV = CHAIN_DIR / "analysis" / "posterior_summary.csv"
NAME = "fig_FIG4_dc_antenna"
DOGS = (1, 3, 4)
MEDIAN_MINUTES = 5.0  # median lines of panels (c) and (d)
ANOMALY_LIMITS_M = (-1.2, 0.8)
MISFIT_LIMITS_M = (0.0, 0.5)
ANTENNA4_COLOR = paper_style.CATEGORY_COLORS[1]
OTHERS_COLOR = "0.45"
HEIGHT_MM = 150.0
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
    """GNSS epochs (h), antenna-4 height anomaly (m), misfit (n, 4) m, steps."""
    inputs = ms.load_bermuda_inputs(config)
    coords = inputs["GPS_Coordinates"]
    grid = np.asarray(config["gps1_to_others"], dtype=float)
    # Rigid-body misfit exactly as in antenna_consistency.py.
    rotation, shift = ms.vessel_rotations(coords, config["gps1_to_others"])
    fitted = np.einsum("nab,kb->nka", rotation, grid) + shift[:, None, :]
    misfit = np.linalg.norm(coords - fitted, axis=2)
    # Same anomaly and step detection as the FX-5 triangles.
    gps_s, anomaly, step_s, step_m = fc.antenna4_steps(config)
    hours = paper_style.hours_since_start(gps_s)
    return hours, anomaly, misfit, paper_style.hours_since_start(step_s), step_m


def draw(edges_h, common, offsets, hours, anomaly, misfit, steps_h):
    """Panels (a)-(d) on one time axis; returns the figure."""
    fig = paper_style.figure("double", HEIGHT_MM)[0]
    fig.clear()
    grid = fig.add_gridspec(3, 2, width_ratios=(6, 1), height_ratios=(1.3, 1, 1))
    ax_dc = fig.add_subplot(grid[0, 0])
    ax_off = fig.add_subplot(grid[0, 1])
    ax_an = fig.add_subplot(grid[1, 0], sharex=ax_dc)
    ax_mis = fig.add_subplot(grid[2, 0], sharex=ax_dc)
    key_an, key_mis = fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 1])
    key_an.axis("off")
    key_mis.axis("off")

    ax_dc.stairs(
        common["common_p97.5"],
        edges_h,
        baseline=common["common_p2.5"],
        fill=True,
        color=paper_style.CATEGORY_COLORS[0],
        alpha=0.3,
        lw=0,
        label="95 % interval",
    )
    ax_dc.stairs(
        common["common_mean"],
        edges_h,
        baseline=None,
        color=paper_style.CATEGORY_COLORS[0],
        lw=1.0,
        label=f"Common {fc.DC_SYMBOL}(t), posterior mean",
    )
    ax_dc.axhline(0.0, color="0.5", lw=0.5)
    ax_dc.set_ylabel(f"Common {fc.DC_SYMBOL} (mm/s)")

    for x, (dog, (mean, low, high)) in enumerate(zip(DOGS, offsets, strict=True)):
        style = paper_style.DOG_STYLE[dog]
        ax_off.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            marker=style["marker"],
            color=style["color"],
            ms=5,
            capsize=2,
            lw=1.0,
        )
    ax_off.axhline(offsets[:, 0].mean(), color="0.4", lw=0.6, ls="--", zorder=0)
    ax_off.set_xticks(range(len(DOGS)), [str(dog) for dog in DOGS])
    ax_off.set_xlim(-0.6, len(DOGS) - 0.4)
    ax_off.set_xlabel("DOG")
    ax_off.set_ylabel(f"{fc.DC_NAME} offset (mm/s)")

    edges = np.arange(
        paper_style.TIME_LIMITS_H[0],
        paper_style.TIME_LIMITS_H[1] + 1e-9,
        MEDIAN_MINUTES / 60.0,
    )
    points = ax_an.scatter(
        hours,
        anomaly,
        s=0.5,
        lw=0,
        color=ANTENNA4_COLOR,
        alpha=0.25,
        label="1-s epochs",
    )
    paper_style.rasterize_dense(points, anomaly.size)
    centres, median, _ = fc.binned_percentiles(
        hours, anomaly, edges, (50.0,), min_count=20
    )
    ax_an.plot(centres, median[:, 0], color="black", lw=0.7, label="5-min median")
    ax_an.axhline(0.0, color="0.5", lw=0.5)
    ax_an.set_ylim(*ANOMALY_LIMITS_M)
    ax_an.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax_an.set_ylabel("Antenna-4 height\nanomaly (m)")

    centres, median4, _ = fc.binned_percentiles(
        hours, misfit[:, 3], edges, (50.0,), min_count=20
    )
    ax_mis.plot(centres, median4[:, 0], color=ANTENNA4_COLOR, lw=1.0, label="Antenna 4")
    for antenna in range(3):
        centres, median, _ = fc.binned_percentiles(
            hours, misfit[:, antenna], edges, (50.0,), min_count=20
        )
        ax_mis.plot(
            centres,
            median[:, 0],
            color=OTHERS_COLOR,
            lw=0.7,
            ls="--",
            label="Antennas 1–3" if antenna == 0 else None,
        )
    ax_mis.set_ylim(*MISFIT_LIMITS_M)
    ax_mis.set_yticks([0.0, 0.2, 0.4])
    ax_mis.set_ylabel("Rigid-body misfit,\n5-min median (m)")
    ax_mis.set_xlabel(paper_style.TIME_LABEL)
    ax_mis.set_xlim(*paper_style.TIME_LIMITS_H)

    for ax in (ax_dc, ax_an, ax_mis):
        for k, step in enumerate(steps_h):
            ax.axvline(
                step,
                color="0.25",
                lw=0.6,
                ls=":",
                zorder=0,
                label=(
                    f"Antenna-4 height step > {fc.ANTENNA4_STEP_MIN_M:g} m"
                    if ax is ax_dc and k == 0
                    else None
                ),
            )
    for ax in (ax_dc, ax_an):
        ax.tick_params(labelbottom=False)
    for ax, letter in zip((ax_dc, ax_off, ax_an, ax_mis), "abcd", strict=True):
        paper_style.panel_label(ax, letter)

    handles, labels = ax_dc.get_legend_handles_labels()
    ax_dc.legend(
        [handles[1], handles[0], handles[2]],
        [labels[1], labels[0], labels[2]],
        loc="upper left",
    )
    legend = key_an.legend(
        *ax_an.get_legend_handles_labels(),
        loc="center left",
        borderaxespad=0.0,
        markerscale=6,
        title="(c) Antenna 4",
    )
    legend.legend_handles[0].set_alpha(1.0)
    key_mis.legend(
        *ax_mis.get_legend_handles_labels(),
        loc="center left",
        borderaxespad=0.0,
        title="(d)",
    )
    return fig


if __name__ == "__main__":
    fc.require(CHAIN_DIR / "chain_00.npz", DC_WINDOWS_CSV, SUMMARY_CSV)
    paper_style.apply()
    pooled = fc.pooled_run(RUN)
    config = fc.run_config(pooled)
    edges_h, common = common_dc(config)
    offsets = dog_offsets()
    hours, anomaly, misfit, steps_h, steps_m = antenna_series(config)

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
    fig = draw(edges_h, common, offsets, hours, anomaly, misfit, steps_h)
    fc.save(fig, NAME)
