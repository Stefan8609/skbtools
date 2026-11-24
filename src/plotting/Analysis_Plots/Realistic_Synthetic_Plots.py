"""Figures of the realistic synthetic experiments (FIG-6, FIG-7, per-obstacle figure).

- ``obstacle_figure``: for each knob alone, the range residual it leaves at the true
  DOG position, clock and lever arm, against time and against the ship's azimuth
  from the DOG (Bermuda track, DOG 3). It shows how each obstacle enters the data
  before any fit.
- ``realism_figure`` (FIG-6): (a) residual vs time, real data and synthetic
  realizations; (b) error budget, position error per obstacle; (c) coverage of the
  true position by the formal intervals.
- ``geometry_figure`` (FIG-7): horizontal and vertical errors per track, lever arm
  fixed vs free.

Inputs are the CSV/npz files of ``Realistic_Synthetic_Ensemble`` and
``revision/analysis/realistic_synthetic/real_reference.py``. Run from the
repository root with PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Synthetic import Realistic_Synthetic as rs
from Inversion_Workflow.Synthetic import Realistic_Synthetic_Ensemble as ens
from plotting import paper_style

RESULTS = ens.OUTPUT_DIR
C = paper_style.RANGE_SOUND_SPEED_MPS
OBSTACLES = {
    "Observed common δc(t)": {"dc_series": "observed"},
    "Random-walk δc(t), 0.044 m/s/√h": {
        "dc_series": "random_walk",
        "dc_random_walk_mps_per_sqrt_h": 0.044,
    },
    "δc sinusoid, 10 mm/s, M2": {"dc_sinusoid_mps": 0.01, "dc_sinusoid_phase_rad": 0.0},
    "δc gradient, 5 mm/s/km east": {"dc_gradient_mps_per_km": (0.005, 0.0)},
    "Current, 3 cm/s east": {"current_mps": (0.03, 0.0)},
    "M2 tidal current, 2 cm/s": {"current_m2_mps": 0.02, "current_m2_phase_rad": 0.0},
    "Clock drift, 1 ppb": {"drift_rate_s_per_s": 1e-9},
    "Transmit delay, 30 µs": {"transmit_delay_s": 30e-6},
    "GNSS common mode, 2/2/5 cm": {"gnss_common_sd_m": (0.02, 0.02, 0.05)},
    "GNSS antenna misfit (real)": {"gnss_relative": "real_misfit"},
}

BUDGET_LABELS = {
    "all": "All sources",
    "dc_series": "Random-walk δc(t)",
    "dc_sinusoid": "δc M2, 10 mm/s",
    "dc_gradient": "δc gradient, 5 mm/s/km",
    "current_steady": "Current, 3 cm/s",
    "current_m2": "M2 current, 2 cm/s",
    "clock_drift": "Clock drift, 1 ppb",
    "gnss_common": "GNSS common mode",
    "gnss_relative": "GNSS antenna misfit",
    "timing_noise": "Timing noise, 20 µs",
}


def residual_at_truth_cm(knobs, DOG_num, seed, tables):
    """Observed minus modelled range (cm) at the true parameters, per GPS epoch.

    The clock offset is removed with the true value only; drift and transmit delay
    stay in the residual, because the inversion has to absorb them.
    """
    knobs = {**knobs, "offset_s": 1991.0}
    CDOG_data, CDOG, GPS_Coordinates, GPS_data, _, truth = rs.realistic_synthetic(
        DOG_num, knobs, seed, *tables
    )
    observed = CDOG_data[:, 0] + CDOG_data[:, 1] - GPS_data - knobs["offset_s"]
    transponder = findTransponder(GPS_Coordinates, rs.GPS1_TO_OTHERS, truth["lever"])
    modelled, _ = calculateTimesRayTracing_Bias_Real(CDOG, transponder, 0.0, *tables)
    to_enu = rs.enu_rotation()
    line = (transponder - CDOG) @ to_enu.T
    azimuth = np.degrees(np.arctan2(line[:, 0], line[:, 1])) % 360
    hours = paper_style.hours_since_start(GPS_data)
    return hours, azimuth, (observed - modelled) * C * 100


def obstacle_figure(DOG_num=3, seed=0):
    """Per-obstacle residual at the truth vs time and azimuth (supplementary)."""
    paper_style.apply()
    tables = ens.load_table(ens.GENERATION_TABLE)
    n = len(OBSTACLES)
    fig, axes = paper_style.figure(
        "double", 24.0 * n, nrows=n, ncols=2, sharex="col", squeeze=False
    )
    style = paper_style.DOG_STYLE[DOG_num]
    for row, (label, knobs) in enumerate(OBSTACLES.items()):
        hours, azimuth, residual = residual_at_truth_cm(knobs, DOG_num, seed, tables)
        for ax, x in zip(axes[row], (hours, azimuth), strict=True):
            points = ax.scatter(x, residual, s=0.3, color=style["color"], lw=0)
            paper_style.rasterize_dense(points, len(x))
            ax.axhline(0.0, color="0.6", lw=0.5)
        axes[row, 0].set_ylabel("cm")
        axes[row, 0].set_title(label, loc="left")
    axes[-1, 0].set_xlabel(paper_style.TIME_LABEL)
    axes[-1, 0].set_xlim(*paper_style.TIME_LIMITS_H)
    axes[-1, 1].set_xlabel(f"Azimuth of the ship from {style['label']} (deg)")
    axes[-1, 1].set_xlim(0, 360)
    return fig


def realism_figure():
    """FIG-6: (a) residual vs time, (b) error budget, (c) coverage."""
    paper_style.apply()
    fig, axes = paper_style.figure("double", 82.0, ncols=3)

    # (a) DOG 3 residual in 15-min bins, real (lever fixed) and five synthetics.
    ax = axes[0]
    bins_h = np.arange(0.0, 17.25, 0.25)
    centre = 0.5 * (bins_h[1:] + bins_h[:-1])
    for seed in range(5):
        for preset, colour in (("observed", "0.55"), ("random", "#56B4E9")):
            path = RESULTS / "coverage" / f"{preset}_lever_fixed_seed{seed:04d}.npz"
            if path.exists():
                values = np.load(path)["DOG3_residual_by_time_cm"]
                label = f"Synthetic, {preset} δc(t)" if seed == 0 else None
                ax.plot(centre, values, color=colour, lw=0.6, label=label)
    real = np.load(RESULTS / "real_reference.npz")["lever_fixed_DOG3_by_time"]
    ax.plot(centre, real, color="black", lw=1.2, label="Bermuda data")
    ax.set(xlabel=paper_style.TIME_LABEL, ylabel="DOG 3 residual, 15-min mean (cm)")
    ax.set_xlim(*paper_style.TIME_LIMITS_H)
    ax.set_ylim(-90, 90)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.2)

    # (b) Error budget: rms horizontal and vertical error per obstacle alone.
    ax = axes[1]
    budget = pd.read_csv(RESULTS / "budget.csv")
    budget = budget[budget["failed"].fillna("") == ""]
    budget = budget[budget["cell"].str.startswith("random_")]
    rows = []
    for cell, group in budget.groupby("cell"):
        if not (cell.startswith("random_only_") or cell == "random_all"):
            continue
        horizontal = np.sqrt(np.mean(group["E_err_cm"] ** 2 + group["N_err_cm"] ** 2))
        vertical = np.sqrt(np.mean(group["U_err_cm"] ** 2))
        name = cell.replace("random_only_", "").replace("random_", "")
        if horizontal + vertical > 0.5:  # transmit delay, dropouts: no effect
            rows.append((BUDGET_LABELS.get(name, name), horizontal, vertical))
    rows.sort(key=lambda r: (r[0] == "All sources", r[1] + r[2]))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[1] for r in rows], 0.4, color="#0072B2", label="Horizontal")
    ax.barh(y - 0.2, [r[2] for r in rows], 0.4, color="#D55E00", label="Vertical")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("rms position error (cm)")
    ax.legend(loc="center right")

    # (c) Coverage: |error| / formal sd, white and AR(1), "random" preset.
    ax = axes[2]
    coverage = pd.read_csv(RESULTS / "coverage.csv")
    coverage = coverage[coverage["failed"].fillna("") == ""]
    cell = coverage[coverage["cell"] == "random_lever_fixed"]
    z_edges = np.logspace(-1, 3, 41)
    for kind, colour in (("white", "#999999"), ("ar1", "#0072B2")):
        z = np.concatenate(
            [
                np.abs(cell[f"{a}_err_cm"] / cell[f"{a}_sd_{kind}_cm"])
                for a in ("E", "N", "U")
            ]
        )
        label = {"white": "White-noise sd", "ar1": "AR(1)-inflated sd"}[kind]
        ax.hist(z, z_edges, histtype="step", color=colour, label=label)
    ax.axvline(1.96, color="black", lw=0.6, ls="--")
    ax.set_ylim(0, 1.35 * ax.get_ylim()[1])  # room for the legend
    ax.set_xscale("log")
    ax.set(xlabel="|error| / formal sd (E, N, U)", ylabel="Count")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    for ax, letter in zip(axes, "abc", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


def geometry_figure():
    """FIG-7: horizontal and vertical error per track, lever fixed vs free."""
    paper_style.apply()
    data = pd.read_csv(RESULTS / "geometry.csv")
    data = data[(data["failed"].fillna("") == "") & (data["wrong_second"] == 0)]
    # A single circle leaves dt, dc and U inseparable: the paper's model is singular
    # there (stated in the caption), so only solvable tracks are plotted.
    data = data[~data["cell"].str.startswith("circle")]
    fig, axes = paper_style.figure("double", 60.0, ncols=2)
    cells = list(
        dict.fromkeys(data["cell"].str.replace(r"_lever_(fixed|free)$", "", regex=True))
    )
    x = np.arange(len(cells))
    for offset, lever, marker in (
        (-0.15, "lever_fixed", "o"),
        (0.15, "lever_free", "s"),
    ):
        for ax, column in zip(axes, ("horizontal", "U"), strict=True):
            values = []
            for name in cells:
                group = data[data["cell"] == f"{name}_{lever}"]
                if column == "horizontal":
                    err = np.hypot(group["E_err_cm"], group["N_err_cm"])
                else:
                    err = np.abs(group["U_err_cm"])
                values.append(np.sqrt(np.mean(err**2)))
            label = {"lever_fixed": "Lever arm fixed", "lever_free": "Lever arm free"}
            ax.plot(
                x + offset,
                values,
                marker,
                label=label[lever],
                color="black" if lever == "lever_fixed" else "#D55E00",
            )
    for ax, label in zip(
        axes, ("rms horizontal error (cm)", "rms vertical error (cm)"), strict=True
    ):
        ax.set_xticks(
            x, [c.capitalize().replace("_lost", "\nlost quadrant") for c in cells]
        )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].legend()
    for ax, letter in zip(axes, "ab", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


if __name__ == "__main__":
    for name, make in (
        ("fig_realistic_obstacles", obstacle_figure),
        ("fig_realistic_synthetic", realism_figure),
        ("fig_track_geometry", geometry_figure),
    ):
        figure = make()
        print(paper_style.save(figure, name))
        plt.close(figure)
