"""FIG-7: survey track geometry against position error, lever arm z fixed vs free.

Message: under realistic errors the real Bermuda track locates the DOGs about as
well as a uniform grid; estimating the vertical lever arm widens the vertical
error on the Bermuda track but hardly changes the horizontal error on any track.

Answers R1 3e (uniform track vs the real uneven track), R3 M7(i) (lost quadrant),
R2 5 and R4 2 (vertical lever arm: externally fixed vs estimated).
Realistic synthetic surveys ("random" preset: random-walk δc(t), M2 δc, δc gradient,
currents, GNSS errors, clock drift, real dropout pattern), 30 seeds × DOGs 1/3/4 per
track, inverted with the paper's Gauss–Newton model (one δc). Errors are estimate
minus truth in local ENU at the site origin; the three DOGs are pooled.

One column per track, three rows:
- (a) The track (ship GPS1 position, east/north from the site origin). Bermuda is
  the processed real track; "− NW" also drops every epoch with the ship north-west
  of the DOG-array centre (light grey; R3 M7(i)).
- (b) East–north error of each survey (lever arm z fixed) and the 95 % ellipse
  across surveys with lever arm z fixed (solid) and free (dashed).
- (c) Up error of each survey with its mean ± 1 sd, lever arm z fixed and free.
The rms horizontal errors, the condition numbers, the l_z error and the single
circle (not invertible) are printed for the caption or a table.

Inputs: revision/analysis/realistic_synthetic/geometry.csv; the processed Bermuda
GPS file (track); the grid comes from ``track_en_m``, the function the generator
uses.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import _figure_common as fc
from bermuda_data import gps_data_path
from Inversion_Workflow.Synthetic import Realistic_Synthetic as rs
from plotting import paper_style

# --- configuration ------------------------------------------------------------------
NAME = "fig_FIG7_track_geometry"
GEOMETRY_CSV = fc.REPO / "revision/analysis/realistic_synthetic/geometry.csv"
BERMUDA_GPS = gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz")
DOGS = (1, 3, 4)
# (cell prefix, column title); "_lost" tracks drop the north-west part.
TRACKS = (
    ("bermuda", "Bermuda (real)"),
    ("grid", "Grid (uniform)"),
    ("bermuda_lost", "Bermuda − NW"),
    ("grid_lost", "Grid − NW"),
)
# Lever arm z: (label on the x axis of (c), ellipse line style, marker face).
LEVERS = {"lever_fixed": ("fixed", "-", "black"), "lever_free": ("free", "--", "white")}
ELLIPSE_PROBABILITY = 0.95
LOST_COLOR = "0.8"
HEIGHT_MM = 138.0
TRACK_LIMIT_KM = 10.0
UP_LIMIT_CM = 1000.0


def solved(table, cell):
    """Fits of a cell that converged on the right integer second."""
    rows = table[table["cell"] == cell]
    return rows[rows["failed"].isna() & (rows["wrong_second"] == 0)]


def track_shapes():
    """Ship E, N (m) per track shape and the DOG E, N (m), site-origin frame."""
    to_enu = rs.enu_rotation()
    data = np.load(BERMUDA_GPS)
    bermuda = (data["GPS_Coordinates"][:, 0] - rs.SITE_ORIGIN_ECEF) @ to_enu.T
    n = int(data["GPS_data"][-1] - data["GPS_data"][0]) + 1  # as idealized_trajectory
    dogs = {d: to_enu @ rs.CDOG_AUGMENTS[d] for d in DOGS}
    centre = np.mean(list(dogs.values()), axis=0)[:2]
    tracks = {"bermuda": bermuda[:, :2], "grid": rs.track_en_m("grid", n, centre)[0]}
    return tracks, dogs, centre


def draw_track(ax, track_en_m, dogs, centre, lost):
    """Track in black; with ``lost`` the NW-of-centre part is drawn light grey."""
    km = track_en_m / 1000.0
    north_west = (track_en_m[:, 0] < centre[0]) & (track_en_m[:, 1] > centre[1])
    kept = km.copy()
    if lost:
        kept[north_west] = np.nan
        dropped = km.copy()
        dropped[~north_west] = np.nan
        ax.plot(dropped[:, 0], dropped[:, 1], color=LOST_COLOR, lw=0.8)
    ax.plot(kept[:, 0], kept[:, 1], color="black", lw=0.8)
    for dog, enu in dogs.items():
        style = paper_style.DOG_STYLE[dog]
        ax.plot(
            enu[0] / 1000,
            enu[1] / 1000,
            marker=style["marker"],
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=3.5,
            zorder=3,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-TRACK_LIMIT_KM, TRACK_LIMIT_KM)
    ax.set_ylim(-TRACK_LIMIT_KM, TRACK_LIMIT_KM)
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])
    ax.set_xlabel("East (km)")


def ellipse(rows):
    """95 % ellipse (n, 2) of the E, N errors of a set of fits, cm."""
    en = rows[["E_err_cm", "N_err_cm"]].to_numpy()
    return fc.ellipse_points(en.mean(axis=0), np.cov(en.T), ELLIPSE_PROBABILITY)


def draw_ellipses(ax, table, prefix, limit_cm):
    """E-N errors (lever z fixed) and the 95 % ellipses, lever z fixed and free."""
    en = solved(table, f"{prefix}_lever_fixed")[["E_err_cm", "N_err_cm"]].to_numpy()
    ax.plot(en[:, 0], en[:, 1], ".", color=fc.RAW_POINTS, markersize=2.5)
    ax.axhline(0, color=fc.CONTEXT, lw=0.4, zorder=0)
    ax.axvline(0, color=fc.CONTEXT, lw=0.4, zorder=0)
    for lever, (_, line_style, _) in LEVERS.items():
        boundary = ellipse(solved(table, f"{prefix}_{lever}"))
        ax.plot(boundary[:, 0], boundary[:, 1], color="black", ls=line_style, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-limit_cm, limit_cm)
    ax.set_ylim(-limit_cm, limit_cm)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    ax.set_xlabel("East error (cm)")


def draw_up(ax, table, prefix):
    """Up error of each survey and its mean ± 1 sd, lever z fixed and free."""
    for x, (lever, (_, _, face)) in enumerate(LEVERS.items()):
        u = solved(table, f"{prefix}_{lever}")["U_err_cm"].to_numpy()
        ax.plot(
            x - 0.22 + np.linspace(-0.07, 0.07, u.size),
            u,
            ".",
            color=fc.RAW_POINTS,
            markersize=2.5,
        )
        ax.errorbar(
            x + 0.1,
            u.mean(),
            u.std(ddof=1),
            color="black",
            marker="o",
            markerfacecolor=face,
            capsize=2,
            lw=1.0,
            ls="none",
            markersize=4,
        )
    ax.axhline(0, color=fc.CONTEXT, lw=0.4, zorder=0)
    ax.set_xticks(range(len(LEVERS)), [label for label, _, _ in LEVERS.values()])
    ax.set_xlim(-0.6, len(LEVERS) - 0.4)
    ax.set_ylim(-UP_LIMIT_CM, UP_LIMIT_CM)
    ax.set_yticks([-800, -400, 0, 400, 800])
    ax.set_xlabel(f"Lever arm {fc.LEVER_LABELS[2]}")


def plot(table):
    fig, axes = paper_style.figure(
        "double", HEIGHT_MM, nrows=3, ncols=len(TRACKS), height_ratios=[1, 1, 0.8]
    )
    tracks, dogs, centre = track_shapes()
    limit = 1.05 * max(
        np.abs(ellipse(solved(table, f"{prefix}_{lever}"))).max()
        for prefix, _ in TRACKS
        for lever in LEVERS
    )
    for column, (prefix, title) in enumerate(TRACKS):
        ax_track, ax_en, ax_up = axes[:, column]
        shape = prefix.replace("_lost", "")
        draw_track(ax_track, tracks[shape], dogs, centre, prefix.endswith("_lost"))
        ax_track.set_title(title)
        draw_ellipses(ax_en, table, prefix, limit)
        draw_up(ax_up, table, prefix)
        if column:
            for ax in (ax_track, ax_en, ax_up):
                ax.tick_params(labelleft=False)
    fc.despine(*axes.flat)
    axes[0, 0].set_ylabel("North (km)")
    axes[1, 0].set_ylabel("North error (cm)")
    axes[2, 0].set_ylabel("Up error (cm)")
    for ax, letter in zip(axes[:, 0], "abc", strict=True):
        paper_style.panel_label(ax, letter)
    handles = [
        Line2D([], [], color="black", ls=line_style, label=f"lever arm z {label}")
        for label, line_style, _ in LEVERS.values()
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=2,
        title="95 % ellipse of the horizontal error in (b)",
    )
    return fig


def print_caption_numbers(table):
    """Numbers that left the figure: rms errors, condition number, l_z, circle."""
    for prefix, label in TRACKS:
        for lever in LEVERS:
            rows = solved(table, f"{prefix}_{lever}")
            en = rows[["E_err_cm", "N_err_cm"]]
            line = (
                f"{label:16s} {lever:12s} n={len(rows)} rms horizontal "
                f"{fc.horizontal_rms(en['E_err_cm'], en['N_err_cm']):.0f} cm; "
                f"E/N/U sd {rows['E_err_cm'].std():.0f}/{rows['N_err_cm'].std():.0f}/"
                f"{rows['U_err_cm'].std():.0f} cm; "
                f"U mean {rows['U_err_cm'].mean():+.0f}"
            )
            if lever == "lever_free":
                dz = rows["lever_z_err_cm"]
                line += f"; l_z error {dz.mean():+.0f} ± {dz.std(ddof=1):.0f} cm"
            print(line)
    for prefix in [prefix for prefix, _ in TRACKS] + ["circle"]:
        cells = table["cell"].str.rsplit("_lever", n=1).str[0]
        chosen = cells.isin([prefix, "circle_lost"] if prefix == "circle" else [prefix])
        values = table[chosen]["condition_number"].dropna()
        low, mid, high = np.percentile(values, [5, 50, 95])
        print(f"{prefix:16s} condition number {mid:.1e} ({low:.1e} to {high:.1e})")
    circle = table[table["cell"].str.startswith("circle")]
    n_solved = (circle["failed"].isna() & (circle["wrong_second"] == 0)).sum()
    print(f"circle: {n_solved}/{len(circle)} fits solved")


if __name__ == "__main__":
    fc.require(GEOMETRY_CSV, BERMUDA_GPS)
    paper_style.apply()  # after imports: some modules turn usetex on at import
    table = pd.read_csv(GEOMETRY_CSV)
    print_caption_numbers(table)
    figure = plot(table)
    fc.save(figure, NAME)
    plt.close(figure)
