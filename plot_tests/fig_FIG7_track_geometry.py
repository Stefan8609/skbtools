"""FIG-7: survey track geometry against position error, lever arm z fixed vs free.

Answers R1 3e (uniform track vs the real uneven track), R3 M7(i) (lost quadrant,
condition number), R2 5 and R4 2 (vertical lever arm: externally fixed vs estimated).
Realistic synthetic surveys ("random" preset: random-walk δc(t), M2 δc, δc gradient,
currents, GNSS errors, clock drift, real dropout pattern), 30 seeds × DOGs 1/3/4 per
track, inverted with the paper's Gauss–Newton model (one δc). Errors are estimate
minus truth in local ENU at the site origin; the three DOGs are pooled.

- (a) The tracks (ship GPS1 position, east/north from the site origin). Bermuda is
  the processed real track; "− NW" also drops every epoch with the ship north-west
  of the DOG-array centre (R3 M7(i)). A single circle cannot be inverted.
- (b, c) East–north error of each survey and its 95 % ellipse across surveys, lever
  arm z fixed (at the truth) and free; the legend gives the rms horizontal error.
- (d) Up error, mean ± 1 sd across surveys, lever arm fixed vs free.
- (e) Condition number of the scaled normal matrix (E, N, U, δt, δc): median and
  5–95 % over all fits that have one.
- (f) Error of the estimated l_z (lever free).

Inputs: revision/analysis/realistic_synthetic/geometry.csv; the processed Bermuda
GPS file (track icon); the circle and grid icons come from ``track_en_m``, the
function the generator uses.
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
# (cell prefix, label, colour, line style): colour = track shape, dashed = NW lost.
TRACKS = (
    ("bermuda", "Bermuda (real)", paper_style.CATEGORY_COLORS[0], "-"),
    ("grid", "Grid (uniform)", paper_style.CATEGORY_COLORS[3], "-"),
    ("bermuda_lost", "Bermuda − NW", paper_style.CATEGORY_COLORS[0], "--"),
    ("grid_lost", "Grid − NW", paper_style.CATEGORY_COLORS[3], "--"),
)
CIRCLE_COLOR = paper_style.CATEGORY_COLORS[5]
LEVERS = {"lever_fixed": "lever z fixed", "lever_free": "lever z free"}
ELLIPSE_PROBABILITY = 0.95
LOST_COLOR = "0.8"


def solved(table, cell):
    """Fits of a cell that converged on the right integer second."""
    rows = table[table["cell"] == cell]
    return rows[rows["failed"].isna() & (rows["wrong_second"] == 0)]


def track_icons():
    """Ship E, N (km) per track shape and the DOG E, N (km), site-origin frame."""
    to_enu = rs.enu_rotation()
    data = np.load(BERMUDA_GPS)
    bermuda = (data["GPS_Coordinates"][:, 0] - rs.SITE_ORIGIN_ECEF) @ to_enu.T
    n = int(data["GPS_data"][-1] - data["GPS_data"][0]) + 1  # as idealized_trajectory
    dogs = {d: to_enu @ rs.CDOG_AUGMENTS[d] for d in DOGS}
    centre = np.mean(list(dogs.values()), axis=0)[:2]
    tracks = {"bermuda": bermuda[:, :2]}
    for shape in ("grid", "circle"):
        tracks[shape] = rs.track_en_m(shape, n, centre)[0]
    return tracks, dogs, centre


def draw_icon(ax, track_en_m, dogs, centre, color, lost):
    """Track in colour; with ``lost`` the NW-of-centre part is drawn light grey."""
    km = track_en_m / 1000.0
    north_west = (track_en_m[:, 0] < centre[0]) & (track_en_m[:, 1] > centre[1])
    kept = km.copy()
    if lost:
        kept[north_west] = np.nan
        dropped = km.copy()
        dropped[~north_west] = np.nan
        ax.plot(dropped[:, 0], dropped[:, 1], color=LOST_COLOR, lw=0.8)
    ax.plot(kept[:, 0], kept[:, 1], color=color, lw=0.8)
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
        )
    ax.set_aspect("equal")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])


def plot_icons(subfig, table):
    tracks, dogs, centre = track_icons()
    axes = subfig.subplots(1, 5)
    for ax, (prefix, label, color, _) in zip(axes, TRACKS, strict=False):
        shape = prefix.replace("_lost", "")
        draw_icon(ax, tracks[shape], dogs, centre, color, prefix.endswith("_lost"))
        ax.set_title(label, color=color, fontweight="bold")
    circle_ok = sum(
        len(solved(table, f"circle{lost}_lever_fixed")) for lost in ("", "_lost")
    )
    circle_n = sum(
        (table["cell"] == f"circle{lost}_lever_fixed").sum() for lost in ("", "_lost")
    )
    draw_icon(axes[4], tracks["circle"], dogs, centre, CIRCLE_COLOR, False)
    axes[4].set_title(f"Circle: {circle_ok}/{circle_n} solved", color="0.3")
    axes[0].set_xlabel("E (km)")
    axes[0].set_ylabel("N (km)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    paper_style.panel_label(axes[0], "a")
    return axes


def ellipse(rows):
    """95 % ellipse (n, 2) of the E, N errors of a set of fits, cm."""
    en = rows[["E_err_cm", "N_err_cm"]].to_numpy()
    return fc.ellipse_points(en.mean(axis=0), np.cov(en.T), ELLIPSE_PROBABILITY)


def plot_ellipses(ax, table, lever, limit_cm):
    """Error points (full tracks), 95 % ellipses and mean; returns rms per track."""
    rms = {}
    for prefix, _, color, style in TRACKS:
        rows = solved(table, f"{prefix}_{lever}")
        en = rows[["E_err_cm", "N_err_cm"]].to_numpy()
        if style == "-":
            ax.plot(en[:, 0], en[:, 1], ".", color=color, markersize=2, alpha=0.5)
        mean = en.mean(axis=0)
        boundary = ellipse(rows)
        width = 1.5 if style == "-" else 1.0
        ax.plot(boundary[:, 0], boundary[:, 1], color=color, ls=style, lw=width)
        ax.plot(*mean, marker="+", color=color, markersize=6, markeredgewidth=1.2)
        rms[prefix] = np.sqrt(np.mean(np.sum(en**2, axis=1)))
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.axvline(0, color="0.6", lw=0.5, ls=":")
    ax.set_aspect("equal")
    ax.set_xlim(-limit_cm, limit_cm)
    ax.set_ylim(-limit_cm, limit_cm)
    ax.set_xlabel("E error (cm)")
    ax.set_ylabel("N error (cm)")
    ax.set_title(f"Horizontal, {LEVERS[lever]}", loc="left")
    return rms


def plot_up(ax, table):
    x = np.arange(len(TRACKS))
    for offset, lever, marker in (
        (-0.17, "lever_fixed", "o"),
        (0.17, "lever_free", "s"),
    ):
        for k, (prefix, _, color, _) in enumerate(TRACKS):
            u = solved(table, f"{prefix}_{lever}")["U_err_cm"].to_numpy()
            ax.plot(
                k + offset + np.linspace(-0.05, 0.05, u.size),
                u,
                ".",
                color=color,
                alpha=0.35,
                markersize=2,
            )
            ax.errorbar(
                k + offset,
                u.mean(),
                u.std(ddof=1),
                color=color,
                marker=marker,
                markerfacecolor=color if lever == "lever_fixed" else "white",
                capsize=2,
                lw=1.0,
                ls="none",
                markersize=4,
            )
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(x, [t[1].replace(" (", "\n(").replace(" − ", "\n− ") for t in TRACKS])
    ax.set_ylim(ax.get_ylim()[0], 1.3 * ax.get_ylim()[1])  # room for the legend
    ax.set_ylabel("U error (cm)")
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="none", label="lever z fixed"),
        Line2D(
            [],
            [],
            color="0.3",
            marker="s",
            markerfacecolor="white",
            ls="none",
            label="lever z free",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", ncols=2)
    ax.set_title("Vertical: mean ± 1 sd", loc="right")


def plot_lever_z(ax, table):
    """Estimated l_z minus truth per track (lever free), rows top to bottom."""
    for k, (prefix, _, color, _) in enumerate(TRACKS):
        dz = solved(table, f"{prefix}_lever_free")["lever_z_err_cm"].to_numpy()
        ax.plot(
            dz,
            -k + np.linspace(-0.08, 0.08, dz.size),
            ".",
            color=color,
            alpha=0.35,
            markersize=2,
        )
        ax.errorbar(
            dz.mean(),
            -k,
            xerr=dz.std(ddof=1),
            color=color,
            marker="s",
            markerfacecolor="white",
            capsize=2,
            ls="none",
            markersize=4,
        )
    ax.axvline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xlabel(f"{fc.LEVER_LABELS[2]} error, lever free (cm)")


def plot_condition(ax, table):
    """Median and 5-95 % of the condition number per track (all fits that have one)."""
    groups = [(t[0], t[2]) for t in TRACKS] + [("circle", CIRCLE_COLOR)]
    for k, (prefix, color) in enumerate(groups):
        cells = [
            c for c in table["cell"].unique() if c.rsplit("_lever", 1)[0] == prefix
        ]
        if prefix == "circle":
            cells += [c for c in table["cell"].unique() if c.startswith("circle_lost")]
        values = table[table["cell"].isin(cells)]["condition_number"].dropna()
        low, mid, high = np.percentile(values, [5, 50, 95])
        ax.errorbar(
            mid,
            -k,
            xerr=[[mid - low], [high - mid]],
            color=color,
            marker="D",
            capsize=2,
            ls="none",
            markersize=4,
        )
    ax.set_xscale("log")
    ax.set_xticks([1e4, 1e6, 1e8])
    ax.set_xlabel("Condition number")
    labels = [t[1] for t in TRACKS] + ["Circle (all)"]
    ax.set_yticks(-np.arange(len(labels)), labels)
    ax.set_ylim(-len(labels) + 0.5, 0.5)


def plot(table):
    fig = plt.figure(figsize=(paper_style.DOUBLE_COLUMN_MM / 25.4, 170 / 25.4))
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[0.68, 1.25, 0.95])
    plot_icons(top, table)
    ax_fixed, ax_free = middle.subplots(1, 2)
    limit = 1.05 * max(
        np.abs(ellipse(solved(table, f"{p}_{lever}"))).max()
        for p, *_ in TRACKS
        for lever in LEVERS
    )
    rms_fixed = plot_ellipses(ax_fixed, table, "lever_fixed", limit)
    rms_free = plot_ellipses(ax_free, table, "lever_free", limit)
    handles = [
        Line2D(
            [],
            [],
            color=color,
            ls=style,
            lw=1.3,
            label=f"{label}: {rms_fixed[prefix]:.0f} | {rms_free[prefix]:.0f} cm",
        )
        for prefix, label, color, style in TRACKS
    ]
    handles.append(Line2D([], [], color="0.3", marker="+", ls="none", label="mean"))
    middle.legend(
        handles=handles,
        loc="outside right center",
        title="95 % ellipse; rms horizontal\nerror, lever z fixed | free",
    )
    ax_up, ax_cond, ax_lz = bottom.subplots(1, 3, width_ratios=[1.9, 1.0, 1.0])
    ax_lz.sharey(ax_cond)
    plot_up(ax_up, table)
    plot_condition(ax_cond, table)
    plot_lever_z(ax_lz, table)
    ax_lz.tick_params(labelleft=False)
    panels = (ax_fixed, ax_free, ax_up, ax_cond, ax_lz)
    for ax, letter in zip(panels, "bcdef", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


if __name__ == "__main__":
    fc.require(GEOMETRY_CSV, BERMUDA_GPS)
    paper_style.apply()  # after imports: some modules turn usetex on at import
    table = pd.read_csv(GEOMETRY_CSV)
    for prefix, label, *_ in TRACKS:
        for lever in LEVERS:
            rows = solved(table, f"{prefix}_{lever}")
            print(
                f"{label:16s} {lever:12s} n={len(rows)} "
                f"E/N/U sd {rows['E_err_cm'].std():.0f}/{rows['N_err_cm'].std():.0f}/"
                f"{rows['U_err_cm'].std():.0f} cm"
            )
    figure = plot(table)
    fc.save(figure, NAME)
    plt.close(figure)
