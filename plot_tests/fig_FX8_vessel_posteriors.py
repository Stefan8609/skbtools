"""FX-8 (remake of Fig. 8): GNSS receivers, transducer lever arm and DOG posteriors.

(a, b) Top and side view of the R/V Atlantic Explorer with the four GNSS receivers,
the posterior-mean transducer position and the 95 % region of the lever-arm prior.
(c) Each receiver's scatter about its body-frame grid point (epoch-wise rigid-body
misfit) and the lever-arm posterior, at one common scale. (d, e) DOG position
posteriors in local ENU about the posterior mean: E–N and E–U. The σ and ρ values are
in tables beside the panels, not over the data.

Reviewer comments: R3 #11 (overlapping, duplicated annotations in Fig. 8), R4 m6
(ξ meant the ray's horizontal distance in Table A1; the second DOG panel is now plain
E–U), R3 specific point 1 (the σ values printed here are the ones to quote in the
Abstract), R4 2b (prior vs posterior).

Inputs: the final run's chains (``fc.FINAL_RUN``), the GNSS file named in its
configuration, and the hull outline digitised for the submitted figure
(``Atlantic_RV_Plot_Data_ar1.npz``: ``segments``, ``side_segments``,
``rotation_deg``, ``gps1_offset``; these do not depend on the run).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
OUTLINE_FILE = (
    fc.REPO / "Bermuda_Data/Output/Plot_Data/Atlantic_RV_Plot_Data_ar1.npz"
)  # hull outline and the drawing frame of the submitted Fig. 8
DOGS = (1, 3, 4)
HEIGHT_MM = 215.0
ZOOM_HALF_RANGE_CM = 30.0  # common scale of the receiver and transducer zooms
DOG_EN_HALF_RANGE_CM = 10.0
DOG_U_HALF_RANGE_CM = 60.0
N_DOG_POINTS = 5000  # thinned draws drawn per DOG panel
ELLIPSE_PROBABILITIES = (0.68, 0.95)
RECEIVER_COLOR = paper_style.CATEGORY_COLORS[0]
TRANSDUCER_COLOR = paper_style.CATEGORY_COLORS[1]
HULL_COLOR = "0.45"
NAME = "fig_FX8_vessel_posteriors"
# -----------------------------------------------------------------------------------

N_BRIDGE_SEGMENTS = 10  # segments = bridge (10), hull (7), moon pool (4)
N_SIDE_HULL_SEGMENTS = 10  # side_segments = hull (10), moon pool (4)


def rotate_z(points, angle_deg):
    """Rotate (..., 3) points about the vertical axis by angle_deg."""
    a = np.deg2rad(angle_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return points @ rz.T


def draw_segments(ax, segments):
    for (x1, y1), (x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], color=HULL_COLOR, lw=0.7, zorder=1)


def draw_ellipses(ax, xy, color):
    cov = np.cov(xy.T)
    for probability, style in zip(ELLIPSE_PROBABILITIES, ("-", "--"), strict=True):
        boundary = fc.ellipse_points(xy.mean(axis=0), cov, probability)
        ax.plot(*boundary.T, color=color, ls=style, lw=0.8, zorder=4)


def sd_table(ax, rows, row_labels, col_labels, title):
    """A small table that fills an axes (the axes itself is hidden)."""
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(paper_style.FONT_SIZE_PT)
    table.scale(1.0, 1.35)
    table.auto_set_column_width(list(range(-1, len(col_labels))))
    ax.set_title(title, loc="left", fontsize=paper_style.FONT_SIZE_PT)
    return table


def vessel_panels(
    ax_top, ax_side, outline, receivers_body, lever, prior_center, prior_sd
):
    """Top (X–Y) and side (X–Z) views in the drawing frame of the submitted figure."""
    angle = float(outline["rotation_deg"])
    origin = np.asarray(outline["gps1_offset"], dtype=float)
    receivers = rotate_z(receivers_body.mean(axis=0), angle) + origin
    transducer = rotate_z(lever.mean(axis=0), angle) + origin
    prior_mean = rotate_z(prior_center, angle) + origin
    # The prior is isotropic in x–y, so its top-view region does not rotate.
    prior_cov_xy = np.diag(prior_sd[:2] ** 2)
    prior_cov_xz = np.diag(prior_sd[[0, 2]] ** 2)

    segments = outline["segments"]
    side_segments = outline["side_segments"]
    draw_segments(ax_top, segments)
    draw_segments(ax_side, side_segments)
    for ax, (i, j), cov in (
        (ax_top, (0, 1), prior_cov_xy),
        (ax_side, (0, 2), prior_cov_xz),
    ):
        boundary = fc.ellipse_points(prior_mean[[i, j]], cov, 0.95)
        ax.plot(*boundary.T, color=TRANSDUCER_COLOR, ls=":", lw=0.9, zorder=2)
        ax.plot(
            receivers[:, i],
            receivers[:, j],
            ls="none",
            marker="D",
            ms=3.5,
            color=RECEIVER_COLOR,
            zorder=3,
        )
        ax.plot(
            transducer[i],
            transducer[j],
            ls="none",
            marker="*",
            ms=7,
            color=TRANSDUCER_COLOR,
            mec="k",
            mew=0.4,
            zorder=5,
        )
        ax.set_aspect("equal")

    # Direct labels: receiver numbers, bridge and moon pool.
    top_offsets = ((0.7, 0.6), (0.7, -1.6), (-1.4, -1.6), (-0.6, 1.0))
    for k, (dx, dy) in enumerate(top_offsets):
        ax_top.text(receivers[k, 0] + dx, receivers[k, 1] + dy, str(k + 1))
    bridge = segments[:N_BRIDGE_SEGMENTS].reshape(-1, 2)
    moon_pool = segments[-4:].reshape(-1, 2)
    ax_top.text(
        bridge[:, 0].mean(), bridge[:, 1].mean(), "bridge", ha="center", va="center"
    )
    ax_top.text(
        moon_pool[:, 0].mean() - 1.5,
        moon_pool[:, 1].mean(),
        "moon pool",
        ha="right",
        va="center",
    )
    side_bridge = side_segments[1:4].reshape(-1, 2)
    side_pool = side_segments[N_SIDE_HULL_SEGMENTS:].reshape(-1, 2)
    ax_side.text(
        side_bridge[:, 0].mean(), side_bridge[:, 1].max() + 0.8, "bridge", ha="center"
    )
    ax_side.text(
        side_pool[:, 0].min() - 1.5,
        side_pool[:, 1].min() - 2.0,
        "moon pool",
        ha="right",
        va="top",
    )

    # Body-frame axes, so the reader can relate (c) and the table to the hull.
    base = np.array([10.0, -2.5])
    for axis, label in ((0, "x"), (1, "y")):
        tip = rotate_z(np.eye(3)[axis] * 3.0, angle)[:2]
        ax_top.annotate(
            "",
            xy=base + tip,
            xytext=base,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.2"),
        )
        ax_top.text(*(base + 1.4 * tip), label, ha="center", va="center")
    ax_top.text(base[0], base[1] - 0.6, "body frame", ha="center", va="top")

    ax_top.set_xlim(0, 57)
    ax_top.set_ylim(-7.0, 7.0)
    ax_side.set_xlim(0, 57)
    ax_side.set_ylim(-5.5, 16.5)
    ax_top.set_ylabel("Across-hull Y (m)")
    ax_side.set_ylabel("Height Z (m)")
    ax_side.set_xlabel("Along-hull X (m), drawing frame")
    ax_top.tick_params(labelbottom=False)


def zoom_panels(axes, receivers_body, lever):
    """Receiver scatter and lever posterior about their means (body x–y, cm)."""
    clouds = [receivers_body[:, k, :2] for k in range(4)] + [lever[:, :2]]
    titles = [f"GNSS {k + 1}" for k in range(4)] + ["Transducer"]
    labels = [("Δx body (cm)", "Δy body (cm)")] * 4 + [
        (f"Δ{fc.LEVER_LABELS[0]} body (cm)", f"Δ{fc.LEVER_LABELS[1]} body (cm)")
    ]
    for ax, xy, title, (xlabel, ylabel), color in zip(
        axes,
        clouds,
        titles,
        labels,
        [RECEIVER_COLOR] * 4 + [TRANSDUCER_COLOR],
        strict=True,
    ):
        xy_cm = (xy - xy.mean(axis=0)) * 100.0
        layer = ax.scatter(*xy_cm.T, s=0.3, color=color, alpha=0.08, lw=0)
        paper_style.rasterize_dense(layer, len(xy_cm))
        draw_ellipses(ax, xy_cm, "0.1")
        ax.set_xlim(-ZOOM_HALF_RANGE_CM, ZOOM_HALF_RANGE_CM)
        ax.set_ylim(-ZOOM_HALF_RANGE_CM, ZOOM_HALF_RANGE_CM)
        ax.set_aspect("equal")
        ax.set_xticks([-20, 0, 20])
        ax.set_yticks([-20, 0, 20])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ax is axes[0] or ax is axes[-1]:
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(labelleft=False)


def dog_panels(axes_en, axes_eu, enu_by_dog, rng):
    """E–N and E–U posterior draws (cm from the posterior mean) for each DOG."""
    for ax_en, ax_eu, dog in zip(axes_en, axes_eu, DOGS, strict=True):
        enu = enu_by_dog[dog]
        style = paper_style.DOG_STYLE[dog]
        shown = enu[rng.choice(len(enu), N_DOG_POINTS, replace=False)]
        for ax, (i, j) in ((ax_en, (0, 1)), (ax_eu, (0, 2))):
            layer = ax.scatter(
                shown[:, i],
                shown[:, j],
                s=1.0,
                color=style["color"],
                marker=style["marker"],
                alpha=0.25,
                lw=0,
            )
            paper_style.rasterize_dense(layer, 2 * paper_style.RASTERIZE_ABOVE_POINTS)
            draw_ellipses(ax, enu[:, [i, j]], "0.1")
            ax.set_xlim(-DOG_EN_HALF_RANGE_CM, DOG_EN_HALF_RANGE_CM)
            ax.set_xticks([-5, 0, 5])
        ax_en.set_ylim(-DOG_EN_HALF_RANGE_CM, DOG_EN_HALF_RANGE_CM)
        ax_en.set_yticks([-5, 0, 5])
        ax_en.set_aspect("equal")
        ax_eu.set_ylim(-DOG_U_HALF_RANGE_CM, DOG_U_HALF_RANGE_CM)
        ax_eu.set_box_aspect(1.0)  # same panel size as E–N; axes not to equal scale
        ax_en.set_title(style["label"])
        ax_en.set_xlabel("ΔE (cm)")
        ax_eu.set_xlabel("ΔE (cm)")
        if dog == DOGS[0]:
            ax_en.set_ylabel("ΔN (cm)")
            ax_eu.set_ylabel("ΔU (cm)")
        else:
            ax_en.tick_params(labelleft=False)
            ax_eu.tick_params(labelleft=False)


def main():
    pooled = fc.pooled_run(RUN)
    config = fc.run_config(pooled)
    outline = np.load(OUTLINE_FILE)
    gps = np.load(fc.REPO / "Bermuda_Data" / config["gps_file"])
    gps_grid = np.asarray(config["gps1_to_others"], dtype=float)
    receivers_body = fc.receivers_in_body_frame(gps["GPS_Coordinates"], gps_grid)
    lever = pooled["lever"]
    names = list(pooled["param_names"])
    index = [names.index(f"lever_{axis}") for axis in "xyz"]
    prior_center = pooled["prior_center"][index]
    prior_sd = pooled["prior_sd"][index]
    enu_by_dog = {dog: fc.dog_enu_cm(pooled, dog)[0] for dog in DOGS}
    # Prior sd of each DOG's U (cm): its ECEF prior covariance rotated into local ENU.
    prior_u_sd_cm = {}
    for dog in DOGS:
        dog_index = [names.index(f"CDOG_aug_DOG{dog}_{axis}") for axis in "xyz"]
        rotation = fc.enu_rotation_at(fc.dog_enu_cm(pooled, dog)[1])
        cov = rotation @ np.diag(pooled["prior_sd"][dog_index] ** 2) @ rotation.T
        prior_u_sd_cm[dog] = float(np.sqrt(cov[2, 2])) * 100.0
    if not np.allclose(list(prior_u_sd_cm.values()), prior_u_sd_cm[DOGS[0]]):
        raise ValueError(f"DOG U prior sds differ: {prior_u_sd_cm}; label each")

    receiver_sd_cm = receivers_body.std(axis=0, ddof=1) * 100.0  # (4, 3)
    lever_sd_cm = lever.std(axis=0, ddof=1) * 100.0
    dog_sd_cm = {dog: enu.std(axis=0, ddof=1) for dog, enu in enu_by_dog.items()}
    dog_rho = {
        dog: (
            fc.correlation(enu[:, 0], enu[:, 1]),
            fc.correlation(enu[:, 0], enu[:, 2]),
        )
        for dog, enu in enu_by_dog.items()
    }
    print(f"{RUN}: {len(lever)} draws; GNSS epochs {len(receivers_body)}")
    for k in range(4):
        print(f"GNSS {k + 1} sd (cm) x y z", np.round(receiver_sd_cm[k], 2))
    print("lever sd (cm)", np.round(lever_sd_cm, 2), "mean (m)", lever.mean(axis=0))
    for dog in DOGS:
        print(
            f"DOG {dog} sd E N U (cm)",
            np.round(dog_sd_cm[dog], 2),
            "rho EN EU",
            np.round(dog_rho[dog], 3),
            f"sigma_U / prior sd {dog_sd_cm[dog][2] / prior_u_sd_cm[dog]:.4f}",
        )

    paper_style.apply()
    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=(92, 40, 83))

    vessel_grid = top.add_gridspec(
        2, 2, width_ratios=(3.3, 1.0), height_ratios=(14, 22)
    )
    ax_top = top.add_subplot(vessel_grid[0, 0])
    ax_side = top.add_subplot(vessel_grid[1, 0], sharex=ax_top)
    ax_table = top.add_subplot(vessel_grid[0, 1])
    ax_legend = top.add_subplot(vessel_grid[1, 1])
    vessel_panels(
        ax_top, ax_side, outline, receivers_body, lever, prior_center, prior_sd
    )
    rows = [[f"{v:.1f}" for v in receiver_sd_cm[k]] for k in range(4)]
    rows.append([f"{v:.1f}" for v in lever_sd_cm])
    sd_table(
        ax_table,
        rows,
        [f"GNSS {k + 1}" for k in range(4)] + ["Transducer"],
        ["σx", "σy", "σz"],
        "Body-frame sd (cm)",
    )
    ax_legend.axis("off")
    ax_legend.legend(
        handles=[
            Line2D([], [], ls="none", marker="D", ms=3.5, color=RECEIVER_COLOR),
            Line2D(
                [], [], ls="none", marker="*", ms=7, color=TRANSDUCER_COLOR, mec="k"
            ),
            Line2D([], [], color=TRANSDUCER_COLOR, ls=":", lw=0.9),
            Line2D([], [], color="0.1", lw=0.8),
            Line2D([], [], color="0.1", lw=0.8, ls="--"),
        ],
        labels=[
            "GNSS receiver (mean)",
            "Transducer\n(posterior mean)",
            "Lever-arm prior, 95 %",
            "68 % region",
            "95 % region",
        ],
        loc="center left",
        handlelength=1.6,
    )
    paper_style.panel_label(ax_top, "a")
    paper_style.panel_label(ax_side, "b")

    zoom_axes = middle.subplots(1, 5)
    zoom_panels(zoom_axes, receivers_body, lever)
    paper_style.panel_label(zoom_axes[0], "c")

    dog_grid = bottom.add_gridspec(
        2, 4, width_ratios=(1, 1, 1, 1.75), height_ratios=(1.0, 1.0)
    )
    axes_en = [bottom.add_subplot(dog_grid[0, k]) for k in range(3)]
    axes_eu = [bottom.add_subplot(dog_grid[1, k]) for k in range(3)]
    dog_panels(axes_en, axes_eu, enu_by_dog, np.random.default_rng(20261118))
    paper_style.panel_label(axes_en[0], "d")
    paper_style.panel_label(axes_eu[0], "e")
    ax_dog_table = bottom.add_subplot(dog_grid[:, 3])
    sd_table(
        ax_dog_table,
        [
            [f"{v:.1f}" for v in dog_sd_cm[dog]]
            + [
                fc.signed(r, "+.2f") if abs(r) >= 0.005 else "0.00"
                for r in dog_rho[dog]
            ]
            + [f"{dog_sd_cm[dog][2] / prior_u_sd_cm[dog]:.3f}"]
            for dog in DOGS
        ],
        [paper_style.DOG_STYLE[dog]["label"] for dog in DOGS],
        ["σE", "σN", "σU", "ρEN", "ρEU", "σU/σp"],
        f"Posterior sd (cm), ρ\nσp: prior sd of U = {prior_u_sd_cm[DOGS[0]]:.0f} cm",
    )

    fc.save(fig, NAME)
    return fig


if __name__ == "__main__":
    fc.require(fc.CHAINS_DIR / RUN, OUTLINE_FILE)
    main()
