"""FX-8 (remake of Fig. 8): GNSS receivers, transducer lever arm and DOG posteriors.

Message: the transducer position in the vessel frame is known far better than the
GNSS antennas agree with a rigid body, and the DOG positions are an order of
magnitude better determined horizontally than vertically.

(a, b) Plan and side view of the R/V Atlantic Explorer with the four GNSS antennas
and the posterior-mean transducer. Around each antenna: the 95 % ellipse of its
scatter about the rigid body (epoch-wise four-antenna misfit); around the
transducer: the 95 % ellipse of the lever-arm posterior; both enlarged by
``ELLIPSE_SCALE``. The dotted ellipse is the 95 % region of the lever-arm prior, to
scale. (c, d) DOG position posteriors in local ENU about the posterior mean, E–N
and E–U, one small panel per DOG. The σ and ρ values that were in tables on the
figure are printed for a paper table.

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
HEIGHT_MM = 142.0
ELLIPSE_SCALE = 10.0  # enlargement of the antenna and transducer ellipses in (a, b)
DOG_EN_HALF_RANGE_CM = 10.0
DOG_U_HALF_RANGE_CM = 60.0
N_DOG_POINTS = 5000  # thinned draws drawn per DOG panel
ELLIPSE_PROBABILITIES = (0.68, 0.95)
RECEIVER_COLOR = "black"
TRANSDUCER_COLOR = fc.ACCENT
HULL_COLOR = "0.6"
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


def enlarged_ellipse(ax, centre, cloud, color):
    """95 % ellipse of a 2-D cloud, enlarged by ELLIPSE_SCALE and drawn at centre."""
    cov = np.cov(cloud.T) * ELLIPSE_SCALE**2
    boundary = fc.ellipse_points(centre, cov, 0.95)
    ax.plot(*boundary.T, color=color, lw=0.8, zorder=4)


def vessel_panels(
    ax_top, ax_side, outline, receivers_body, lever, prior_center, prior_sd
):
    """Top (X–Y) and side (X–Z) views in the drawing frame of the submitted figure."""
    angle = float(outline["rotation_deg"])
    origin = np.asarray(outline["gps1_offset"], dtype=float)
    receivers = rotate_z(receivers_body.mean(axis=0), angle) + origin
    transducer = rotate_z(lever.mean(axis=0), angle) + origin
    # Scatter of each antenna and of the lever arm, rotated into the drawing frame.
    receiver_clouds = rotate_z(receivers_body, angle)
    lever_cloud = rotate_z(lever, angle)
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
        ax.plot(*boundary.T, color=fc.CONTEXT, ls=":", lw=0.9, zorder=2)
        ax.plot(
            receivers[:, i],
            receivers[:, j],
            ls="none",
            marker="D",
            ms=3.0,
            color=RECEIVER_COLOR,
            zorder=3,
        )
        for k in range(receivers.shape[0]):
            enlarged_ellipse(
                ax, receivers[k, [i, j]], receiver_clouds[:, k][:, [i, j]], "black"
            )
        ax.plot(
            transducer[i],
            transducer[j],
            ls="none",
            marker="o",
            ms=3.5,
            color=TRANSDUCER_COLOR,
            zorder=5,
        )
        enlarged_ellipse(
            ax, transducer[[i, j]], lever_cloud[:, [i, j]], TRANSDUCER_COLOR
        )
        ax.set_aspect("equal")

    # Direct labels: receiver numbers, bridge and moon pool.
    top_offsets = ((2.4, 0.0), (2.4, 0.0), (-3.0, 0.0), (-3.4, 0.0))
    for k, (dx, dy) in enumerate(top_offsets):
        ax_top.text(
            receivers[k, 0] + dx,
            receivers[k, 1] + dy,
            str(k + 1),
            ha="center",
            va="center",
        )
    bridge = segments[:N_BRIDGE_SEGMENTS].reshape(-1, 2)
    moon_pool = segments[-4:].reshape(-1, 2)
    ax_top.text(
        bridge[:, 0].mean(), bridge[:, 1].mean(), "bridge", ha="center", va="center"
    )
    ax_top.text(
        moon_pool[:, 0].mean() - 2.2,
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
    ax_top.set_ylabel("Across hull (m)")
    ax_side.set_ylabel("Height (m)")
    ax_side.set_xlabel("Along hull (m)")
    ax_top.set_yticks([-5, 0, 5])
    ax_side.set_yticks([0, 5, 10, 15])
    ax_top.tick_params(labelbottom=False)
    fc.despine(ax_top, ax_side)


def dog_panels(axes_en, axes_eu, enu_by_dog, rng):
    """E–N and E–U posterior draws (cm from the posterior mean) for each DOG."""
    for ax_en, ax_eu, dog in zip(axes_en, axes_eu, DOGS, strict=True):
        enu = enu_by_dog[dog]
        label = paper_style.DOG_STYLE[dog]["label"]
        shown = enu[rng.choice(len(enu), N_DOG_POINTS, replace=False)]
        for ax, (i, j) in ((ax_en, (0, 1)), (ax_eu, (0, 2))):
            layer = ax.scatter(
                shown[:, i], shown[:, j], s=1.0, color=fc.RAW_POINTS, lw=0
            )
            paper_style.rasterize_dense(layer, 2 * paper_style.RASTERIZE_ABOVE_POINTS)
            draw_ellipses(ax, enu[:, [i, j]], "black")
            ax.set_xlim(-DOG_EN_HALF_RANGE_CM, DOG_EN_HALF_RANGE_CM)
            ax.set_xticks([-5, 5])
            ax.xaxis.set_major_formatter(lambda v, _: fc.signed(v, "g"))
            ax.yaxis.set_major_formatter(lambda v, _: fc.signed(v, "g"))
            ax.set_box_aspect(1.0)
            ax.set_title(label)
            ax.set_xlabel("ΔE (cm)")
        ax_en.set_ylim(-DOG_EN_HALF_RANGE_CM, DOG_EN_HALF_RANGE_CM)
        ax_en.set_yticks([-5, 5])
        ax_eu.set_ylim(-DOG_U_HALF_RANGE_CM, DOG_U_HALF_RANGE_CM)
        ax_eu.set_yticks([-40, 40])
        if dog != DOGS[0]:
            ax_en.tick_params(labelleft=False)
            ax_eu.tick_params(labelleft=False)
    axes_en[0].set_ylabel("ΔN (cm)", labelpad=1)
    axes_eu[0].set_ylabel("ΔU (cm)", labelpad=1)
    fc.despine(*axes_en, *axes_eu)


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
    top, bottom = fig.subfigures(2, 1, height_ratios=(98, 44))

    vessel_grid = top.add_gridspec(
        2, 2, width_ratios=(3.3, 1.0), height_ratios=(14, 22)
    )
    ax_top = top.add_subplot(vessel_grid[0, 0])
    ax_side = top.add_subplot(vessel_grid[1, 0], sharex=ax_top)
    ax_legend = top.add_subplot(vessel_grid[:, 1])
    vessel_panels(
        ax_top, ax_side, outline, receivers_body, lever, prior_center, prior_sd
    )
    ax_legend.axis("off")
    ax_legend.legend(
        handles=[
            Line2D([], [], ls="none", marker="D", ms=3.0, color=RECEIVER_COLOR),
            Line2D([], [], ls="none", marker="o", ms=3.5, color=TRANSDUCER_COLOR),
            Line2D([], [], color="black", lw=0.8),
            Line2D([], [], color=fc.CONTEXT, ls=":", lw=0.9),
            Line2D([], [], color="black", lw=0.8),
            Line2D([], [], color="black", lw=0.8, ls="--"),
        ],
        labels=[
            "GNSS antenna",
            "Transducer",
            f"(a, b) 95 % ellipse,\nenlarged × {ELLIPSE_SCALE:g}",
            "Lever-arm prior,\n95 %, to scale",
            "(c, d) 68 % region",
            "(c, d) 95 % region",
        ],
        loc="center left",
        handlelength=1.6,
    )
    paper_style.panel_label(ax_top, "a")
    paper_style.panel_label(ax_side, "b")

    dog_axes = bottom.subplots(1, 2 * len(DOGS))
    axes_en, axes_eu = dog_axes[: len(DOGS)], dog_axes[len(DOGS) :]
    dog_panels(axes_en, axes_eu, enu_by_dog, np.random.default_rng(20261118))
    paper_style.panel_label(axes_en[0], "c")
    paper_style.panel_label(axes_eu[0], "d")

    fc.save(fig, NAME)
    return fig


if __name__ == "__main__":
    fc.require(fc.CHAINS_DIR / RUN, OUTLINE_FILE)
    main()
