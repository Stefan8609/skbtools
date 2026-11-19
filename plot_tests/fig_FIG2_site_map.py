"""Site map of the Bermuda survey (new Fig. 2, merged with the remade Fig. 3 track).

The vessel track (mean of the four GNSS antennas) is drawn at the epochs kept by
the GNSS height filter, coloured by time since survey start, over the epochs the
filter rejected (grey; same epochs as FX-A3). DOGs are at the MCMC posterior mean
of the final run. Bathymetry is the GEBCO 2025 grid at its true cell footprint;
an inset locates the site relative to Bermuda (0 m contour of the regional grid).

Reviewer comments: R1 4a (map of the station locations), R1 4e, R1 3f and R3 #10
(distinct DOG markers, "DOG" naming, time colourbar with the Fig. 4 time range),
R3 M7(i) (which part of the hourglass the filter removed), R1 4d (acquisition
geometry), R3 #18 (deployment positions: none exist in the repo, so none shown).
Inputs: the paths in the configuration block. Run from the repo root with
PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pymap3d import ecef2geodetic, geodetic2enu

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
DOGS = (1, 3, 4)
PROCESSED_FILE = fc.REPO / "Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz"
GNSS_FILES = [
    fc.REPO / f"Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{k}-camp.mat"
    for k in (1, 2, 3, 4)
]
# GEBCO 2025, 15 arc-second grid, heights in m relative to mean sea level.
SURVEY_GRID = fc.REPO / "Bermuda_Data/gebco_2025_bermuda_survey_ascii.asc"
SURVEY_TID_GRID = fc.REPO / "Bermuda_Data/gebco_2025_bermuda_survey_tid_ascii.asc"
REGION_GRID = fc.REPO / "Bermuda_Data/gebco_2025_bermuda_region_ascii.asc"
# Fallback when the survey grid is absent: the 7 x 12-cell tile around the DOGs.
SMALL_GRID = fc.REPO / "Bermuda_Data/gebco_2025_n31.47_s31.42_w-68.71_e-68.68_ascii.asc"
MULTIBEAM_TID = 11  # GEBCO source type identifier: multibeam sounding
NAME = "fig_FIG2_site_map"
HEIGHT_MM = 125.0
SCALE_BAR_KM = 2.0
CONTOUR_STEP_M = 25.0
MAP_PAD_DEG = 0.003
BATHY_CMAP = ListedColormap(plt.get_cmap("Blues_r")(np.linspace(0.30, 0.92, 256)))
REJECTED_COLOR = "0.15"
SEA_COLOR = "#DCEAF5"
MULTIBEAM_LABEL = "GEBCO cells\nfrom multibeam\n(TID 11)"
# -----------------------------------------------------------------------------------


def lon_label(value, _=None):
    return f"{abs(value):.2f}° {'W' if value < 0 else 'E'}"


def lat_label(value, _=None):
    return f"{abs(value):.2f}° {'S' if value < 0 else 'N'}"


def metres_per_degree(lat_deg, lon_deg):
    """East metres per degree of longitude and north metres per degree of latitude."""
    east = geodetic2enu(lat_deg, lon_deg + 0.01, 0, lat_deg, lon_deg, 0)[0] / 0.01
    north = geodetic2enu(lat_deg + 0.01, lon_deg, 0, lat_deg, lon_deg, 0)[1] / 0.01
    return east, north


def antenna_mean_lat_lon(ecef_m):
    """Latitude and longitude (deg, WGS84) of the mean of the antennas, (n, 4, 3)."""
    centre = ecef_m.mean(axis=1)
    lat, lon, _ = ecef2geodetic(centre[:, 0], centre[:, 1], centre[:, 2])
    return lat, lon


def draw_locator(ax, survey_box):
    """Land of the regional GEBCO grid, its 0 m contour and the map's footprint."""
    grid, extent = fc.read_esri_ascii(REGION_GRID)
    # Plain sea: the depth colourbar belongs to panel (a) only.
    ax.set_facecolor(SEA_COLOR)
    land = np.where(grid >= 0, 1.0, np.nan)
    ax.imshow(land, extent=extent, cmap=ListedColormap(["0.35"]))
    cell = (extent[1] - extent[0]) / grid.shape[1]
    lon = extent[0] + cell * (np.arange(grid.shape[1]) + 0.5)
    lat = extent[3] - cell * (np.arange(grid.shape[0]) + 0.5)
    ax.contour(lon, lat, grid, levels=[0.0], colors="k", linewidths=0.6)
    west, east, south, north = survey_box
    ax.add_patch(
        Rectangle((west, south), east - west, north - south, fill=False, lw=1.0)
    )
    rows, cols = np.nonzero(grid >= 0)
    ax.annotate(
        "Bermuda",
        (lon[cols].mean(), lat[rows].min()),
        xytext=(0, -3),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    ax.annotate(
        "Survey",
        (west, 0.5 * (south + north)),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
    )
    east_m, north_m = metres_per_degree(np.mean(extent[2:]), np.mean(extent[:2]))
    ax.set_aspect(north_m / east_m)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xticks(np.arange(np.ceil(extent[0]) + 1, extent[1], 3.0))
    ax.set_yticks(np.arange(np.ceil(extent[2]), extent[3] + 1e-9, 1.0))
    ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}° W")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}° N")
    # Distance from the map centre to the nearest land cell, for the caption.
    centre_lat, centre_lon = 0.5 * (south + north), 0.5 * (west + east)
    e, n, _ = geodetic2enu(lat[rows], lon[cols], 0, centre_lat, centre_lon, 0)
    print(f"nearest Bermuda land cell: {np.hypot(e, n).min() / 1000:.0f} km")


def main():
    survey = SURVEY_GRID if SURVEY_GRID.exists() else SMALL_GRID
    fc.require(PROCESSED_FILE, *GNSS_FILES, survey)
    if survey == SMALL_GRID:
        print(f"BLOCKED (partial): {SURVEY_GRID} missing; bathymetry only near DOGs")
    paper_style.apply()
    # Manual layout (constrained layout cannot hold an equal-distance map + inset).
    plt.rcParams["figure.constrained_layout.use"] = False

    processed = np.load(PROCESSED_FILE)
    kept_lat, kept_lon = antenna_mean_lat_lon(processed["GPS_Coordinates"])
    kept_hours = paper_style.hours_since_start(processed["GPS_data"])
    # Rejected = epochs common to all antennas in the filter window that the
    # processed file does not hold (FX-A3 checks that the filter reproduces it).
    epochs = fc.load_gnss_epochs(GNSS_FILES)
    rejected = ~np.isin(epochs["gps_s"], processed["GPS_data"])
    rej_lat, rej_lon = antenna_mean_lat_lon(
        epochs["ecef_m"][:, rejected].swapaxes(0, 1)
    )
    print(f"kept {kept_hours.size}, rejected {int(rejected.sum())}")

    pooled = fc.pooled_run(RUN)
    dog_ecef = pooled["CDOG_reference"] + pooled["CDOG_aug"].mean(axis=0)
    dog_lat, dog_lon, dog_height = ecef2geodetic(*dog_ecef.T)

    grid, extent = fc.read_esri_ascii(survey)
    all_lon = np.r_[kept_lon, rej_lon, dog_lon]
    all_lat = np.r_[kept_lat, rej_lat, dog_lat]
    xlim = (all_lon.min() - MAP_PAD_DEG, all_lon.max() + MAP_PAD_DEG)
    ylim = (all_lat.min() - MAP_PAD_DEG, all_lat.max() + MAP_PAD_DEG)
    east_m, north_m = metres_per_degree(np.mean(ylim), np.mean(xlim))

    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    # (a) kept track by time, (b) rejected epochs; same extent and bathymetry.
    ax_kept = fig.add_axes([0.12, 0.215, 0.315, 0.745])
    ax_rejected = fig.add_axes([0.45, 0.215, 0.315, 0.745])
    cell = (extent[1] - extent[0]) / grid.shape[1]
    lon_edges = extent[0] + cell * np.arange(grid.shape[1] + 1)
    lat_edges = extent[3] - cell * np.arange(grid.shape[0] + 1)
    lon_centres, lat_centres = lon_edges[:-1] + cell / 2, lat_edges[:-1] - cell / 2
    levels = np.arange(
        np.ceil(np.nanmin(grid) / CONTOUR_STEP_M) * CONTOUR_STEP_M,
        np.nanmax(grid),
        CONTOUR_STEP_M,
    )
    tid = None
    if survey == SURVEY_GRID and SURVEY_TID_GRID.exists():
        tid, _ = fc.read_esri_ascii(SURVEY_TID_GRID)
        dog_tid = [
            tid[int((extent[3] - la) / cell), int((lo - extent[0]) / cell)]
            for la, lo in zip(dog_lat, dog_lon, strict=True)
        ]
        print(f"GEBCO TID at the DOG cells (DOGs {DOGS}): {dog_tid}")
    plt.rcParams["hatch.linewidth"] = 0.4
    plt.rcParams["hatch.color"] = "0.35"
    halo = [patheffects.withStroke(linewidth=2.0, foreground="w")]
    for ax in (ax_kept, ax_rejected):
        bathy = ax.pcolormesh(lon_edges, lat_edges, grid, cmap=BATHY_CMAP, zorder=0)
        ax.contour(lon_centres, lat_centres, grid, levels, colors="w", linewidths=0.4)
        if tid is not None:
            ax.contourf(
                lon_centres,
                lat_centres,
                (tid == MULTIBEAM_TID).astype(float),
                [0.5, 1.5],
                colors="none",
                hatches=["///"],
            )
        if survey == SMALL_GRID:
            ax.add_patch(
                Rectangle(
                    (extent[0], extent[2]),
                    extent[1] - extent[0],
                    extent[3] - extent[2],
                    fill=False,
                    lw=0.6,
                    ls="--",
                )
            )
        for j, dog in enumerate(DOGS):
            style = paper_style.DOG_STYLE[dog]
            ax.plot(
                dog_lon[j],
                dog_lat[j],
                ls="none",
                marker=style["marker"],
                ms=6,
                mfc=style["color"],
                mec="k",
                mew=0.8,
                zorder=5,
            )
            ax.annotate(
                style["label"],
                (dog_lon[j], dog_lat[j]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontweight="bold",
                path_effects=halo,
                zorder=6,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(north_m / east_m)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
        ax.xaxis.set_major_formatter(lon_label)
        ax.yaxis.set_major_formatter(lat_label)
        ax.set_xlabel("Longitude (WGS84)")

    track = ax_kept.scatter(
        kept_lon,
        kept_lat,
        c=kept_hours,
        s=0.8,
        lw=0,
        cmap="viridis",
        vmin=paper_style.TIME_LIMITS_H[0],
        vmax=paper_style.TIME_LIMITS_H[1],
        zorder=3,
    )
    paper_style.rasterize_dense(track, kept_lon.size)
    ax_kept.set_ylabel("Latitude (WGS84)")
    ax_kept.set_title(f"Kept GNSS epochs ({kept_lon.size:,})".replace(",", "\u2009"))
    points = ax_rejected.scatter(
        rej_lon, rej_lat, s=0.8, color=REJECTED_COLOR, lw=0, zorder=3
    )
    paper_style.rasterize_dense(points, rej_lon.size)
    ax_rejected.tick_params(labelleft=False)
    ax_rejected.set_title(
        f"Rejected GNSS epochs ({rej_lon.size:,})".replace(",", "\u2009")
    )

    # Scale bar and north arrow in the lower-left corner of (a).
    x0 = xlim[0] + 0.08 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    bar_deg = SCALE_BAR_KM * 1000.0 / east_m
    ax_kept.plot([x0, x0 + bar_deg], [y0, y0], color="k", lw=2.0, solid_capstyle="butt")
    ax_kept.text(
        x0 + bar_deg / 2,
        y0 + 0.01 * (ylim[1] - ylim[0]),
        f"{SCALE_BAR_KM:.0f} km",
        ha="center",
        va="bottom",
        path_effects=halo,
    )
    arrow_deg = 0.07 * (ylim[1] - ylim[0])
    ax_kept.annotate(
        "N",
        xy=(x0, y0 + 2.9 * arrow_deg),
        xytext=(x0, y0 + 1.5 * arrow_deg),
        ha="center",
        va="top",
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 1.0},
        path_effects=halo,
    )

    inset = fig.add_axes([0.80, 0.60, 0.185, 0.36])
    draw_locator(inset, (*xlim, *ylim))
    for ax, letter in ((ax_kept, "a"), (ax_rejected, "b"), (inset, "c")):
        paper_style.panel_label(ax, letter)

    time_axes = fig.add_axes([0.10, 0.075, 0.30, 0.02])
    colorbar = fig.colorbar(track, cax=time_axes, orientation="horizontal")
    colorbar.set_label(paper_style.TIME_LABEL)
    depth_axes = fig.add_axes([0.45, 0.075, 0.30, 0.02])
    colorbar = fig.colorbar(bathy, cax=depth_axes, orientation="horizontal")
    colorbar.ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    colorbar.ax.xaxis.set_major_formatter(lambda v, _: fc.signed(v, ".0f"))
    colorbar.set_label("GEBCO 2025 height (m, relative to mean sea level)")
    if tid is not None:
        hatch = Rectangle((0, 0), 1, 1, fc="none", hatch="///", lw=0.4, ec="0.35")
        fig.legend(
            [hatch],
            [MULTIBEAM_LABEL],
            loc="upper left",
            bbox_to_anchor=(0.785, 0.40),
            handlelength=1.6,
            handleheight=1.6,
        )
    fc.save(fig, NAME)


if __name__ == "__main__":
    main()
