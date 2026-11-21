"""Site map of the Bermuda survey (new Fig. 2, merged with the remade Fig. 3 track).

Message: the vessel crossed above the three DOGs on an hourglass track over a
14-hour survey on a nearly flat, 5.2-5.3 km deep seafloor west of Bermuda.

One map. The vessel track (mean of the four GNSS antennas at the epochs kept by
the GNSS height filter) is coloured by time since survey start, the only
colourbar. DOGs are at the MCMC posterior mean of the final run. Bathymetry is
the GEBCO 2025 grid at its true cell footprint in light greys with labelled
contours; an inset locates the site relative to Bermuda (0 m contour of the
regional grid). The rejected epochs are in FX-A3; the GEBCO source goes in the
caption.

Reviewer comments: R1 4a (map of the station locations), R1 4e, R1 3f and R3 #10
(distinct DOG markers, "DOG" naming, time colourbar with the Fig. 4 time range),
R1 4d (acquisition geometry), R3 #18 (deployment positions: none exist in the
repo, so none shown). R3 M7(i) (what the filter removed) is answered by FX-A3.
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
HEIGHT_MM = 128.0
MAP_LEFT_MM, MAP_BOTTOM_MM, MAP_WIDTH_MM = 16.0, 24.0, 64.0
SCALE_BAR_KM = 2.0
CONTOUR_STEP_M = 50.0
MAP_PAD_DEG = 0.004
# Deeper is darker; kept light so that the track and the DOGs carry the colour.
BATHY_CMAP = ListedColormap(plt.get_cmap("Greys_r")(np.linspace(0.62, 0.97, 256)))
CONTOUR_COLOR = "0.45"
# Locator inset: lower-left corner in axes fractions and width in mm.
INSET_LEFT, INSET_BOTTOM, INSET_WIDTH_MM = 0.03, 0.44, 22.0
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
    """Land of the regional GEBCO grid and the map's footprint, without ticks."""
    grid, extent = fc.read_esri_ascii(REGION_GRID)
    land = np.where(grid >= 0, 1.0, np.nan)
    ax.imshow(land, extent=extent, cmap=ListedColormap(["0.2"]))
    cell = (extent[1] - extent[0]) / grid.shape[1]
    lon = extent[0] + cell * (np.arange(grid.shape[1]) + 0.5)
    lat = extent[3] - cell * (np.arange(grid.shape[0]) + 0.5)
    west, east, south, north = survey_box
    ax.add_patch(
        Rectangle((west, south), east - west, north - south, fill=False, lw=0.8)
    )
    rows, cols = np.nonzero(grid >= 0)
    ax.annotate(
        "Bermuda",
        (lon[cols].min(), lat[rows].mean()),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
    )
    ax.annotate(
        "Survey",
        (0.5 * (west + east), south),
        xytext=(0, -3),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    east_m, north_m = metres_per_degree(np.mean(extent[2:]), np.mean(extent[:2]))
    ax.set_aspect(north_m / east_m)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    # Distance from the map centre to the nearest land cell, for the caption.
    centre_lat, centre_lon = 0.5 * (south + north), 0.5 * (west + east)
    e, n, _ = geodetic2enu(lat[rows], lon[cols], 0, centre_lat, centre_lon, 0)
    print(f"nearest Bermuda land cell: {np.hypot(e, n).min() / 1000:.0f} km")
    print(f"locator extent (W, E, S, N): {extent}")


def main():
    survey = SURVEY_GRID if SURVEY_GRID.exists() else SMALL_GRID
    fc.require(PROCESSED_FILE, REGION_GRID, survey)
    if survey == SMALL_GRID:
        print(f"BLOCKED (partial): {SURVEY_GRID} missing; bathymetry only near DOGs")
    paper_style.apply()
    # Manual layout (constrained layout cannot hold an equal-distance map + inset).
    plt.rcParams["figure.constrained_layout.use"] = False

    processed = np.load(PROCESSED_FILE)
    track_lat, track_lon = antenna_mean_lat_lon(processed["GPS_Coordinates"])
    track_hours = paper_style.hours_since_start(processed["GPS_data"])

    pooled = fc.pooled_run(RUN)
    dog_ecef = pooled["CDOG_reference"] + pooled["CDOG_aug"].mean(axis=0)
    dog_lat, dog_lon, dog_height = ecef2geodetic(*dog_ecef.T)

    grid, extent = fc.read_esri_ascii(survey)
    xlim = (track_lon.min() - MAP_PAD_DEG, track_lon.max() + MAP_PAD_DEG)
    ylim = (track_lat.min() - MAP_PAD_DEG, track_lat.max() + MAP_PAD_DEG)
    east_m, north_m = metres_per_degree(np.mean(ylim), np.mean(xlim))

    width_mm = paper_style.SINGLE_COLUMN_MM
    fig = plt.figure(
        figsize=(
            width_mm / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    map_height_mm = (
        MAP_WIDTH_MM * (ylim[1] - ylim[0]) * north_m / ((xlim[1] - xlim[0]) * east_m)
    )
    ax = fig.add_axes(
        [
            MAP_LEFT_MM / width_mm,
            MAP_BOTTOM_MM / HEIGHT_MM,
            MAP_WIDTH_MM / width_mm,
            map_height_mm / HEIGHT_MM,
        ]
    )
    cell = (extent[1] - extent[0]) / grid.shape[1]
    lon_edges = extent[0] + cell * np.arange(grid.shape[1] + 1)
    lat_edges = extent[3] - cell * np.arange(grid.shape[0] + 1)
    lon_centres, lat_centres = lon_edges[:-1] + cell / 2, lat_edges[:-1] - cell / 2
    levels = np.arange(
        np.ceil(np.nanmin(grid) / CONTOUR_STEP_M) * CONTOUR_STEP_M,
        np.nanmax(grid),
        CONTOUR_STEP_M,
    )
    if survey == SURVEY_GRID and SURVEY_TID_GRID.exists():
        tid, _ = fc.read_esri_ascii(SURVEY_TID_GRID)
        dog_tid = [
            tid[int((extent[3] - la) / cell), int((lo - extent[0]) / cell)]
            for la, lo in zip(dog_lat, dog_lon, strict=True)
        ]
        print(f"GEBCO TID at the DOG cells (DOGs {DOGS}): {dog_tid}")
        print(f"multibeam (TID {MULTIBEAM_TID}) cells: {np.mean(tid == 11):.0%}")
    # Contours only where the map shows them (two cells inside the frame), so
    # that every contour label lands inside the map.
    rows = (lat_centres > ylim[0] + 2 * cell) & (lat_centres < ylim[1] - 2 * cell)
    cols = (lon_centres > xlim[0] + 2 * cell) & (lon_centres < xlim[1] - 2 * cell)
    inside = grid[np.ix_(rows, cols)]
    print(f"GEBCO height on the map: {inside.min():.0f} to {inside.max():.0f} m")
    halo = [patheffects.withStroke(linewidth=2.0, foreground="w")]
    ax.pcolormesh(lon_edges, lat_edges, grid, cmap=BATHY_CMAP, zorder=0)
    contours = ax.contour(
        lon_centres[cols],
        lat_centres[rows],
        inside,
        levels,
        colors=CONTOUR_COLOR,
        linewidths=0.5,
        linestyles="solid",
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
    track = ax.scatter(
        track_lon,
        track_lat,
        c=track_hours,
        s=1.2,
        lw=0,
        cmap="viridis",
        vmin=paper_style.TIME_LIMITS_H[0],
        vmax=paper_style.TIME_LIMITS_H[1],
        zorder=3,
    )
    paper_style.rasterize_dense(track, track_lon.size)
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
    # Contour labels after the limits are set, so that they land inside the map.
    ax.clabel(
        contours,
        fmt=lambda v: f"{fc.signed(v, '.0f')} m",
        colors="0.25",
        inline_spacing=2,
    )

    # Scale bar and north arrow between the two southern legs of the track.
    x0 = np.mean(xlim) - 0.5 * SCALE_BAR_KM * 1000.0 / east_m
    y0 = ylim[0] + 0.035 * (ylim[1] - ylim[0])
    bar_deg = SCALE_BAR_KM * 1000.0 / east_m
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="k", lw=1.5, solid_capstyle="butt")
    ax.text(
        x0 + bar_deg / 2,
        y0 + 0.008 * (ylim[1] - ylim[0]),
        f"{SCALE_BAR_KM:.0f} km",
        ha="center",
        va="bottom",
    )
    arrow_deg = 0.05 * (ylim[1] - ylim[0])
    x_north = x0 + bar_deg + 0.08 * (xlim[1] - xlim[0])
    ax.annotate(
        "N",
        xy=(x_north, y0 + 1.9 * arrow_deg),
        xytext=(x_north, y0),
        ha="center",
        va="bottom",
        arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 0.8},
    )

    # Locator inset in the empty area west of the track crossing; its height
    # follows from the region's own equal-distance aspect.
    region_extent = fc.read_esri_ascii(REGION_GRID)[1]
    region_east_m, region_north_m = metres_per_degree(
        np.mean(region_extent[2:]), np.mean(region_extent[:2])
    )
    inset_height_mm = (
        INSET_WIDTH_MM
        * (region_extent[3] - region_extent[2])
        * region_north_m
        / ((region_extent[1] - region_extent[0]) * region_east_m)
    )
    inset = ax.inset_axes(
        [
            INSET_LEFT,
            INSET_BOTTOM,
            INSET_WIDTH_MM / MAP_WIDTH_MM,
            inset_height_mm / map_height_mm,
        ],
        zorder=4,
    )
    inset.set_facecolor("white")
    draw_locator(inset, (*xlim, *ylim))

    time_axes = fig.add_axes(
        [MAP_LEFT_MM / width_mm, 11.0 / HEIGHT_MM, MAP_WIDTH_MM / width_mm, 0.016]
    )
    colorbar = fig.colorbar(track, cax=time_axes, orientation="horizontal")
    colorbar.set_ticks(np.arange(0, 15, 2))
    colorbar.set_label(paper_style.TIME_LABEL)
    colorbar.outline.set_linewidth(0.6)
    fc.save(fig, NAME)


if __name__ == "__main__":
    main()
