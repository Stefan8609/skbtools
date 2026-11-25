"""Survey map.

Question: where are the DOGs, and how did the vessel sail over them?
Message: three DOGs ~2 km apart on a flat seafloor 5.2 km deep, crossed by an
hourglass track with a diagonal leg.

Inputs: processed GNSS epochs, the final MCMC run, the GEBCO 2025 grid (drawn cell
by cell, unsmoothed).
"""

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from pymap3d import ecef2geodetic

import _figure_common as fc
import _style as st

NAME = "fig01_survey_geometry"
PROCESSED_FILE = fc.REPO / "Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz"
GEBCO_GRID = fc.REPO / "Bermuda_Data/gebco_2025_bermuda_region_ascii.asc"
HALF_WIDTH_DEG = (0.095, 0.115)  # longitude, latitude half-widths of the map
FONT_PT = 10.0  # larger than the 8 pt of the other figures: the map has little text
TRACK_GAP_S = 60.0  # the track line is broken where GNSS epochs are missing
# Ocean blues, light (shallow) to dark (deep); kept light enough for black labels.
OCEAN = LinearSegmentedColormap.from_list(
    "ocean_blues", ["#e3f2fb", "#a9d3ee", "#62a8d8", "#2f7fbf", "#1b5a9c"]
)
DOG_MARKER = {"marker": "^", "ms": 7.0, "mfc": "#ffd23f", "mec": "black", "mew": 0.8}
LABEL_OFFSET_PT = {1: (-8, 4), 3: (9, 0), 4: (-8, -6)}  # clear of the track
TRACK_CMAP = "plasma"  # time along the track; warm colours stand out on the blues
TRACK_EDGE = [patheffects.withStroke(linewidth=2.2, foreground="black")]
HALO = [patheffects.withStroke(linewidth=2.0, foreground="white")]


def gebco_cells(centre_lat, centre_lon):
    """Cell edges (lon, lat) and depth (m, positive down) of the cells on the map."""
    grid, extent = fc.read_esri_ascii(GEBCO_GRID)
    cell = (extent[1] - extent[0]) / grid.shape[1]
    lon_edges = extent[0] + cell * np.arange(grid.shape[1] + 1)
    lat_edges = extent[3] - cell * np.arange(grid.shape[0] + 1)
    cols = np.flatnonzero(
        np.abs(lon_edges[:-1] + cell / 2 - centre_lon) < HALF_WIDTH_DEG[0] + cell
    )
    rows = np.flatnonzero(
        np.abs(lat_edges[:-1] - cell / 2 - centre_lat) < HALF_WIDTH_DEG[1] + cell
    )
    return (
        lon_edges[cols[0] : cols[-1] + 2],
        lat_edges[rows[0] : rows[-1] + 2],
        -grid[np.ix_(rows, cols)],
    )


def main():
    """Draw and save the figure."""
    st.apply()
    plt.rcParams["axes.grid"] = False
    plt.rcParams.update(
        {
            key: FONT_PT
            for key in (
                "font.size",
                "axes.labelsize",
                "xtick.labelsize",
                "ytick.labelsize",
                "legend.fontsize",
            )
        }
    )
    processed = np.load(PROCESSED_FILE)
    pooled = fc.pooled_run(fc.FINAL_RUN)
    dog_ecef = pooled["CDOG_reference"] + pooled["CDOG_aug"].mean(axis=0)
    dog_lat, dog_lon, dog_height = ecef2geodetic(*dog_ecef.T)
    centre_lat, centre_lon = dog_lat.mean(), dog_lon.mean()

    track_lat, track_lon, _ = ecef2geodetic(
        *processed["GPS_Coordinates"].mean(axis=1).T
    )
    track_hours = st.hours(processed["GPS_data"])
    points = np.column_stack([track_lon, track_lat])
    connected = np.diff(processed["GPS_data"]) <= TRACK_GAP_S
    segments = np.stack([points[:-1], points[1:]], axis=1)[connected]

    fig, ax = st.figure(st.MEDIUM_MM, 119.0)
    fig.get_layout_engine().set(w_pad=0.08)
    lon_edges, lat_edges, depth = gebco_cells(centre_lat, centre_lon)
    cells = ax.pcolormesh(lon_edges, lat_edges, depth, cmap=OCEAN, rasterized=True)
    colourbar = fig.colorbar(cells, ax=ax, shrink=1.0, aspect=30, pad=0.03)
    colourbar.set_label("GEBCO depth (m)")
    colourbar.ax.invert_yaxis()  # deeper is lower

    ax.add_collection(LineCollection(segments, color="black", lw=2.4, capstyle="round"))
    track = LineCollection(
        segments,
        array=track_hours[:-1][connected],
        cmap=TRACK_CMAP,
        norm=Normalize(*st.TIME_LIMITS_H),
        lw=1.4,
        capstyle="round",
    )
    ax.add_collection(track)
    time_bar = fig.colorbar(
        track, ax=ax, location="top", shrink=0.95, aspect=32, pad=0.03
    )
    time_bar.set_label(st.TIME_LABEL)
    time_bar.set_ticks(st.TIME_TICKS_H)
    for dog, lon, lat in zip(st.DOGS, dog_lon, dog_lat, strict=True):
        ax.plot(lon, lat, ls="", zorder=5, **DOG_MARKER)
        offset = LABEL_OFFSET_PT[dog]
        ax.annotate(
            st.DOG[dog]["label"],
            (lon, lat),
            offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            va="center",
            path_effects=HALO,
            zorder=6,
        )
    ax.plot(
        [],
        [],
        color=plt.get_cmap(TRACK_CMAP)(0.5),
        lw=1.4,
        path_effects=TRACK_EDGE,
        label="Vessel track",
    )
    ax.plot([], [], ls="", label="DOG", **DOG_MARKER)
    ax.legend(loc="lower center", ncols=2, columnspacing=2.5, frameon=False)
    ax.set_xlim(centre_lon - HALF_WIDTH_DEG[0], centre_lon + HALF_WIDTH_DEG[0])
    ax.set_ylim(
        centre_lat - HALF_WIDTH_DEG[1] + 0.02, centre_lat + HALF_WIDTH_DEG[1] - 0.02
    )
    ax.set_aspect(1.0 / np.cos(np.deg2rad(centre_lat)))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Longitude $\\lambda$ (°E)")
    ax.set_ylabel("Latitude $\\phi$ (°N)")

    for dog, lat, lon, height in zip(
        st.DOGS, dog_lat, dog_lon, dog_height, strict=True
    ):
        print(
            f"DOG {dog}: {lat:.5f}° N, {lon:.5f}° E, ellipsoidal height {height:.1f} m"
        )
    print(f"depth of the cells on the map: {depth.min():.0f}–{depth.max():.0f} m")
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
