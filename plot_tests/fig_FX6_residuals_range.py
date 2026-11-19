"""FX-6: range residuals of the final model against slant range, all three DOGs.

Replaces the submitted Figs. 6, C14 and C15. One row per DOG: every residual as a
rasterized point coloured by time since survey start (labelled colourbar, R1 4j),
with the median and 16-84 % band of slant-range bins (explained in the legend; they
replace the unexplained red ±σ lines, R1 4j). The top axis gives the elevation
angle at the DOG, so low-elevation behaviour (R2 7) can be read off. Colour is time
rather than elevation because elevation is a fixed function of slant range at a
fixed depth (top axis) and would repeat the x-axis; time shows the hour-scale
shifts R3 M3 describes. Residuals as in FX-5 (posterior mean of the final run,
range at 1515 m/s, R1 4i).

Inputs: the chains of FINAL_RUN (Bermuda_Data/Output/MCMC_Chains/ar1_common30/) and
the data they were sampled on (rebuilt by _figure_common.final_residuals).
"""

import numpy as np
from matplotlib import patheffects

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
CHAIN_DIR = fc.CHAINS_DIR / RUN
NAME = "fig_FX6_residuals_range"
DOGS = (1, 3, 4)
BIN_KM = 0.25  # slant-range bins of the median and 16-84 % band
MIN_PER_BIN = 50
RANGE_LIMITS_KM = (5.0, 13.0)
Y_LIMITS_CM = (-75.0, 100.0)  # holds every residual of the final run (no clipping)
ELEVATION_TICKS_DEG = (70, 60, 45, 35, 30, 25)
SEED = 20261118  # random drawing order, so late epochs do not hide early ones
CMAP = "cividis"
HEIGHT_MM = 150.0
# -----------------------------------------------------------------------------------


def draw(residuals):
    """Three rows (DOG 1, 3, 4) of residual against slant range; returns the figure."""
    fig, axes = paper_style.figure("double", HEIGHT_MM, nrows=3, sharex=True)
    edges_km = np.arange(RANGE_LIMITS_KM[0], RANGE_LIMITS_KM[1] + 1e-9, BIN_KM)
    rng = np.random.default_rng(SEED)
    for k, (ax, dog) in enumerate(zip(axes, DOGS, strict=True)):
        style = paper_style.DOG_STYLE[dog]
        res_cm = residuals[dog]["residual_cm"]
        range_km = residuals[dog]["slant_range_m"] / 1000.0
        hours = paper_style.hours_since_start(residuals[dog]["gps_time_s"])
        if res_cm.min() < Y_LIMITS_CM[0] or res_cm.max() > Y_LIMITS_CM[1]:
            raise ValueError(f"DOG {dog} residuals exceed {Y_LIMITS_CM} cm")
        if range_km.min() < RANGE_LIMITS_KM[0] or range_km.max() > RANGE_LIMITS_KM[1]:
            raise ValueError(f"DOG {dog} slant ranges exceed {RANGE_LIMITS_KM} km")
        centres, pct, _ = fc.binned_percentiles(
            range_km, res_cm, edges_km, min_count=MIN_PER_BIN
        )
        ax.axhline(0.0, color="0.5", lw=0.5, zorder=1)
        order = rng.permutation(res_cm.size)
        points = ax.scatter(
            range_km[order],
            res_cm[order],
            c=hours[order],
            cmap=CMAP,
            vmin=paper_style.TIME_LIMITS_H[0],
            vmax=paper_style.TIME_LIMITS_H[1],
            s=0.8,
            lw=0,
            zorder=2,
        )
        paper_style.rasterize_dense(points, res_cm.size)
        ax.fill_between(
            centres,
            pct[:, 0],
            pct[:, 2],
            color="#D55E00",
            alpha=0.3,
            lw=0,
            zorder=3,
            label=f"16–84 % of residuals per {BIN_KM:g}-km bin",
        )
        ax.plot(
            centres,
            pct[:, 1],
            color="black",
            lw=1.0,
            # White halo: the line stays readable over the points in greyscale.
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
            zorder=4,
            label=f"Median of residuals per {BIN_KM:g}-km bin",
        )
        count = f"{res_cm.size:,}".replace(",", " ")
        marker = ax.plot([], [], ls="", marker=style["marker"], color=style["color"])
        ax.legend(
            marker,
            [f"{style['label']}: {count} pairs, σ = {res_cm.std():.1f} cm"],
            loc="upper right",
        )
        ax.set_ylim(*Y_LIMITS_CM)
        ax.set_yticks([-50, 0, 50])
        ax.set_ylabel("Range residual (cm)")
        paper_style.panel_label(ax, "abc"[k])
    # Elevation for DOG 1's vertical separation (straight ray); beyond 5.5 km the
    # other DOGs differ by < 1 deg (printed in __main__).
    height_km = vertical_separation_km(residuals[DOGS[0]])

    def to_elevation(r_km):
        ratio = height_km / np.maximum(np.asarray(r_km, dtype=float), height_km)
        return np.rad2deg(np.arcsin(ratio))

    def to_range(elevation_deg):
        return height_km / np.sin(np.deg2rad(np.clip(elevation_deg, 1.0, 90.0)))

    top = axes[0].secondary_xaxis("top", functions=(to_elevation, to_range))
    top.set_xticks(ELEVATION_TICKS_DEG)
    top.set_xlabel(f"Elevation angle at DOG {DOGS[0]} (°)")
    axes[-1].set_xlim(*RANGE_LIMITS_KM)
    axes[-1].set_xlabel("Slant range, transducer to DOG (km)")
    colorbar = fig.colorbar(points, ax=axes, location="right", aspect=40, pad=0.01)
    colorbar.set_label(paper_style.TIME_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center", ncols=2)
    return fig


def vertical_separation_km(dog_residuals):
    """Median transducer height above the DOG (km), from range and elevation."""
    vertical_m = dog_residuals["slant_range_m"] * np.sin(
        np.deg2rad(dog_residuals["elevation_deg"])
    )
    return float(np.median(vertical_m)) / 1000.0


if __name__ == "__main__":
    fc.require(CHAIN_DIR / "chain_00.npz")
    paper_style.apply()
    residuals, _ = fc.final_residuals(fc.pooled_run(RUN))
    heights = {dog: vertical_separation_km(residuals[dog]) for dog in DOGS}
    print("vertical separation (km):", {d: round(h, 4) for d, h in heights.items()})
    r_km = np.linspace(5.5, RANGE_LIMITS_KM[1], 200)
    spread = [np.rad2deg(np.arcsin(h / r_km)) for h in heights.values()]
    print(
        "max elevation difference between DOGs beyond 5.5 km: "
        f"{np.ptp(spread, axis=0).max():.2f}°"
    )
    for dog in DOGS:
        res = residuals[dog]
        low = res["elevation_deg"] < 30.0
        print(
            f"DOG {dog}: elevation {res['elevation_deg'].min():.1f}-"
            f"{res['elevation_deg'].max():.1f}°, slant range "
            f"{res['slant_range_m'].min() / 1e3:.2f}-"
            f"{res['slant_range_m'].max() / 1e3:.2f} km; below 30°: "
            f"n = {low.sum()}, sd = {res['residual_cm'][low].std():.2f} cm, "
            f"above: sd = {res['residual_cm'][~low].std():.2f} cm"
        )
    fig = draw(residuals)
    fc.save(fig, NAME)
