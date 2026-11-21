"""FX-6: range residuals of the final model against slant range, all three DOGs.

Message: the residuals of the final model show no trend with slant range or
elevation angle; their median stays within about 10 cm at all ranges.

Replaces the submitted Figs. 6, C14 and C15. One panel per DOG on shared axes:
every residual as a light grey, rasterized point, with the median and 16-84 % band
of slant-range bins in the DOG colour (they replace the unexplained red ±σ lines,
R1 4j). The top axis gives the elevation angle at the DOG, so low-elevation
behaviour (R2 7) can be read off. The points are not coloured by time any more;
single passes at particular times show as clumps (FX-5 has the time axis).
Residuals as in FX-5 (posterior mean of the final run, range at 1515 m/s, R1 4i).

Inputs: the chains of FINAL_RUN (Bermuda_Data/Output/MCMC_Chains/ar1_common30/) and
the data they were sampled on (rebuilt by _figure_common.final_residuals).
"""

import numpy as np

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
ELEVATION_TICKS_DEG = (60, 40, 30, 25)
HEIGHT_MM = 72.0
# -----------------------------------------------------------------------------------


def draw(residuals):
    """One panel per DOG of residual against slant range; returns the figure."""
    fig, axes = paper_style.figure(
        "double", HEIGHT_MM, ncols=3, sharex=True, sharey=True
    )
    edges_km = np.arange(RANGE_LIMITS_KM[0], RANGE_LIMITS_KM[1] + 1e-9, BIN_KM)
    # Elevation for DOG 1's vertical separation (straight ray); beyond 5.5 km the
    # other DOGs differ by < 1 deg (printed in __main__).
    height_km = vertical_separation_km(residuals[DOGS[0]])

    def to_elevation(r_km):
        ratio = height_km / np.maximum(np.asarray(r_km, dtype=float), height_km)
        return np.rad2deg(np.arcsin(ratio))

    def to_range(elevation_deg):
        return height_km / np.sin(np.deg2rad(np.clip(elevation_deg, 1.0, 90.0)))

    for k, (ax, dog) in enumerate(zip(axes, DOGS, strict=True)):
        style = paper_style.DOG_STYLE[dog]
        res_cm = residuals[dog]["residual_cm"]
        range_km = residuals[dog]["slant_range_m"] / 1000.0
        if res_cm.min() < Y_LIMITS_CM[0] or res_cm.max() > Y_LIMITS_CM[1]:
            raise ValueError(f"DOG {dog} residuals exceed {Y_LIMITS_CM} cm")
        if range_km.min() < RANGE_LIMITS_KM[0] or range_km.max() > RANGE_LIMITS_KM[1]:
            raise ValueError(f"DOG {dog} slant ranges exceed {RANGE_LIMITS_KM} km")
        centres, pct, _ = fc.binned_percentiles(
            range_km, res_cm, edges_km, min_count=MIN_PER_BIN
        )
        ax.axhline(0.0, color=fc.CONTEXT, lw=0.5, zorder=1)
        points = ax.scatter(
            range_km, res_cm, color=fc.RAW_POINTS, s=0.8, lw=0, zorder=2
        )
        paper_style.rasterize_dense(points, res_cm.size)
        ax.fill_between(
            centres,
            pct[:, 0],
            pct[:, 2],
            color=style["color"],
            alpha=0.45,
            lw=0,
            zorder=3,
        )
        ax.plot(centres, pct[:, 1], color=style["color"], lw=1.6, zorder=4)
        ax.text(
            0.97, 0.96, style["label"], transform=ax.transAxes, ha="right", va="top"
        )
        ax.spines["right"].set_visible(False)
        top = ax.secondary_xaxis("top", functions=(to_elevation, to_range))
        top.set_xticks(ELEVATION_TICKS_DEG)
        top.set_xlabel("Elevation angle (°)")
        ax.set_xlabel("Slant range (km)")
        paper_style.panel_label(ax, "abc"[k])
    axes[0].set_xlim(*RANGE_LIMITS_KM)
    axes[0].set_xticks([6, 8, 10, 12])
    axes[0].set_ylim(*Y_LIMITS_CM)
    axes[0].set_yticks([-50, 0, 50])
    axes[0].set_ylabel("Range residual (cm)")
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
            f"DOG {dog}: n = {res['residual_cm'].size}, sd = "
            f"{res['residual_cm'].std():.2f} cm; elevation "
            f"{res['elevation_deg'].min():.1f}-"
            f"{res['elevation_deg'].max():.1f}°, slant range "
            f"{res['slant_range_m'].min() / 1e3:.2f}-"
            f"{res['slant_range_m'].max() / 1e3:.2f} km; below 30°: "
            f"n = {low.sum()}, sd = {res['residual_cm'][low].std():.2f} cm, "
            f"above: sd = {res['residual_cm'][~low].std():.2f} cm"
        )
    fig = draw(residuals)
    fc.save(fig, NAME)
