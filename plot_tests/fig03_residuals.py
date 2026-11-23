"""Travel-time residuals, one row per DOG.

Question: how large are the travel-time residuals, are they random in time, and do
the three DOGs share them?
Message: the residuals are close to normal in distribution but not white in time:
they wander over tens of minutes, and that slow part is largely common to the
three DOGs (correlations of the 5-min medians are in the row titles).

Layout follows Plot_Modular.time_series_plot: histogram with the fitted normal
curve on the left, residuals against time on the right. All rows share one
symmetric residual axis that holds every residual.

Input: Output/Plot_Data/integer_pick_metrics_DOG{n}.npz, the observed and modelled
travel times of the Gauss-Newton solution (the MCMC chains predate the GNSS
reprocessing of 2026-11-21; switch to their residuals once they are rerun).
"""

import numpy as np
from scipy.stats import norm

import _figure_common as fc
import _style as st

NAME = "fig03_residuals"
PICK_METRICS = str(
    fc.REPO / "Bermuda_Data/Output/Plot_Data/integer_pick_metrics_DOG{}.npz"
)
SOUND_SPEED_MPS = 1515.0  # c in c·sigma_epsilon, as in Table 1
MEDIAN_WINDOW_S = 300.0  # running median that shows the slow part
MEDIAN_MIN_COUNT = 30
N_BINS = 60
TIME_LIMITS_H = (0.0, 14.0)
MS_PER_CM = 10.0 / SOUND_SPEED_MPS


def load_residuals(dog):
    """Emission times (GPS s) and travel-time residuals (ms), observed − modelled."""
    data = np.load(PICK_METRICS.format(dog))
    return data["GPS_clock"], (data["CDOG_full"] - data["GPS_full"]) * 1e3


def main():
    st.apply()
    series = {dog: load_residuals(dog) for dog in st.DOGS}
    start_s = min(t_s[0] for t_s, _ in series.values())
    stop_s = max(t_s[-1] for t_s, _ in series.values())
    window_edges_s = np.arange(start_s, stop_s + MEDIAN_WINDOW_S, MEDIAN_WINDOW_S)
    largest_ms = max(np.abs(residual_ms).max() for _, residual_ms in series.values())
    limit_ms = np.ceil(1.03 * largest_ms / 0.05) * 0.05
    bin_edges_ms = np.linspace(-limit_ms, limit_ms, N_BINS + 1)

    fig, axes = st.figure(
        st.DOUBLE_MM,
        125.0,
        nrows=3,
        ncols=2,
        sharex="col",
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 5.0], "wspace": 0.03},
    )
    medians = {}
    for (ax_hist, ax_time), dog in zip(axes, st.DOGS, strict=True):
        t_s, residual_ms = series[dog]
        mean_ms, sd_ms = norm.fit(residual_ms)

        ax_hist.hist(
            residual_ms,
            bins=bin_edges_ms,
            density=True,
            orientation="horizontal",
            color=st.DATA_BLUE,
            alpha=0.5,
        )
        grid_ms = np.linspace(-limit_ms, limit_ms, 300)
        ax_hist.plot(norm.pdf(grid_ms, mean_ms, sd_ms), grid_ms, "k", lw=1.2)
        ax_hist.invert_xaxis()
        ax_hist.set_ylabel("Residuals $\\epsilon$ (ms)")

        points = ax_time.scatter(
            st.hours(t_s),
            residual_ms,
            s=1.0,
            color=st.DATA_BLUE,
            linewidths=0,
            rasterized=True,
            label="Residuals $\\epsilon$",
        )
        centres_s, median, _ = fc.binned_percentiles(
            t_s,
            residual_ms,
            window_edges_s,
            percentiles=(50.0,),
            min_count=MEDIAN_MIN_COUNT,
        )
        medians[dog] = median[:, 0]
        (median_line,) = ax_time.plot(
            st.hours(centres_s), medians[dog], "k", lw=1.0, label="5-min median"
        )
        ax_time.set_title(
            f"{st.DOG[dog]['label']}:  $\\sigma_\\epsilon$ = {sd_ms:.3f} ms,  "
            f"$c\\,\\sigma_\\epsilon$ = {sd_ms / MS_PER_CM:.1f} cm",
            loc="left",
            pad=3,
        )
        range_axis = ax_time.secondary_yaxis(
            "right", functions=(lambda ms: ms / MS_PER_CM, lambda cm: cm * MS_PER_CM)
        )
        range_axis.set_ylabel("Range residuals $c\\,\\epsilon$ (cm)")
        ax_time.tick_params(right=False)
        print(
            f"DOG {dog}: n {residual_ms.size}, mean {mean_ms * 1e3:+.2f} µs, "
            f"sigma_eps {sd_ms:.4f} ms, c sigma_eps {sd_ms / MS_PER_CM:.2f} cm, "
            f"largest |eps| {np.abs(residual_ms).max():.3f} ms"
        )

    for ax, dog in zip(axes[:, 1], st.DOGS, strict=True):
        parts = []
        for other in st.DOGS:
            if other == dog:
                continue
            both = np.isfinite(medians[dog]) & np.isfinite(medians[other])
            r = fc.correlation(medians[dog][both], medians[other][both])
            parts.append(f"DOG {other}: $r$ = {r:.2f}")
            print(f"5-min medians, DOG {dog} vs {other}: r = {r:.3f}, n = {both.sum()}")
        ax.set_title("5-min median vs " + ",  ".join(parts), loc="right", pad=3)
    fig.legend(
        handles=[points, median_line],
        loc="outside upper right",
        ncols=2,
        markerscale=4,
        columnspacing=1.2,
    )
    axes[0, 0].set_ylim(-limit_ms, limit_ms)
    axes[0, 1].set_xlim(TIME_LIMITS_H)
    axes[-1, 1].set_xticks(np.arange(0, 15, 2))
    axes[-1, 1].set_xlabel(st.TIME_LABEL)
    axes[-1, 0].set_xlabel("Density (1/ms)")
    for ax, letter in zip(axes[:, 0], "abc", strict=True):
        st.panel_letter(ax, letter, x_pt=-34.0)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
