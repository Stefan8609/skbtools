"""ESV bias over the survey under models of increasing time resolution.

Question: does the ESV bias change during the survey, and does the answer depend
on how its time variation is modelled?
Message: a single constant hides changes of about 0.1 m/s. Four blocks per DOG
and a 30-min series common to the three DOGs agree on those changes, and the
series hardly moves when white noise or a clock drift is added to the model; only
the level differs between models (it trades off with the timing bias and height).

Each line is the posterior mean of the ESV bias applied to one DOG: the common
series plus that DOG's offset. The band is the 95 % interval of the 30-min model.

Input: MCMC runs listed in MODELS.
"""

import numpy as np

import _figure_common as fc
import _style as st

NAME = "fig07_esv_correction"
# (run, legend label, colour, line width)
MODELS = (
    ("ar1_splits1", "Constant per DOG", "#D55E00", 1.2),
    ("ar1", "4 blocks per DOG", "#56B4E9", 1.2),
    (fc.FINAL_RUN, "Common 30-min series", "black", 1.5),
    ("ar1_white_common30", "Common 30-min series, added white noise", "#009E73", 0.9),
    ("ar1_common30_drift", "Common 30-min series, clock drift", "#CC79A7", 0.9),
)


def esv_series(run):
    """Window edges (h) and draws (n, DOG, window) of the ESV bias (m/s) of a run."""
    pooled = fc.pooled_run(run)
    config = fc.run_config(pooled)
    if config.get("esv_window_edges_s") is not None:  # common windows
        edges_s = np.asarray(config["esv_window_edges_s"], dtype=float)
    else:  # blocks of equal numbers of GNSS epochs, as np.array_split cuts them
        gps_s = np.load(fc.REPO / "Bermuda_Data" / config["gps_file"])["GPS_data"]
        blocks = np.array_split(np.arange(gps_s.size), config["num_esv_splits"])
        edges_s = np.array([gps_s[block[0]] for block in blocks] + [gps_s[-1]])
    return st.hours(edges_s), pooled["esv_bias"]


def main():
    st.apply()
    fig, axes = st.figure(st.DOUBLE_MM, 120.0, nrows=3, sharex=True, sharey=True)
    for run, label, color, width in MODELS:
        edges_h, draws = esv_series(run)
        mean = draws.mean(axis=0)
        for j, ax in enumerate(axes):
            if run == fc.FINAL_RUN:
                low, high = np.percentile(draws[:, j, :], [2.5, 97.5], axis=0)
                ax.stairs(high, edges_h, baseline=low, fill=True, color="0.8", lw=0)
            ax.stairs(
                mean[j], edges_h, baseline=None, color=color, lw=width, label=label
            )
        print(
            f"{run}: range of the posterior-mean ESV bias over the survey, DOG 1/3/4: "
            f"{np.ptp(mean, axis=1).round(3)} m/s; mean {mean.mean(axis=1).round(3)}"
        )
    for ax, dog, letter in zip(axes, st.DOGS, "abc", strict=True):
        ax.set_title(st.DOG[dog]["label"], loc="left", pad=3)
        ax.set_ylabel("ESV bias $\\delta c$ (m/s)")
        st.panel_letter(ax, letter, x_pt=-36.0, y_pt=3.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, columnspacing=1.2)
    axes[-1].set_xlim(st.TIME_LIMITS_H)
    axes[-1].set_xticks(st.TIME_TICKS_H)
    axes[-1].set_xlabel(st.TIME_LABEL)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
