"""Separation between DOG positions estimated from two parts of the survey.

Question: do two parts of the survey give the same DOG positions?
Message: interleaved 15-min blocks agree within their uncertainty; the two halves,
on different arms of the track, differ by 44-82 cm horizontally and up to 2.3 m in
height. Error bars and the dashed 95 % line come from the summed posterior
covariances.

Input: MCMC runs of the final model on parts of the survey.
"""

import numpy as np

import _figure_common as fc
import _style as st

NAME = "fig08_repeatability"
# (legend label, run of the first part, run of the second part, colour)
SPLITS = (
    (
        "Interleaved 15-min blocks",
        "ar1_common30_alt15a",
        "ar1_common30_alt15b",
        "#56B4E9",
    ),
    (
        "First half and second half",
        "ar1_common30_half1",
        "ar1_common30_half2",
        "#D55E00",
    ),
)
N_DRAWS = 20000
SEED = 20261121
BAR_WIDTH = 0.36


def separation_statistics(difference_cm, covariance, rng):
    """Distance, its 95 % interval and the 95 % distance expected from noise alone."""
    difference_cm = np.atleast_1d(difference_cm)
    covariance = np.atleast_2d(covariance)
    noise = rng.multivariate_normal(np.zeros(difference_cm.size), covariance, N_DRAWS)
    with_difference = np.linalg.norm(noise + difference_cm, axis=1)
    return (
        np.linalg.norm(difference_cm),
        np.percentile(with_difference, [2.5, 97.5]),
        np.percentile(np.linalg.norm(noise, axis=1), 95.0),
    )


def main():
    """Draw and save the figure."""
    st.apply()
    final = fc.pooled_run(fc.FINAL_RUN)
    reference = {
        dog: final["CDOG_reference"] + final["CDOG_aug"][:, j, :].mean(axis=0)
        for j, dog in enumerate(fc.run_config(final)["dogs"])
    }
    rng = np.random.default_rng(SEED)
    fig, (ax_h, ax_v) = st.figure(st.DOUBLE_MM, 76.0, ncols=2)
    for s, (label, first, second, color) in enumerate(SPLITS):
        parts = (
            fc.run_positions_cm(first, reference),
            fc.run_positions_cm(second, reference),
        )
        for k, dog in enumerate(st.DOGS):
            difference = parts[1][dog][0] - parts[0][dog][0]
            covariance = parts[0][dog][1] + parts[1][dog][1]
            x = k + (s - 0.5) * BAR_WIDTH
            for ax, index in ((ax_h, slice(0, 2)), (ax_v, slice(2, 3))):
                distance, interval, noise_95 = separation_statistics(
                    difference[index], covariance[index, index], rng
                )
                ax.bar(
                    x,
                    distance,
                    BAR_WIDTH * 0.92,
                    color=color,
                    label=label if k == 0 else None,
                )
                ax.errorbar(
                    x,
                    distance,
                    yerr=[[distance - interval[0]], [interval[1] - distance]],
                    fmt="none",
                    ecolor="black",
                    capsize=2.5,
                    lw=0.9,
                )
                ax.plot(
                    [x - BAR_WIDTH / 2, x + BAR_WIDTH / 2],
                    [noise_95, noise_95],
                    "k--",
                    lw=1.2,
                    label="95 % limit if the parts differed only by their\n"
                    "posterior uncertainty"
                    if (k, s) == (0, 0)
                    else None,
                )
                ax.annotate(
                    f"{distance:.0f}",
                    (x, interval[1]),
                    (0, 2),
                    textcoords="offset points",
                    ha="center",
                )
                print(
                    f"{label}, DOG {dog}, {'horiz.' if ax is ax_h else 'vertical'}: "
                    f"{distance:.1f} cm (95 % {interval.round(1)}), noise alone < "
                    f"{noise_95:.1f} cm"
                )
    for ax, ylabel, letter in (
        (ax_h, "Horizontal separation (cm)", "a"),
        (ax_v, "Vertical separation (cm)", "b"),
    ):
        ax.set_xticks(range(len(st.DOGS)))
        ax.set_xticklabels([st.DOG[dog]["label"] for dog in st.DOGS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.08 * ax.get_ylim()[1])
        ax.grid(axis="x", visible=False)
        st.panel_letter(ax, letter)
    handles, labels = ax_h.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, columnspacing=1.5)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
