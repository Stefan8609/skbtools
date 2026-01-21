import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)


@njit(cache=True)
def _rmse_cm_for_offset(
    offset,
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    thresh=0.5,
):
    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset,
        thresh,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )[1:4]

    N = len(CDOG_full)
    if N == 0:
        return np.nan, 0

    # integer-cycle correction (same as elsewhere)
    for i in range(N):
        diff = GPS_full[i] - CDOG_full[i]
        if abs(diff) >= 0.9:
            CDOG_full[i] += np.round(diff)

    rmse_cm = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515.0 * 100.0
    return rmse_cm, N


@njit(cache=True)
def _coherence_obj_for_offset(
    offset,
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    thresh=0.5,
):
    """
    Phase-coherence objective:
      r = CDOG_full - GPS_full
      w = wrap(r) into [-0.5, 0.5)
      R = |mean(exp(i 2π w))|
      obj = (1 - R) / sqrt(N)   (smaller is better)
    """
    CDOG_full, GPS_clock, GPS_full = two_pointer_index(
        offset,
        thresh,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )[1:4]

    N = len(CDOG_full)
    if N < 5:
        return np.nan, N, np.nan

    r = CDOG_full - GPS_full
    w = (r + 0.5) % 1.0 - 0.5  # wrap to [-0.5, 0.5)
    # Numba supports complex128; this is fine
    R = np.abs(np.mean(np.exp(1j * 2.0 * np.pi * w)))

    obj = (1.0 - R) / np.sqrt(N)
    return obj, N, R


def plot_integer_pick_metrics_dog(
    center_offset,
    CDOG_data,
    GPS_data,
    travel_times,
    transponder_coordinates,
    esv,
    half_window=100.0,
    step=0.1,
    thresh=0.5,
    vlines=None,
    title_suffix="",
):
    """
    vlines: optional dict like {"best_offset": value, "offset_hint": value}
    """
    offsets = np.round(
        np.arange(
            center_offset - half_window, center_offset + half_window + step, step
        ),
        8,
    )

    rmse = np.empty(offsets.shape[0], dtype=np.float64)
    nmatch = np.empty(offsets.shape[0], dtype=np.float64)
    coh_obj = np.empty(offsets.shape[0], dtype=np.float64)
    coh_R = np.empty(offsets.shape[0], dtype=np.float64)

    for i in range(offsets.shape[0]):
        off = offsets[i]
        rmse[i], nmatch[i] = _rmse_cm_for_offset(
            off,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            thresh=thresh,
        )
        coh_obj[i], _, coh_R[i] = _coherence_obj_for_offset(
            off,
            CDOG_data,
            GPS_data,
            travel_times,
            transponder_coordinates,
            esv,
            thresh=thresh,
        )

    finite = np.isfinite(coh_obj)
    best_off = offsets[np.nanargmin(coh_obj)] if finite.any() else np.nan

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axes[0].plot(offsets, rmse)
    axes[0].set_ylabel("RMSE (cm)")
    axes[0].grid(True)

    axes[1].plot(offsets, coh_obj)
    axes[1].set_ylabel("Circular Variance $(1 - R)/(sqrt(M))$")
    axes[1].grid(True)

    axes[2].plot(offsets, nmatch)
    axes[2].set_ylabel("Matched Pairs (N)")
    axes[2].set_xlabel("Timing Bias (s)")
    axes[2].grid(True)

    # Annotate best by Circular Variance
    if np.isfinite(best_off):
        for ax in axes:
            ax.axvline(best_off, linestyle="--")
        # axes[0].set_title(
        #     f"Integer-pick metrics (best by coherence: {best_off:.3f}{title_suffix}"
        # )

    # Additional vlines (hint / solved offset etc.)
    if vlines is not None:
        for _, val in vlines.items():
            if val is None:
                continue
            for ax in axes:
                ax.axvline(val, linestyle=":")
        # Don't overwrite the title if coherence already set it
        if not np.isfinite(best_off):
            axes[0].set_title(f"Integer-pick metrics{title_suffix}")

    plt.tight_layout()
    plt.show()

    return offsets, rmse, coh_obj, nmatch, coh_R, best_off
