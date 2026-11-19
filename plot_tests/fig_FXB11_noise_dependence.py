"""FX-B11 (remake of Fig. B11): ranging residual against GNSS and timing noise.

Grid of synthetic inversions (combined_dependence.py: Bermuda DOG 3 trajectory,
receiver noise σd and DOG timing noise σt, free clock offset, every realisation
kept). (a) Mean residual sd c·σε over the realisations at each grid point (filled
contours) with the Eq. B36 prediction c·sqrt(σt² + (g σd / c)²) as black contours,
g computed from the antenna geometry (K8; the submitted B36 used g = 1). (b) c·σε
against σd at σt = 0 and (c) against σt at σd = 0: mean ± 1 sd across realisations
(band), with B36 for the computed g (solid) and for g = 1 (dashed).

Reviewer comments: R3 #12 (one realisation per grid point; ≥ 30 realisations with
uncertainty bands), K8 (B36 underestimates the GNSS term by g ≈ 1.6; the submitted
grid used a retry loop that discarded failed fits).

Input: ``Plot_Data/Combined_Dependence_<table>.npz`` with the per-realisation arrays
``Z_std_all`` (n_real, n_t, n_d) in s, axes ``X`` (σd, m) and ``Y`` (σt, s), and
the gain ``g``.
"""

import numpy as np

import _figure_common as fc
from acoustics.synthetic_esv_tables import ESV_MODERATE
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
INVERSION_ESV = ESV_MODERATE
INPUT = (
    fc.REPO / f"Bermuda_Data/Output/Plot_Data/Combined_Dependence_{INVERSION_ESV}.npz"
)
MIN_REALIZATIONS = 30  # R3 #12
CONTOUR_STEP_CM = 2.0
FILL_STEP_CM = 1.0
HEIGHT_MM = 72.0
BAND_COLOR = paper_style.CATEGORY_COLORS[0]
NAME = "fig_FXB11_noise_dependence"
# -----------------------------------------------------------------------------------

C_MPS = paper_style.RANGE_SOUND_SPEED_MPS


def dependence_figure(sigma_d_m, sigma_t_s, range_sd_cm_all, gain):
    """Map and two sections; inputs are the producer's X, Y (n_t, n_d) grids and
    the per-realisation range sd (n_real, n_t, n_d) in cm."""
    mean_cm = range_sd_cm_all.mean(axis=0)
    sd_cm = range_sd_cm_all.std(axis=0, ddof=1)
    d_cm, t_us = sigma_d_m * 100.0, sigma_t_s * 1e6
    expected_cm = fc.b36_range_sd_cm(sigma_d_m, sigma_t_s, gain, C_MPS)

    paper_style.apply()
    fig, axes = paper_style.figure(
        "double", HEIGHT_MM, ncols=3, gridspec_kw={"width_ratios": (1.25, 1, 1)}
    )
    ax_map, ax_d, ax_t = axes
    top = np.ceil(max(mean_cm.max(), expected_cm.max()))
    fill = ax_map.contourf(
        d_cm, t_us, mean_cm, levels=np.arange(0.0, top + FILL_STEP_CM, FILL_STEP_CM)
    )
    lines = ax_map.contour(
        d_cm,
        t_us,
        expected_cm,
        levels=np.arange(CONTOUR_STEP_CM, top + CONTOUR_STEP_CM, CONTOUR_STEP_CM),
        colors="k",
        linewidths=0.7,
    )
    ax_map.clabel(lines, fmt="%.0f", fontsize=paper_style.FONT_SIZE_PT)
    colorbar = fig.colorbar(fill, ax=ax_map)
    colorbar.set_label("Mean range residual sd cσε (cm)")
    ax_map.set_xlabel("Receiver noise σd (cm)")
    ax_map.set_ylabel("Timing noise σt (µs)")

    sections = (
        (ax_d, d_cm[0], mean_cm[0], sd_cm[0], sigma_d_m[0], np.zeros(d_cm.shape[1])),
        (
            ax_t,
            t_us[:, 0],
            mean_cm[:, 0],
            sd_cm[:, 0],
            np.zeros(t_us.shape[0]),
            sigma_t_s[:, 0],
        ),
    )
    for ax, x, mean, sd, d_m, t_s in sections:
        ax.fill_between(x, mean - sd, mean + sd, color=BAND_COLOR, alpha=0.3, lw=0)
        ax.plot(x, mean, color=BAND_COLOR, marker="o", ms=2.5, label="Synthetic, ±1 sd")
        ax.plot(
            x,
            fc.b36_range_sd_cm(d_m, t_s, gain, C_MPS),
            color="k",
            label=f"B36, g = {gain:.2f}",
        )
        ax.plot(
            x,
            fc.b36_range_sd_cm(d_m, t_s, 1.0, C_MPS),
            color="k",
            ls="--",
            label="B36, g = 1",
        )
        ax.set_ylabel("Range residual sd cσε (cm)")
        ax.set_ylim(bottom=0.0)
    ax_d.set_xlabel("Receiver noise σd (cm), σt = 0")
    ax_t.set_xlabel("Timing noise σt (µs), σd = 0")
    ax_d.legend(loc="upper left")
    for ax, letter in zip(axes, "abc", strict=True):
        paper_style.panel_label(ax, letter)
    return fig


def main():
    with np.load(INPUT) as npz:
        sigma_d_m, sigma_t_s = npz["X"], npz["Y"]
        range_sd_cm_all = npz["Z_std_all"] * C_MPS * 100.0
        gain = float(npz["g"])
    n_real = range_sd_cm_all.shape[0]
    print(f"{n_real} realisations per grid point, g = {gain:.3f}")
    if n_real < MIN_REALIZATIONS:
        print(f"BLOCKED: {INPUT} has {n_real} realisations, need {MIN_REALIZATIONS}")
        raise SystemExit(fc.SKIP_EXIT_CODE)
    if not (sigma_d_m[0, 0] == 0.0 and sigma_t_s[0, 0] == 0.0):
        raise ValueError("the sections need a grid that starts at σd = σt = 0")
    fig = dependence_figure(sigma_d_m, sigma_t_s, range_sd_cm_all, gain)
    fc.save(fig, NAME)


if __name__ == "__main__":
    fc.require(INPUT)
    main()
