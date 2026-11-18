"""Shared pieces of the revision figures in plot_tests/ (not yet paper code).

- Names and notation used on every figure (``FINAL_RUN``, ``DC_NAME``, lever labels).
- Loaders: pooled MCMC draws, the run's model, the residuals of the final model at
  its posterior mean, DOG positions in local ENU.
- Pure functions behind every number printed on a figure (covariance ellipses,
  correlation, autocorrelation, coverage, binned percentiles, circular variance,
  GNSS rejection counts). Each has a test in tests/test_plot_tests_helpers.py.
- ``save``: checks the printed size, the 8 pt minimum and text overlaps, then writes
  the PDF and PNG with ``paper_style.save`` plus a 600 dpi PNG and greyscale and
  deuteranopia previews for the colour checks.

Scripts run from the repository root with PYTHONPATH=src.
"""

import json
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib.text import Text
from pymap3d import ecef2geodetic
from scipy.stats import chi2, norm

from plotting import paper_style

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = paper_style.FIGURE_DIR.parent / "plot_tests"  # gitignored
CHECKS_DIR = OUTPUT_DIR / "checks"
ITERATIONS_DIR = OUTPUT_DIR / "iterations"
CHAINS_DIR = REPO / "Bermuda_Data" / "Output" / "MCMC_Chains"

# The reported solution (FIG-1) and the run behind every "final" figure: common
# 30-min delta-c(t) windows plus a per-DOG offset, AR(1) noise, one dt per DOG.
FINAL_SOLUTION = "MCMC posterior mean"
FINAL_RUN = "ar1_common30"
# R3 #7: delta-c is an empirical correction, not an oceanographic parameter.
DC_NAME = "ESV correction"
DC_SYMBOL = "δc"
DT_SYMBOL = "δt"
LEVER_LABELS = ("$l_x$", "$l_y$", "$l_z$")  # body frame
MINUS = "−"

# Okabe-Ito colours for signed quantities (correlation signs).
POSITIVE_COLOR = "#D55E00"
NEGATIVE_COLOR = "#0072B2"

ALLOWED_WIDTHS_MM = (paper_style.SINGLE_COLUMN_MM, paper_style.DOUBLE_COLUMN_MM)
# SpringerOpen production pages cap figure plus legend at 225 mm (Springer: 234 mm).
MAX_HEIGHT_MM = 225.0
MIN_FONT_PT = 8.0
CHECK_DPI = 600
# Exit code of a script whose input does not exist yet (make_plot_tests: SKIPPED).
SKIP_EXIT_CODE = 3

# Machado et al. (2009) deuteranopia simulation, severity 1, linear RGB.
DEUTERANOPIA = np.array(
    [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]
)


def signed(value, fmt="+.2f"):
    """Number formatted with a Unicode minus (e.g. "−0.47")."""
    return format(value, fmt).replace("-", MINUS)


# --- data loaders --------------------------------------------------------------


def require(*paths):
    """Exit with SKIP_EXIT_CODE, naming the missing inputs, unless all paths exist."""
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        print("BLOCKED: missing " + ", ".join(missing))
        raise SystemExit(SKIP_EXIT_CODE)


def pooled_run(run):
    """All saved draws of an MCMC run, pooled over its chains (see MCMC_diagnostics)."""
    from Inversion_Workflow.MCMC_Functions.MCMC_diagnostics import pooled_run as pool

    return pool(run)


def run_config(pooled):
    """The configuration a run was sampled with (includes the dc window edges)."""
    return json.loads(str(pooled["config_json"]))


def run_model_and_problem(pooled):
    """Sampler model and fixed pairs of a run, rebuilt from its saved configuration.

    Warns when the ESV table on disk differs from the one the chains used, since
    the residuals would then not be those the chains saw.
    """
    from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms

    config = run_config(pooled)
    model, problem = ms.setup_bermuda(config)
    used = json.loads(str(np.load(_first_chain(pooled["run_name"]))["metadata_json"]))
    used_sha = used["input_files"]["esv_table"]["sha256"]
    disk_sha = problem["input_files"]["esv_table"]["sha256"]
    if used_sha != disk_sha:
        print(
            f"WARNING: {pooled['run_name']} used ESV table {used_sha[:8]}, "
            f"disk has {disk_sha[:8]}"
        )
    return model, problem


def _first_chain(run):
    return sorted((CHAINS_DIR / run).glob("chain_*.npz"))[0]


def final_residuals(pooled):
    """Range residuals (cm, 1515 m/s) of a run at its posterior mean, per DOG.

    Returns {dog: dict(gps_time_s, residual_cm, transducer_ecef_m, dog_ecef_m,
    slant_range_m, elevation_deg)} in time order, plus the posterior-mean theta.
    """
    from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms

    model, problem = run_model_and_problem(pooled)
    theta = pooled["theta_samples"].mean(axis=0)
    residuals = ms.residuals_cm(theta, model, problem)
    params = ms.unpack(theta, model)
    trans = (problem["R_stacked"] @ params["lever"]).reshape(-1, 3) + problem["d"]
    gps = np.load(REPO / "Bermuda_Data" / run_config(pooled)["gps_file"])["GPS_data"]
    out = {}
    for j, dog in enumerate(model["config"]["dogs"]):
        sel = problem["pair_slices"][j]
        index = problem["pair_gps_index"][sel]
        dog_ecef = problem["CDOG_reference"] + params["CDOG_aug"][j]
        out[dog] = {
            "gps_time_s": gps[index],
            "residual_cm": residuals[sel],
            "transducer_ecef_m": trans[index],
            "dog_ecef_m": dog_ecef,
            "slant_range_m": np.linalg.norm(trans[index] - dog_ecef, axis=1),
            "elevation_deg": elevation_deg(dog_ecef, trans[index]),
        }
    return out, theta


def enu_rotation_at(ecef_m):
    """Rows E, N, U (3, 3) of the local frame at an ECEF point (WGS84)."""
    from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import enu_rotation

    lat, lon, _ = ecef2geodetic(*ecef_m)
    return enu_rotation(lat, lon)


def elevation_deg(origin_ecef_m, targets_ecef_m):
    """Elevation (deg) of each target above the local horizon at the origin."""
    up = enu_rotation_at(origin_ecef_m)[2]
    vectors = np.atleast_2d(targets_ecef_m) - origin_ecef_m
    return np.rad2deg(np.arcsin(vectors @ up / np.linalg.norm(vectors, axis=1)))


def dog_enu_cm(pooled, dog):
    """Draws of one DOG in local ENU at its posterior mean, minus that mean (n, 3) cm.

    Returns the draws and the posterior-mean ECEF position (m).
    """
    j = run_config(pooled)["dogs"].index(dog)
    offsets = pooled["CDOG_aug"][:, j, :]
    mean_ecef = pooled["CDOG_reference"] + offsets.mean(axis=0)
    rotation = enu_rotation_at(mean_ecef)
    return (offsets - offsets.mean(axis=0)) @ rotation.T * 100.0, mean_ecef


# --- pure helpers behind printed numbers ------------------------------------------


def chi2_radius_squared(probability, dof=2):
    """Squared Mahalanobis radius that holds ``probability`` of a Gaussian."""
    return chi2.ppf(probability, dof)


def ellipse_points(mean, cov, probability, n=200):
    """Boundary (n, 2) of the 2-D Gaussian region holding ``probability``."""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"cov must be (2, 2); got {cov.shape}")
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    scale = np.sqrt(chi2_radius_squared(probability) * values)
    return (vectors @ (scale[:, None] * circle)).T + np.asarray(mean, dtype=float)


def correlation(x, y):
    """Pearson correlation coefficient of two equal-length samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be equal 1-D arrays; got {x.shape}, {y.shape}")
    return float(np.corrcoef(x, y)[0, 1])


def autocorrelation(x, max_lag):
    """Sample autocorrelation of a 1-D series for lags 0..max_lag (biased, FFT)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or max_lag >= x.size:
        raise ValueError(f"need a 1-D series longer than max_lag={max_lag}")
    x = x - x.mean()
    n = x.size
    spectrum = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1] / n
    return acov / acov[0]


def coverage_fraction(errors, sds, probability):
    """Fraction of |error| / sd inside the central Gaussian ``probability`` interval."""
    errors, sds = np.asarray(errors, dtype=float), np.asarray(sds, dtype=float)
    if errors.shape != sds.shape:
        raise ValueError(f"errors {errors.shape} and sds {sds.shape} differ in shape")
    if np.any(sds <= 0):
        raise ValueError("standard deviations must be positive")
    z = norm.ppf(0.5 + probability / 2.0)
    return float(np.mean(np.abs(errors / sds) <= z))


def binned_percentiles(x, y, edges, percentiles=(16.0, 50.0, 84.0), min_count=1):
    """Percentiles of y in bins of x; NaN where a bin has fewer than min_count points.

    Returns bin centres (n_bins,), percentiles (n_bins, len(percentiles)), counts.
    """
    x, y, edges = (np.asarray(a, dtype=float) for a in (x, y, edges))
    index = np.digitize(x, edges) - 1
    n_bins = edges.size - 1
    values = np.full((n_bins, len(percentiles)), np.nan)
    counts = np.bincount(index[(index >= 0) & (index < n_bins)], minlength=n_bins)
    for k in range(n_bins):
        if counts[k] >= min_count:
            values[k] = np.percentile(y[index == k], percentiles)
    return 0.5 * (edges[1:] + edges[:-1]), values, counts


def circular_variance(residuals_s, period_s=1.0):
    """1 - |mean phasor| of residuals wrapped on the pulse period (0: aligned)."""
    phase = 2.0 * np.pi * np.asarray(residuals_s, dtype=float) / period_s
    return float(1.0 - np.abs(np.mean(np.exp(1j * phase))))


def rejection_counts(reject_masks):
    """Rejected epochs per receiver and combined under "any receiver fails".

    reject_masks is (n_receivers, n_epochs) boolean on a common epoch grid.
    """
    masks = np.asarray(reject_masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError(f"reject_masks must be 2-D; got {masks.shape}")
    return masks.sum(axis=1), int(masks.any(axis=0).sum())


# --- figure checks and saving -------------------------------------------------------


def _drawn_texts(fig):
    """Visible, non-empty Text artists that are actually drawn (ticks in view)."""
    tick_labels, drawn = set(), []
    for ax in fig.axes:
        axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
        for axis in axes:
            low, high = sorted(axis.get_view_interval())
            tolerance = 1e-9 * max(1.0, abs(high - low))
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for label in (tick.label1, tick.label2):
                    tick_labels.add(id(label))
                    inside = low - tolerance <= tick.get_loc() <= high + tolerance
                    if inside and label.get_visible() and label.get_text():
                        drawn.append(label)
    for text in fig.findobj(Text):
        if id(text) in tick_labels or not text.get_visible() or not text.get_text():
            continue
        if text.get_alpha() == 0:
            continue
        drawn.append(text)
    return drawn


def figure_problems(fig, allowed_widths_mm=ALLOWED_WIDTHS_MM):
    """List of style-contract violations: size, fonts < 8 pt, overlapping text."""
    problems = []
    width_mm, height_mm = fig.get_size_inches() * paper_style.MM_PER_INCH
    if not any(abs(width_mm - w) < 0.05 for w in allowed_widths_mm):
        problems.append(f"width {width_mm:.1f} mm not in {allowed_widths_mm}")
    if height_mm > MAX_HEIGHT_MM + 0.05:
        problems.append(f"height {height_mm:.1f} mm > {MAX_HEIGHT_MM} mm")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    texts = _drawn_texts(fig)
    boxes = []
    for text in texts:
        if text.get_fontsize() < MIN_FONT_PT - 1e-6:
            problems.append(f"{text.get_text()!r} is {text.get_fontsize():.1f} pt")
        box = text.get_window_extent(renderer)
        if (
            box.x0 < figure_box.x0 - 1
            or box.y0 < figure_box.y0 - 1
            or box.x1 > figure_box.x1 + 1
            or box.y1 > figure_box.y1 + 1
        ):
            problems.append(f"{text.get_text()!r} falls outside the figure")
        boxes.append(box.expanded(1.0, 1.0).padded(-0.5))
    for a in range(len(texts)):
        for b in range(a + 1, len(texts)):
            if boxes[a].overlaps(boxes[b]):
                problems.append(
                    f"text overlap: {texts[a].get_text()!r} / {texts[b].get_text()!r}"
                )
    return problems


def colour_previews(png_path, out_dir=CHECKS_DIR):
    """Write greyscale and deuteranopia versions of a PNG; return their paths."""
    rgb = mpimg.imread(png_path)[..., :3].astype(float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    grey = np.where(
        luminance <= 0.0031308,
        12.92 * luminance,
        1.055 * luminance ** (1 / 2.4) - 0.055,
    )
    deutan = np.clip(linear @ DEUTERANOPIA.T, 0.0, 1.0)
    deutan = np.where(
        deutan <= 0.0031308, 12.92 * deutan, 1.055 * deutan ** (1 / 2.4) - 0.055
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(png_path).stem
    grey_path, deutan_path = (
        out_dir / f"{stem}_grey.png",
        out_dir / f"{stem}_deutan.png",
    )
    mpimg.imsave(grey_path, np.clip(grey, 0, 1), cmap="gray", vmin=0, vmax=1)
    mpimg.imsave(deutan_path, np.clip(deutan, 0, 1))
    return grey_path, deutan_path


def save(fig, name, allowed_widths_mm=ALLOWED_WIDTHS_MM, strict=True):
    """Check the figure, then save PDF + PNG (outputs) and the check images.

    With strict=True a violation of the style contract raises; otherwise it prints.
    """
    problems = figure_problems(fig, allowed_widths_mm)
    if problems:
        message = f"{name}: " + "; ".join(problems)
        if strict:
            raise AssertionError(message)
        print("CHECK", message)
    pdf = paper_style.save(fig, name, OUTPUT_DIR)
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHECKS_DIR / f"{name}_{CHECK_DPI}dpi.png", dpi=CHECK_DPI)
    colour_previews(OUTPUT_DIR / f"{name}.png")
    print(f"saved {pdf}")
    return pdf


def keep_iteration(name, label):
    """Copy the current PNG of a figure to iterations/<name>_<label>.png."""
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = ITERATIONS_DIR / f"{name}_{label}.png"
    shutil.copy(OUTPUT_DIR / f"{name}.png", target)
    return target
