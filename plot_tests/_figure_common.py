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

# Imported here, before any script calls paper_style.apply(): MCMC_sampler turns
# LaTeX text on at import, which would otherwise override the paper style.
from Inversion_Workflow.MCMC_Functions import MCMC_diagnostics, MCMC_sampler
from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import enu_rotation
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

# One accent colour for the one thing the eye should find on a figure (the final
# solution, the chosen offset, the antenna-4 steps, rejected epochs); everything
# that is context (alternatives, raw points, reference lines) is grey.
ACCENT = "#D55E00"
CONTEXT = "0.6"
RAW_POINTS = "0.75"

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
    return MCMC_diagnostics.pooled_run(run)


def run_config(pooled):
    """The configuration a run was sampled with (includes the dc window edges)."""
    return json.loads(str(pooled["config_json"]))


def run_model_and_problem(pooled):
    """Sampler model and fixed pairs of a run, rebuilt from its saved configuration.

    Warns when the ESV table or GNSS file on disk differs from the one the chains used:
    the residuals would then not be those the chains saw.
    """
    config = run_config(pooled)
    model, problem = MCMC_sampler.setup_bermuda(config)
    used = json.loads(str(np.load(_first_chain(pooled["run_name"]))["metadata_json"]))
    for key, name in (("esv_table", "ESV table"), ("gps_file", "GNSS file")):
        used_sha = used["input_files"][key]["sha256"]
        disk_sha = problem["input_files"][key]["sha256"]
        if used_sha != disk_sha:
            print(
                f"WARNING: {pooled['run_name']} used {name} {used_sha[:8]}, "
                f"disk has {disk_sha[:8]}"
            )
    return model, problem


def _first_chain(run):
    return sorted((CHAINS_DIR / run).glob("chain_*.npz"))[0]


def best_sample(pooled):
    """The saved draw with the highest posterior density (the run's MAP estimate)."""
    return pooled["theta_samples"][int(np.argmax(pooled["logpost"]))]


def final_residuals(pooled, theta=None):
    """Range residuals (cm, 1515 m/s) of a run at one parameter vector, per DOG.

    theta defaults to the posterior mean. Returns {dog: dict(gps_time_s,
    residual_cm, transducer_ecef_m, dog_ecef_m, slant_range_m, elevation_deg)} in
    time order, plus the theta used.
    """
    model, problem = run_model_and_problem(pooled)
    if theta is None:
        theta = pooled["theta_samples"].mean(axis=0)
    residuals = MCMC_sampler.residuals_cm(theta, model, problem)
    params = MCMC_sampler.unpack(theta, model)
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


def despine(*axes):
    """Left and bottom spines only (no frame) on each of the axes."""
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)


def keep_iteration(name, label):
    """Copy the current PNG of a figure to iterations/<name>_<label>.png."""
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = ITERATIONS_DIR / f"{name}_{label}.png"
    shutil.copy(OUTPUT_DIR / f"{name}.png", target)
    return target


# --- helpers added for FX-7 and FIG-9 (MCMC) ------------------------------------------


def linear_gaussian_prior(coefficients, prior_center, prior_sd, constant=0.0):
    """Prior mean and sd of q = coefficients @ theta + constant.

    The sampler's priors are independent Gaussians (``prior_center``, ``prior_sd``
    per parameter). Parameters with a zero coefficient are ignored, so their prior
    may be NaN (e.g. phi, which has a bounded uniform prior).
    """
    coefficients, center, sd = (
        np.asarray(a, dtype=float) for a in (coefficients, prior_center, prior_sd)
    )
    if not coefficients.shape == center.shape == sd.shape or coefficients.ndim != 1:
        raise ValueError(
            f"need equal 1-D arrays; got {coefficients.shape}, {center.shape}, "
            f"{sd.shape}"
        )
    used = coefficients != 0.0
    if not np.all(np.isfinite(center[used]) & np.isfinite(sd[used])):
        raise ValueError("q loads on a parameter without a Gaussian prior")
    mean = coefficients[used] @ center[used] + constant
    return float(mean), float(np.sqrt(np.sum((coefficients[used] * sd[used]) ** 2)))


def density_levels(density, probabilities):
    """Levels of a gridded density whose super-level sets hold each probability.

    ``density`` holds non-negative values on a uniform grid (e.g. a 2-D histogram).
    The region {density >= level} holds at least the given fraction of the total,
    so the 68 % level is higher than the 95 % level.
    """
    values = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    if values.size == 0 or values[-1] < 0 or values.sum() <= 0:
        raise ValueError("density must be non-negative with a positive total")
    cumulative = np.cumsum(values) / values.sum()
    levels = []
    for probability in probabilities:
        if not 0.0 < probability < 1.0:
            raise ValueError(f"probability must be in (0, 1); got {probability}")
        levels.append(float(values[np.searchsorted(cumulative, probability)]))
    return levels


# --- helpers added for FIG-3 and FX-4 (clock offset, travel times) ------------------


def wrap_to_period(values_s, period_s=1.0):
    """Values wrapped into [-period/2, period/2): the fractional part of Eq. A22."""
    values = np.asarray(values_s, dtype=float)
    return (values + 0.5 * period_s) % period_s - 0.5 * period_s


def alignment_scan(dt_values_s, CDOG_data, GPS_data, travel_times_s, gate_s=0.5):
    """Circular variance, RMSE (s) and pair count of the pairing at each trial δt.

    CDOG_data (M, 2) holds integer and fractional DOG-clock seconds, GPS_data (N,)
    the emission epochs (GPS s) and travel_times_s (N,) the modelled travel times
    (s). Each δt re-derives the pairing with ``two_pointer_index`` (Eq. A20); the
    0.5 s gate is the one of the offset search (``find_int_offset``).
    """
    from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index

    CDOG_data = np.ascontiguousarray(CDOG_data, dtype=float)
    GPS_data = np.ascontiguousarray(GPS_data, dtype=float)
    travel_times_s = np.ascontiguousarray(travel_times_s, dtype=float)
    if CDOG_data.ndim != 2 or CDOG_data.shape[1] != 2:
        raise ValueError(f"CDOG_data must be (M, 2); got {CDOG_data.shape}")
    if GPS_data.shape != travel_times_s.shape or GPS_data.ndim != 1:
        raise ValueError(
            f"GPS_data {GPS_data.shape} and travel_times_s {travel_times_s.shape} "
            "must be equal 1-D arrays"
        )
    unused_coordinates = np.zeros((GPS_data.size, 3))
    unused_esv = np.zeros(GPS_data.size)
    dt_values_s = np.asarray(dt_values_s, dtype=float)
    cv = np.empty(dt_values_s.size)
    rmse_s = np.empty(dt_values_s.size)
    n_pairs = np.empty(dt_values_s.size, dtype=int)
    for i, dt in enumerate(dt_values_s):
        observed, _, modelled = two_pointer_index(
            dt,
            gate_s,
            CDOG_data,
            GPS_data,
            travel_times_s,
            unused_coordinates,
            unused_esv,
        )[1:4]
        residuals = observed - modelled
        cv[i] = circular_variance(residuals)
        rmse_s[i] = np.sqrt(np.mean(residuals**2))
        n_pairs[i] = residuals.size
    return cv, rmse_s, n_pairs


def line_fit(x, y):
    """Least-squares slope and intercept of y on x over the finite pairs; and n."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be equal 1-D arrays; got {x.shape}, {y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        raise ValueError("need at least two finite (x, y) pairs")
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return float(slope), float(intercept), int(keep.sum())


def vertical_travel_time_s(dog_ecef_m, transducer_ecef_m, sound_speed_mps):
    """Travel time (s) straight down from each transducer epoch to the DOG's depth.

    Height difference (WGS84 ellipsoid) divided by ``sound_speed_mps``; no
    one-way travel time can be shorter when that is the fastest effective speed.
    """
    transducer = np.atleast_2d(np.asarray(transducer_ecef_m, dtype=float))
    if transducer.shape[1] != 3:
        raise ValueError(f"transducer_ecef_m must be (n, 3); got {transducer.shape}")
    dog_height = ecef2geodetic(*np.asarray(dog_ecef_m, dtype=float))[2]
    heights = ecef2geodetic(transducer[:, 0], transducer[:, 1], transducer[:, 2])[2]
    return (np.asarray(heights) - dog_height) / sound_speed_mps


# --- helpers added for FX-5, FX-6 and FIG-4 (residuals, delta-c) -------------------


def window_steps(draws):
    """Posterior mean and sd of the step between consecutive windows.

    draws is (n_draws, n_windows), e.g. the common ESV correction per window; step
    k is window k + 1 minus window k, so the result has n_windows - 1 entries.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] < 2:
        raise ValueError(f"draws must be (n_draws, >= 2 windows); got {draws.shape}")
    steps = np.diff(draws, axis=1)
    return steps.mean(axis=0), steps.std(axis=0)


def level_shifts(times_s, values, half_window_s, min_shift, min_count=20):
    """Abrupt level changes in a time series (e.g. antenna height jumps).

    At each sample the shift is median(values in [t, t + w)) minus median(values in
    [t - w, t)), w = half_window_s; NaN where either side has < min_count samples.
    Samples with |shift| > min_shift are grouped into events (a gap > w starts a
    new event). Returns the per-sample shifts and, per event, the time and shift
    of its largest |shift|. times_s must be strictly increasing.
    """
    times_s, values = np.asarray(times_s, dtype=float), np.asarray(values, float)
    if times_s.ndim != 1 or times_s.shape != values.shape:
        raise ValueError(
            f"need equal 1-D times and values; got {times_s.shape}, {values.shape}"
        )
    if np.any(np.diff(times_s) <= 0):
        raise ValueError("times_s must be strictly increasing")
    start = np.searchsorted(times_s, times_s - half_window_s, side="left")
    stop = np.searchsorted(times_s, times_s + half_window_s, side="left")
    shifts = np.full(times_s.size, np.nan)
    for i in range(times_s.size):
        if i - start[i] >= min_count and stop[i] - i >= min_count:
            after, before = values[i : stop[i]], values[start[i] : i]
            shifts[i] = np.median(after) - np.median(before)
    large = np.flatnonzero(np.abs(np.nan_to_num(shifts)) > min_shift)
    new_event = np.flatnonzero(np.diff(times_s[large]) > half_window_s) + 1
    groups = np.split(large, new_event)
    peaks = []
    for group in groups:
        if group.size:
            # A clean step gives a run of equal maxima; its middle is the step time.
            size = np.abs(shifts[group])
            top = group[size == size.max()]
            peaks.append(top[top.size // 2])
    peaks = np.asarray(peaks, dtype=int)
    return shifts, times_s[peaks], shifts[peaks]


def antenna4_height_anomaly(gps_coordinates, gps_grid, up):
    """Height of antenna 4 above where the rigid grid fitted to antennas 1-3 puts it.

    gps_coordinates is (n, 4, 3) ECEF m, gps_grid (4, 3) body-frame m (the run's
    gps1_to_others), up (3,) the local unit up vector. Antennas 1-3 fix the
    vessel's position and attitude at each epoch, so ship motion cancels and what
    remains (m) is antenna 4's own height error relative to the other three.
    """
    from geometry.rigid_body import findRotationAndDisplacement

    gps_coordinates = np.asarray(gps_coordinates, dtype=float)
    gps_grid = np.asarray(gps_grid, dtype=float)
    if gps_grid.shape != (4, 3) or gps_coordinates.shape[1:] != (4, 3):
        raise ValueError(
            f"need (n, 4, 3) receivers and a (4, 3) grid; got "
            f"{gps_coordinates.shape}, {gps_grid.shape}"
        )
    anomaly = np.empty(gps_coordinates.shape[0])
    for i, epoch in enumerate(gps_coordinates):
        rotation, shift = findRotationAndDisplacement(gps_grid[:3].T, epoch[:3].T)
        anomaly[i] = (epoch[3] - (rotation @ gps_grid[3] + shift)) @ up
    return anomaly


# Antenna-4 height steps shown in FX-5 and FIG-4: |median of 3 min after minus
# 3 min before| > 0.15 m, with >= 20 epochs on each side (none across data gaps).
ANTENNA4_STEP_HALF_WINDOW_S = 180.0
ANTENNA4_STEP_MIN_M = 0.15


def antenna4_steps(config):
    """Antenna-4 height anomaly of a run's GNSS data and its steps.

    Returns GPS_data (s), the anomaly (m), and the step times (GPS_data s) and
    sizes (m) found with the ANTENNA4_STEP_* settings.
    """
    inputs = MCMC_sampler.load_bermuda_inputs(config)
    coords = inputs["GPS_Coordinates"]
    up = enu_rotation_at(coords[0, 0])[2]
    anomaly = antenna4_height_anomaly(coords, config["gps1_to_others"], up)
    _, step_s, step_m = level_shifts(
        inputs["GPS_data"],
        anomaly,
        ANTENNA4_STEP_HALF_WINDOW_S,
        ANTENNA4_STEP_MIN_M,
    )
    return inputs["GPS_data"], anomaly, step_s, step_m


# --- helpers added for FIG-1, FIG-2 and FX-A3 (flowchart, map, GNSS) ---------------

# GNSS quality control as run by Inversion_Workflow/Bermuda/Initialize_Bermuda_Data.py
# (initialize_bermuda and its __main__), which wrote Processed_GPS_Receivers_DOG_*.npz.
# Epoch time there is (MJD - 59015) days + seconds of day; GPS_data subtracts 70000 s.
GNSS_DAY0_MJD = 59015
GNSS_CLOCK_ADJUSTMENT_S = 70000.0
GNSS_WINDOW_H = (25.0, 39.0)  # epochs kept, hours since MJD 59015 0 h
GNSS_HEIGHT_LIMITS_M = (-38.0, -35.0)  # hard cut-offs, WGS84 ellipsoidal height
GNSS_MEDIAN_WINDOW = 5000  # epochs in the running median and MAD
GNSS_MAD_FACTOR = 2.0


def running_median_mad(values, window):
    """Running median and median absolute deviation over ``window`` samples.

    Same definition as Initialize_Bermuda_Data.running_median/running_abs_dev:
    sample i uses values[i - window // 2 : i + window // 2 + 1], cut at the ends,
    and the MAD is taken about that window's own median.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or window < 1:
        raise ValueError(f"need a 1-D series and window >= 1; got {values.shape}")
    n, half = values.size, window // 2
    median, mad = np.empty(n), np.empty(n)
    first_full, last_full = half, n - half - 1
    edges = [i for i in range(n) if i < first_full or i > last_full]
    for i in edges:
        part = values[max(0, i - half) : i + half + 1]
        median[i] = np.median(part)
        mad[i] = np.median(np.abs(part - median[i]))
    if last_full >= first_full:
        windows = sliding_window_view(values, 2 * half + 1)
        for start in range(0, windows.shape[0], 500):  # chunks bound the memory
            part = windows[start : start + 500]
            centre = np.median(part, axis=1)
            median[first_full + start : first_full + start + part.shape[0]] = centre
            mad[first_full + start : first_full + start + part.shape[0]] = np.median(
                np.abs(part - centre[:, None]), axis=1
            )
    return median, mad


def gnss_rejection_masks(
    heights_m,
    limits_m=GNSS_HEIGHT_LIMITS_M,
    window=GNSS_MEDIAN_WINDOW,
    mad_factor=GNSS_MAD_FACTOR,
):
    """Two-stage GNSS height filter of initialize_bermuda, per receiver.

    heights_m is (n_receivers, n_epochs). Stage 1 drops an epoch if any receiver is
    outside the open interval ``limits_m``. Stage 2, on the epochs left, flags a
    receiver whose height is outside running median ± mad_factor·MAD. An epoch is
    kept only if no receiver fails either stage. Returns dict of (n_receivers,
    n_epochs) arrays hard_fail, band_fail (False where stage 1 dropped the epoch),
    median_m and mad_m (NaN there), and kept (n_epochs,).
    """
    heights = np.asarray(heights_m, dtype=float)
    if heights.ndim != 2:
        raise ValueError(
            f"heights_m must be (n_receivers, n_epochs); got {heights.shape}"
        )
    hard_fail = ~((heights > limits_m[0]) & (heights < limits_m[1]))
    stage2 = ~hard_fail.any(axis=0)
    band_fail = np.zeros_like(hard_fail)
    median, mad = np.full(heights.shape, np.nan), np.full(heights.shape, np.nan)
    for r, height in enumerate(heights):
        median[r, stage2], mad[r, stage2] = running_median_mad(height[stage2], window)
        low = median[r, stage2] - mad_factor * mad[r, stage2]
        high = median[r, stage2] + mad_factor * mad[r, stage2]
        band_fail[r, stage2] = ~((height[stage2] >= low) & (height[stage2] <= high))
    return {
        "hard_fail": hard_fail,
        "band_fail": band_fail,
        "median_m": median,
        "mad_m": mad,
        "kept": stage2 & ~band_fail.any(axis=0),
    }


def load_gnss_epochs(paths, window_h=GNSS_WINDOW_H):
    """Epochs common to all receivers inside window_h, as in initialize_bermuda.

    Returns dict gps_s (n,) on the GPS_data scale of Processed_GPS_Receivers_DOG_*,
    ecef_m (n_receivers, n, 3) and height_m (n_receivers, n, WGS84 ellipsoidal).
    """
    from scipy.io import loadmat

    receivers = []
    for path in paths:
        data = loadmat(path)
        t = (data["days"].ravel() - GNSS_DAY0_MJD) * 86400.0 + data["times"].ravel()
        inside = (t / 3600.0 >= window_h[0]) & (t / 3600.0 <= window_h[1])
        receiver = {key: data[key].ravel()[inside] for key in ("x", "y", "z", "elev")}
        receiver["t"] = t[inside]
        receivers.append(receiver)
    common = receivers[0]["t"]
    for receiver in receivers[1:]:
        common = np.intersect1d(common, receiver["t"])
    ecef, height = [], []
    for receiver in receivers:
        index = np.flatnonzero(np.isin(receiver["t"], common))
        if not np.array_equal(receiver["t"][index], common):
            raise ValueError("receiver epochs are not unique and time-ordered")
        ecef.append(np.stack([receiver[key][index] for key in "xyz"], axis=1))
        height.append(receiver["elev"][index])
    return {
        "gps_s": common - GNSS_CLOCK_ADJUSTMENT_S,
        "ecef_m": np.array(ecef),
        "height_m": np.array(height),
    }


def read_esri_ascii(path):
    """Grid (nrows, ncols; NaN for no data, first row north) and its cell-edge extent.

    Extent is (west, east, south, north) in degrees from the xllcorner/yllcorner
    header of an ESRI ASCII grid such as the GEBCO exports (the GEBCO TID grids
    come without the NODATA_value line).
    """
    header = {}
    with open(path) as handle:
        for line in handle:  # 5 or 6 header lines (NODATA_value is optional)
            key, *value = line.split()
            if not key[0].isalpha():
                break
            header[key.lower()] = float(value[0])
    grid = np.loadtxt(path, skiprows=len(header), ndmin=2)
    shape = (int(header["nrows"]), int(header["ncols"]))
    if grid.shape != shape:
        raise ValueError(f"{path}: header says {shape}, data are {grid.shape}")
    grid[grid == header.get("nodata_value", np.nan)] = np.nan
    west, south, cell = header["xllcorner"], header["yllcorner"], header["cellsize"]
    extent = (west, west + shape[1] * cell, south, south + shape[0] * cell)
    return grid, extent


# --- helpers added for FIG-5, FIG-6 and FIG-7 (models, synthetics) -----------------


def spread_over_sd(means, sds):
    """Range of the means (max - min) divided by the median sd (FIG-5)."""
    means, sds = np.asarray(means, dtype=float), np.asarray(sds, dtype=float)
    if means.shape != sds.shape or means.size < 2:
        raise ValueError(f"need >= 2 equal-length means and sds; got {means.shape}")
    return float((means.max() - means.min()) / np.median(sds))


def gapped_autocorrelation(t_s, x, max_lag):
    """Autocorrelation at lags 0..max_lag (s) of a series on whole seconds with gaps.

    t_s are increasing integer seconds (missing seconds are gaps, not zeros). Each lag
    averages the products of all pairs exactly that far apart, about the mean of x.
    """
    t_s, x = np.asarray(t_s, dtype=float), np.asarray(x, dtype=float)
    if t_s.shape != x.shape or t_s.ndim != 1:
        raise ValueError(f"t_s, x must be equal 1-D arrays; got {t_s.shape}, {x.shape}")
    index = np.round(t_s - t_s[0]).astype(int)
    if np.any(np.diff(index) <= 0) or not np.allclose(index, t_s - t_s[0]):
        raise ValueError("t_s must be strictly increasing whole seconds")
    if max_lag >= index[-1]:
        raise ValueError(f"max_lag={max_lag} s must be shorter than the series")
    n = index[-1] + 1
    values, mask = np.zeros(n), np.zeros(n)
    values[index], mask[index] = x - x.mean(), 1.0

    def lagged_sum(a, b):
        spectrum = np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n))
        return np.fft.irfft(spectrum)[: max_lag + 1]

    pairs = np.round(lagged_sum(mask, mask))
    with np.errstate(invalid="ignore", divide="ignore"):
        acov = np.where(pairs > 0, lagged_sum(values, values) / pairs, np.nan)
    return acov / acov[0]


def abs_normal_bin_probability(edges):
    """Probability that |Z| falls in each bin of ``edges``, Z ~ N(0, 1)."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("edges must be increasing and non-negative")
    return 2.0 * np.diff(norm.cdf(edges))


def horizontal_rms(east_errors, north_errors):
    """Rms horizontal error sqrt(mean(E² + N²)) of paired E and N errors."""
    east, north = np.asarray(east_errors, float), np.asarray(north_errors, float)
    if east.shape != north.shape or east.size == 0:
        raise ValueError(
            f"need equal, non-empty E and N errors; got {east.shape}, {north.shape}"
        )
    return float(np.sqrt(np.mean(east**2 + north**2)))


# --- helpers added for FX-8, FIG-8 and appendix B (vessel, levers, synthetics) -----


def receivers_in_body_frame(gps_coordinates, gps_grid):
    """Each epoch's receivers mapped onto the body-frame grid by a rigid-body fit.

    gps_coordinates is (n, 4, 3) ECEF m, gps_grid (4, 3) body-frame m (receiver 1 at
    the origin). Returns (n, 4, 3) body-frame m; the scatter of a receiver about its
    grid point is the epoch-wise misfit of the four-antenna rigid body.
    """
    from geometry.rigid_body import findRotationAndDisplacement

    gps_coordinates = np.asarray(gps_coordinates, dtype=float)
    gps_grid = np.asarray(gps_grid, dtype=float)
    if gps_coordinates.ndim != 3 or gps_coordinates.shape[1:] != gps_grid.shape:
        raise ValueError(
            f"need (n, {gps_grid.shape[0]}, 3) receivers for a {gps_grid.shape} "
            f"grid; got {gps_coordinates.shape}"
        )
    body = np.empty_like(gps_coordinates)
    for i, epoch in enumerate(gps_coordinates):
        rotation, shift = findRotationAndDisplacement(epoch.T, gps_grid.T)
        body[i] = epoch @ rotation.T + shift
    return body


def central_interval(samples, probability=0.95):
    """Mean and central ``probability`` interval (low, high) of samples along axis 0."""
    samples = np.asarray(samples, dtype=float)
    tail = 50.0 * (1.0 - probability)
    low, high = np.percentile(samples, [tail, 100.0 - tail], axis=0)
    return samples.mean(axis=0), low, high


def mean_with_ci(values, probability=0.95):
    """Mean along axis 0 and the half-width of its normal-theory confidence interval.

    NaN rows are not allowed: drop failed realisations first and report how many.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("values contain NaN; drop failed realisations first")
    if values.shape[0] < 2:
        raise ValueError(f"need at least 2 values; got {values.shape[0]}")
    z = norm.ppf(0.5 + probability / 2.0)
    half_width = z * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return values.mean(axis=0), half_width


def b36_range_sd_cm(
    sigma_d_m, sigma_t_s, gain=1.0, sound_speed_mps=paper_style.RANGE_SOUND_SPEED_MPS
):
    """Eq. B36 ranging sd c·σε (cm) = c·sqrt(σt² + (g σd / c)²).

    gain g maps receiver noise σd to transducer range noise (submitted B36: g = 1;
    K8: g ≈ 1.6 for the Bermuda antenna geometry).
    """
    sigma_d_m, sigma_t_s = np.asarray(sigma_d_m), np.asarray(sigma_t_s)
    sigma_eps_s = np.sqrt(sigma_t_s**2 + (gain * sigma_d_m / sound_speed_mps) ** 2)
    return sigma_eps_s * sound_speed_mps * 100.0


# --- helpers added for the 3-D posterior figure -------------------------------------


# --- helper added for the 2026-11-21 figures (model sensitivity, repeatability) -----


def run_positions_cm(run, reference_ecef_m):
    """Posterior mean and covariance of each DOG position of a run, in local ENU.

    reference_ecef_m maps DOG number to an ECEF position (m); the result maps each
    DOG of the run to (mean (3,), covariance (3, 3)) of E, N, U in cm relative to
    that reference, in the local frame at the reference.
    """
    pooled = pooled_run(run)
    out = {}
    for j, dog in enumerate(run_config(pooled)["dogs"]):
        to_enu = enu_rotation_at(reference_ecef_m[dog])
        ecef = pooled["CDOG_reference"] + pooled["CDOG_aug"][:, j, :]
        enu_cm = (ecef - reference_ecef_m[dog]) @ to_enu.T * 100.0
        out[dog] = (enu_cm.mean(axis=0), np.cov(enu_cm.T))
    return out


# --- helper added for the density-coloured scatters (2026-11-24) ----------------------


def nearby_counts(x, y, x_radius, y_radius):
    """Number of points within an ellipse of the given radii about each point.

    The count includes the point itself, as in the density scatters of
    plotting/Analysis_Plots/PuertoRico_GPS_Plots.py. x and y are equal 1-D arrays.
    """
    from scipy.spatial import cKDTree

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be equal 1-D arrays; got {x.shape}, {y.shape}")
    if x_radius <= 0 or y_radius <= 0:
        raise ValueError(f"radii must be positive; got {x_radius}, {y_radius}")
    scaled = np.column_stack([x / x_radius, y / y_radius])
    return cKDTree(scaled).query_ball_point(scaled, r=1.0, return_length=True)
