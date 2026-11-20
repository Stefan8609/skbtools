from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    NOMINAL_SOUND_SPEED_MPS,
)
from geometry.ECEF_Geodetic import ECEF_Geodetic
from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from plotting.Analysis_Plots.metric_plots import (
    _coherence_obj_for_offset,
    _rmse_cm_for_offset,
)
from puertorico_data import gps_data_path, gps_output_path


clock_segment = "before"
simulated_annealing = True
save = True

derivative_gap = 5.0
preliminary_gate = 0.2
final_gate = 0.05

receiver_names = np.array(["PF", "SF", "PA", "SA"])
receiver_sets = (
    "PF-SF-PA-SA",
    "PF-SF-PA",
    "PF-SF-SA",
    "PF-PA-SA",
    "SF-PA-SA",
)


def nearest_indices(reference, values):
    """Return indices of the nearest values in a sorted reference array."""
    indices = np.searchsorted(reference, values)
    indices = np.clip(indices, 0, len(reference) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(reference[previous] - values) < np.abs(
        reference[indices] - values
    )
    return np.where(use_previous, previous, indices)


def rotation_matrices(coordinates, body_coordinates):
    """Calculate the body-to-ECEF rotation at each accepted epoch."""
    rotations = np.empty((len(coordinates), 3, 3))
    for index, positions in enumerate(coordinates):
        rotations[index] = findRotationAndDisplacement(
            body_coordinates.T,
            positions.T,
        )[0]
    return rotations


def local_axes(coordinates):
    """Return local east and north unit vectors at ECEF coordinates."""
    latitude, longitude, _ = ECEF_Geodetic(coordinates)
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)
    east = np.column_stack(
        (-np.sin(longitude), np.cos(longitude), np.zeros(len(longitude)))
    )
    north = np.column_stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    return east, north


def segmented_gradient(values, times, maximum_gap=derivative_gap):
    """Differentiate within contiguous time segments only."""
    gradient = np.full(len(values), np.nan)
    boundaries = np.flatnonzero(np.diff(times) > maximum_gap) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(times)]
    for start, end in zip(starts, ends, strict=False):
        if end - start >= 2:
            gradient[start:end] = np.gradient(values[start:end], times[start:end])
    return gradient


def circular_smooth(heading, times, window=61):
    """Smooth headings without averaging across temporal gaps."""
    smoothed = np.full(len(heading), np.nan)
    boundaries = np.flatnonzero(np.diff(times) > derivative_gap) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(times)]
    for start, end in zip(starts, ends, strict=False):
        count = end - start
        if count == 0:
            continue
        width = min(window, count if count % 2 else count - 1)
        if width < 3:
            smoothed[start:end] = heading[start:end]
            continue
        half = width // 2
        values = np.exp(1j * np.radians(heading[start:end]))
        padded = np.pad(values, half, mode="edge")
        mean_values = np.convolve(padded, np.ones(width) / width, mode="valid")
        smoothed[start:end] = np.mod(np.degrees(np.angle(mean_values)), 360.0)
    return smoothed


def identify_legs(heading, times):
    """Identify straight survey legs from smoothed heading turn rate."""
    smoothed = circular_smooth(heading, times)
    unwrapped = np.unwrap(np.radians(smoothed))
    turn_rate = np.degrees(segmented_gradient(unwrapped, times))
    turning = np.abs(turn_rate) > 0.5
    turning = np.convolve(turning.astype(int), np.ones(61, dtype=int), mode="same") > 0

    cores = []
    start = None
    for index in range(len(times) + 1):
        gap = index > 0 and index < len(times) and times[index] - times[index - 1] > 5
        unavailable = index == len(times) or turning[index] or gap
        if not unavailable and start is None:
            start = index
        if unavailable and start is not None:
            if index - start >= 100 and times[index - 1] - times[start] >= 120:
                cores.append((start, index))
            start = None

    if len(cores) == 0:
        return np.zeros(len(times), dtype=int), turn_rate

    boundaries = [
        0.5 * (times[first[1] - 1] + times[second[0]])
        for first, second in zip(cores[:-1], cores[1:], strict=False)
    ]
    return np.searchsorted(boundaries, times), turn_rate


def receiver_set_labels(used_receivers):
    """Convert receiver masks to the requested compact labels."""
    labels = np.array(
        ["-".join(receiver_names[mask]) for mask in used_receivers],
        dtype="U16",
    )
    unknown = sorted(set(labels) - set(receiver_sets))
    if unknown:
        raise ValueError(f"Unexpected receiver configurations: {unknown}")
    return labels


def load_diagnostic_data():
    """Load the saved solution and preserve its accepted observation indices."""
    method_tag = "_SimAnn" if simulated_annealing else ""
    file_tag = f"PuertoRico_{clock_segment}{method_tag}"
    inversion_path = gps_output_path("Plot_Data", f"{file_tag}_inversion.npz")
    inversion = np.load(inversion_path)
    processed = np.load(
        gps_data_path(
            f"GPS_Data/Processed_GPS_Receivers_DOG_1_{clock_segment}_clock_loss.npz"
        )
    )

    gps_indices = nearest_indices(processed["GPS_data"], inversion["GPS_data"])
    if (
        np.max(np.abs(processed["GPS_data"][gps_indices] - inversion["GPS_data"]))
        > 1e-6
    ):
        raise ValueError("Saved inversion epochs do not align with processed GNSS data")

    paired_gps_times = (
        inversion["GPS_clock"] - inversion["GPS_full"] + inversion["time_bias"]
    )
    observation_gps_indices = nearest_indices(inversion["GPS_data"], paired_gps_times)
    if (
        np.max(
            np.abs(inversion["GPS_data"][observation_gps_indices] - paired_gps_times)
        )
        > 1e-6
    ):
        raise ValueError("Accepted acoustic observations do not align with GNSS epochs")

    tag_times = inversion["CDOG_data"][:, 0] + inversion["CDOG_data"][:, 1]
    observation_tag_times = inversion["CDOG_clock"] + inversion["best_offset"]
    observation_tag_indices = nearest_indices(tag_times, observation_tag_times)
    if (
        np.max(np.abs(tag_times[observation_tag_indices] - observation_tag_times))
        > 1e-6
    ):
        raise ValueError("Accepted acoustic observations do not align with input tags")

    processed_indices = gps_indices[observation_gps_indices]
    coordinates = processed["GPS_Coordinates"][processed_indices]
    used_receivers = processed["used_receivers"][processed_indices]
    rotations = rotation_matrices(coordinates, processed["gps1_to_others"])
    transducer = inversion["transponder_coordinates"][observation_gps_indices]
    dog = inversion["inversion_result"][:3]
    times = inversion["GPS_data"][observation_gps_indices]
    time_hours = (times - times[0]) / 3600.0
    residual = (
        inversion["CDOG_full"] - inversion["GPS_full"]
    ) * NOMINAL_SOUND_SPEED_MPS
    slant_range = np.linalg.norm(transducer - dog, axis=1)
    radial_velocity = segmented_gradient(slant_range, times)

    east, north = local_axes(transducer)
    forward = rotations[:, :, 0]
    heading = np.mod(
        np.degrees(
            np.arctan2(
                np.sum(forward * east, axis=1),
                np.sum(forward * north, axis=1),
            )
        ),
        360.0,
    )
    line_of_sight = (dog - transducer) / slant_range[:, None]
    body_line_of_sight = np.einsum("nji,nj->ni", rotations, line_of_sight)
    azimuth = np.mod(
        np.degrees(
            np.arctan2(
                np.sum(line_of_sight * east, axis=1),
                np.sum(line_of_sight * north, axis=1),
            )
        ),
        360.0,
    )
    legs, turn_rate = identify_legs(heading, times)

    return SimpleNamespace(
        file_tag=file_tag,
        inversion_path=inversion_path,
        inversion=inversion,
        dog=dog,
        times=times,
        time_hours=time_hours,
        residual=residual,
        slant_range=slant_range,
        radial_velocity=radial_velocity,
        heading=heading,
        azimuth=azimuth,
        body_line_of_sight=body_line_of_sight,
        receiver_labels=receiver_set_labels(used_receivers),
        four_receivers=np.all(used_receivers, axis=1),
        legs=legs,
        turn_rate=turn_rate,
        rotations=rotations,
        transducer=transducer,
        gps_observation_indices=processed_indices,
        acoustic_observation_indices=observation_tag_indices,
    )


def linear_fit(x, y):
    """Fit a line and return compact regression diagnostics."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    correlation = np.corrcoef(x, y)[0, 1]
    return SimpleNamespace(
        intercept=coefficients[0],
        slope=coefficients[1],
        correlation=correlation,
        r_squared=correlation**2,
        rmse=np.sqrt(np.mean((y - predicted) ** 2)),
        count=len(x),
        x=x,
        y=y,
    )


def residual_stats(values):
    """Return standard and robust residual statistics."""
    median = np.median(values)
    return SimpleNamespace(
        count=len(values),
        mean=np.mean(values),
        median=median,
        rmse=np.sqrt(np.mean(values**2)),
        sigma=np.std(values),
        robust_sigma=1.4826 * np.median(np.abs(values - median)),
    )


def binned_summary(x, y, bins):
    """Calculate median and 16-84 percentile summaries in fixed bins."""
    group = np.digitize(x, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    median = np.full(len(centers), np.nan)
    lower = np.full(len(centers), np.nan)
    upper = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        values = y[group == index]
        if len(values) >= 10:
            lower[index], median[index], upper[index] = np.percentile(
                values, [16, 50, 84]
            )
    return centers, median, lower, upper


def harmonic_fit(angle, residual):
    """Measure first-harmonic residual dependence on a circular angle."""
    radians = np.radians(angle)
    design = np.column_stack((np.ones(len(angle)), np.sin(radians), np.cos(radians)))
    coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
    predicted = design @ coefficients
    ss_total = np.sum((residual - np.mean(residual)) ** 2)
    r_squared = 1.0 - np.sum((residual - predicted) ** 2) / ss_total
    return SimpleNamespace(
        coefficients=coefficients,
        amplitude=np.hypot(coefficients[1], coefficients[2]),
        r_squared=r_squared,
    )


def output_directory():
    """Return the Puerto Rico diagnostic figure directory."""
    path = gps_data_path("Figs", "Inversion_Workflow", "PuertoRico_Diagnostics")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(figure, filename):
    """Save and close one diagnostic figure."""
    path = output_directory() / filename
    if save:
        figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return path


def plot_radial_velocity(data):
    """Plot residual against radial velocity and fit timing correction."""
    fit = linear_fit(data.radial_velocity, data.residual)
    figure, axis = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(data.radial_velocity)
    scatter = axis.scatter(
        data.radial_velocity[valid],
        data.residual[valid],
        c=data.time_hours[valid],
        s=4,
        alpha=0.55,
        cmap="viridis",
    )
    x_line = np.linspace(np.min(fit.x), np.max(fit.x), 200)
    axis.plot(x_line, fit.intercept + fit.slope * x_line, color="black", linewidth=2)
    limits = np.percentile(fit.x, [1, 99])
    bins = np.linspace(limits[0], limits[1], 21)
    centers, median, _, _ = binned_summary(fit.x, fit.y, bins)
    axis.plot(centers, median, color="tab:red", marker="o", linewidth=2)
    axis.set_xlabel("Radial velocity (m/s)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Time since first observation (h)")
    axis.set_title(
        f"Timing correction = {fit.slope:.4f} s, intercept = {fit.intercept:.2f} m\n"
        f"r = {fit.correlation:.3f}, $R^2$ = {fit.r_squared:.3f}, "
        f"fit RMSE = {fit.rmse:.2f} m, N = {fit.count:,}"
    )
    figure.tight_layout()
    return fit, save_figure(figure, "residual_vs_radial_velocity.pdf")


def plot_heading(data):
    """Plot residual against circular vessel heading."""
    bins = np.linspace(0, 360, 25)
    centers, median, lower, upper = binned_summary(data.heading, data.residual, bins)
    x = np.r_[centers[-1] - 360, centers, centers[0] + 360]
    med = np.r_[median[-1], median, median[0]]
    low = np.r_[lower[-1], lower, lower[0]]
    high = np.r_[upper[-1], upper, upper[0]]
    fit = harmonic_fit(data.heading, data.residual)

    figure, axis = plt.subplots(figsize=(9, 5))
    scatter = axis.scatter(
        data.heading,
        data.residual,
        c=data.time_hours,
        s=4,
        alpha=0.5,
        cmap="viridis",
    )
    axis.fill_between(x, low, high, color="tab:red", alpha=0.2)
    axis.plot(x, med, color="tab:red", linewidth=2)
    axis.set_xlim(0, 360)
    axis.set_xticks(np.arange(0, 361, 45))
    axis.set_xlabel("Vessel heading (degrees)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.set_title(
        f"Heading first-harmonic amplitude = {fit.amplitude:.2f} m, "
        f"$R^2$ = {fit.r_squared:.3f}"
    )
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Time since first observation (h)")
    figure.tight_layout()
    return fit, save_figure(figure, "residual_vs_heading.pdf")


def plot_body_line_of_sight(data):
    """Fit and plot the body-frame line-of-sight lever diagnostic."""
    design = np.column_stack((np.ones(len(data.residual)), data.body_line_of_sight))
    coefficients = np.linalg.lstsq(design, data.residual, rcond=None)[0]
    predicted = design @ coefficients
    corrected = data.residual - predicted
    ss_total = np.sum((data.residual - np.mean(data.residual)) ** 2)
    r_squared = 1.0 - np.sum(corrected**2) / ss_total

    labels = ("Forward", "Starboard", "Vertical")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        sharey=True,
        layout="constrained",
    )
    scatter = None
    for index, axis in enumerate(axes):
        scatter = axis.scatter(
            data.body_line_of_sight[:, index],
            data.residual,
            c=data.time_hours,
            s=3,
            alpha=0.45,
            cmap="viridis",
        )
        axis.set_xlabel(f"Body LOS: {labels[index].lower()}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Range-equivalent residual (m)")
    figure.colorbar(
        scatter,
        ax=axes,
        label="Time since first observation (h)",
        pad=0.02,
    )
    figure.suptitle(
        f"Implied lever magnitude = {np.linalg.norm(coefficients[1:]):.2f} m, "
        f"$R^2$ = {r_squared:.3f}"
    )
    result = SimpleNamespace(
        intercept=coefficients[0],
        vector=coefficients[1:],
        magnitude=np.linalg.norm(coefficients[1:]),
        r_squared=r_squared,
        rmse_before=np.sqrt(np.mean(data.residual**2)),
        rmse_after=np.sqrt(np.mean(corrected**2)),
    )
    return result, save_figure(figure, "residual_vs_body_los.pdf")


def receiver_statistics(data):
    """Return residual statistics in the requested receiver-set order."""
    return {
        label: residual_stats(data.residual[data.receiver_labels == label])
        for label in receiver_sets
        if np.any(data.receiver_labels == label)
    }


def plot_receiver_sets(data):
    """Plot receiver-set time series and grouped distributions."""
    statistics = receiver_statistics(data)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(statistics)))

    figure, axis = plt.subplots(figsize=(11, 5))
    for color, label in zip(colors, statistics, strict=False):
        mask = data.receiver_labels == label
        axis.scatter(
            data.time_hours[mask],
            data.residual[mask],
            s=5,
            alpha=0.65,
            color=color,
            label=label,
        )
    axis.set_xlabel("Time since first observation (h)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    figure.tight_layout()
    time_path = save_figure(figure, "residual_vs_receiver_set_time.pdf")

    figure, axis = plt.subplots(figsize=(10, 5))
    labels = list(statistics)
    values = [data.residual[data.receiver_labels == label] for label in labels]
    axis.boxplot(values, labels=labels, showfliers=False, whis=(5, 95))
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    distribution_path = save_figure(figure, "residual_by_receiver_set.pdf")
    return statistics, (time_path, distribution_path)


def plot_four_receiver_comparison(data):
    """Compare all accepted observations against four-receiver epochs."""
    all_stats = residual_stats(data.residual)
    four_stats = residual_stats(data.residual[data.four_receivers])
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    axes[0].scatter(data.time_hours, data.residual, s=3, alpha=0.5)
    axes[0].set_title(
        f"Three- or four-receiver solution: N={all_stats.count:,}, "
        f"RMSE={all_stats.rmse:.2f} m"
    )
    axes[1].scatter(
        data.time_hours[data.four_receivers],
        data.residual[data.four_receivers],
        s=5,
        alpha=0.65,
        color="tab:orange",
    )
    axes[1].set_title(
        f"Four receivers only: N={four_stats.count:,}, RMSE={four_stats.rmse:.2f} m"
    )
    axes[1].set_xlabel("Time since first observation (h)")
    for axis in axes:
        axis.set_ylabel("Residual (m)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return (
        all_stats,
        four_stats,
        save_figure(figure, "four_receiver_comparison.pdf"),
    )


def circular_mean_degrees(values):
    """Return a circular mean in degrees."""
    return np.mod(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(values))))), 360)


def leg_statistics(data):
    """Calculate per-leg residual and timing statistics."""
    results = []
    for leg in np.unique(data.legs):
        mask = data.legs == leg
        stats = residual_stats(data.residual[mask])
        timing = linear_fit(data.radial_velocity[mask], data.residual[mask])
        results.append(
            SimpleNamespace(
                leg=int(leg),
                start=data.times[mask][0],
                end=data.times[mask][-1],
                midpoint=np.mean(data.time_hours[mask]),
                duration=(data.times[mask][-1] - data.times[mask][0]) / 60,
                heading=circular_mean_degrees(data.heading[mask]),
                stats=stats,
                timing=timing,
            )
        )
    return results


def plot_legs(data):
    """Plot residual diagnostics separated by automatically detected legs."""
    statistics = leg_statistics(data)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(statistics)))

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    distributions = []
    for color, result in zip(colors, statistics, strict=False):
        mask = data.legs == result.leg
        axes[0].scatter(
            data.time_hours[mask], data.residual[mask], s=4, color=color, alpha=0.6
        )
        distributions.append(data.residual[mask])
    axes[0].set_xlabel("Time since first observation (h)")
    axes[0].set_ylabel("Residual (m)")
    axes[0].grid(alpha=0.25)
    axes[1].boxplot(
        distributions,
        labels=[str(result.leg + 1) for result in statistics],
        showfliers=False,
        whis=(5, 95),
    )
    axes[1].set_xlabel("Survey leg")
    axes[1].set_ylabel("Residual (m)")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    leg_path = save_figure(figure, "residual_by_survey_leg.pdf")

    figure, axis = plt.subplots(figsize=(10, 6))
    for color, result in zip(colors, statistics, strict=False):
        mask = data.legs == result.leg
        axis.scatter(
            data.slant_range[mask] / 1000,
            data.residual[mask],
            s=4,
            color=color,
            alpha=0.6,
            label=str(result.leg + 1),
        )
    axis.set_xlabel("Slant range (km)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.grid(alpha=0.25)
    axis.legend(title="Leg", ncol=4, fontsize=7)
    figure.tight_layout()
    range_path = save_figure(figure, "residual_vs_range_by_leg.pdf")
    return statistics, (leg_path, range_path)


def plot_azimuth(data):
    """Plot residual against source-to-DOG azimuth."""
    bins = np.linspace(0, 360, 25)
    centers, median, _, _ = binned_summary(data.azimuth, data.residual, bins)
    fit = harmonic_fit(data.azimuth, data.residual)
    figure, axis = plt.subplots(figsize=(9, 5))
    scatter = axis.scatter(
        data.azimuth,
        data.residual,
        c=data.time_hours,
        s=4,
        alpha=0.5,
        cmap="viridis",
    )
    axis.plot(centers, median, color="tab:red", marker="o", linewidth=2)
    axis.set_xlim(0, 360)
    axis.set_xticks(np.arange(0, 361, 45))
    axis.set_xlabel("Source-to-DOG azimuth (degrees)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.set_title(
        f"Azimuth first-harmonic amplitude = {fit.amplitude:.2f} m, "
        f"$R^2$ = {fit.r_squared:.3f}"
    )
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Time since first observation (h)")
    figure.tight_layout()
    return fit, save_figure(figure, "residual_vs_azimuth.pdf")


def plot_range_branches(data):
    """Color residual-range branches by heading and receiver set."""
    figure, axis = plt.subplots(figsize=(9, 5))
    scatter = axis.scatter(
        data.slant_range / 1000,
        data.residual,
        c=data.heading,
        s=4,
        alpha=0.55,
        cmap="twilight",
    )
    axis.set_xlabel("Slant range (km)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Vessel heading (degrees)")
    figure.tight_layout()
    heading_path = save_figure(figure, "residual_vs_range_by_heading.pdf")

    figure, axis = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(receiver_sets)))
    for color, label in zip(colors, receiver_sets, strict=False):
        mask = data.receiver_labels == label
        axis.scatter(
            data.slant_range[mask] / 1000,
            data.residual[mask],
            s=4,
            alpha=0.55,
            color=color,
            label=label,
        )
    axis.set_xlabel("Slant range (km)")
    axis.set_ylabel("Range-equivalent residual (m)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    receiver_path = save_figure(figure, "residual_vs_range_by_receiver_set.pdf")
    return heading_path, receiver_path


def calculate_timing_metrics(data):
    """Reuse the existing timing-offset metric calculations near the solution."""
    inversion = data.inversion
    selected_offset = inversion["best_offset"].item() - inversion["time_bias"].item()
    half_window = 0.1
    step = 0.0005
    for _ in range(3):
        offsets = np.arange(
            selected_offset - half_window,
            selected_offset + half_window + step / 2,
            step,
        )
        rmse = np.empty(len(offsets))
        coherence = np.empty(len(offsets))
        matches = np.empty(len(offsets))
        for index, offset in enumerate(offsets):
            rmse[index], matches[index] = _rmse_cm_for_offset(
                offset,
                inversion["CDOG_data"],
                inversion["GPS_data"],
                inversion["times_guess"],
                inversion["transponder_coordinates"],
                inversion["esv"],
            )
            coherence[index] = _coherence_obj_for_offset(
                offset,
                inversion["CDOG_data"],
                inversion["GPS_data"],
                inversion["times_guess"],
                inversion["transponder_coordinates"],
                inversion["esv"],
            )[0]
        rmse_index = np.nanargmin(rmse)
        coherence_index = np.nanargmin(coherence)
        edge = min(
            rmse_index,
            coherence_index,
            len(offsets) - rmse_index - 1,
            len(offsets) - coherence_index - 1,
        )
        if edge > 3:
            break
        half_window *= 2
    return SimpleNamespace(
        offsets=offsets,
        rmse=rmse / 100.0,
        coherence=coherence,
        matches=matches,
        current_offset=inversion["best_offset"].item(),
        selected_offset=selected_offset,
        rmse_offset=offsets[rmse_index],
        coherence_offset=offsets[coherence_index],
        rmse_index=rmse_index,
        coherence_index=coherence_index,
    )


def plot_timing_metrics(metrics):
    """Plot timing metrics against offset and against each other."""
    markers = (
        (metrics.current_offset, "Current", "tab:gray", ":"),
        (metrics.selected_offset, "Selected", "black", "-"),
        (metrics.rmse_offset, "Minimum RMSE", "tab:red", "--"),
        (metrics.coherence_offset, "Best circular", "tab:blue", "--"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(metrics.offsets, metrics.coherence, color="tab:blue")
    axes[0].set_ylabel("Circular-variance objective")
    axes[1].plot(metrics.offsets, metrics.rmse, color="tab:red")
    axes[1].set_ylabel("RMSE (m)")
    axes[1].set_xlabel("Timing offset (s)")
    for value, label, color, style in markers:
        for axis in axes:
            axis.axvline(value, color=color, linestyle=style, label=label)
            axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[0].legend(unique.values(), unique.keys(), ncol=2, fontsize=9)
    figure.tight_layout()
    offset_path = save_figure(figure, "timing_offset_metrics.pdf")

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(metrics.coherence, metrics.rmse, color="0.65", linewidth=1)
    scatter = axis.scatter(
        metrics.coherence,
        metrics.rmse,
        c=metrics.offsets,
        cmap="viridis",
        s=16,
    )
    selected_indices = (
        np.argmin(np.abs(metrics.offsets - metrics.current_offset)),
        np.argmin(np.abs(metrics.offsets - metrics.selected_offset)),
        metrics.rmse_index,
        metrics.coherence_index,
    )
    for index, (_, label, color, _) in zip(selected_indices, markers, strict=False):
        axis.scatter(
            metrics.coherence[index],
            metrics.rmse[index],
            s=70,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
            label=label,
        )
    axis.set_xlabel("Circular-variance objective")
    axis.set_ylabel("RMSE (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.colorbar(scatter, ax=axis, label="Timing offset (s)")
    figure.tight_layout()
    space_path = save_figure(figure, "timing_offset_metric_space.pdf")
    return offset_path, space_path


def plot_leg_timing(statistics):
    """Plot timing corrections inferred independently for each survey leg."""
    midpoints = np.array([result.midpoint for result in statistics])
    corrections = np.array([result.timing.slope for result in statistics])
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(midpoints, corrections, color="0.6", linewidth=1)
    scatter = axis.scatter(
        midpoints,
        corrections,
        c=np.arange(len(statistics)),
        cmap="turbo",
        s=45,
    )
    for result in statistics:
        axis.annotate(
            str(result.leg + 1),
            (result.midpoint, result.timing.slope),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Time since first observation (h)")
    axis.set_ylabel("Inferred timing correction (s)")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Survey leg")
    figure.tight_layout()
    return save_figure(figure, "inferred_timing_offset_by_leg.pdf")


def print_summary(
    data,
    timing_fit,
    heading_fit,
    body_fit,
    receivers,
    all_stats,
    four_stats,
    legs,
    azimuth_fit,
    timing_metrics,
):
    """Print the requested concise numerical diagnostic summary."""
    range_correlation = np.corrcoef(data.slant_range, data.residual)[0, 1]
    leg_rmse = np.array([result.stats.rmse for result in legs])
    leg_medians = np.array([result.stats.median for result in legs])
    timing_by_leg = np.array([result.timing.slope for result in legs])

    print("\nTIMING")
    print(f"  residual/radial-velocity correlation: {timing_fit.correlation:.4f}")
    print(f"  implied global timing correction: {timing_fit.slope:.6f} s")
    print(f"  regression intercept: {timing_fit.intercept:.4f} m")
    print(f"  regression R2: {timing_fit.r_squared:.4f}")
    print(f"  regression RMSE: {timing_fit.rmse:.4f} m")
    print(f"  regression observations: {timing_fit.count:,}")
    print(f"  stored offset: {timing_metrics.current_offset:.6f} s")
    print(f"  final effective offset: {timing_metrics.selected_offset:.6f} s")
    print(f"  RMSE-optimal timing offset: {timing_metrics.rmse_offset:.6f} s")
    print(
        "  circular-variance-optimal timing offset: "
        f"{timing_metrics.coherence_offset:.6f} s"
    )
    print(
        "  timing metric optimum difference: "
        f"{timing_metrics.rmse_offset - timing_metrics.coherence_offset:.6f} s"
    )
    print("  timing correction by leg:")
    for result in legs:
        print(
            f"    leg {result.leg + 1:02d}: {result.timing.slope: .6f} s "
            f"(R2={result.timing.r_squared:.3f}, N={result.stats.count})"
        )

    print("\nLEVER ARM")
    print(f"  forward coefficient: {body_fit.vector[0]: .4f} m")
    print(f"  starboard coefficient: {body_fit.vector[1]: .4f} m")
    print(f"  vertical coefficient: {body_fit.vector[2]: .4f} m")
    print(f"  implied vector magnitude: {body_fit.magnitude:.4f} m")
    print(f"  body LOS R2: {body_fit.r_squared:.4f}")
    print(
        f"  RMSE before/after LOS fit: "
        f"{body_fit.rmse_before:.4f} / {body_fit.rmse_after:.4f} m"
    )

    print("\nGNSS")
    for label, stats in receivers.items():
        print(
            f"  {label:11s}: N={stats.count:5d}, mean={stats.mean: .3f}, "
            f"median={stats.median: .3f}, RMSE={stats.rmse:.3f}, "
            f"sigma={stats.sigma:.3f}, robust={stats.robust_sigma:.3f} m"
        )
    print(f"  all vs four-only RMSE: {all_stats.rmse:.4f} / {four_stats.rmse:.4f} m")
    print(
        f"  all vs four-only median: {all_stats.median:.4f} / {four_stats.median:.4f} m"
    )
    print(f"  all vs four-only sigma: {all_stats.sigma:.4f} / {four_stats.sigma:.4f} m")

    print("\nLEGS")
    for result in legs:
        print(
            f"  leg {result.leg + 1:02d}: {result.start:.1f}-{result.end:.1f} s, "
            f"{result.duration:.1f} min, heading={result.heading:.1f}, "
            f"N={result.stats.count}, median={result.stats.median:.3f}, "
            f"RMSE={result.stats.rmse:.3f}, sigma={result.stats.sigma:.3f} m"
        )

    print("\nGEOMETRY")
    print(f"  residual/slant-range correlation: {range_correlation:.4f}")
    print(
        f"  heading harmonic: amplitude={heading_fit.amplitude:.4f} m, "
        f"R2={heading_fit.r_squared:.4f}"
    )
    print(
        f"  azimuth harmonic: amplitude={azimuth_fit.amplitude:.4f} m, "
        f"R2={azimuth_fit.r_squared:.4f}"
    )

    print("\nDIAGNOSTIC EVIDENCE")
    print(
        f"  leg median span={np.ptp(leg_medians):.3f} m; "
        f"leg RMSE range={np.min(leg_rmse):.3f}-{np.max(leg_rmse):.3f} m"
    )
    print(
        f"  leg timing range={np.nanmin(timing_by_leg):.4f} to "
        f"{np.nanmax(timing_by_leg):.4f} s"
    )
    print(
        "  These values are diagnostic evidence only; they do not by themselves "
        "prove a timing, lever, GNSS, DOG-location, or ESV cause."
    )

    print("\nINTERPRETATION")
    print(
        "  The strongest evidence is body-frame/heading, receiver-set, and "
        "survey-leg structure."
    )
    print(
        "  The selected effective offset is at the RMSE minimum, while the "
        "radial-velocity slopes vary strongly by leg; one remaining constant "
        "timing correction is therefore not the leading explanation."
    )
    print(
        "  The near-zero range correlation and weak azimuth harmonic provide "
        "little support for a simple ESV or DOG-location error. These are "
        "relative diagnostic weights, not proof."
    )


def run_diagnostics():
    """Generate the Puerto Rico inversion diagnostic figure set."""
    data = load_diagnostic_data()
    timing_fit, _ = plot_radial_velocity(data)
    heading_fit, _ = plot_heading(data)
    body_fit, _ = plot_body_line_of_sight(data)
    receivers, _ = plot_receiver_sets(data)
    all_stats, four_stats, _ = plot_four_receiver_comparison(data)
    legs, _ = plot_legs(data)
    azimuth_fit, _ = plot_azimuth(data)
    plot_range_branches(data)
    timing_metrics = calculate_timing_metrics(data)
    plot_timing_metrics(timing_metrics)
    plot_leg_timing(legs)
    print_summary(
        data,
        timing_fit,
        heading_fit,
        body_fit,
        receivers,
        all_stats,
        four_stats,
        legs,
        azimuth_fit,
        timing_metrics,
    )
    print(f"\nSaved diagnostic figures to {output_directory()}")


if __name__ == "__main__":
    run_diagnostics()
