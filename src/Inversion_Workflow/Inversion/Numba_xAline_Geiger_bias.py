import numpy as np
import scipy.io as sio

from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned
from Inversion_Workflow.Inversion.Numba_xAline import (
    filter_close_receptions,
    find_int_offset,
    find_subint_offset,
    two_pointer_index,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    esv_gradient,
    calculateTimesRayTracing_Bias,
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from numba import njit
from plotting.Plot_Modular import time_series_plot
from bermuda_data import gps_data_path


@njit(cache=True)
def compute_Jacobian_biased(guess, transponder_coordinates, times, esv, grad_x_c0=None):
    """Jacobian of the travel-time residual with respect to (x, y, z, dt, dc).

    ``esv`` must already include the ESV bias (c_eff + dc), as returned by
    ``calculateTimesRayTracing_Bias(_Real)``. The residual is
    (T_mod - time_bias) - T_obs with T_mod = range / esv.

    ``grad_x_c0`` (N, 3), the gradient of the geometry-dependent reference ESV c0
    with respect to the DOG (``esv_gradient``), adds the term -(T / esv) * grad c0
    of the full derivative. Without it c0 is held fixed (frozen-ESV Jacobian).
    """
    diffs = transponder_coordinates - guess

    J = np.zeros((len(transponder_coordinates), 5))

    # Compute different partial derivatives
    J[:, 0] = -diffs[:, 0] / (times[:] * (esv[:] ** 2))
    J[:, 1] = -diffs[:, 1] / (times[:] * (esv[:] ** 2))
    J[:, 2] = -diffs[:, 2] / (times[:] * (esv[:] ** 2))
    if grad_x_c0 is not None:
        J[:, :3] -= times[:, None] / esv[:, None] * grad_x_c0
    J[:, 3] = -1.0
    J[:, 4] = -times[:] / esv[:]

    return J


@njit(cache=True)
def residual_inliers(residuals, sigma=6.0):
    """Keep residuals within a conservative median-based limit."""
    if len(residuals) < 5:
        return np.ones(len(residuals), dtype=np.bool_)

    center = np.median(residuals)
    scale = 1.4826 * np.median(np.abs(residuals - center))
    if scale == 0.0:
        return np.ones(len(residuals), dtype=np.bool_)

    inliers = np.abs(residuals - center) <= sigma * scale
    if np.sum(inliers) < 5:
        return np.ones(len(residuals), dtype=np.bool_)
    return inliers


@njit(cache=True)
def paired_time_mid(emission_times):
    """Reference epoch t_mid (s) of the clock drift.

    Half-way between the first and last paired GPS emission times of the DOG, as
    in the MCMC (``MCMC_sampler``): ``dt(t) = dt0 + drift_rate * (t - t_mid)``.
    """
    return 0.5 * (emission_times[0] + emission_times[-1])


@njit(cache=True)
def drift_step(J, residuals, t_from_mid):
    """Gauss-Newton step for (x, y, z, time_bias, esv_bias, drift_rate).

    ``J`` is the 5-column Jacobian of ``compute_Jacobian_biased``. The residual is
    ``(T_mod - time_bias + drift_rate * (t - t_mid)) - T_obs`` (internal
    ``time_bias - drift_rate * (t - t_mid)``), so the drift column is
    ``+(t - t_mid)``. It is up to ~4e4 s against ~1e-9 s/s, so it is divided by
    the largest ``|t - t_mid|`` (the solved parameter is then the accumulated clock
    error at the survey end, s) and the step is a least-squares solve rather than
    an inverse of ``J.T @ J``.
    """
    half_span = np.max(np.abs(t_from_mid))
    J_drift = np.empty((J.shape[0], 6))
    J_drift[:, :5] = J
    J_drift[:, 5] = t_from_mid / half_span
    delta = np.linalg.lstsq(J_drift, -residuals, rcond=-1.0)[0]
    delta[5] = delta[5] / half_span
    return delta


# Largest integer-second shift tried per pass when re-selecting the offset.
INT_OFFSET_SHIFT_SECONDS = 3
MAX_INT_OFFSET_PASSES = 20


@njit(cache=True, fastmath=True)
def _geiger_at_offset(
    guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data,
    initial_offset,
    clock_drift,
):
    """Alignment and Gauss-Newton fit for one starting offset.

    With ``initial_offset=None`` the offset is searched over the full range with the
    circular-variance score at ``guess``; otherwise only within 0.5 s of the rounded
    ``initial_offset``.
    """
    epsilon = 10**-5
    k = 0
    delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    time_bias = 0.0
    esv_bias = 0.0
    drift_rate = 0.0
    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    if clock_drift:
        estimate = np.append(estimate, drift_rate)

    # Initial offset estimate
    if not real_data:
        times_guess, esv = calculateTimesRayTracing_Bias(
            guess,
            transponder_coordinates,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            guess,
            transponder_coordinates,
            esv_bias,
            dz_array,
            angle_array,
            esv_matrix,
        )
    offset_data = filter_close_receptions(CDOG_data)
    if initial_offset is None:
        offset = find_int_offset(
            offset_data, GPS_data, times_guess, transponder_coordinates, esv
        )
    else:
        offset = find_int_offset(
            offset_data,
            GPS_data,
            times_guess,
            transponder_coordinates,
            esv,
            offset=np.round(initial_offset),
            halfwindow=0.5,
        )
    offset = find_subint_offset(
        offset,
        offset_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )

    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if not real_data:
            times_guess, esv = calculateTimesRayTracing_Bias(
                guess,
                transponder_coordinates,
                esv_bias,
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times_guess, esv = calculateTimesRayTracing_Bias_Real(
                guess,
                transponder_coordinates,
                esv_bias,
                dz_array,
                angle_array,
                esv_matrix,
            )

        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset, 0.2, CDOG_data, GPS_data, times_guess, transponder_coordinates, esv
        )
        residuals = (GPS_full - time_bias) - CDOG_full
        t_from_mid = np.zeros(len(residuals))
        if clock_drift:
            # Pairs change between iterations here, so t_mid follows them.
            emission_times = GPS_clock - GPS_full
            t_from_mid = emission_times - paired_time_mid(emission_times)
            residuals = residuals + drift_rate * t_from_mid
        inliers = residual_inliers(residuals)
        grad_x_c0 = esv_gradient(
            guess,
            transponder_coordinates_full[inliers],
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
        )[0]
        J = compute_Jacobian_biased(
            guess,
            transponder_coordinates_full[inliers],
            GPS_full[inliers],
            esv_full[inliers],
            grad_x_c0,
        )
        if clock_drift:
            delta = drift_step(J, residuals[inliers], t_from_mid[inliers])
        else:
            delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ residuals[inliers]
        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        if clock_drift:
            drift_rate = estimate[5]
        k += 1
    """Refine offset in local region"""
    offset = find_subint_offset(
        offset,
        offset_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )
    return estimate, offset


@njit(cache=True, fastmath=True)
def misfit_rms(
    estimate,
    offset,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data,
    clock_drift=False,
):
    """RMS travel-time misfit (s) of the final fit at ``offset``."""
    drift_rate = estimate[5] if clock_drift else 0.0
    result = final_bias_geiger(
        estimate[:3],
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        estimate[4],
        estimate[3],
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
        clock_drift,
        drift_rate,
    )
    misfit = result[1] - result[2]
    if len(misfit) < 5:
        return np.inf
    return np.sqrt(np.mean(misfit**2))


@njit(cache=True, fastmath=True)
def initial_bias_geiger(
    guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=False,
    initial_offset=None,
    clock_drift=False,
):
    """Initial inversion including time and ESV bias parameters.

    Returns ``(estimate, offset)`` with ``estimate = [x, y, z, time_bias, esv_bias]``.
    The clock offset of Eq. 6 is ``offset - time_bias``. With ``clock_drift`` the
    estimate gets a sixth entry, ``drift_rate`` (s/s, Eq. 6 convention), and
    ``offset - time_bias`` is dt0, the offset at ``t_mid`` (see
    ``paired_time_mid``).

    With ``initial_offset`` the alignment is searched within 0.5 s of it. Without
    one, the integer second picked by the circular-variance score at ``guess`` is
    biased by about the position error divided by the vessel speed, so it is
    re-selected with the converged fit: shifts of up to
    ``INT_OFFSET_SHIFT_SECONDS`` are refitted around the best offset until none
    lowers the misfit RMS.
    """
    estimate, offset = _geiger_at_offset(
        guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
        initial_offset,
        clock_drift,
    )
    if initial_offset is not None:
        return estimate, offset

    best_rms = misfit_rms(
        estimate,
        offset,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
        real_data,
        clock_drift,
    )
    for _ in range(MAX_INT_OFFSET_PASSES):
        center_estimate, center_offset = estimate, offset
        for shift in range(-INT_OFFSET_SHIFT_SECONDS, INT_OFFSET_SHIFT_SECONDS + 1):
            if shift == 0:
                continue
            candidate, candidate_offset = _geiger_at_offset(
                center_estimate[:3],
                CDOG_data,
                GPS_data,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
                real_data,
                center_offset + shift,
                clock_drift,
            )
            rms = misfit_rms(
                candidate,
                candidate_offset,
                CDOG_data,
                GPS_data,
                transponder_coordinates,
                dz_array,
                angle_array,
                esv_matrix,
                real_data,
                clock_drift,
            )
            if rms < best_rms:
                best_rms, estimate, offset = rms, candidate, candidate_offset
        if offset == center_offset:
            break
    return estimate, offset


@njit(cache=True, fastmath=True)
def final_bias_geiger(
    guess,
    CDOG_data,
    GPS_data,
    transponder_coordinates,
    offset,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=False,
    clock_drift=False,
    drift_rate=0.0,
):
    """Final bias-aware inversion with fixed offset.

    With ``clock_drift`` the estimate gets a sixth entry, ``drift_rate`` (s/s, Eq. 6
    convention), started from ``drift_rate``; t_mid is set by the pairs made here.
    """
    epsilon = 10**-5
    k = 0
    delta = np.array([1.0, 1.0, 1.0])
    estimate = np.array([guess[0], guess[1], guess[2], time_bias, esv_bias])
    if clock_drift:
        estimate = np.append(estimate, drift_rate)

    if not real_data:
        times_guess, esv = calculateTimesRayTracing_Bias(
            guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
        )

    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        offset,
        0.05,
        CDOG_data,
        GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )
    t_mid = 0.0
    t_from_mid = np.zeros(len(GPS_clock))
    if clock_drift:
        emission_times = GPS_clock - GPS_full
        t_mid = paired_time_mid(emission_times)
        t_from_mid = emission_times - t_mid

    while np.linalg.norm(delta) > epsilon and k < 100:
        # Find the best offset
        if not real_data:
            GPS_full, esv = calculateTimesRayTracing_Bias(
                guess,
                transponder_coordinates_full,
                esv_bias,
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            GPS_full, esv = calculateTimesRayTracing_Bias_Real(
                guess,
                transponder_coordinates_full,
                esv_bias,
                dz_array,
                angle_array,
                esv_matrix,
            )

        residuals = (GPS_full - time_bias) - CDOG_full
        if clock_drift:
            residuals = residuals + drift_rate * t_from_mid
        inliers = residual_inliers(residuals)
        grad_x_c0 = esv_gradient(
            guess,
            transponder_coordinates_full[inliers],
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
        )[0]
        J = compute_Jacobian_biased(
            guess,
            transponder_coordinates_full[inliers],
            GPS_full[inliers],
            esv[inliers],
            grad_x_c0,
        )
        if clock_drift:
            delta = drift_step(J, residuals[inliers], t_from_mid[inliers])
        else:
            delta = -1 * np.linalg.inv(J.T @ J) @ J.T @ residuals[inliers]
        estimate = estimate + delta
        guess = estimate[:3]
        time_bias = estimate[3]
        esv_bias = estimate[4]
        if clock_drift:
            drift_rate = estimate[5]
        k += 1

    if not real_data:
        times_guess, esv = calculateTimesRayTracing_Bias(
            guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
        )
    else:
        times_guess, esv = calculateTimesRayTracing_Bias_Real(
            guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
        )

    """Note doing GPS_data - time_bias to include time_bias
       in offset when calculating travel times"""
    shifted_GPS_data = GPS_data - time_bias
    if clock_drift:
        # The drift moves arrivals by ~40 us at most, far inside the 0.05 s gate,
        # but is included so the returned pairs carry the fitted clock model.
        shifted_GPS_data = shifted_GPS_data + drift_rate * (GPS_data - t_mid)
    (
        CDOG_clock,
        CDOG_full,
        GPS_clock,
        GPS_full,
        transponder_coordinates_full,
        esv_full,
    ) = two_pointer_index(
        offset,
        0.05,
        CDOG_data,
        shifted_GPS_data,
        times_guess,
        transponder_coordinates,
        esv,
    )
    inliers = residual_inliers(GPS_full - CDOG_full)
    return (
        estimate,
        CDOG_full[inliers],
        GPS_full[inliers],
        CDOG_clock[inliers],
        GPS_clock[inliers],
    )


# Gauss-Newton offset search: integer seconds tried on each side of the aligned
# offset, and the smallest best-vs-second-best Delta chi^2 accepted. In small
# synthetic tests (DOG 3, 20 trials) right picks had Delta chi^2 >= 156
# and picks outside the window <= 23.
OFFSET_SEARCH_HALFWINDOW_SECONDS = 15
MIN_OFFSET_DELTA_CHI2 = 50.0
# Largest accepted misfit RMS of the best second: 50 cm of range at 1515 m/s. Right
# seconds fit at 13-24 cm on the real data; a local minimum at a wrong second
# (Puerto Rico crossover legs, -10 s instead of -19 s) fit at 94 cm and still had
# Delta chi^2 = 176 against its neighbours.
MAX_OFFSET_RMS_SECONDS = 0.5 / 1515.0
LEVER_GN_MAX_ITERATIONS = 50
# Iterations paired with the wide gate before narrowing it, so the time bias can
# absorb a fractional misalignment of a few tenths of a second.
LEVER_GN_WIDE_GATE_ITERATIONS = 6


def lever_basis(GPS_Coordinates, gps1_to_others):
    """Return ``(d, R)`` with transducer positions ``d + R @ lever`` at every epoch.

    ``d`` (N, 3) is GPS1 and ``R`` (N, 3, 3) the body-to-ECEF rotation fitted to the
    receiver grid, so the transducer position is exactly linear in the lever arm.
    """
    d = findTransponder(GPS_Coordinates, gps1_to_others, np.zeros(3))
    R = np.empty((len(d), 3, 3))
    for axis in range(3):
        unit_lever = np.eye(3)[axis]
        R[:, :, axis] = findTransponder(GPS_Coordinates, gps1_to_others, unit_lever) - d
    return d, R


def lever_bias_geiger(
    guess,
    CDOG_data,
    GPS_data,
    lever_origin,
    lever_rotation,
    lever,
    offset,
    dz_array,
    angle_array,
    esv_matrix,
    real_data=False,
    clock_drift=False,
):
    """Gauss-Newton for position, biases and the lever arm at a fixed clock offset.

    Fits ``[x, y, z, time_bias, esv_bias]`` (plus ``drift_rate`` with ``clock_drift``)
    and the lever arm, with transducer positions ``lever_origin + lever_rotation @
    lever`` (see ``lever_basis``). Pairs are re-made every iteration at
    ``offset - time_bias``, first within 0.5 s, then 0.05 s. The vertical lever arm is
    nearly degenerate with the DOG depth; the least-squares solve takes the
    minimum-norm step along that direction.

    Returns ``(estimate, lever, rms, n_pairs)``: ``rms`` (s) of the inlier residuals,
    ``np.inf`` with fewer than 20 pairs.
    """
    n_estimate = 6 if clock_drift else 5
    estimate = np.zeros(n_estimate)
    estimate[:3] = guess
    lever = np.array(lever, dtype=np.float64)
    rms, n_pairs = np.inf, 0
    for iteration in range(LEVER_GN_MAX_ITERATIONS):
        transponder_coordinates = lever_origin + lever_rotation @ lever
        if real_data:
            times, esv = calculateTimesRayTracing_Bias_Real(
                estimate[:3],
                transponder_coordinates,
                estimate[4],
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times, esv = calculateTimesRayTracing_Bias(
                estimate[:3],
                transponder_coordinates,
                estimate[4],
                dz_array,
                angle_array,
                esv_matrix,
            )
        gate = 0.5 if iteration < LEVER_GN_WIDE_GATE_ITERATIONS else 0.05
        (
            CDOG_clock,
            CDOG_full,
            GPS_clock,
            GPS_full,
            transponder_coordinates_full,
            esv_full,
        ) = two_pointer_index(
            offset - estimate[3],
            gate,
            CDOG_data,
            GPS_data,
            times,
            transponder_coordinates,
            esv,
        )
        n_pairs = len(GPS_full)
        if n_pairs < 20:
            return estimate, lever, np.inf, n_pairs

        # Pairs were made at offset - time_bias; the observed travel time is
        # arrival - offset - emission time.
        observed = CDOG_full - estimate[3]
        residuals = (GPS_full - estimate[3]) - observed
        emission_times = GPS_clock - GPS_full
        t_from_mid = emission_times - paired_time_mid(emission_times)
        if clock_drift:
            residuals = residuals + estimate[5] * t_from_mid
        inliers = residual_inliers(residuals)
        rms = np.sqrt(np.mean(residuals[inliers] ** 2))
        n_pairs = int(np.sum(inliers))

        epoch = np.searchsorted(GPS_data, emission_times - 1e-6)
        # dT/ds = line_of_sight / c - (T / c) * grad_s c0, then dT/dlever = dT/ds @ R.
        grad_x_c0, grad_s_c0 = esv_gradient(
            estimate[:3],
            transponder_coordinates_full,
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
        )
        line_of_sight = transponder_coordinates_full - estimate[:3]
        line_of_sight /= np.sqrt(np.sum(line_of_sight**2, axis=1))[:, None]
        dT_ds = (line_of_sight - GPS_full[:, None] * grad_s_c0) / esv_full[:, None]
        J_lever = np.einsum("ni,nij->nj", dT_ds, lever_rotation[epoch])
        columns = [
            compute_Jacobian_biased(
                estimate[:3],
                transponder_coordinates_full,
                GPS_full,
                esv_full,
                grad_x_c0,
            )
        ]
        half_span = np.max(np.abs(t_from_mid))
        if clock_drift:
            # Scaled as in drift_step: the solved value is the clock error at the end.
            columns.append((t_from_mid / half_span)[:, None])
        columns.append(J_lever)
        J = np.hstack(columns)[inliers]
        delta = np.linalg.lstsq(J, -residuals[inliers], rcond=None)[0]
        if clock_drift:
            delta[5] = delta[5] / half_span
        estimate = estimate + delta[:n_estimate]
        lever = lever + delta[n_estimate:]
        if (
            iteration >= LEVER_GN_WIDE_GATE_ITERATIONS
            and np.linalg.norm(delta) < 10**-5
        ):
            break
    return estimate, lever, rms, n_pairs


def gauss_newton_offset_search(
    CDOG_data,
    GPS_data,
    GPS_Coordinates,
    gps1_to_others,
    initial_guess,
    initial_lever,
    dz_array,
    angle_array,
    esv_matrix,
    initial_offset=None,
    real_data=False,
    clock_drift=False,
    halfwindow=OFFSET_SEARCH_HALFWINDOW_SECONDS,
):
    """Find the DOG clock offset and lever arm when the integer second is unknown.

    A wrong integer second pairs each arrival with the ship position that many
    seconds earlier. The ship's body-frame velocity is nearly constant, so a lever
    arm moved by about that many seconds of travel absorbs most of it, and a misfit
    compared at one fixed lever arm prefers the wrong second. Here the data are
    aligned once at ``initial_lever`` (circular variance and one Gauss-Newton fit,
    or within 0.5 s of ``initial_offset`` when given). ``lever_bias_geiger`` then
    solves position, biases and the lever arm at every integer second within
    ``halfwindow`` of that alignment, and the lowest misfit RMS wins. The window is
    recentred while the best second lies on its edge.

    Confidence: ``delta_chi2 = n_pairs * (rms_2**2 - rms**2) / rms**2`` between the
    best and second-best seconds (white noise). Below ``MIN_OFFSET_DELTA_CHI2``, or
    with a best-second RMS above ``MAX_OFFSET_RMS_SECONDS`` (a local minimum at a
    wrong second), the search is repeated once over four times the window, then
    raises ``ValueError``.

    Returns ``(lever, offset, estimate, diagnostics)``. The time bias is folded into
    the offset: ``estimate[3]`` is 0 and ``offset`` is the Eq. 6 clock offset dt
    (dt0 at t_mid with ``clock_drift``). ``diagnostics`` holds ``rms`` and
    ``second_rms`` (s), ``second_offset`` (s), ``delta_chi2`` and ``n_pairs``.
    """
    lever_origin, lever_rotation = lever_basis(GPS_Coordinates, gps1_to_others)
    transponder_coordinates = lever_origin + lever_rotation @ initial_lever
    try:
        aligned_offset = _geiger_at_offset(
            initial_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            dz_array,
            angle_array,
            esv_matrix,
            real_data,
            initial_offset,
            clock_drift,
        )[1]
    except np.linalg.LinAlgError:
        # Too few pairs for the fit after the circular-variance alignment: keep
        # that alignment; the search below refits at every candidate second.
        if real_data:
            times, esv = calculateTimesRayTracing_Bias_Real(
                initial_guess,
                transponder_coordinates,
                0.0,
                dz_array,
                angle_array,
                esv_matrix,
            )
        else:
            times, esv = calculateTimesRayTracing_Bias(
                initial_guess,
                transponder_coordinates,
                0.0,
                dz_array,
                angle_array,
                esv_matrix,
            )
        offset_data = filter_close_receptions(CDOG_data)
        if initial_offset is None:
            aligned_offset = find_int_offset(
                offset_data, GPS_data, times, transponder_coordinates, esv
            )
        else:
            aligned_offset = find_int_offset(
                offset_data,
                GPS_data,
                times,
                transponder_coordinates,
                esv,
                offset=np.round(initial_offset),
                halfwindow=0.5,
            )
        aligned_offset = find_subint_offset(
            aligned_offset,
            offset_data,
            GPS_data,
            times,
            transponder_coordinates,
            esv,
        )

    fits = {}
    for window in (halfwindow, 4 * halfwindow):
        center = 0
        for _ in range(MAX_INT_OFFSET_PASSES):
            for shift in range(center - window, center + window + 1):
                if shift not in fits:
                    fits[shift] = lever_bias_geiger(
                        initial_guess,
                        CDOG_data,
                        GPS_data,
                        lever_origin,
                        lever_rotation,
                        initial_lever,
                        aligned_offset + shift,
                        dz_array,
                        angle_array,
                        esv_matrix,
                        real_data,
                        clock_drift,
                    )
            best = min(fits, key=lambda s: fits[s][2])
            if abs(best - center) < window:
                break
            center = best
        best, second = sorted(fits, key=lambda s: fits[s][2])[:2]
        estimate, lever, rms, n_pairs = fits[best]
        second_rms = fits[second][2]
        delta_chi2 = n_pairs * (second_rms**2 - rms**2) / rms**2
        if delta_chi2 >= MIN_OFFSET_DELTA_CHI2 and rms <= MAX_OFFSET_RMS_SECONDS:
            break
    else:
        if delta_chi2 >= MIN_OFFSET_DELTA_CHI2:
            raise ValueError(
                "Best integer second of the DOG clock offset fits too poorly: "
                f"expected RMS <= {MAX_OFFSET_RMS_SECONDS * 1515 * 100:.0f} cm, got "
                f"{rms * 1515 * 100:.1f} cm ({best:+d} s from the alignment, "
                f"{n_pairs} pairs). Check the starting position, lever arm and data, "
                "or pass initial_offset."
            )
        raise ValueError(
            "Integer second of the DOG clock offset is ambiguous: expected "
            f"delta chi^2 >= {MIN_OFFSET_DELTA_CHI2:.0f} between the best and "
            f"second-best seconds, got {delta_chi2:.1f} (best {best:+d} s from the "
            f"alignment, RMS {rms * 1515 * 100:.1f} cm; second {second:+d} s, RMS "
            f"{second_rms * 1515 * 100:.1f} cm; {n_pairs} pairs). Check the starting "
            "position, lever arm and data."
        )

    estimate = estimate.copy()
    offset = aligned_offset + best - estimate[3]
    estimate[3] = 0.0
    diagnostics = {
        "rms": rms,
        "second_rms": second_rms,
        "second_offset": aligned_offset + second,
        "delta_chi2": delta_chi2,
        "n_pairs": n_pairs,
    }
    return lever, offset, estimate, diagnostics


if __name__ == "__main__":
    # Table to generate synthetic times
    esv_table = sio.loadmat(gps_data_path("ESV_Tables/global_table_esv.mat"))
    dz_array = esv_table["distance"].flatten()
    angle_array = esv_table["angle"].flatten()
    esv_matrix = esv_table["matrice"]

    position_noise = 2 * 10**-2
    time_noise = 2 * 10**-5
    downsample = 10

    esv_bias = 3.0
    time_bias = 0
    true_offset = np.random.rand() * 10000

    gps1_to_others = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.4054, -4.20905, 0.060621],
            [-12.1105, -0.956145, 0.00877],
            [-8.70446831, 5.165195, 0.04880436],
        ]
    )
    gps1_to_transponder = np.array([-12.48862757, 0.22622633, -15.89601934])

    """Either generate a realistic or use bermuda trajectory"""
    type = "bermuda"

    if type == "bermuda":
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = bermuda_trajectory(
            time_noise,
            position_noise,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            offset=true_offset,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=gps1_to_transponder,
        )
        real = True
    if type == "unaligned":
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = generateUnaligned(
            20000,
            time_noise,
            position_noise,
            true_offset,
            esv_bias,
            time_bias,
            dz_array,
            angle_array,
            esv_matrix,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=gps1_to_transponder,
        )
        real = False

    GPS_data = GPS_data[::downsample]
    GPS_Coordinates = GPS_Coordinates[::downsample]

    print("True Offset: ", true_offset)

    """After Generating run through the analysis"""

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )

    CDOG_position_adjustment = (np.random.rand(3) * 400) - 200
    print("CDOG Position Adjustment:", CDOG_position_adjustment)
    guess = CDOG + CDOG_position_adjustment

    print("\n ------Starting Initial Geiger------ \n")
    inversion_result, offset = initial_bias_geiger(
        guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
        real_data=real,
    )
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print(
        "SUB-INT Offset: {:.4f}".format(offset),
        "DIFF: {:.4f}".format(offset - true_offset),
    )
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion_Workflow:", np.around(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))
    print("\n")

    print("------Starting Final Geiger------ \n")
    inversion_result, CDOG_full, GPS_full, CDOG_clock, GPS_clock = final_bias_geiger(
        inversion_guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        esv_bias,
        time_bias,
        dz_array,
        angle_array,
        esv_matrix,
        real_data=real,
    )
    inversion_guess = inversion_result[:3]
    time_bias = inversion_result[3]
    esv_bias = inversion_result[4]
    print("CDOG:", np.around(CDOG, 2))
    print("Inversion_Workflow:", np.around(inversion_result, 2))
    print("Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100))

    # Plot the results
    time_series_plot(
        CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise
    )
