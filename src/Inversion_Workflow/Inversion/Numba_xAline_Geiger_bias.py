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
from Inversion_Workflow.Inversion.Numba_Geiger_bias import (
    compute_Jacobian_biased,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from numba import njit
from plotting.Plot_Modular import time_series_plot
from bermuda_data import gps_data_path


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
        J = compute_Jacobian_biased(
            guess,
            transponder_coordinates_full[inliers],
            GPS_full[inliers],
            esv_full[inliers],
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
        J = compute_Jacobian_biased(
            guess,
            transponder_coordinates_full[inliers],
            GPS_full[inliers],
            esv[inliers],
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
