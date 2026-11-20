import contextlib
import io

import numpy as np
import matplotlib.pyplot as plt

from Inversion_Workflow.Synthetic.Generate_Unaligned import (
    generateUnaligned,
)
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)
from acoustics.synthetic_esv_tables import ESV_GENERATE, ESV_MODERATE, ESV_STRONG
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import check_esv_coverage
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    gauss_newton_offset_search,
    initial_bias_geiger,
    final_bias_geiger,
)
from Inversion_Workflow.Inversion.Numba_xAline_Annealing_bias import (
    anneal_with_offset_search,
)
from plotting.Plot_Modular import time_series_plot
from bermuda_data import gps_data_path, gps_output_path, leg_mask

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 20,  # change this freely
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)

# True lever arm (m, GPS1 to transponder) of each generator, keyed by generate_type.
TRUE_LEVER = {
    0: np.array([-10.0, 3.0, -15.0]),
    1: np.array([-12.4659, 9.6021, -13.2993]),
}
# Receiver positions (m) relative to GPS1 in the Bermuda body frame.
BERMUDA_GPS1_TO_OTHERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.4054, -4.20905, 0.060621],
        [-12.1105, -0.956145, 0.00877],
        [-8.70446831, 5.165195, 0.04880436],
    ]
)
# A recovered clock offset (Eq. 6 dt) further than this from the truth is a failure.
OFFSET_FAIL_S = 0.01
SOUND_SPEED_MPS = 1515.0


def modular_synthetic(
    time_noise,
    position_noise,
    in_esv_bias,
    in_time_bias,
    esv1=ESV_GENERATE,
    esv2=ESV_MODERATE,
    generate_type=1,
    inversion_type=1,
    plot=True,
    DOG_num=3,
    downsample=50,
    use_leg_mask=False,
    in_drift_rate=0.0,
    clock_drift=False,
    true_offset=None,
):
    """Generate synthetic data and invert them.

    ``true_offset`` is the DOG clock offset in seconds put into the synthetic data;
    it is drawn from U(1000, 10000) s when None. It is never given to the inversion.

    ``in_drift_rate`` is the true clock drift (s/s, Eq. 6 convention) put into the
    synthetic data; ``clock_drift`` adds drift_rate to the inversion as a sixth
    entry of ``inversion_result``.
    """
    np.set_printoptions(suppress=True)
    # Choose ESV table for generation and to run synthetic
    #   Perhaps make the file link a parameter of the function

    dz_array_generate, angle_array_generate, esv_matrix_generate = load_esv_table(
        gps_data_path(f"ESV_Tables/{esv1}.mat")
    )

    dz_array_inversion, angle_array_inversion, esv_matrix_inversion = load_esv_table(
        gps_data_path(f"ESV_Tables/{esv2}.mat")
    )

    # Choose Generate type:
    #   0: Generate Unaligned Realistic Data
    #   1: Use Bermuda Dataset
    if generate_type == 0:
        # Generate Unaligned Realistic Data
        if true_offset is None:
            true_offset = np.random.rand() * 9000 + 1000
        gps1_to_others = np.array(
            [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
        )
        gps1_to_transponder = TRUE_LEVER[0]
        print(true_offset)
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
            in_esv_bias,
            in_time_bias,
            dz_array_generate,
            angle_array_generate,
            esv_matrix_generate,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=gps1_to_transponder,
            drift_rate=in_drift_rate,
        )
        z_sample = False
    else:
        # Use Bermuda Dataset
        gps1_to_others = BERMUDA_GPS1_TO_OTHERS
        gps1_to_transponder = TRUE_LEVER[1]
        if true_offset is None:
            true_offset = np.random.rand() * 9000 + 1000
        (
            CDOG_data,
            CDOG,
            GPS_Coordinates,
            GPS_data,
            true_transponder_coordinates,
        ) = bermuda_trajectory(
            time_noise,
            position_noise,
            in_esv_bias,
            in_time_bias,
            dz_array_generate,
            angle_array_generate,
            esv_matrix_generate,
            offset=true_offset,
            DOG_num=DOG_num,
            gps1_to_others=gps1_to_others,
            gps1_to_transponder=gps1_to_transponder,
            drift_rate=in_drift_rate,
        )

        # Paper I synthetics use the full trajectory, downsampled. The leg mask
        # (9-11 h and 12.4-15 h, not downsampled) is a sensitivity test only.
        if use_leg_mask:
            keep = leg_mask(GPS_data)
        else:
            keep = slice(None, None, downsample)
        GPS_Coordinates = GPS_Coordinates[keep]
        GPS_data = GPS_data[keep]
        true_transponder_coordinates = true_transponder_coordinates[keep]
        z_sample = True
    # The DOG clock offset of Eq. 6 is offset + time_bias at generation, and
    # offset - time_bias at inversion; only this combination is identified.
    true_dt = true_offset + in_time_bias
    print("True clock offset", true_dt)
    # Choose Inversion_Workflow Type
    #   0: Just xAline Geiger (true lever arm)
    #   1: Gauss-Newton offset search (integer second and lever arm)
    #   2: Simulated annealing over the integer seconds (validation of 1)
    real_data = True if generate_type == 1 else False
    for table_name, dz_array, angle_array in (
        (esv1, dz_array_generate, angle_array_generate),
        (esv2, dz_array_inversion, angle_array_inversion),
    ):
        check_esv_coverage(
            CDOG,
            true_transponder_coordinates,
            dz_array,
            angle_array,
            real_data,
            name=table_name,
            on_outside="warn",
        )
    initial_guess = CDOG + np.random.normal(0, 50, 3)
    if inversion_type == 0:
        # Just xAline Geiger
        lever = gps1_to_transponder

        transponder_coordinates = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        inversion_result, offset = initial_bias_geiger(
            initial_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
            clock_drift=clock_drift,
        )

        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        drift_rate = inversion_result[5] if clock_drift else 0.0

        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )
        print("\n")
        (
            inversion_result,
            CDOG_full,
            GPS_full,
            CDOG_clock,
            GPS_clock,
        ) = final_bias_geiger(
            inversion_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            offset,
            esv_bias,
            time_bias,
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
            clock_drift=clock_drift,
            drift_rate=drift_rate,
        )
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )

    elif inversion_type in (1, 2):
        real_lever = gps1_to_transponder
        initial_lever = real_lever - np.random.normal(0, 3, 3)

        if inversion_type == 1:
            lever, offset, inversion_result, _ = gauss_newton_offset_search(
                CDOG_data,
                GPS_data,
                GPS_Coordinates,
                gps1_to_others,
                initial_guess,
                initial_lever,
                dz_array_inversion,
                angle_array_inversion,
                esv_matrix_inversion,
                real_data=real_data,
                clock_drift=clock_drift,
            )
        else:
            lever, offset, inversion_result = anneal_with_offset_search(
                300,
                CDOG_data,
                GPS_data,
                GPS_Coordinates,
                gps1_to_others,
                initial_guess,
                initial_lever,
                dz_array_inversion,
                angle_array_inversion,
                esv_matrix_inversion,
                real_data=real_data,
                z_sample=z_sample,
                clock_drift=clock_drift,
            )
        inversion_guess = inversion_result[:3]
        time_bias = inversion_result[3]
        esv_bias = inversion_result[4]
        drift_rate = inversion_result[5] if clock_drift else 0.0
        print("CDOG:", np.around(CDOG, 2))
        print("Inversion_Workflow:", np.round(inversion_result, 3))
        print(
            "Distance: {:.2f} cm".format(np.linalg.norm(inversion_guess - CDOG) * 100)
        )
        print(
            f"Lever Error: {np.round(np.linalg.norm(lever - real_lever) * 100, 2)} cm"
        )

        transponder_coordinates = findTransponder(
            GPS_Coordinates, gps1_to_others, lever
        )
        (
            inversion_result,
            CDOG_full,
            GPS_full,
            CDOG_clock,
            GPS_clock,
        ) = final_bias_geiger(
            inversion_guess,
            CDOG_data,
            GPS_data,
            transponder_coordinates,
            offset,
            esv_bias,
            time_bias,
            dz_array_inversion,
            angle_array_inversion,
            esv_matrix_inversion,
            real_data=real_data,
            clock_drift=clock_drift,
            drift_rate=drift_rate,
        )
    else:
        raise ValueError(f"inversion_type must be 0, 1 or 2, got {inversion_type!r}")

    print(f"Offset Error: {offset - inversion_result[3] - true_dt:.6f} s")
    if clock_drift:
        print(f"Drift Rate Error: {inversion_result[5] - in_drift_rate:.3e} s/s")

    if plot:
        time_series_plot(
            CDOG_clock, CDOG_full, GPS_clock, GPS_full, position_noise, time_noise
        )

    return (
        inversion_result,
        CDOG_data,
        CDOG_full,
        GPS_data,
        GPS_full,
        CDOG_clock,
        GPS_clock,
        CDOG,
        transponder_coordinates,
        GPS_Coordinates,
        offset,
        lever,
    )


def synthetic_trials(
    num_trials,
    time_noise,
    position_noise,
    in_time_bias=0.0,
    esv2=ESV_MODERATE,
    generate_type=1,
    inversion_type=1,
    DOG_num=1,
    downsample=50,
    save_name=None,
):
    """Repeat ``modular_synthetic`` with an unknown clock offset and summarize.

    Each trial draws a new true offset, noise and starting guesses. A trial that
    raises is recorded as NaN. Returns per-trial arrays: ``dt_error_s`` (Eq. 6
    clock offset error), ``residual_std_cm`` and ``residual_mean_cm`` (c times the
    CDOG-GPS residual), ``position_error_cm`` (CDOG, 3-D) and ``lever_error_cm``.
    Saved to ``Plot_Data/<save_name>.npz`` when ``save_name`` is given.
    """
    names = (
        "dt_error_s",
        "residual_std_cm",
        "residual_mean_cm",
        "position_error_cm",
        "lever_error_cm",
    )
    results = {name: np.full(num_trials, np.nan) for name in names}
    for i in range(num_trials):
        true_offset = np.random.rand() * 9000 + 1000
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = modular_synthetic(
                    time_noise,
                    position_noise,
                    0,
                    in_time_bias,
                    esv2=esv2,
                    generate_type=generate_type,
                    inversion_type=inversion_type,
                    plot=False,
                    DOG_num=DOG_num,
                    downsample=downsample,
                    true_offset=true_offset,
                )
        except Exception as e:
            print(f"{i + 1}/{num_trials} trial failed: {e}")
            continue
        inversion_result, CDOG_full, GPS_full, CDOG, offset, lever = (
            out[0],
            out[2],
            out[4],
            out[7],
            out[10],
            out[11],
        )
        residual_cm = (CDOG_full - GPS_full) * SOUND_SPEED_MPS * 100
        results["dt_error_s"][i] = (
            offset - inversion_result[3] - (true_offset + in_time_bias)
        )
        results["residual_std_cm"][i] = np.std(residual_cm)
        results["residual_mean_cm"][i] = np.mean(residual_cm)
        results["position_error_cm"][i] = (
            np.linalg.norm(inversion_result[:3] - CDOG) * 100
        )
        results["lever_error_cm"][i] = (
            np.linalg.norm(lever - TRUE_LEVER[generate_type]) * 100
        )
        print(
            f"{i + 1}/{num_trials} dt error {results['dt_error_s'][i]:.6f} s, "
            f"residual std {results['residual_std_cm'][i]:.2f} cm"
        )

    n_failed = np.sum(np.abs(results["dt_error_s"]) > OFFSET_FAIL_S)
    n_raised = np.sum(np.isnan(results["dt_error_s"]))
    print(f"\n{esv2}, generate_type {generate_type}, {num_trials} trials")
    print(f"Offset failures (|dt error| > {OFFSET_FAIL_S} s): {n_failed}")
    print(f"Trials that raised: {n_raised}")
    for name in names[1:]:
        print(f"{name}: median {np.nanmedian(results[name]):.2f}")
    if save_name is not None:
        np.savez(gps_output_path(f"Plot_Data/{save_name}"), **results)
    return results


if __name__ == "__main__":
    time_noise = 2 * 10**-5  # s
    position_noise = 2 * 10**-2  # m
    num_trials = 100
    offset_trials = 1000
    time_bias_offset_test = 0.37  # s

    # Fig. B10 statistics: strong table, plus the identical-table control.
    for esv2 in (ESV_STRONG, ESV_GENERATE):
        synthetic_trials(
            num_trials,
            time_noise,
            position_noise,
            esv2=esv2,
            save_name=f"synthetic_trials_{esv2}",
        )

    # Offset recovery on the Bermuda (1) and unaligned (0) generators.
    for generate_type in (1, 0):
        synthetic_trials(
            offset_trials,
            time_noise,
            position_noise,
            in_time_bias=time_bias_offset_test,
            generate_type=generate_type,
            save_name=f"offset_recovery_type{generate_type}",
        )
