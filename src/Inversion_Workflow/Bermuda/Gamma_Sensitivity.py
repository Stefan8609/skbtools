"""Sensitivity of the solution to the pairing gate gamma.

For each gamma in GAMMAS_S, reruns the Gauss-Newton workflow of
Bermuda_Data_time_bias.py with that gate at every pairing stage, and re-makes the
MCMC fixed pairs of FINAL_RUN (counts only). Reports per DOG and gamma the pair
counts, c*sigma_eps, and the change of lever arm, delta_t and DOG ENU position from
the solution at ``PAIRING_GATE_S``.

Run from the repository root with ``PYTHONPATH=src`` (about 10 min). Outputs go to
``revision/analysis/gamma_sensitivity/`` (gitignored).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bermuda_data import (
    CDOG_GUESS_AUGMENTS,
    CDOG_REFERENCE,
    GPS1_TO_OTHERS,
    OFFSET_HINTS_S,
    gps_data_path,
)
from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    NOMINAL_SOUND_SPEED_MPS,
    modelled_times,
)
from Inversion_Workflow.Forward_Model.Find_ESV import load_esv_table
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline import (
    OFFSET_SEARCH_THRESHOLD_SECONDS,
    PAIRING_GATE_S,
    two_pointer_index,
)
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    gauss_newton_offset_search,
)
from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from Inversion_Workflow.MCMC_Functions.MCMC_posterior_summary import enu_rotation

# Config
DOGS = (1, 3, 4)
# Gates (s). The pulses repeat at 1 Hz, so a gate below the 0.5 s half-period
# cannot reach the neighbouring pulse; fitted residuals are ~0.1 ms.
GAMMAS_S = np.array(
    [2e-4, 5e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
)
HALF_PERIOD_S = 0.5
# No drift, as the current GN table (Plot_Data/integer_pick_metrics_DOG<n>.npz)
# and the final model.
CLOCK_DRIFT = False
INITIAL_LEVER_GUESS = np.array([-12.69796149, 9.51739301, -15.0743129])
ESV_TABLE = "ESV_Tables/global_table_esv_extended.mat"
MCMC_RUN = "ar1_common30"  # FINAL_RUN of plot_tests/_figure_common.py
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "revision/analysis/gamma_sensitivity"
DOG_COLORS = {1: "#0072B2", 3: "#D55E00", 4: "#009E73"}  # plot_tests/_style.py


def gauss_newton_at_gate(dog_num, gamma, esv_table):
    """Bermuda_Data_time_bias.py (Gauss-Newton search, final fit) at gate ``gamma``.

    Returns a row dict; a search that fails at this gate is recorded with its error.
    """
    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{dog_num}.npz"))
    GPS_Coordinates, GPS_data, CDOG_data = (
        data["GPS_Coordinates"],
        data["GPS_data"],
        data["CDOG_data"],
    )
    row = {"DOG": dog_num, "gamma_s": float(gamma)}
    try:
        lever, offset, estimate, _ = gauss_newton_offset_search(
            CDOG_data,
            GPS_data,
            GPS_Coordinates,
            GPS1_TO_OTHERS,
            CDOG_REFERENCE + CDOG_GUESS_AUGMENTS[dog_num],
            INITIAL_LEVER_GUESS,
            *esv_table,
            real_data=True,
            clock_drift=CLOCK_DRIFT,
            gate=gamma,
        )
    except ValueError as error:
        row["error"] = str(error)
        return row
    transponder_coordinates = findTransponder(GPS_Coordinates, GPS1_TO_OTHERS, lever)
    # Pairs entering the final fit: final_bias_geiger pairs once at its start.
    start_times, start_esv = modelled_times(
        estimate[:3], transponder_coordinates, estimate[4], *esv_table, True
    )
    pairs_in_gate = len(
        two_pointer_index(
            offset,
            gamma,
            CDOG_data,
            GPS_data,
            start_times,
            transponder_coordinates,
            start_esv,
        )[1]
    )
    result, CDOG_full, GPS_full, _, _ = final_bias_geiger(
        estimate[:3],
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        offset,
        estimate[4],
        estimate[3],
        *esv_table,
        real_data=True,
        clock_drift=CLOCK_DRIFT,
        drift_rate=estimate[5] if CLOCK_DRIFT else 0.0,
        gate=gamma,
    )
    row.update(
        {
            "integer_second_ok": bool(np.round(offset) == OFFSET_HINTS_S[dog_num]),
            "pairs_in_gate": pairs_in_gate,
            "pairs_kept": len(CDOG_full),
            "c_sigma_cm": float(
                np.std(CDOG_full - GPS_full) * NOMINAL_SOUND_SPEED_MPS * 100
            ),
            "lever_arm_m": lever.tolist(),
            "position_ecef_m": result[:3].tolist(),
            "delta_t_s": float(offset - result[3]),
            "delta_c_m_s": float(result[4]),
        }
    )
    return row


def mcmc_pair_counts(gammas):
    """MCMC fixed pairs per DOG (after its outlier rule) of MCMC_RUN at each gate."""
    config = ms.run_config(MCMC_RUN)
    inputs = ms.load_bermuda_inputs(config)
    GPS_data, GPS_Coordinates = ms.select_epochs(
        config, inputs["GPS_data"], inputs["GPS_Coordinates"]
    )
    counts = {}
    for gamma in gammas:
        problem = ms.build_problem(
            {**config, "pair_threshold_s": float(gamma)},
            GPS_Coordinates,
            GPS_data,
            inputs["CDOG_all_data"],
            inputs["dz_array"],
            inputs["angle_array"],
            inputs["esv_matrix"],
        )
        counts[float(gamma)] = dict(
            zip(config["dogs"], problem["pair_counts"].tolist(), strict=True)
        )
    return counts


def add_changes(rows):
    """Lever, delta_t and ENU position change of each row from its DOG's reference."""
    for dog_num in DOGS:
        dog_rows = [r for r in rows if r["DOG"] == dog_num and "error" not in r]
        reference = next(r for r in dog_rows if r["gamma_s"] == PAIRING_GATE_S)
        reference_position = np.array(reference["position_ecef_m"])
        lat, lon, _ = ECEF_Geodetic(reference_position[None, :])
        rotation = enu_rotation(lat[0], lon[0])
        for row in dog_rows:
            position = np.array(row["position_ecef_m"])
            change_cm = rotation @ (position - reference_position) * 100
            row["dE_cm"], row["dN_cm"], row["dU_cm"] = change_cm.tolist()
            lever_change = np.subtract(row["lever_arm_m"], reference["lever_arm_m"])
            row["d_lever_cm"] = float(np.linalg.norm(lever_change) * 100)
            row["d_delta_t_us"] = (row["delta_t_s"] - reference["delta_t_s"]) * 1e6


def plot_sweep(rows):
    """Pairs, c*sigma_eps and changes of position, lever arm and delta_t vs gamma."""
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex=True)
    for dog_num in DOGS:
        dog_rows = [r for r in rows if r["DOG"] == dog_num and "error" not in r]
        gamma = [r["gamma_s"] for r in dog_rows]
        color = DOG_COLORS[dog_num]
        style = {"color": color, "marker": "o", "ms": 3, "lw": 1}
        axes[0, 0].plot(
            gamma, [r["pairs_kept"] for r in dog_rows], label=f"DOG {dog_num}", **style
        )
        axes[0, 0].plot(
            gamma, [r["mcmc_pairs"] for r in dog_rows], color=color, ls=":", lw=1
        )
        axes[0, 1].plot(gamma, [r["c_sigma_cm"] for r in dog_rows], **style)
        axes[0, 2].plot(gamma, [r["d_delta_t_us"] for r in dog_rows], **style)
        axes[1, 0].plot(
            gamma, [np.hypot(r["dE_cm"], r["dN_cm"]) for r in dog_rows], **style
        )
        axes[1, 1].plot(gamma, [r["dU_cm"] for r in dog_rows], **style)
        axes[1, 2].plot(gamma, [r["d_lever_cm"] for r in dog_rows], **style)

    reference = rf"from $\gamma$={PAIRING_GATE_S} s"
    axes[0, 0].set_ylabel("Pairs (solid: GN kept, dotted: MCMC)")
    axes[0, 1].set_ylabel(r"$c\sigma_\epsilon$ (cm)")
    axes[0, 2].set_ylabel(rf"$\delta t$ change {reference} ($\mu$s)")
    axes[1, 0].set_ylabel(f"Horizontal change {reference} (cm)")
    axes[1, 1].set_ylabel(f"Vertical change {reference} (cm)")
    axes[1, 2].set_ylabel(f"Lever-arm change {reference} (cm)")
    axes[0, 0].legend(frameon=False)
    for ax, letter in zip(axes.flat, "abcdef", strict=True):
        ax.set_xscale("log")
        ax.axvline(PAIRING_GATE_S, color="0.3", lw=0.8, ls="--")
        ax.axvline(HALF_PERIOD_S, color="0.3", lw=0.8, ls=":")
        ax.text(0.02, 0.95, f"({letter})", transform=ax.transAxes, va="top")
    for ax in axes[1]:
        ax.set_xlabel(r"Pairing gate $\gamma$ (s)")
    axes[0, 1].text(
        PAIRING_GATE_S, 1.01, r"$\gamma$  ½ period",
        transform=axes[0, 1].get_xaxis_transform(), ha="left", va="bottom", fontsize=7,
    )  # fmt: skip
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    esv_table = load_esv_table(gps_data_path(ESV_TABLE))
    mcmc_pairs = mcmc_pair_counts(GAMMAS_S)
    rows = []
    for dog_num in DOGS:
        for gamma in GAMMAS_S:
            row = gauss_newton_at_gate(dog_num, gamma, esv_table)
            row["mcmc_pairs"] = mcmc_pairs[float(gamma)][dog_num]
            rows.append(row)
            print(f"DOG {dog_num}, gamma {gamma:g} s: {row.get('error', 'ok')}")
    add_changes(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "gamma_sensitivity.json", "w") as handle:
        json.dump(
            {"esv_table": ESV_TABLE, "clock_drift": CLOCK_DRIFT, "mcmc_run": MCMC_RUN,
             "pairing_gate_s": PAIRING_GATE_S,
             "search_window_s": OFFSET_SEARCH_THRESHOLD_SECONDS, "rows": rows},
            handle,
            indent=2,
        )  # fmt: skip

    columns = (
        "DOG", "gamma_s", "pairs_in_gate", "pairs_kept", "mcmc_pairs", "c_sigma_cm",
        "dE_cm", "dN_cm", "dU_cm", "d_lever_cm", "d_delta_t_us", "delta_c_m_s",
    )  # fmt: skip
    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(str(r.get(name, "")) for name in columns))
    (OUTPUT_DIR / "gamma_sensitivity.csv").write_text("\n".join(lines) + "\n")

    fig = plot_sweep(rows)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"gamma_sensitivity.{ext}", dpi=300)
    print(f"Saved to {OUTPUT_DIR}")
