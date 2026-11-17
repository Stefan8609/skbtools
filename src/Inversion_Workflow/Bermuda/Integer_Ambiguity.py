"""Test whether the integer-second part of delta_t is uniquely identified.

Reviewer 3, M1. Starting from the best solution for one DOG, delta_t is shifted by
k whole seconds and the emission/arrival pairing is re-derived from scratch with
``two_pointer_index`` (paper Eq. A20, gate gamma). Travel-time residuals are then
evaluated (a) with the best-solution parameters held fixed and (b) after
re-fitting position, ESV bias and fine delta_t with ``final_bias_geiger``.

Run from the repository root with ``PYTHONPATH=src``. Outputs go to
``revision/analysis/integer_ambiguity/`` (gitignored).
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from bermuda_data import gps_data_path
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline import two_pointer_index
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    initial_bias_geiger,
    residual_inliers,
)

# -------------------------
# Config
# -------------------------
DOG_num = 3
N_SHIFT = 5
GATE_SECONDS = 0.4  # gamma of paper Eq. A20
# Paper I (Table 1) used the table from before the 2026-10-08 ray-tracing fix.
# Switch to "global_table_esv_extended.mat" to use the current table.
ESV_TABLE = "ESV_Tables/global_table_esv_extended_before_ray_fix_20261008.mat"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "revision/analysis/integer_ambiguity"

# Same constants as Bermuda_Data_time_bias.py. Table 1 lever arms and published
# values (delta_t in s, c*sigma_eps in cm with c = 1515 m/s) are used for the check.
CDOG_BASE = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
GPS1_TO_OTHERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)
START_AUGMENT = {
    1: [-398.16, 371.90, 773.02],
    3: [825.182985, -111.05670221, -734.10011698],
    4: [236.428385, -1307.98390221, -2189.21991698],
}
TABLE1_LEVER = {
    1: [-12.77, 9.58, -12.07],
    3: [-12.89, 9.45, -12.28],
    4: [-13.36, 8.65, -12.95],
}
TABLE1_DELTA_T = {1: 1866.016, 3: 3175.017, 4: 1939.018}
TABLE1_C_SIGMA_CM = {1: 22.80, 3: 23.75, 4: 21.99}
PAPER_SOUND_SPEED = 1515.0  # m/s, used only for the paper's c*sigma convention


def load_dog(dog_num, esv_table_file=ESV_TABLE):
    """Load GPS coordinates, GPS epochs, DOG arrivals and the ESV table."""
    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{dog_num}.npz"))
    table = sio.loadmat(gps_data_path(esv_table_file))
    esv_table = (
        table["distance"].flatten(),
        table["angle"].flatten(),
        table["matrice"],
    )
    return data["GPS_Coordinates"], data["GPS_data"], data["CDOG_data"], esv_table


def best_solution(dog_num, CDOG_data, GPS_data, transponder_coordinates, esv_table):
    """Geiger inversion (initial then final) as in Bermuda_Data_time_bias.py.

    Returns (estimate, offset, time_bias, esv_bias) where estimate is
    [x, y, z, time_bias, esv_bias] and delta_t = offset - time_bias.
    """
    dz_array, angle_array, esv_matrix = esv_table
    guess = CDOG_BASE + np.array(START_AUGMENT[dog_num])
    estimate, offset = initial_bias_geiger(
        guess,
        CDOG_data,
        GPS_data,
        transponder_coordinates,
        dz_array,
        angle_array,
        esv_matrix,
        real_data=True,
        initial_offset=float(np.round(TABLE1_DELTA_T[dog_num])),
    )
    estimate, CDOG_full, GPS_full, _, _ = final_bias_geiger(
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
        real_data=True,
    )
    return estimate, offset, estimate[3], estimate[4], CDOG_full, GPS_full


def radial_velocity(transponder_coordinates, GPS_data, dog_position, max_span_s=4.0):
    """Line-of-sight speed (m/s) of the transponder relative to the DOG.

    Central difference of the modeled range between the neighbouring emission
    epochs; NaN where those epochs are more than ``max_span_s`` apart (data gaps).
    """
    slant_range = np.linalg.norm(transponder_coordinates - dog_position, axis=1)
    velocity = np.full(len(GPS_data), np.nan)
    span = GPS_data[2:] - GPS_data[:-2]
    velocity[1:-1] = (slant_range[2:] - slant_range[:-2]) / span
    velocity[1:-1][span > max_span_s] = np.nan
    return velocity


def shifted_pairing_residuals(
    offset, gate, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv
):
    """Re-derive the pairing at ``offset`` and return the travel-time residuals.

    Residual = observed - modeled travel time (s). The returned dict holds the
    emission index of each pair, the residuals, the outlier mask from
    ``residual_inliers`` and the pair ESV.
    """
    CDOG_clock, CDOG_full, GPS_clock, GPS_full, _, esv_full = two_pointer_index(
        offset,
        gate,
        CDOG_data,
        GPS_data,
        travel_times,
        transponder_coordinates,
        esv,
    )
    # GPS_clock is a copy of GPS_data + travel_times, so the match is exact.
    emission_index = np.searchsorted(GPS_data + travel_times, GPS_clock)
    residuals = CDOG_full - GPS_full
    return {
        "emission_index": emission_index,
        "clock": GPS_clock,
        "residuals": residuals,
        "inliers": residual_inliers(residuals),
        "esv": esv_full,
    }


def rmse(residuals):
    return float(np.sqrt(np.mean(residuals**2)))


def fixed_parameter_row(k, offset, time_bias, gate, CDOG_data, GPS_data, model):
    """Variant (a): best-solution parameters, pairing re-derived at delta_t_k."""
    pairing = shifted_pairing_residuals(offset + k, gate, CDOG_data, GPS_data, *model)
    inliers = pairing["inliers"]
    return pairing, {
        "pairs": len(pairing["residuals"]),
        "pairs_after_rejection": int(inliers.sum()),
        "rmse_s": rmse(pairing["residuals"][inliers]),
        "rmse_no_rejection_s": rmse(pairing["residuals"]),
        "esv_mean": float(pairing["esv"][inliers].mean()),
    }


def mechanism_fit(pairing, velocity):
    """Correlation and OLS slope (s) of range residual (m) vs radial velocity."""
    inliers = pairing["inliers"]
    v = velocity[pairing["emission_index"]][inliers]
    r = (pairing["residuals"] * pairing["esv"])[inliers]
    keep = np.isfinite(v)
    v, r = v[keep], r[keep]
    slope, intercept = np.polyfit(v, r, 1)
    return {
        "n": int(len(v)),
        "correlation": float(np.corrcoef(v, r)[0, 1]),
        "slope_s": float(slope),
        "intercept_m": float(intercept),
        "v": v,
        "r": r,
    }


def run_analysis(dog_num=DOG_num, n_shift=N_SHIFT, gate=GATE_SECONDS):
    GPS_Coordinates, GPS_data, CDOG_data, esv_table = load_dog(dog_num)
    dz_array, angle_array, esv_matrix = esv_table
    transponder_coordinates = findTransponder(
        GPS_Coordinates, GPS1_TO_OTHERS, np.array(TABLE1_LEVER[dog_num])
    )

    estimate, offset, time_bias, esv_bias, CDOG_full, GPS_full = best_solution(
        dog_num, CDOG_data, GPS_data, transponder_coordinates, esv_table
    )
    dog_position = estimate[:3]
    delta_t = offset - time_bias
    c_sigma_cm = np.std(CDOG_full - GPS_full) * PAPER_SOUND_SPEED * 100
    print(
        f"DOG {dog_num}: delta_t = {delta_t:.6f} s, delta_c = {esv_bias:.4f} m/s, "
        f"c*sigma = {c_sigma_cm:.2f} cm over {len(CDOG_full)} pairs "
        f"(Table 1: {TABLE1_DELTA_T[dog_num]} s, {TABLE1_C_SIGMA_CM[dog_num]} cm)"
    )

    # Model travel times from the transponder at each emission epoch.
    travel_times, esv = calculateTimesRayTracing_Bias_Real(
        dog_position,
        transponder_coordinates,
        esv_bias,
        dz_array,
        angle_array,
        esv_matrix,
    )
    # Fold the fine delta_t into the emission epochs as final_bias_geiger does.
    emission_epochs = GPS_data - time_bias
    model = (travel_times, transponder_coordinates, esv)
    velocity = radial_velocity(transponder_coordinates, GPS_data, dog_position)

    rows, pairings = [], {}
    for k in range(-n_shift, n_shift + 1):
        pairing, row = fixed_parameter_row(
            k, offset, time_bias, gate, CDOG_data, emission_epochs, model
        )
        row["k"] = k
        row["delta_t_s"] = delta_t + k
        row["fit"] = {}
        # (b) re-fit starting from the best solution at the shifted offset
        try:
            refit, refit_CDOG, refit_GPS, _, _ = final_bias_geiger(
                dog_position,
                CDOG_data,
                GPS_data,
                transponder_coordinates,
                offset + k,
                esv_bias,
                time_bias,
                dz_array,
                angle_array,
                esv_matrix,
                real_data=True,
            )
            row["refit_pairs_after_rejection"] = len(refit_CDOG)
            row["refit_rmse_s"] = rmse(refit_CDOG - refit_GPS)
            row["refit_dog_shift_m"] = float(np.linalg.norm(refit[:3] - dog_position))
            row["refit_delta_c_m_s"] = float(refit[4])
            row["refit_delta_t_s"] = float(offset + k - refit[3])
        except Exception as error:  # a singular fit is itself a result
            row["refit_error"] = f"{type(error).__name__}: {error}"
            row["refit_rmse_s"] = np.nan
            row["refit_pairs_after_rejection"] = 0
        rows.append(row)
        pairings[k] = pairing

    base = next(r for r in rows if r["k"] == 0)
    for row in rows:
        row["rmse_us"] = row["rmse_s"] * 1e6
        row["rmse_cm"] = row["rmse_s"] * row["esv_mean"] * 100
        row["rmse_ratio"] = row["rmse_s"] / base["rmse_s"]
        row["refit_rmse_us"] = row["refit_rmse_s"] * 1e6
        row["refit_rmse_cm"] = row["refit_rmse_s"] * base["esv_mean"] * 100
        row["refit_rmse_ratio"] = row["refit_rmse_s"] / base["refit_rmse_s"]

    mechanism = {k: mechanism_fit(pairings[k], velocity) for k in (-1, 1)}
    for k, fit in mechanism.items():
        rows[[r["k"] for r in rows].index(k)]["fit"] = {
            name: fit[name] for name in ("n", "correlation", "slope_s", "intercept_m")
        }
    summary = {
        "DOG": dog_num,
        "esv_table": ESV_TABLE,
        "gate_s": gate,
        "delta_t_best_s": float(delta_t),
        "delta_c_best_m_s": float(esv_bias),
        "dog_position_ecef_m": dog_position.tolist(),
        "lever_arm_m": TABLE1_LEVER[dog_num],
        "c_sigma_cm_paper_convention": float(c_sigma_cm),
        "c_sigma_cm_published": TABLE1_C_SIGMA_CM[dog_num],
        "rows": rows,
    }
    return summary, pairings, mechanism, velocity


if __name__ == "__main__":
    summary, pairings, mechanism, velocity = run_analysis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / f"integer_ambiguity_DOG{DOG_num}.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    header = (
        "k,M_k,M_k_after_rejection,rmse_us,rmse_cm,ratio,"
        "refit_M_k,refit_rmse_us,refit_rmse_cm,refit_ratio,corr,slope_s"
    )
    print(header)
    lines = [header]
    for r in summary["rows"]:
        fit = r["fit"]
        line = (
            f"{r['k']},{r['pairs']},{r['pairs_after_rejection']},"
            f"{r['rmse_us']:.2f},{r['rmse_cm']:.2f},{r['rmse_ratio']:.3f},"
            f"{r['refit_pairs_after_rejection']},{r['refit_rmse_us']:.2f},"
            f"{r['refit_rmse_cm']:.2f},{r['refit_rmse_ratio']:.3f},"
            f"{fit.get('correlation', float('nan')):.4f},"
            f"{fit.get('slope_s', float('nan')):.4f}"
        )
        lines.append(line)
        print(line)
    (OUTPUT_DIR / f"integer_ambiguity_DOG{DOG_num}.csv").write_text("\n".join(lines))

    np.savez_compressed(
        OUTPUT_DIR / f"integer_ambiguity_DOG{DOG_num}_pairs.npz",
        **{
            f"k{k}_{name}": pairings[k][name]
            for k in (-1, 0, 1)
            for name in ("clock", "residuals", "inliers", "esv")
        },
    )

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "mathtext.fontset": "cm",
        }
    )
    rows = summary["rows"]
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.semilogy(ks, [r["rmse_ratio"] for r in rows], "o-", label="(a) fixed parameters")
    ax.semilogy(ks, [r["refit_rmse_ratio"] for r in rows], "s--", label="(b) re-fit")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"Integer shift $k$ (s), $\delta t_k=\delta t_{best}+k$")
    ax.set_ylabel(r"RMSE$(k)/$RMSE$(0)$")
    ax.legend()
    ax.set_title("(i)")

    ax = axes[0, 1]
    for k, color in ((0, "k"), (-1, "C0"), (1, "C3")):
        p = pairings[k]
        keep = p["inliers"]
        ax.plot(
            p["clock"][keep] / 3600,
            p["residuals"][keep] * p["esv"][keep] * 100,
            ".",
            ms=1,
            color=color,
            label=rf"$k={k}$",
        )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(r"$c\,\epsilon$ (cm)")
    ax.legend(markerscale=8)
    ax.set_title("(ii)")

    for ax, k in zip(axes[1], (-1, 1)):
        fit = mechanism[k]
        ax.plot(fit["v"], fit["r"], ".", ms=1, color="C0" if k < 0 else "C3")
        vv = np.array([np.nanmin(fit["v"]), np.nanmax(fit["v"])])
        ax.plot(vv, fit["slope_s"] * vv + fit["intercept_m"], "k-", lw=1)
        ax.set_xlabel("Radial velocity of vessel relative to DOG (m/s)")
        ax.set_ylabel(r"$c\,\epsilon$ (m)")
        ax.set_title(
            rf"(iii) $k={k}$: $r={fit['correlation']:.2f}$, "
            rf"slope $={fit['slope_s']:.2f}$ s"
        )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"integer_ambiguity_DOG{DOG_num}.{ext}", dpi=300)
