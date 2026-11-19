"""Posterior summaries of a Bermuda MCMC run (Paper I revision, items B2-B4).

- Summary per parameter and DOG: MAP (the highest-posterior saved draw), mean, sd
  and the 2.5/50/97.5 percentiles. DOG positions are also given in local ENU,
  relative to the submitted Table 1 position (so ENU is the change since the
  submission), and in WGS84 geodetic coordinates. A markdown table in the layout
  of Table 1, and the posterior lever arm against its drawing-based prior.
- Linearized comparison at the MAP: J^T W J (W from the per-DOG sigma), its
  condition number, the formal covariance (J^T W J)^-1 and its correlations,
  against the posterior; also with AR(1) weighting (Reviewer 4, point 2a).
- Prior sensitivity: posterior sd / prior sd for each parameter.

Outputs go to Output/MCMC_Chains/<run_name>/analysis/.
"""

import json

import numpy as np
from pymap3d import ecef2geodetic

from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from Inversion_Workflow.MCMC_Functions.MCMC_diagnostics import load_run, write_csv
from bermuda_data import gps_output_path

# Submitted Table 1: DOG positions as ECEF offsets from CDOG_reference (m), and
# the joint lever arm (body frame, m). ENU origins and comparison values.
SUBMITTED_TABLE1_AUG_M = {
    1: [-396.91, 369.80, 774.24],
    3: [826.22, -112.94, -733.06],
    4: [236.20, -1306.98, -2189.99],
}
SUBMITTED_TABLE1_JOINT_LEVER_M = [-12.87, 9.44, -13.36]
# Posterior sd / prior sd above which a parameter is reported as prior-dominated,
# and below which it is data-dominated.
PRIOR_DOMINATED_RATIO = 0.5
DATA_DOMINATED_RATIO = 0.1


def enu_rotation(lat_deg, lon_deg):
    """Rows E, N, U (3, 3) rotating ECEF differences into local ENU (WGS84 lat)."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def reported_quantities(model, offsets, CDOG_reference):
    """Linear map from theta to the reported quantities.

    Returns names, units, the matrix A (n_quantities, n_params) and the constant
    b, so that quantities = theta @ A.T + b. Covers the lever arm, each DOG's
    ECEF offset and ENU change since Table 1, Eq. 6 dt, each ESV split and their
    mean (for "common_windows": the common series, each DOG's dc per window and
    its offset, the mean over windows), drift, sigma and phi (log sigma is
    exponentiated separately), and the paper's lever_z - DOG U.
    """
    layout = model["layout"]
    n_params = len(model["names"])
    dogs = model["config"]["dogs"]
    rows, constants, names, units = [], [], [], []

    def add(name, unit, coefficients, constant=0.0):
        row = np.zeros(n_params)
        for index, value in coefficients:
            row[index] = value
        rows.append(row)
        constants.append(constant)
        names.append(name)
        units.append(unit)

    for a, axis in enumerate("xyz"):
        add(f"lever_{axis}", "m", [(layout["lever"].start + a, 1.0)])
    splits = model["num_splits"]
    if "esv_common" in layout:
        # dc of DOG j in window k = esv_offset[j] + common[k]; common has zero mean.
        common_map = ms.common_series_matrix(splits)
        common_terms = [
            [
                (layout["esv_common"].start + i, common_map[k, i])
                for i in range(splits - 1)
            ]
            for k in range(splits)
        ]
        for k in range(splits):
            add(f"common_dc_w{k}", "m/s", common_terms[k])
    for j, dog in enumerate(dogs):
        aug_start = layout["CDOG_aug"].start + 3 * j
        for a, axis in enumerate("xyz"):
            add(f"DOG{dog}_xR_{axis}", "m", [(aug_start + a, 1.0)])
        origin = CDOG_reference + np.array(SUBMITTED_TABLE1_AUG_M[dog])
        lat, lon, _ = ecef2geodetic(*origin)
        rotation = enu_rotation(lat, lon)
        offset_from_table1 = rotation @ (CDOG_reference - origin)
        for e, axis in enumerate("ENU"):
            add(
                f"DOG{dog}_{axis}_minus_Table1",
                "m",
                [(aug_start + a, rotation[e, a]) for a in range(3)],
                offset_from_table1[e],
            )
        add(
            f"DOG{dog}_dt_Eq6", "s", [(layout["time_bias"].start + j, -1.0)], offsets[j]
        )
        if "esv_common" in layout:
            offset = (layout["esv_offset"].start + j, 1.0)
            for k in range(splits):
                add(f"DOG{dog}_dc_w{k}", "m/s", [offset] + common_terms[k])
            add(f"DOG{dog}_dc_mean", "m/s", [offset])
        else:
            esv_start = layout["esv_bias"].start + splits * j
            for s in range(splits):
                add(f"DOG{dog}_dc_split{s}", "m/s", [(esv_start + s, 1.0)])
            add(
                f"DOG{dog}_dc_mean",
                "m/s",
                [(esv_start + s, 1.0 / splits) for s in range(splits)],
            )
        if "drift_rate" in layout:
            add(f"DOG{dog}_drift_rate", "s/s", [(layout["drift_rate"].start + j, 1.0)])
        if "phi" in layout:
            add(f"DOG{dog}_phi", "", [(layout["phi"].start + j, 1.0)])
        add(
            f"lever_z_minus_DOG{dog}_U",
            "m",
            [(layout["lever"].start + 2, 1.0)]
            + [(aug_start + a, -rotation[2, a]) for a in range(3)],
            -offset_from_table1[2],
        )
    return names, units, np.array(rows), np.array(constants)


def percentile_row(name, unit, draws, map_value):
    p2_5, p50, p97_5 = np.percentile(draws, [2.5, 50.0, 97.5])
    return {
        "quantity": name,
        "unit": unit,
        "MAP": map_value,
        "mean": draws.mean(),
        "sd": draws.std(ddof=1),
        "p2.5": p2_5,
        "p50": p50,
        "p97.5": p97_5,
    }


def posterior_summary(theta, logpost, model, offsets, CDOG_reference):
    """Rows of MAP, mean, sd and percentiles for every reported quantity."""
    names, units, A, b = reported_quantities(model, offsets, CDOG_reference)
    quantities = theta @ A.T + b
    best = int(np.argmax(logpost))
    rows = [
        percentile_row(n, u, quantities[:, k], quantities[best, k])
        for k, (n, u) in enumerate(zip(names, units, strict=False))
    ]

    layout = model["layout"]
    for j, dog in enumerate(model["config"]["dogs"]):
        if "log_sigma" in layout:
            sigma = np.exp(theta[:, layout["log_sigma"].start + j])
            rows.append(percentile_row(f"DOG{dog}_c_sigma", "cm", sigma, sigma[best]))
        if "atanh_phi" in layout:
            phi = np.tanh(theta[:, layout["atanh_phi"].start + j])
            rows.append(percentile_row(f"DOG{dog}_phi", "", phi, phi[best]))
        if "log_sigma_white" in layout:
            white = np.exp(theta[:, layout["log_sigma_white"].start + j])
            rows.append(
                percentile_row(f"DOG{dog}_c_sigma_white", "cm", white, white[best])
            )
        aug = (
            theta[
                :,
                layout["CDOG_aug"].start + 3 * j : layout["CDOG_aug"].start + 3 * j + 3,
            ]
            + CDOG_reference
        )
        lat, lon, height = ecef2geodetic(aug[:, 0], aug[:, 1], aug[:, 2])
        for label, unit, values in (
            ("lat", "deg", lat),
            ("lon", "deg", lon),
            ("height", "m", height),
        ):
            rows.append(percentile_row(f"DOG{dog}_{label}", unit, values, values[best]))
    return rows


def table1_markdown(rows, model, run_name):
    """Table 1 layout (posterior mean +- sd); the joint lever arm is one row."""
    value = {row["quantity"]: row for row in rows}

    def fmt(name, digits):
        row = value[name]
        return f"{row['mean']:.{digits}f} ± {row['sd']:.{digits}f}"

    has_drift = "drift_rate" in model["layout"]
    has_sigma = "log_sigma" in model["layout"]
    has_phi = "phi" in model["layout"] or "atanh_phi" in model["layout"]
    has_white = "log_sigma_white" in model["layout"]
    header = ["DOG", "xR (m)", "lat, lon (deg), height (m)", "δt (s)", "δ̄c (m/s)"]
    header += ["drift (ns/s)"] if has_drift else []
    header += ["cσε (cm)" if not has_white else "cσ AR(1) (cm)"] if has_sigma else []
    header += ["φ"] if has_phi else []
    header += ["cσ white (cm)"] if has_white else []
    lines = [
        f"Posterior mean ± sd, run `{run_name}`.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for dog in model["config"]["dogs"]:
        xR = ", ".join(fmt(f"DOG{dog}_xR_{a}", 3) for a in "xyz")
        geodetic = (
            f"{value[f'DOG{dog}_lat']['mean']:.7f}, "
            f"{value[f'DOG{dog}_lon']['mean']:.7f}, {fmt(f'DOG{dog}_height', 3)}"
        )
        cells = [
            str(dog),
            f"[{xR}]",
            geodetic,
            fmt(f"DOG{dog}_dt_Eq6", 6),
            fmt(f"DOG{dog}_dc_mean", 4),
        ]
        if has_drift:
            row = value[f"DOG{dog}_drift_rate"]
            cells.append(f"{1e9 * row['mean']:.2f} ± {1e9 * row['sd']:.2f}")
        if has_sigma:
            cells.append(fmt(f"DOG{dog}_c_sigma", 2))
        if has_phi:
            cells.append(fmt(f"DOG{dog}_phi", 3))
        if has_white:
            cells.append(fmt(f"DOG{dog}_c_sigma_white", 2))
        lines.append("| " + " | ".join(cells) + " |")
    lever = ", ".join(fmt(f"lever_{a}", 3) for a in "xyz")
    lines.append("| Joint p (m) | " + f"[{lever}]" + " |" * (len(header) - 1))
    lines += [
        "",
        "xR: ECEF offset from CDOG_reference. δt: Eq. 6 convention "
        "(offset − time_bias). δ̄c: mean over the ESV splits or windows. cσε: sampled "
        "residual sd in range at 1515 m/s.",
    ]
    return "\n".join(lines) + "\n"


def lever_against_prior(rows, model):
    """Posterior lever arm against the drawing-based prior, per component."""
    value = {row["quantity"]: row for row in rows}
    config = model["config"]
    out = []
    for a, axis in enumerate("xyz"):
        row = value[f"lever_{axis}"]
        center = config["prior_lever_center_m"][a]
        sd = config["prior_lever_sd_m"][a]
        out.append(
            {
                "component": axis,
                "prior_center_m": center,
                "prior_sd_m": sd,
                "posterior_mean_m": row["mean"],
                "posterior_sd_m": row["sd"],
                "posterior_p2.5_m": row["p2.5"],
                "posterior_p97.5_m": row["p97.5"],
                "shift_in_prior_sd": (row["mean"] - center) / sd,
                "posterior_sd_over_prior_sd": row["sd"] / sd,
                "submitted_Table1_joint_m": SUBMITTED_TABLE1_JOINT_LEVER_M[a],
            }
        )
    return out


def prior_sensitivity(theta, model, offsets, CDOG_reference):
    """Posterior sd / prior sd for each Gaussian parameter and each DOG's ENU."""
    names = model["names"]
    rows = []
    for k, name in enumerate(names):
        if not model["gaussian"][k]:
            continue
        rows.append((name, theta[:, k].std(ddof=1), model["prior_sd"][k]))
    q_names, _, A, b = reported_quantities(model, offsets, CDOG_reference)
    enu = [k for k, name in enumerate(q_names) if "_minus_Table1" in name]
    for k in enu:
        # The DOG position prior is isotropic, so each ENU axis has the same sd.
        rows.append(
            (
                q_names[k],
                (theta @ A[k]).std(ddof=1),
                model["config"]["prior_CDOG_aug_sd_m"],
            )
        )
    out = []
    for name, posterior_sd, prior_sd in rows:
        ratio = posterior_sd / prior_sd
        if ratio > PRIOR_DOMINATED_RATIO:
            verdict = "prior-dominated"
        elif ratio < DATA_DOMINATED_RATIO:
            verdict = "data-dominated"
        else:
            verdict = "data and prior"
        out.append(
            {
                "parameter": name,
                "posterior_sd": posterior_sd,
                "prior_sd": prior_sd,
                "ratio": ratio,
                "verdict": verdict,
            }
        )
    return sorted(out, key=lambda row: -row["ratio"])


def linearized_comparison(theta, logpost, model, problem, offsets, CDOG_reference):
    """Formal (Gauss-Newton) covariance at the MAP against the posterior.

    Returns standard-deviation rows, correlation rows for the pairs discussed in
    Sec. 6.2, and the condition numbers. The formal covariance has no prior;
    "iid" weights residuals by the MAP marginal sd of each DOG; "model" whitens
    them with the run's own noise model at the MAP (AR(1), or AR(1) plus white
    noise; equal to "iid" for the iid modes).
    """
    best = int(np.argmax(logpost))
    theta_map = theta[best]
    n_geo = model["geometric_end"]
    jacobian = ms.residual_jacobian(theta_map, model, problem)
    sigma, phi = ms.noise_parameters(theta_map, model)
    sigma_white = ms.white_noise_sd(theta_map, model)
    marginal_sd = sigma if sigma_white is None else np.sqrt(sigma**2 + sigma_white**2)

    # An ESV split with no fixed pairs (DOG 4 has gaps) has a zero Jacobian
    # column: the data say nothing about it, so the prior-free formal covariance
    # is computed over the data-constrained columns only, and any quantity that
    # loads on an unconstrained column gets a NaN formal covariance.
    constrained = np.any(jacobian != 0.0, axis=0)
    jacobian = jacobian[:, constrained]
    J_iid, _ = ms.whiten_ar1(
        jacobian,
        problem["segment_start"],
        problem["pair_dog"],
        marginal_sd,
        np.zeros_like(phi),
    )
    normal_matrix = J_iid.T @ J_iid
    scaling = 1.0 / np.sqrt(np.diag(normal_matrix))
    condition = {
        "cond_JTWJ_raw_units": np.linalg.cond(normal_matrix),
        "cond_JTWJ_unit_columns": np.linalg.cond(
            normal_matrix * np.outer(scaling, scaling)
        ),
        "unconstrained_parameters": [
            model["names"][k] for k in np.flatnonzero(~constrained)
        ],
    }
    J_model, _ = ms.whiten_residuals(jacobian, problem, sigma, phi, sigma_white)
    formal = {
        "iid": np.linalg.inv(normal_matrix),
        "model": np.linalg.inv(J_model.T @ J_model),
    }

    q_names, units, A, _ = reported_quantities(model, offsets, CDOG_reference)
    A_geo = A[:, :n_geo]
    linear = np.flatnonzero(np.all(A[:, n_geo:] == 0.0, axis=1))
    posterior_cov = np.cov((theta @ A.T)[:, linear], rowvar=False)
    covariances = {"posterior": posterior_cov}
    A_formal = A_geo[linear][:, constrained[:n_geo]]
    unconstrained_quantity = np.any(A_geo[linear][:, ~constrained[:n_geo]], axis=1)
    for key, cov in formal.items():
        cov = A_formal @ cov @ A_formal.T
        cov[unconstrained_quantity, :] = np.nan
        cov[:, unconstrained_quantity] = np.nan
        covariances[f"formal_{key}"] = cov

    sd_rows = []
    for i, k in enumerate(linear):
        row = {"quantity": q_names[k], "unit": units[k]}
        for key, cov in covariances.items():
            row[f"{key}_sd"] = np.sqrt(cov[i, i])
        row["posterior_over_formal_iid"] = row["posterior_sd"] / row["formal_iid_sd"]
        sd_rows.append(row)

    position = {q_names[k]: i for i, k in enumerate(linear)}
    pairs = []
    for dog in model["config"]["dogs"]:
        pairs += [
            (f"DOG{dog}_dc_mean", f"DOG{dog}_dt_Eq6"),
            ("lever_z", f"DOG{dog}_U_minus_Table1"),
            (f"lever_z_minus_DOG{dog}_U", f"DOG{dog}_dt_Eq6"),
            (f"lever_z_minus_DOG{dog}_U", f"DOG{dog}_dc_mean"),
            (f"DOG{dog}_U_minus_Table1", f"DOG{dog}_dc_mean"),
        ]
    correlation_rows = []
    for first, second in pairs:
        i, k = position[first], position[second]
        row = {"pair": f"{first} vs {second}"}
        for key, cov in covariances.items():
            row[key] = cov[i, k] / np.sqrt(cov[i, i] * cov[k, k])
        correlation_rows.append(row)
    return sd_rows, correlation_rows, condition


def common_dc_rows(theta, model, problem):
    """Per dc window: GPS-time span, pairs per DOG, common series and DOG dc.

    Times are hours from the first window edge (the first paired GPS time).
    dc values are posterior mean and 2.5/97.5 percentiles in m/s.
    """
    edges = np.array(model["config"]["esv_window_edges_s"])
    hours = (edges - edges[0]) / 3600.0
    n_windows = model["num_splits"]
    common = theta[:, model["layout"]["esv_common"]] @ (
        ms.common_series_matrix(n_windows).T
    )
    dc = ms.esv_bias_matrix(theta, model)
    rows = []
    for k in range(n_windows):
        row = {
            "window": k,
            "start_h": hours[k],
            "end_h": hours[k + 1],
            "common_mean": common[:, k].mean(),
            "common_p2.5": np.percentile(common[:, k], 2.5),
            "common_p97.5": np.percentile(common[:, k], 97.5),
        }
        for j, dog in enumerate(model["config"]["dogs"]):
            in_window = problem["pair_window"][problem["pair_slices"][j]] == k
            row[f"DOG{dog}_pairs"] = int(in_window.sum())
            row[f"DOG{dog}_dc_mean"] = dc[:, j, k].mean()
            row[f"DOG{dog}_dc_sd"] = dc[:, j, k].std(ddof=1)
        rows.append(row)
    return rows


def plot_common_dc(rows, model, run_name, out_path):
    """dc(t): the common series with its 95% band, and each DOG's dc per window.

    A DOG's dc is drawn only in windows where it has pairs; elsewhere it would be
    its offset plus the common series, not a measurement.
    """
    import matplotlib.pyplot as plt

    start = np.array([row["start_h"] for row in rows])
    end = np.array([row["end_h"] for row in rows])
    centre = 0.5 * (start + end)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for row in rows:
        top.fill_between(
            [row["start_h"], row["end_h"]],
            1e3 * row["common_p2.5"],
            1e3 * row["common_p97.5"],
            color="0.8",
            linewidth=0,
        )
        top.plot([row["start_h"], row["end_h"]], [1e3 * row["common_mean"]] * 2, "k-")
    top.axhline(0.0, color="0.5", linewidth=0.5)
    # No "%" or bare "_" in the title: they break LaTeX text rendering.
    name = run_name.replace("_", r"\_") if plt.rcParams["text.usetex"] else run_name
    top.set(
        ylabel="common dc (mm/s)",
        title=f"{name}: common dc(t), zero mean, 2.5-97.5 percentile band",
    )
    for dog in model["config"]["dogs"]:
        has_pairs = np.array([row[f"DOG{dog}_pairs"] > 0 for row in rows])
        mean = np.array([row[f"DOG{dog}_dc_mean"] for row in rows])
        sd = np.array([row[f"DOG{dog}_dc_sd"] for row in rows])
        bottom.errorbar(
            centre[has_pairs],
            mean[has_pairs],
            sd[has_pairs],
            fmt="o",
            markersize=3,
            label=f"DOG {dog}",
        )
    bottom.set(
        xlabel="hours from first paired GPS time",
        ylabel="dc (m/s), mean ± sd",
    )
    bottom.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize_run(run_name, linearized=True):
    """Write the B2-B4 tables of a run and print the key numbers."""
    chains = load_run(run_name)
    config = json.loads(str(chains[0]["config_json"]))
    model = ms.build_model(config)
    if chains[0]["fixed_phi"].size:
        model["fixed_phi"] = chains[0]["fixed_phi"]
    theta = np.concatenate([chain["theta_samples"] for chain in chains])
    logpost = np.concatenate([chain["logpost"] for chain in chains])
    offsets = chains[0]["offsets_s"]
    CDOG_reference = chains[0]["CDOG_reference"]
    out_dir = gps_output_path("MCMC_Chains", run_name, "analysis")
    out_dir.mkdir(exist_ok=True)

    rows = posterior_summary(theta, logpost, model, offsets, CDOG_reference)
    write_csv(rows, out_dir / "posterior_summary.csv")
    (out_dir / "table1_posterior.md").write_text(table1_markdown(rows, model, run_name))
    lever = lever_against_prior(rows, model)
    write_csv(lever, out_dir / "lever_vs_prior.csv")
    sensitivity = prior_sensitivity(theta, model, offsets, CDOG_reference)
    write_csv(sensitivity, out_dir / "prior_sensitivity.csv")
    common_windows = "esv_common" in model["layout"]
    problem = None
    if linearized or common_windows:
        _, problem = ms.setup_bermuda(config)
    if common_windows:
        dc_rows = common_dc_rows(theta, model, problem)
        write_csv(dc_rows, out_dir / "dc_windows.csv")
        plot_common_dc(dc_rows, model, run_name, out_dir / "dc_t.pdf")

    print(f"\nRun {run_name}: {theta.shape[0]} draws from {len(chains)} chains")
    print((out_dir / "table1_posterior.md").read_text())
    print("Lever arm vs drawing-based prior (m):")
    for row in lever:
        print(
            f"  {row['component']}: prior {row['prior_center_m']:.2f} ± "
            f"{row['prior_sd_m']:.2f}, posterior {row['posterior_mean_m']:.3f} ± "
            f"{row['posterior_sd_m']:.3f} (shift {row['shift_in_prior_sd']:+.2f} "
            f"prior sd, sd ratio {row['posterior_sd_over_prior_sd']:.3f}; "
            f"submitted {row['submitted_Table1_joint_m']:.2f})"
        )
    print("Prior sensitivity (largest posterior/prior sd ratios):")
    for row in sensitivity[:6]:
        print(f"  {row['parameter']:28s} {row['ratio']:.4f}  {row['verdict']}")

    if linearized:
        sd_rows, correlation_rows, condition = linearized_comparison(
            theta, logpost, model, problem, offsets, CDOG_reference
        )
        write_csv(sd_rows, out_dir / "linearized_sd.csv")
        write_csv(correlation_rows, out_dir / "linearized_correlations.csv")
        (out_dir / "linearized_condition.json").write_text(
            json.dumps(condition, indent=2)
        )
        print(
            f"cond(J^T W J): {condition['cond_JTWJ_raw_units']:.3g} raw units, "
            f"{condition['cond_JTWJ_unit_columns']:.3g} with unit columns"
        )
        if condition["unconstrained_parameters"]:
            print(
                "No data constrain (left out of the formal covariance): "
                + ", ".join(condition["unconstrained_parameters"])
            )
        print("Correlations: posterior / formal iid / formal (run's noise model)")
        for row in correlation_rows:
            print(
                f"  {row['pair']:45s} {row['posterior']:+.3f} "
                f"{row['formal_iid']:+.3f} {row['formal_model']:+.3f}"
            )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    RUN_NAMES = [
        "ar1",
        "ar1_common15",
        "ar1_white",
        "ar1_white_common30",
        "iid",
    ]

    LINEARIZED = True

    for run_name in RUN_NAMES:
        summarize_run(run_name, linearized=LINEARIZED)
