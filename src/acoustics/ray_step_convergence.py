"""Accuracy of the ray-integration step behind the ESV table.

``ray_tracing`` integrates layer by layer, so its step is the profile spacing (0.10 m
in the Bermuda table). The cast is thinned to steps from 5.2 mm (the reference) to
10 m and traced for a few (dz, elevation) pairs; the range and travel-time errors are
reported at a fixed launch angle and at fixed geometry (what the table stores).

Run from the repository root with ``PYTHONPATH=src``. Outputs go to
``revision/analysis/ray_step/`` (gitignored).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustics.ray_tracing import ray_trace_locate, ray_tracing
from bermuda_data import PROJECT_ROOT, gps_data_path

# Config
TRANSDUCER_DEPTH_M = 2.0  # z_a, as in ESV_table_Bermuda.construct_esv
PROFILE_BOTTOM_M = 5350.0  # production extension depth
DEEP_GRADIENT_WINDOW_M = 10.0
PRODUCTION_STRIDE = 20  # ESV_table_Bermuda: cast[::20]
REFERENCE_STRIDE = 1
STRIDES = (1, 4, 20, 50, 200, 500, 1000, 2000)  # cast samples are 5.225 mm apart
# dz = z_b - z_a: table edges (5150, 5300 m) and the DOG 1/3/4 range (5199-5226 m)
DZ_M = (5150.0, 5200.0, 5225.0, 5300.0)
# Bermuda data span 22.9-89.8 deg; the table starts at 20 deg
ELEVATION_DEG = (20.0, 23.0, 30.0, 45.0, 60.0, 75.0, 90.0)
TIMING_NOISE_S = 20e-6  # timing noise of the synthetics
INTERPOLATION_RMS_S = 0.3e-6  # bilinear lookup error
OUTPUT_DIR = PROJECT_ROOT / "revision" / "analysis" / "ray_step"


def thinned_profile(stride):
    """Cast 2 thinned by ``stride`` and extended linearly, as ESV_table_Bermuda.

    The deep gradient is fitted on the production profile for every stride, so the
    extension below the cast is the same and only the step changes (a 10 m window
    holds too few samples of the coarse profiles for a fit).
    """
    depth = np.genfromtxt(gps_data_path("SVP_Data/depth_cast2_smoothed.txt"))
    cz = np.genfromtxt(gps_data_path("SVP_Data/cz_cast2_smoothed.txt"))
    depth_prod = depth[::PRODUCTION_STRIDE]
    deep = depth_prod >= depth_prod[-1] - DEEP_GRADIENT_WINDOW_M
    slope = np.polyfit(depth_prod[deep], cz[::PRODUCTION_STRIDE][deep], 1)[0]
    depth, cz = depth[::stride], cz[::stride]
    spacing = float(np.median(np.diff(depth)))
    extra = np.arange(
        np.nextafter(depth[-1], np.inf), PROFILE_BOTTOM_M + 0.5 * spacing, spacing
    )
    cz = np.concatenate([cz, cz[-1] + slope * (extra - depth[-1])])
    depth = np.concatenate([depth, extra])
    return np.ascontiguousarray(depth), np.ascontiguousarray(cz), spacing


def trace_at_geometry(dz, elevation_deg, depth, cz):
    """Launch angle (deg), T (s) at the exact range, and the bisection floor (s)."""
    z_b = TRANSDUCER_DEPTH_M + dz
    x_target = dz / np.tan(np.radians(elevation_deg))
    launch_deg = ray_trace_locate(TRANSDUCER_DEPTH_M, z_b, x_target, depth, cz)
    x_found, _, travel_time = ray_tracing(
        launch_deg, TRANSDUCER_DEPTH_M, z_b, depth, cz
    )
    # nu = dT/dx at fixed end depths (theta_0 from the vertical, c at z_a)
    nu = np.sin(np.radians(90.0 - launch_deg)) / np.interp(
        TRANSDUCER_DEPTH_M, depth, cz
    )
    tolerance_floor = nu * (x_target - x_found)
    return launch_deg, travel_time + tolerance_floor, tolerance_floor


def run():
    """Trace every step, dz and elevation; return the rows of the comparison."""
    profiles = {stride: thinned_profile(stride) for stride in STRIDES}
    depth_ref, cz_ref, _ = profiles[REFERENCE_STRIDE]
    rows = []
    for dz in DZ_M:
        z_b = TRANSDUCER_DEPTH_M + dz
        for elevation in ELEVATION_DEG:
            launch_ref, t_ref, _ = trace_at_geometry(dz, elevation, depth_ref, cz_ref)
            x_ref, _, t_ref_launch = ray_tracing(
                launch_ref, TRANSDUCER_DEPTH_M, z_b, depth_ref, cz_ref
            )
            slant_m = dz / np.sin(np.radians(elevation))
            for stride, (depth, cz, step) in profiles.items():
                x_h, _, t_h_launch = ray_tracing(
                    launch_ref, TRANSDUCER_DEPTH_M, z_b, depth, cz
                )
                _, t_h, floor = trace_at_geometry(dz, elevation, depth, cz)
                rows.append(
                    {
                        "dz_m": dz,
                        "elevation_deg": elevation,
                        "stride": stride,
                        "step_m": step,
                        "launch_elevation_ref_deg": launch_ref,
                        "slant_range_m": slant_m,
                        "travel_time_ref_s": t_ref,
                        # fixed launch angle: the integrals alone
                        "dx_fixed_launch_mm": (x_h - x_ref) * 1e3,
                        "dT_fixed_launch_us": (t_h_launch - t_ref_launch) * 1e6,
                        # fixed geometry: the tabulated quantity
                        "dT_fixed_geometry_us": (t_h - t_ref) * 1e6,
                        "dESV_mm_per_s": (slant_m / t_h - slant_m / t_ref) * 1e3,
                        "bisection_floor_us": floor * 1e6,
                    }
                )
    return rows


def summarise(rows):
    """Max |difference| over all geometries, per step."""
    summary = {}
    for stride in STRIDES:
        sel = [r for r in rows if r["stride"] == stride]
        summary[f"{sel[0]['step_m']:.4f} m (stride {stride})"] = {
            key: float(np.max(np.abs([r[key] for r in sel])))
            for key in (
                "dx_fixed_launch_mm",
                "dT_fixed_launch_us",
                "dT_fixed_geometry_us",
                "dESV_mm_per_s",
                "bisection_floor_us",
            )
        }
    return summary


def plot(rows, path):
    """Schematic of the layer sum, convergence with step, and errors by angle."""
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(170 / 25.4, 140 / 25.4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ELEVATION_DEG)))
    steps = {r["stride"]: r["step_m"] for r in rows}
    h_prod, h_1m = steps[PRODUCTION_STRIDE], steps[200]

    # (a) Layer sum of ray_tracing: one straight segment per layer
    ax = axes[0, 0]
    layer_z = np.arange(0, 5)
    c_layer = np.array([1.00, 0.93, 0.86, 0.80])  # sound speed falling with depth
    sin0 = np.sin(np.radians(55))
    x = [0.0]
    for c in c_layer:
        theta = np.arcsin(c * sin0)
        x.append(x[-1] + np.tan(theta))
    for z in layer_z:
        ax.axhline(z, color="0.75", lw=0.6)
    ax.plot(x, layer_z, "k-o", ms=3)
    ax.plot([0, x[-1]], [0, layer_z[-1]], "k--", lw=0.8)
    ax.annotate("", (0, 1.2), (0, 0), arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.text(0.06, 0.75, r"$\theta_0$", ha="left")
    ax.text(x[1] + 0.08, 0.75, r"$\theta_1$", ha="left")
    ax.text(x[-1] * 0.5 - 0.1, 2.1, r"$S$", ha="right")
    ax.text(x[-1] + 0.12, 3.5, r"$h$", va="center")
    ax.annotate(
        "", (x[-1] + 0.05, 3), (x[-1] + 0.05, 4), arrowprops={"arrowstyle": "<->"}
    )
    ax.text(-0.15, 0, r"$z_a$ (transducer, 2 m)", ha="right", va="center")
    ax.text(-0.15, 4, r"$z_b = z_a + dz$ (DOG)", ha="right", va="center")
    ax.text(
        -0.15,
        2.0,
        "layer $i$ = one step $h$\nbetween SVP samples:\n"
        r"$\nu = \sin\theta_0 / c(z_a)$"
        "\n"
        r"$\sin\theta_i = \nu\,\bar c_i$"
        "\n"
        r"$\Delta x_i = h\tan\theta_i$"
        "\n"
        r"$\Delta T_i = h / (\bar c_i\cos\theta_i)$"
        "\n"
        r"$\bar c_i$: mean of end speeds",
        ha="right",
        va="center",
    )
    ax.set_xlim(-3.3, x[-1] + 0.4)
    ax.set_ylim(4.3, -0.4)
    ax.axis("off")
    ax.set_title("(a) Layer sum of the ray integrals (not to scale)", loc="left")

    # (b) convergence: |dT| at fixed geometry against step
    ax = axes[0, 1]
    for color, elevation in zip(colors, ELEVATION_DEG, strict=True):
        for dz in DZ_M:
            sel = [
                r
                for r in rows
                if r["elevation_deg"] == elevation
                and r["dz_m"] == dz
                and r["stride"] != REFERENCE_STRIDE
            ]
            ax.loglog(
                [r["step_m"] for r in sel],
                [abs(r["dT_fixed_geometry_us"]) for r in sel],
                color=color,
                lw=0.8,
                label=f"{elevation:.0f}°" if dz == DZ_M[0] else None,
            )
    h = np.array([0.01, 12])
    worst_prod = max(
        abs(r["dT_fixed_geometry_us"]) for r in rows if r["stride"] == PRODUCTION_STRIDE
    )
    ax.loglog(h, worst_prod * (h / h_prod) ** 2, "k:", lw=0.8, label=r"$\propto h^2$")
    ax.axhline(TIMING_NOISE_S * 1e6, color="tab:red", lw=0.8)
    ax.text(0.012, TIMING_NOISE_S * 1e6 * 0.35, "timing noise 20 µs", color="tab:red")
    ax.axhline(INTERPOLATION_RMS_S * 1e6, color="tab:orange", lw=0.8)
    ax.text(
        0.012,
        INTERPOLATION_RMS_S * 1e6 * 0.35,
        "table lookup rms 0.3 µs",
        color="tab:orange",
    )
    for step, name in ((h_prod, "production"), (h_1m, "1 m")):
        ax.axvline(step, color="0.4", lw=0.8, ls="--")
        ax.text(step * 1.08, 2.5e2, f"{name}\n{step:.2f} m", color="0.3", va="top")
    ax.set(
        xlabel="Integration step $h$ (m)",
        ylabel=r"Travel-time error $|\Delta T|$ (µs)",
        xlim=(0.01, 12),
        ylim=(1e-6, 1e3),
    )
    ax.grid(True, which="major", lw=0.3)
    ax.legend(title="elevation", fontsize=7, ncol=2, loc="lower right")
    ax.set_title("(b) Error against step, all (dz, elevation)", loc="left")

    # (c), (d) at the production step and at 1 m, against elevation
    for ax, key, label in (
        (axes[1, 0], "dT_fixed_geometry_us", r"Travel-time error $|\Delta T|$ (µs)"),
        (axes[1, 1], "dx_fixed_launch_mm", r"Range error $|\Delta x|$ (mm)"),
    ):
        for stride, style in ((PRODUCTION_STRIDE, "-"), (200, "--")):
            for dz, color in zip(
                DZ_M, plt.cm.plasma([0.1, 0.4, 0.6, 0.85]), strict=True
            ):
                sel = [r for r in rows if r["stride"] == stride and r["dz_m"] == dz]
                # all differences are negative: the coarse step shortens x and T
                ax.semilogy(
                    [r["elevation_deg"] for r in sel if r["elevation_deg"] < 90],
                    [abs(r[key]) for r in sel if r["elevation_deg"] < 90],
                    style,
                    color=color,
                    lw=1,
                    label=f"dz = {dz:.0f} m" if stride == PRODUCTION_STRIDE else None,
                )
        ax.set(xlabel=r"Elevation angle $\beta$ (°)", ylabel=label)
        ax.grid(True, which="both", lw=0.3)
    axes[1, 0].plot([], [], "k-", label=f"h = {h_prod:.2f} m (production)")
    axes[1, 0].plot([], [], "k--", label=f"h = {h_1m:.2f} m")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("(c) Fixed geometry: T = S / ESV", loc="left")
    axes[1, 1].set_title("(d) Fixed launch angle: horizontal range", loc="left")

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run()
    summary = summarise(rows)

    depth_prod, cz_prod, _ = thinned_profile(PRODUCTION_STRIDE)
    sound_speed = {
        "c_surface_mps": float(cz_prod[0]),
        "c_transducer_mps": float(np.interp(TRANSDUCER_DEPTH_M, depth_prod, cz_prod)),
    }
    for step, values in summary.items():
        print(step, {key: round(value, 4) for key, value in values.items()})
    print(sound_speed)

    keys = list(rows[0])
    with open(OUTPUT_DIR / "ray_step.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.10g}" for k in keys) + "\n")
    with open(OUTPUT_DIR / "ray_step.json", "w") as f:
        json.dump(
            {
                "max_abs_by_step": summary,
                "production_stride": PRODUCTION_STRIDE,
                "reference_stride": REFERENCE_STRIDE,
                "transducer_depth_m": TRANSDUCER_DEPTH_M,
                "dz_m": DZ_M,
                "elevation_deg": ELEVATION_DEG,
                **sound_speed,
            },
            f,
            indent=2,
        )
    plot(rows, Path(OUTPUT_DIR / "ray_step"))
