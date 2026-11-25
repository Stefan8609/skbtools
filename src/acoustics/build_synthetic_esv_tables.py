"""Build the ESV tables used by the synthetic experiments, all on one grid.

``synth_ref`` (generation) and the perturbed tables (inversion) differ only in the
sound-speed profile. Grid: dz 5100-5350 m in 1 m steps (the synthetics query
~5195-5245 m) and 10-90 deg in 0.1 deg steps below 25 deg, 0.25 deg above, which
keeps the bilinear lookup error (rms 0.3 us, see ``check_interpolation_error``) far
below the 20 us timing noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from numba import njit

from acoustics.ray_tracing import ray_trace_locate, ray_tracing
from acoustics.svp_perturbation import SEABED_DEPTH_M, SETTINGS, named_perturbation
from acoustics.synthetic_esv_tables import ESV_TABLE_SETS
from Inversion_Workflow.Forward_Model.Find_ESV import find_esv
from bermuda_data import gps_data_path

TRANSDUCER_DEPTH_M = 2.0
DZ_ARRAY_M = np.arange(5100.0, 5350.0 + 0.5, 1.0)
# ESV curves strongly below about 20 deg, so the step is finer there.
ANGLE_ARRAY_DEG = np.concatenate(
    [np.arange(10.0, 25.0, 0.1), np.arange(25.0, 90.0 + 0.125, 0.25)]
)
PROFILE_STRIDE = 200  # cast samples are 5 mm apart; keep every 1 m
PROFILE_BOTTOM_M = 5400.0  # linear extension of the deep gradient below the cast
DEEP_GRADIENT_WINDOW_M = 10.0
TABLE_NAMES = ESV_TABLE_SETS["new"]
# Window of (angle, dz) the Bermuda synthetics query, used to summarise ESV changes.
QUERY_ANGLE_RANGE_DEG = (22.8, 89.8)
QUERY_DZ_RANGE_M = (5198.9, 5225.7)


def reference_profile():
    """Bermuda cast 2, thinned and extended linearly to ``PROFILE_BOTTOM_M``."""
    depth = np.genfromtxt(gps_data_path("SVP_Data/depth_cast2_smoothed.txt"))
    cz = np.genfromtxt(gps_data_path("SVP_Data/cz_cast2_smoothed.txt"))
    depth, cz = depth[::PROFILE_STRIDE], cz[::PROFILE_STRIDE]
    deep = depth >= depth[-1] - DEEP_GRADIENT_WINDOW_M
    slope = np.polyfit(depth[deep], cz[deep], 1)[0]
    spacing = float(np.median(np.diff(depth)))
    extra = np.arange(depth[-1] + spacing, PROFILE_BOTTOM_M + 0.5 * spacing, spacing)
    cz = np.concatenate([cz, cz[-1] + slope * (extra - depth[-1])])
    depth = np.concatenate([depth, extra])
    return np.ascontiguousarray(depth), np.ascontiguousarray(cz)


@njit
def construct_esv_grid(depth, cz, dz_array, angle_array, z_a):
    """ESV = slant range / travel time, as in ``ESV_table_Bermuda.construct_esv``."""
    esv_matrix = np.zeros((len(dz_array), len(angle_array)))
    tan_beta = np.tan(angle_array * np.pi / 180)
    for i in range(len(dz_array)):
        z = dz_array[i] + z_a
        x_values = dz_array[i] / tan_beta
        for j in range(len(angle_array)):
            x = x_values[j]
            alpha = ray_trace_locate(z_a, z, x, depth, cz)
            _, _, travel_time = ray_tracing(alpha, z_a, z, depth, cz)
            esv_matrix[i, j] = np.sqrt(x**2 + dz_array[i] ** 2) / travel_time
    return esv_matrix


def check_interpolation_error(depth, cz, dz_array, angle_array, esv_matrix, n=1000):
    """Max and rms travel-time error (s) of bilinear lookup against direct tracing.

    Random (angle, dz) inside the table over the Bermuda dz range and all angles.
    """
    rng = np.random.default_rng(0)
    angle = rng.uniform(angle_array[0], angle_array[-1], n)
    dz = rng.uniform(*QUERY_DZ_RANGE_M, n)
    esv_direct = np.array(
        [
            construct_esv_grid(depth, cz, dz[i : i + 1], angle[i : i + 1], 2.0)[0, 0]
            for i in range(n)
        ]
    )
    esv_table = find_esv(angle, dz, dz_array, angle_array, esv_matrix)
    travel_time = dz / np.sin(np.radians(angle)) / esv_direct
    error = travel_time * (esv_direct - esv_table) / esv_direct
    return np.abs(error).max(), np.sqrt(np.mean(error**2))


def change_summary(depth, delta_c, dz_array, angle_array, delta_esv):
    """Mean and shape-only (mean removed) sizes of delta c and delta ESV, m/s.

    The mean is absorbed by the inversion's ESV bias (dc); only the shape tests it.
    """
    in_water = depth <= SEABED_DEPTH_M
    window = np.ix_(
        (dz_array >= QUERY_DZ_RANGE_M[0] - 1) & (dz_array <= QUERY_DZ_RANGE_M[1] + 1),
        (angle_array >= QUERY_ANGLE_RANGE_DEG[0])
        & (angle_array <= QUERY_ANGLE_RANGE_DEG[1]),
    )
    esv = delta_esv[window]
    return {
        "dc_mean": delta_c[in_water].mean(),
        "dc_shape_sd": delta_c[in_water].std(),
        "dc_max_abs": np.abs(delta_c).max(),
        "dc_deep_max_abs": np.abs(delta_c[depth > 3000]).max(),
        "esv_mean": esv.mean(),
        "esv_shape_sd": esv.std(),
        "esv_shape_max_abs": np.abs(esv - esv.mean()).max(),
    }


def save_figure(depth, cz, profiles, dz_array, angle_array, esv_matrices, path):
    """c(z), delta c(z), and the shape-only ESV change of each perturbed table."""
    names = [name for name in profiles if name != "ref"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    shallow = depth <= 2000
    for name, profile in profiles.items():
        axes[0, 0].plot(profile[shallow], depth[shallow], label=name)
    axes[0, 0].set(xlabel="c (m/s)", ylabel="depth (m)", title="Sound speed")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()
    for name in names:
        axes[0, 1].plot(profiles[name] - cz, depth, label=name)
    axes[0, 1].set(xlabel=r"$\Delta c$ (m/s)", title=r"$\Delta c(z)$")
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend()
    window = (dz_array >= 5190) & (dz_array <= 5235)
    for ax, name in zip(axes[1], names, strict=False):
        delta = (esv_matrices[name] - esv_matrices["ref"])[window]
        image = ax.pcolormesh(
            angle_array,
            dz_array[window],
            (delta - delta.mean()) * 1000,
            shading="auto",
        )
        ax.axhspan(*QUERY_DZ_RANGE_M, fill=False, edgecolor="w")
        ax.axvline(QUERY_ANGLE_RANGE_DEG[0], color="w")
        ax.set(
            xlabel="elevation angle (deg)",
            ylabel="dz (m)",
            title=f"{name}: ESV change minus mean {delta.mean():.3f} m/s",
        )
        fig.colorbar(image, ax=ax, label="mm/s")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    depth, cz = reference_profile()
    profiles = {"ref": cz}
    for name in SETTINGS:
        profiles[name] = named_perturbation(name, depth, cz)

    esv_matrices = {}
    for name, profile in profiles.items():
        esv_matrices[name] = construct_esv_grid(
            depth, profile, DZ_ARRAY_M, ANGLE_ARRAY_DEG, TRANSDUCER_DEPTH_M
        )
        parameters = SETTINGS.get(
            name, {"heave_amplitude_m": 0.0, "mixed_layer_dT": 0.0}
        )
        table_name = TABLE_NAMES["generate" if name == "ref" else name]
        sio.savemat(
            gps_data_path(f"ESV_Tables/{table_name}.mat"),
            {
                "angle": ANGLE_ARRAY_DEG,
                "distance": DZ_ARRAY_M,
                "matrice": esv_matrices[name],
                "transducer_depth": TRANSDUCER_DEPTH_M,
                "profile": name,
                "profile_stride": PROFILE_STRIDE,
                "profile_bottom_m": PROFILE_BOTTOM_M,
                "svp_depth_file": "SVP_Data/depth_cast2_smoothed.txt",
                "svp_speed_file": "SVP_Data/cz_cast2_smoothed.txt",
                **parameters,
            },
        )

    worst, rms = check_interpolation_error(
        depth, cz, DZ_ARRAY_M, ANGLE_ARRAY_DEG, esv_matrices["ref"]
    )
    print(
        f"Interpolation travel-time error: max {worst * 1e6:.3f} us, "
        f"rms {rms * 1e6:.3f} us"
    )
    for name in SETTINGS:
        summary = change_summary(
            depth,
            profiles[name] - cz,
            DZ_ARRAY_M,
            ANGLE_ARRAY_DEG,
            esv_matrices[name] - esv_matrices["ref"],
        )
        print(name, {key: round(float(value), 4) for key, value in summary.items()})
    save_figure(
        depth,
        cz,
        profiles,
        DZ_ARRAY_M,
        ANGLE_ARRAY_DEG,
        esv_matrices,
        gps_data_path("Figs/synthetic_esv_tables.png"),
    )
