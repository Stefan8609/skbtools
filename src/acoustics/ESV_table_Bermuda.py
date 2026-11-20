import numpy as np
from numba import njit
from acoustics.ray_tracing import ray_trace_locate, ray_tracing
import matplotlib.pyplot as plt
import scipy.io as sio
from time import time
from bermuda_data import gps_data_path


@njit
def construct_esv(depth, cz):
    """ESV table (Eq. A16): straight-line range over ray travel time.

    Rows are receiver depths z (m; dz = z - z_a below the 2 m transducer), columns
    elevation angles of 20-90 deg. Returns ``(beta_array, z_array, esv_matrix)``.
    """
    beta_array = np.linspace(20, 90, 400)
    # dz = z - z_a = 5150-5300 m, 1 m step: every DOG with at least 48 m of margin
    # (DOG 1 queries 5198-5201 m)
    z_array = np.linspace(5152, 5302, 151)

    esv_matrix = np.zeros((len(z_array), len(beta_array)))
    z_a = 2.0

    beta_rad = beta_array * np.pi / 180
    tan_beta = np.tan(beta_rad)

    for i in range(len(z_array)):
        print("Progress: ", i, "/", len(z_array))
        z = z_array[i]
        x_values = (z - z_a) / tan_beta

        for j in range(len(beta_array)):
            x = x_values[j]
            alpha = ray_trace_locate(z_a, z, x, depth, cz)
            x_found, z_dist, time = ray_tracing(alpha, z_a, z, depth, cz)
            dist = np.sqrt(x**2 + (z - z_a) ** 2)
            esv_matrix[i, j] = dist / time

    return beta_array, z_array, esv_matrix


if __name__ == "__main__":
    depth = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("SVP_Data/depth_cast2_smoothed.txt"))[::20]
    )
    cz = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("SVP_Data/cz_cast2_smoothed.txt"))[::20]
    )

    dz_native = float(np.median(np.diff(depth)))
    z_max_current = float(depth[-1])
    z_target_max = 5350.0  # must exceed the largest z in construct_esv
    window_m = 10.0
    mask = depth >= (z_max_current - window_m)
    m_local, b_local = np.polyfit(depth[mask], cz[mask], 1)

    # Extrapolate linearly at native spacing
    new_depth = np.arange(
        np.nextafter(z_max_current, np.inf), z_target_max + 0.5 * dz_native, dz_native
    )

    cz_tail = cz[-1] + m_local * (new_depth - z_max_current)

    depth_ext = np.concatenate([depth, new_depth])
    cz_ext = np.concatenate([cz, cz_tail])

    depth, cz = depth_ext, cz_ext

    # Keep arrays contiguous for numba calls that follow
    depth = np.ascontiguousarray(depth)
    cz = np.ascontiguousarray(cz)

    # Extended SVP
    plt.figure()
    plt.plot(cz, depth, label="Extended SVP")
    plt.gca().invert_yaxis()
    plt.xlabel("Sound Speed (m/s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.title("SVP extended to 5350 m")
    plt.show()

    print(f"New depth range: {depth.min():.1f} to {depth.max():.1f} m")
    print(f"Deep gradient used: {m_local * 1000:.2f} m/s per km")

    z_a = 2.0
    print(z_a)
    print(cz[-1])

    start_time = time()
    beta_array, z_array, esv_matrix = construct_esv(depth, cz)
    end_time = time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    plt.figure(figsize=(10, 6))
    plt.contourf(beta_array, z_array, esv_matrix, levels=10, cmap="viridis")
    plt.colorbar(label="ESV (m/s)")
    plt.gca().invert_yaxis()
    plt.xlabel("Elevation Angle (degrees)")
    plt.ylabel("Depth (m)")
    plt.title("Effective Sound Velocity (m/s)")
    plt.show()

    dz_array = z_array - z_a

    data_to_save = {
        "angle": beta_array,
        "distance": dz_array,
        "matrice": esv_matrix,
        "transducer_depth": z_a,
    }
    # Overwrites the table of the Bermuda inversions; the table from before the
    # 2026-10-08 ray-tracing fix is kept as
    # global_table_esv_extended_before_ray_fix_20261008.mat
    sio.savemat(gps_data_path("ESV_Tables/global_table_esv_extended.mat"), data_to_save)
