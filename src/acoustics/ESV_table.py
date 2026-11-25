import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numba import njit
from time import time

from acoustics.ray_tracing import ray_trace_locate, ray_tracing
from puertorico_data import gps_data_path


@njit
def construct_esv(depth, cz, transducer_depth, dog_depths, beta_array):
    """ESV table for each DOG depth and elevation angle."""
    esv_matrix = np.zeros((len(dog_depths), len(beta_array)))
    beta_rad = beta_array * np.pi / 180
    tan_beta = np.tan(beta_rad)

    for i in range(len(dog_depths)):
        print("DOG depth:", i + 1, "/", len(dog_depths))

        dog_depth = dog_depths[i]
        x_values = (dog_depth - transducer_depth) / tan_beta

        for j in range(len(beta_array)):
            x = x_values[j]

            alpha = ray_trace_locate(
                transducer_depth,
                dog_depth,
                x,
                depth,
                cz,
            )
            _, _, travel_time = ray_tracing(
                alpha,
                transducer_depth,
                dog_depth,
                depth,
                cz,
            )

            distance = np.sqrt(x**2 + (dog_depth - transducer_depth) ** 2)
            esv_matrix[i, j] = distance / travel_time

    return esv_matrix


if __name__ == "__main__":
    transducer_depth = 60.0
    minimum_dog_depth = 2950.0
    maximum_dog_depth = 3050.0
    dog_depth_step = 1.0
    # This SVP has no finite direct-ray solution below 12 degrees at these depths.
    minimum_elevation_angle = 12.0
    maximum_elevation_angle = 89.75
    angle_step = 0.25
    table_name = "PuertoRico_table_transducer_60_DOG_2950_3050.mat"

    depth = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("SVP_Data/depth_PuertoRico_DelGrosso.txt"))
    )
    cz = np.ascontiguousarray(
        np.genfromtxt(gps_data_path("SVP_Data/cz_PuertoRico_DelGrosso.txt"))
    )

    print(f"SVP depth range: {depth.min():.1f} to {depth.max():.1f} m")
    if depth.max() < maximum_dog_depth:
        raise ValueError(
            f"SVP only reaches {depth.max():.1f} m, but the ESV table requires "
            f"{maximum_dog_depth:.0f} m."
        )

    beta_array = np.arange(
        minimum_elevation_angle,
        maximum_elevation_angle + angle_step / 2,
        angle_step,
    )
    dog_depths = np.arange(
        minimum_dog_depth,
        maximum_dog_depth + dog_depth_step / 2,
        dog_depth_step,
    )

    print(f"Bottom SVP sound speed: {cz[-1]:.3f} m/s")
    print(f"Representative transducer depth: {transducer_depth:.0f} m")
    print(f"DOG depths: {dog_depths[0]:.0f} to {dog_depths[-1]:.0f} m")
    print(f"Elevation angles: {beta_array[0]:.2f} to {beta_array[-1]:.2f} degrees")

    start_time = time()
    esv_matrix = construct_esv(
        depth,
        cz,
        transducer_depth,
        dog_depths,
        beta_array,
    )
    end_time = time()

    if not np.all(np.isfinite(esv_matrix)):
        invalid = np.count_nonzero(~np.isfinite(esv_matrix))
        raise ValueError(f"ESV table contains {invalid} invalid ray paths")

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Number of DOG depths: {len(dog_depths)}")
    print(f"Number of angles: {len(beta_array)}")

    plt.figure(figsize=(10, 6))
    plt.contourf(
        beta_array,
        dog_depths,
        esv_matrix,
        levels=20,
        cmap="viridis",
    )

    plt.colorbar(label="ESV (m/s)")
    plt.gca().invert_yaxis()

    plt.xlabel("Elevation Angle (degrees)")
    plt.ylabel("Receiver Depth (m)")
    plt.title(
        "Puerto Rico Effective Sound Velocity, "
        f"Transducer Depth {transducer_depth:.0f} m"
    )
    plt.tight_layout()
    plt.show()

    data_to_save = {
        "angle": beta_array,
        "distance": dog_depths - transducer_depth,
        "dog_depth": dog_depths,
        "transducer_depth": transducer_depth,
        "matrice": esv_matrix,
    }

    output_file = gps_data_path("ESV_Tables", table_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_file, data_to_save)

    print(f"Saved: {output_file}")
