import numpy as np
import scipy.io as sio

from Numba_xAline_bias import final_bias_geiger
from Numba_Geiger import findTransponder

"""Gaussian Samples for gps1_to_others, and gps1_to_transponder"""

def gaussian_search(num_points,
    output_file='gaussian_output.txt',
    initial_lever_base=np.array([-12.4659, 9.6021, -13.2993]),
    initial_gps_grid = np.array([[0.0, 0.0, 0.0],
                           [-2.39341409, -4.22350344, 0.02941493],
                           [-12.09568416, -0.94568462, 0.0043972],
                           [-8.68674054, 5.16918806, -0.02499322]]),
    sigma_lever=np.array([0.5, 0.5, 2.0]),
    sigma_gps_grid=np.array([0.5, 0.5, 0.1]),
    downsample=50,
):
    """
    Function to sample the GPS grid and GPS to transducer in a Gaussian distribution.
        Calculates the RMSE of the joint CDOGs and saves each sample and RMSE to a file.

    Parameters
    ----------
    num_points : int
        Number of points to sample in the Gaussian distribution.
    output_file : str
        Path to the output file.
    CDOG_guess_augment : np.ndarray
        Augmented guess for CDOG coordinates.
    initial_lever_base : np.ndarray
        Base guess for the initial lever arm.
    initial_gps_grid : np.ndarray
        Initial guess for the GPS grid coordinates.
    downsample : int
        Downsample factor for the data.
    """
    np.set_printoptions(suppress=True)

    # Load the external ESV data
    esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
    dz_array = esv_table['distance'].flatten()
    angle_array = esv_table['angle'].flatten()
    esv_matrix = esv_table['matrice']

    # Load the external GPS data
    data = np.load(f'../../../GPSData/Processed_GPS_Receivers_DOG_1.npz')
    GPS_Coordinates = data['GPS_Coordinates']
    GPS_data = data['GPS_data']
    CDOG_guess = data['CDOG_guess']

    # Load the DOG data for each DOG
    CDOG_all_data = []
    for i in range(1, 5):
        if i == 2:
            continue
        CDOG_data = sio.loadmat(f'../../../GPSData/DOG{i}-camp.mat')['tags'].astype(float)
        CDOG_data[:, 1] = CDOG_data[:, 1] / 1e9
        CDOG_all_data.append(CDOG_data)

    # Initialize the best known augments and offsets
    CDOG_augments = np.array([[-398.16, 371.90, 773.02],
                              [825.182985, -111.05670221, -734.10011698],
                              [236.428385, -1307.98390221, -2189.21991698]])
    offsets = np.array([1866.0, 3175.0, 1939.0])

    esv_bias = 0.0
    time_bias = 0.0

    # Downsample Data
    GPS_Coordinates = GPS_Coordinates[::downsample]
    GPS_data = GPS_data[::downsample]

    # Initialize the output file
    with open(output_file, 'w') as file:
        # Write a header line
        file.write("Grid Search Results\n")
        file.write("[Lever], [[GPS Grid Estimate]], RMSE combined\n")
        for i in range(num_points):
            lever_guess = initial_lever_base + np.random.normal(0, sigma_lever, 3)
            gps1_grid_guess = initial_gps_grid + np.random.normal(0, sigma_gps_grid, (4, 3))

            # Calculate the transponder coordinates
            transponder_coordinates = findTransponder(GPS_Coordinates, gps1_grid_guess, lever_guess)

            # Run final_geiger for each DOG
            RMSE_sum = 0
            for j in range(3):
                offset = offsets[j]
                inversion_guess = CDOG_guess + CDOG_augments[j]
                CDOG_data = CDOG_all_data[j]
                try:
                    inversion_result, CDOG_full, GPS_full, _, _ = final_bias_geiger(inversion_guess, CDOG_data,
                                                                                    GPS_data,
                                                                                    transponder_coordinates,
                                                                                    offset,
                                                                                    esv_bias, time_bias, dz_array,
                                                                                    angle_array, esv_matrix,
                                                                                    real_data=True)
                    RMSE = np.sqrt(np.mean((CDOG_full - GPS_full) ** 2))
                    RMSE_sum += RMSE
                except Exception as error:
                    print(error)
                    print(f"Error in final_bias_geiger, skipping this iteration {j}")
                    RMSE_sum = np.inf
                    break

            file.write(
                f"[{np.array2string(lever_guess, precision=4, separator=', ', max_line_width=1000)[1:-1]}], "
                f"[{str(gps1_grid_guess.round(4)).replace('\n', ' ')[1:-1]}], "
                f"{RMSE_sum * 100 * 1515:.4f}\n"
            )
            file.flush()  # ensures immediate write

            print(
                f"Iteration {i + 1}/{num_points}: \n"
                f"Lever: {np.array2string(lever_guess, precision=4, separator=', ', max_line_width=1000)},\n"
                f"gps1_grid_guess: {str(gps1_grid_guess.round(4)).replace('\n', ' ')},\n"
                f"RMSE: {RMSE_sum * 100 * 1515:.4f}\n"
            )

if __name__ == "__main__":
    gaussian_search(
        num_points=10000,
        output_file='gaussian_output.txt',
        initial_lever_base=np.array([-12.4659, 9.6021, -13.2993]),
        initial_gps_grid=np.array([[0.0, 0.0, 0.0],
                                   [-2.39341409, -4.22350344, 0.02941493],
                                   [-12.09568416, -0.94568462, 0.0043972],
                                   [-8.68674054, 5.16918806, -0.02499322]]),
        sigma_lever=np.array([0.5, 0.5, 2.0]),
        sigma_gps_grid=np.array([0.5, 0.5, 0.1]),
        downsample=50,
    )




