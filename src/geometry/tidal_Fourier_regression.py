import numpy as np
from data import gps_data_path
from geometry.ECEF_Geodetic import ECEF_Geodetic
import matplotlib.pyplot as plt
from plotting.save import save_plot


def fit_tides(elevations, GPS_data, tide_frequencies):
    """
    Fit tidal components to GPS elevation data using linear regression.

    Parameters
    ----------
    elevations : array-like
        Array of elevation data (in meters).
    GPS_data : array-like
        Array of GPS time data seconds.
    tide_frequencies : array-like
        Array of tidal frequencies (in cycles per day).

    Returns
    -------
    dict
        Dictionary containing fitted amplitudes and phases for each tidal frequency.
    """
    # # Convert frequencies to Hz
    # tide_frequencies = np.array(tide_frequencies) / (3600 * 24)

    # Move starting time to 0
    GPS_data = GPS_data - GPS_data[0]
    GPS_data = GPS_data / 86400  # Convert to days

    # Prepare the design matrix for linear regression
    A = np.zeros((len(GPS_data), 2 * len(tide_frequencies) + 1))
    A[:, 0] = 1
    for i, freq in enumerate(tide_frequencies):
        A[:, 2 * i + 1] = np.cos(2 * np.pi * freq * GPS_data)
        A[:, 2 * i + 2] = np.sin(2 * np.pi * freq * GPS_data)

    # Perform linear regression to find coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A, elevations, rcond=None)

    # Extract amplitudes and phases from coefficients
    results = {"constant": coeffs[0]}
    for i, freq in enumerate(tide_frequencies):
        a_cos = coeffs[2 * i + 1]
        a_sin = coeffs[2 * i + 2]
        amplitude = np.sqrt(a_cos**2 + a_sin**2)
        phase = np.arctan2(-a_sin, a_cos)
        results[freq] = {"amplitude": amplitude, "phase": phase}

    return results


def plot_fit(GPS_data, elevations, fitted_components, func_name="tidal_fit"):
    """
    Plot observed elevations and fitted tidal regression.
    """
    fitted_elev = np.zeros_like(elev)
    for freq, params in fitted_components.items():
        if freq == "constant":
            fitted_elev += params
        else:
            fitted_elev += params["amplitude"] * np.cos(
                2 * np.pi * freq * GPS_data + params["phase"]
            )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(GPS_data, elevations, color="blue", s=1, label="Observed")
    ax.plot(GPS_data, fitted_elev, color="r", linewidth=2, label="Fitted")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Elevation (m)")
    ax.legend()
    ax.set_title("Tidal Regression Fit")
    plt.show()

    # Save figure
    save_plot(fig, func_name)
    return


if __name__ == "__main__":
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_full.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    GPS_barycenter = np.mean(GPS_Coordinates, axis=1)

    _, _, elev = ECEF_Geodetic(GPS_barycenter)

    results = fit_tides(
        elev, GPS_data, tide_frequencies=np.array([1.9322736, 1.0027379])
    )
    print(results)

    # [1.9322736, 2.0, 1.8959819, 1.0027379, 0.9295357, 2.0054759, 0.9972621]

    GPS_data = GPS_data - GPS_data[0]
    GPS_data = GPS_data / 86400  # Convert to days
    # plot fit
    plot_fit(GPS_data, elev, results, func_name="tidal_fit_full")
