from pathlib import Path

import numpy as np
import copernicusmarine
import matplotlib.pyplot as plt
import xarray as xr

from puertorico_data import gps_data_path


if __name__ == "__main__":
    output_path = gps_data_path("Ocean_Data")
    output_path.mkdir(parents=True, exist_ok=True)

    # TDOG 02 deployment location.
    dog_latitude = 18.889
    dog_longitude = -66.257

    # Require the selected GLORYS profile to extend at least this deep.
    # Change this once the exact TDOG water depth is known.
    minimum_profile_depth = 2800.0

    neighborhood_file = "PuertoRico_SVP_Neighborhood_20230118.nc"
    profile_file = "PuertoRico_SVP_Profile_20230118.nc"

    # ------------------------------------------------------------
    # Download a small neighborhood around TDOG 02
    # ------------------------------------------------------------

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so"],
        minimum_longitude=dog_longitude - 0.25,
        maximum_longitude=dog_longitude + 0.25,
        minimum_latitude=dog_latitude - 0.10,
        maximum_latitude=dog_latitude + 0.30,
        start_datetime="2023-01-18T00:00:00",
        end_datetime="2023-01-18T00:00:00",
        minimum_depth=0.0,
        maximum_depth=6000.0,
        output_directory=output_path,
        output_filename=neighborhood_file,
    )

    # ------------------------------------------------------------
    # Find deepest valid depth at every GLORYS grid cell
    # ------------------------------------------------------------

    neighborhood_path = output_path / neighborhood_file
    ds = xr.open_dataset(neighborhood_path)

    depth = ds["depth"].values
    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    temperature = ds["thetao"].squeeze("time").values
    salinity = ds["so"].squeeze("time").values

    candidates = []

    for i, latitude in enumerate(latitudes):
        for j, longitude in enumerate(longitudes):
            valid = np.isfinite(temperature[:, i, j]) & np.isfinite(salinity[:, i, j])

            if not np.any(valid):
                continue

            deepest_depth = depth[np.where(valid)[0][-1]]

            north_km = (latitude - dog_latitude) * 111.0
            east_km = (
                (longitude - dog_longitude)
                * 111.0
                * np.cos(np.deg2rad(dog_latitude))
            )
            distance_km = np.sqrt(north_km**2 + east_km**2)

            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance_km": distance_km,
                    "deepest_depth": deepest_depth,
                }
            )

    # Sort all cells by distance from TDOG 02.
    candidates.sort(key=lambda x: x["distance_km"])

    print("\nGLORYS cells near TDOG 02")
    print("--------------------------")
    print(" Latitude   Longitude   Distance (km)   Deepest valid (m)")

    for candidate in candidates:
        print(
            f"{candidate['latitude']:9.4f} "
            f"{candidate['longitude']:10.4f} "
            f"{candidate['distance_km']:13.2f} "
            f"{candidate['deepest_depth']:18.1f}"
        )

    # ------------------------------------------------------------
    # Choose closest cell that is sufficiently deep
    # ------------------------------------------------------------

    deep_candidates = [
        candidate
        for candidate in candidates
        if candidate["deepest_depth"] >= minimum_profile_depth
    ]

    if not deep_candidates:
        raise ValueError(
            f"No downloaded GLORYS cell reaches {minimum_profile_depth:.0f} m. "
            "Increase the neighborhood size."
        )

    best = deep_candidates[0]

    print("\nSelected GLORYS cell")
    print("--------------------")
    print(f"TDOG location:       {dog_latitude:.4f}, {dog_longitude:.4f}")
    print(f"Selected location:   {best['latitude']:.4f}, {best['longitude']:.4f}")
    print(f"Distance from TDOG:  {best['distance_km']:.2f} km")
    print(f"Deepest valid level: {best['deepest_depth']:.1f} m")

    # ------------------------------------------------------------
    # Extract selected profile
    # ------------------------------------------------------------

    selected = ds.isel(
        latitude=best["i"],
        longitude=best["j"],
    )

    profile_temperature = selected["thetao"].squeeze().values
    profile_salinity = selected["so"].squeeze().values

    valid = np.isfinite(profile_temperature) & np.isfinite(profile_salinity)

    profile_depth = depth[valid]
    profile_temperature = profile_temperature[valid]
    profile_salinity = profile_salinity[valid]

    # Save the selected single-cell profile separately.
    selected.to_netcdf(output_path / profile_file)

    print(f"\nSaved selected profile to:\n{output_path / profile_file}")

    # ------------------------------------------------------------
    # Plot temperature and salinity with depth
    # ------------------------------------------------------------

    fig, axs = plt.subplots(1, 2, figsize=(9, 8), sharey=True)

    axs[0].plot(profile_temperature, profile_depth, linewidth=2)
    axs[0].set_xlabel("Potential temperature (°C)")
    axs[0].set_ylabel("Depth (m)")
    axs[0].set_title("Temperature")
    axs[0].grid(alpha=0.25)

    axs[1].plot(profile_salinity, profile_depth, linewidth=2)
    axs[1].set_xlabel("Practical Salinity")
    axs[1].set_title("Salinity")
    axs[1].grid(alpha=0.25)

    axs[0].invert_yaxis()

    fig.suptitle(
        "Puerto Rico Ocean Profile\n"
        f"GLORYS: {best['latitude']:.4f}°N, "
        f"{abs(best['longitude']):.4f}°W — "
        f"{best['distance_km']:.1f} km from TDOG 02"
    )

    fig.tight_layout()
    plt.show()