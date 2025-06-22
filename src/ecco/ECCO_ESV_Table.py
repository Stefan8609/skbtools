import numpy as np
import xarray as xr
import gsw
import scipy.io as sio

from data import gps_data_path

from acoustics.svp import DelGrosso_SV, depth_to_pressure
from examples.ESV_table import construct_esv

"""Build ESV table for every month of ECCO to ECCO Depth"""

# Define coordinates for Bermuda
lat_bermuda = 31.447
lon_bermuda = -68.6896

ecco_dir = str(gps_data_path("ECCO_Temp_Salinity"))
months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
month_to_num = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
for month in months:
    print(f"Starting {month}")
    num = month_to_num[month]
    if num > 9:
        ds = xr.open_dataset(
            ecco_dir + f"/OCEAN_TEMPERATURE_SALINITY_mon_mean_2017-{num}"
            f"_ECCO_V4r4_latlon_0p50deg.nc"
        )
    else:
        ds = xr.open_dataset(
            ecco_dir + f"/OCEAN_TEMPERATURE_SALINITY_mon_mean_2017-0{num}"
            f"_ECCO_V4r4_latlon_0p50deg.nc"
        )

    # Define coordinates for Bermuda
    lat_bermuda = 31.447
    lon_bermuda = -68.6896

    temp_profile = ds.THETA.sel(
        latitude=lat_bermuda, longitude=lon_bermuda, method="nearest"
    )

    salinity_profile = ds.SALT.sel(
        latitude=lat_bermuda, longitude=lon_bermuda, method="nearest"
    )
    z_a = 40
    z_arr = np.linspace(0.0, -5300.0, 53000)
    temp_arr = temp_profile.interp(Z=z_arr)
    salinity_arr = salinity_profile.interp(Z=z_arr)

    temp_data = temp_arr.values[0]
    salinity_data = salinity_arr.values[0]
    nan_indices = np.isnan(temp_data) | np.isnan(salinity_data)

    temp_data = temp_data[~nan_indices]
    salinity_data = salinity_data[~nan_indices]
    z_arr = z_arr[~nan_indices]

    depth = -1 * z_arr
    pressure = depth_to_pressure(-1 * z_arr[: len(salinity_data)], lat_bermuda)
    in_situ_temp = gsw.conversions.t_from_CT(salinity_data, temp_data, pressure)

    pressure = depth_to_pressure(-1 * z_arr[: len(salinity_data)], lat_bermuda)
    cz = DelGrosso_SV(salinity_data, in_situ_temp, pressure)

    beta_array, z_array, esv_matrix = construct_esv(depth, cz)

    dz_array = z_array - z_a

    data_to_save = {"angle": beta_array, "distance": dz_array, "matrice": esv_matrix}

    sio.savemat(
        gps_data_path(f"global_table_esv_ECCO_{month}.mat"),
        {"distance": z_array, "angle": beta_array, "matrice": esv_matrix},
    )
    print(f"Finished {month}")
