import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gsw
import scipy.io as sio

from acoustics.svp import DelGrosso_SV, depth_to_pressure

CTD = sio.loadmat("../GPSData/CTD_Data/AE2008_Cast2.mat")["AE2008_Cast2"]
depth = CTD[:, 0][::100]
temperature = CTD[:, 1][::100]
salinity = CTD[:, 4][::100]

ecco_dir = "../GPSData/ECCO_Temp_Salinity"

month = "January"
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

z_arr = np.linspace(-40, -5300, 5301)
temp_arr = temp_profile.interp(Z=z_arr)
salinity_arr = salinity_profile.interp(Z=z_arr)

temp_data = temp_arr.values[0]
salinity_data = salinity_arr.values[0]

nan_indices = np.isnan(temp_data) | np.isnan(salinity_data)
temp_data = temp_data[~nan_indices]
salinity_data = salinity_data[~nan_indices]

pressure = depth_to_pressure(-1 * z_arr[: len(salinity_data)], lat_bermuda)
pressure = pressure  # - 10.1325  # Convert to sea pressure

# Convert potential temperature to in-situ temperature
in_situ_temp = gsw.conversions.t_from_CT(salinity_data, temp_data, pressure)

# Calculate sound speed
sound_speed1 = DelGrosso_SV(salinity_data, in_situ_temp, pressure)

# Plot salinity and temperature profiles on side by side plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(in_situ_temp, -1 * z_arr[: len(temp_data)], label="ECCO")
plt.plot(temperature, depth, label="CTD")
plt.title("Temperature Profile")
plt.xlabel("Temperature (°C)")
plt.ylabel("Depth (m)")
plt.legend()
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.plot(salinity_data, -1 * z_arr[: len(salinity_data)])
plt.plot(salinity, depth)
plt.title("Salinity Profile")
plt.xlabel("Salinity (PSU)")
plt.gca().invert_yaxis()
plt.show()

# Plot sound speed profile
depth_t = np.ascontiguousarray(np.genfromtxt("../GPSData/depth_cast2_smoothed.txt"))[
    ::100
]
cz_t = np.ascontiguousarray(np.genfromtxt("../GPSData/cz_cast2_smoothed.txt"))[::100]

plt.figure(figsize=(6, 8))
plt.plot(sound_speed1, -1 * z_arr[: len(salinity_data)], label="DelGrosso ECCO")
plt.plot(cz_t, depth_t, label="DelGrosso Thalia")
plt.gca().invert_yaxis()
plt.title("Sound Speed Profile")
plt.xlabel("Sound Speed (m/s)")
plt.ylabel("Depth (m)")
plt.legend()
plt.show()

# Plot the difference between ECCO SVP for January vs every other month
months = [
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
svp_diff = np.zeros((len(months), len(sound_speed1)))
for i, month in enumerate(months):
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

    temp_profile = ds.THETA.sel(
        latitude=lat_bermuda, longitude=lon_bermuda, method="nearest"
    )

    salinity_profile = ds.SALT.sel(
        latitude=lat_bermuda, longitude=lon_bermuda, method="nearest"
    )

    z_arr = np.linspace(-40, -5300, 5301)
    temp_arr = temp_profile.interp(Z=z_arr)
    salinity_arr = salinity_profile.interp(Z=z_arr)

    temp_data = temp_arr.values[0]
    temp_data = temp_data[~np.isnan(temp_data)]
    salinity_data = salinity_arr.values[0]
    salinity_data = salinity_data[~np.isnan(salinity_data)]

    pressure = depth_to_pressure(-1 * z_arr[: len(salinity_data)], lat_bermuda)
    pressure = pressure  # - 10.1325  # Convert to sea pressure

    # Convert potential temperature to in-situ temperature
    in_situ_temp = gsw.conversions.t_from_CT(salinity_data, temp_data, pressure)

    # Calculate sound speed
    sound_speed = DelGrosso_SV(salinity_data, in_situ_temp, pressure)

    svp_diff[i] = sound_speed1 - sound_speed

# Make individual plots for each month
for i, month in enumerate(months):
    plt.figure(figsize=(6, 8))
    plt.plot(svp_diff[i], -1 * z_arr[: len(salinity_data)], label=f"{month} ECCO")
    plt.xlim(-16, 16)
    plt.gca().invert_yaxis()
    plt.title("Sound Speed Profile")
    plt.xlabel("Sound Speed (m/s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.show()
