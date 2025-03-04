import ssl
import certifi
from urllib.request import urlopen

import numpy as np
import xarray as xr
import ecco_v4_py as ecco
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set up SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

ecco_dir = "GPSData/ECCO_Temp_Salinity"

month = 'June'
month_to_num = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
                'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
                'November': 11, 'December': 12}
num = month_to_num[month]
if num > 9:
    ds = xr.open_dataset(ecco_dir + f'/OCEAN_TEMPERATURE_SALINITY_mon_mean_2017-{num}_ECCO_V4r4_latlon_0p50deg.nc')
else:
    ds = xr.open_dataset(ecco_dir + f'/OCEAN_TEMPERATURE_SALINITY_mon_mean_2017-0{num}_ECCO_V4r4_latlon_0p50deg.nc')

# Define coordinates for Bermuda
lat_bermuda = 31.447
lon_bermuda = -68.6896

temp_profile = ds.THETA.sel(
    latitude=lat_bermuda,
    longitude=lon_bermuda,
    method='nearest'
)

salinity_profile = ds.SALT.sel(
    latitude=lat_bermuda,
    longitude=lon_bermuda,
    method='nearest'
)


# Plot temperature profile
import matplotlib.pyplot as plt
temp_profile.plot(y='Z')
plt.title(f'Temperature Profile at Bermuda in {month}')
plt.xlabel('Temperature (°C)')
plt.ylabel('Depth (m)')
plt.show()


depths = [0, -1500, -3000, -4500]  # meters
n_depths = len(depths)
fig = plt.figure(figsize=(10, 6))
for i, depth in enumerate(depths):
    ax = plt.subplot(2, 2, i + 1, projection=ccrs.Robinson())

    # Extract temperature at specific depth
    temp_slice = ds.THETA.sel(Z=depth, method='nearest')

    # Plot with cartopy
    temp_slice.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu_r',
        cbar_kwargs={'label': 'Potential Temperature (°C)'}
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines()

    # Add title
    ax.set_title(f'Potential Temperature at {depth}m depth in {month}')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 6))
for i, depth in enumerate(depths):
    ax = plt.subplot(2, 2, i + 1, projection=ccrs.Robinson())

    # Extract temperature at specific depth
    salinity_slice = ds.SALT.sel(Z=depth, method='nearest')

    # Plot with cartopy
    salinity_slice.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu_r',
        cbar_kwargs={'label': 'Salinity (ppt)'}
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines()

    # Add title
    ax.set_title(f'Salinity at {depth}m depth in {month}')

plt.tight_layout()
plt.show()




# Create zoomed-in plot around Bermuda
fig = plt.figure(figsize=(14, 6))
depth_of_interest = -100

# Extract temperature and salinity at -1500 for bermuda
bermuda_temp = temp_profile.sel(Z=depth_of_interest, method='nearest')
bermuda_salinity = salinity_profile.sel(Z=depth_of_interest, method='nearest')

# Temperature subplot
ax1 = plt.subplot(121, projection=ccrs.PlateCarree())
temp_slice = ds.THETA.sel(Z=depth_of_interest, method='nearest')
temp_slice.plot(
    ax=ax1,
    transform=ccrs.PlateCarree(),
    cmap='RdYlBu_r',
    cbar_kwargs={'label': 'Potential Temperature (°C)'},
    vmin = bermuda_temp - 10,
    vmax = bermuda_temp + 10
)
ax1.set_extent([lon_bermuda-10, lon_bermuda+10, lat_bermuda-10, lat_bermuda+10], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE)
ax1.gridlines(draw_labels=True)
ax1.plot(lon_bermuda, lat_bermuda, 'k*', markersize=10, transform=ccrs.PlateCarree())
plt.title(f'Temperature around Bermuda at {depth_of_interest}m ({month})')

# Salinity subplot
ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
salinity_slice = ds.SALT.sel(Z=depth_of_interest, method='nearest')
salinity_slice.plot(
    ax=ax2,
    transform=ccrs.PlateCarree(),
    cmap='RdYlBu_r',
    cbar_kwargs={'label': 'Salinity (ppt)'},
    vmin = bermuda_salinity - 1,
    vmax = bermuda_salinity + 2
)
ax2.set_extent([lon_bermuda-10, lon_bermuda+10, lat_bermuda-10, lat_bermuda+10], crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE)
ax2.gridlines(draw_labels=True)
ax2.plot(lon_bermuda, lat_bermuda, 'k*', markersize=10, transform=ccrs.PlateCarree())
plt.title(f'Salinity around Bermuda at {depth_of_interest}m ({month})')

plt.tight_layout()
plt.show()