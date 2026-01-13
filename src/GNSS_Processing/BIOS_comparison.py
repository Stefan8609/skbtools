from GNSS_Processing.ppp2mat import save_mat
import numpy as np
import matplotlib.pyplot as plt
from data import gps_data_path

Antenna_num = 4

input_path_me = gps_data_path(f"GPS_data/Bermuda_Testing/Antenna{Antenna_num}-camp.prd")
output_path_me = gps_data_path(
    f"GPS_data/Bermuda_Testing/Antenna{Antenna_num}-camp.mat"
)
output_me = save_mat(input_path_me, output_path_me)

input_path_terry = gps_data_path(
    f"GPS_data/Bermuda_Testing/Antenna{Antenna_num}-camp-Terry.prd"
)
output_path_terry = gps_data_path(
    f"GPS_data/Bermuda_Testing/Antenna{Antenna_num}-camp-Terry.mat"
)
output_terry = save_mat(input_path_terry, output_path_terry)

# Match the times of both datasets for comparison
day = 59015
times_me = np.array(output_me["times"])
days_me = np.array(output_me["days"]) - day
datetimes_me = (days_me * 24 * 3600) + times_me

times_terry = np.array(output_terry["times"])
days_terry = np.array(output_terry["days"]) - day
datetimes_terry = (days_terry * 24 * 3600) + times_terry

common_times = np.intersect1d(datetimes_me, datetimes_terry)
idx_me = np.isin(datetimes_me, common_times)
idx_terry = np.isin(datetimes_terry, common_times)

# Reduce all arrays to common times
x_me = output_me["x"][idx_me]
y_me = output_me["y"][idx_me]
z_me = output_me["z"][idx_me]
times_me = output_me["times"][idx_me]
elev_me = output_me["elev"][idx_me]

x_terry = output_terry["x"][idx_terry]
y_terry = output_terry["y"][idx_terry]
z_terry = output_terry["z"][idx_terry]
times_terry = output_terry["times"][idx_terry]
elev_terry = output_terry["elev"][idx_terry]

# Print mean and std of differences
print(
    f"X Difference: Mean = {np.mean(x_me - x_terry):.4f} m, Std = {np.std(x_me - x_terry):.4f} m"
)
print(
    f"Y Difference: Mean = {np.mean(y_me - y_terry):.4f} m, Std = {np.std(y_me - y_terry):.4f} m"
)
print(
    f"Z Difference: Mean = {np.mean(z_me - z_terry):.4f} m, Std = {np.std(z_me - z_terry):.4f} m"
)
print(
    f"Elevation Difference: Mean = {np.mean(elev_me - elev_terry):.4f} m, Std = {np.std(elev_me - elev_terry):.4f} m"
)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
# My elevation
axes[0].scatter(times_me, elev_me, label="Elevation (m)", s=1)
axes[0].set_ylim(-38, -34.5)
axes[0].set_ylabel("Elevation (m)")
axes[0].set_title("PRIDE 3.0  Elevation Over Time")
axes[0].legend()

# Terry's elevation
axes[1].scatter(times_terry, elev_terry, label="Elevation (m)", color="orange", s=1)
axes[1].set_ylim(-38, -34.5)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Elevation (m)")
axes[1].set_title("PRIDE 2.0 Elevation Over Time")
axes[1].legend()

fig.tight_layout()
plt.show()

path = gps_data_path(f"Figs/GPS/BIOS_comparison_Antenna{Antenna_num}.png")
fig.savefig(path, dpi=300)
