import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load your CSV data
unavco_data = pd.read_csv("GPSData/network-monitoring.csv")

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a Basemap instance with a Robinson projection
m = Basemap(projection='robin', lon_0=0, resolution='c')

# Draw coastlines and fill continents
m.drawcoastlines()
m.fillcontinents(color='lightgray', lake_color='white')

# Draw the edges of the map
m.drawmapboundary(fill_color='white')

# Convert latitude and longitude to the map's projection coordinates
x, y = m(unavco_data['lon'].values, unavco_data['lat'].values)

# Plot the UNAVCO CORS stations
plt.scatter(x, y, color='blue', s=30, alpha=0.9, edgecolors='k')

# Add title
plt.title('UNAVCO CORS Stations', fontsize=18)

# Show the plot
plt.show()
