import numpy as np
from geometry.ECEF_Geodetic import ECEF_Geodetic
from data import gps_data_path

# Find the geodetic coordinates of the CDOG locations
CDOG_augments = [
    [-396.73685216, 369.21165118, 774.34118812],
    [826.22200156, -112.71197144, -733.58577345],
    [235.88302714, -1306.08869213, -2190.76178745],
]
CDOG_reference = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
CDOG_Locations = CDOG_augments + CDOG_reference
lat, lon, alt = ECEF_Geodetic(CDOG_Locations)
print("CDOG Locations in Geodetic Coordinates:")
for i in range(len(CDOG_Locations)):
    print(
        f"Location {i + 1}: Latitude: {lat[i]:.6f}, Longitude: {lon[i]:.6f}," 
        f"Altitude: {alt[i]:.2f} m"
    )


# Quick check against GEBCO data for the same location
def read_arc_ascii_grid(path):
    header = {}
    with open(path, "r") as f:
        for _ in range(6):
            key, val = f.readline().split()
            header[key.lower()] = float(val)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header.get("xllcorner", header.get("xllcenter"))
    yll = header.get("yllcorner", header.get("yllcenter"))
    cellsize = header["cellsize"]
    nodata = header.get("nodata_value", -9999)

    data = np.loadtxt(path, skiprows=6)
    data = np.where(data == nodata, np.nan, data)

    return {
        "ncols": ncols,
        "nrows": nrows,
        "xll": xll,
        "yll": yll,
        "cellsize": cellsize,
        "data": data,
    }


def query_depth_nearest(grid, lat, lon):
    xll = grid["xll"]
    yll = grid["yll"]
    cell = grid["cellsize"]
    nrows = grid["nrows"]
    data = grid["data"]

    col = int(round((lon - xll) / cell))
    row_from_bottom = int(round((lat - yll) / cell))

    row = nrows - 1 - row_from_bottom

    if row < 0 or row >= data.shape[0] or col < 0 or col >= data.shape[1]:
        raise ValueError("Point is outside the grid")

    elev = data[row, col]
    depth = -elev if elev < 0 else 0.0
    return elev, depth


grid = read_arc_ascii_grid(
    gps_data_path("gebco_2025_n31.47_s31.42_w-68.71_e-68.68_ascii.asc")
)

for i in range(len(CDOG_Locations)):
    elev, depth = query_depth_nearest(grid, lat=lat[i], lon=lon[i])
    print(
        f"Location {i + 1}: Latitude: {lat[i]:.6f},"
        f"Longitude: {lon[i]:.6f}, Depth: {-depth:.2f} m"
    )
    diff = alt[i] + depth
    print(f"Difference between GPS altitude and GEBCO depth: {diff:.2f} m")
