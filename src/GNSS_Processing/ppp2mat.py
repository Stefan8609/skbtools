from pathlib import Path
import numpy as np
from scipy.io import savemat
from data import gps_data_path
import matplotlib.pyplot as plt


def save_mat(input_path, output_path):
    fin = Path(input_path)
    fout = Path(output_path)
    fout.parent.mkdir(parents=True, exist_ok=True)

    with fin.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    i, n = 0, len(lines)

    def looks_data(s):
        if not s or "*" in s:
            return False
        toks = s.split()
        if len(toks) < 8:
            return False
        try:
            float(toks[0])
            float(toks[1])
            float(toks[2])
            float(toks[3])
            float(toks[4])
            float(toks[5])
            float(toks[6])
            float(toks[7])
            return True
        except Exception:
            return False

    while (
        i < n
        and not lines[i].upper().endswith("END OF HEADER")
        and not looks_data(lines[i])
    ):
        i += 1
    if i < n and lines[i].upper().endswith("END OF HEADER"):
        i += 1
        while (
            i < n and not looks_data(lines[i]) and not lines[i].lstrip().startswith("*")
        ):
            i += 1
    if i < n and lines[i].lstrip().startswith("*"):
        i += 1

    mjd, sod = [], []
    x, y, z = [], [], []
    lat, lon, h = [], [], []
    nsat_bundle, pdop = [], []

    while i < n:
        ln = lines[i]
        i += 1
        if not ln or "*" in ln or ln.lstrip().startswith("*"):
            continue
        toks = ln.split()
        if len(toks) < 8:
            continue
        j = 0
        try:
            mjd_val = float(toks[j])
            j += 1
            sod_val = float(toks[j])
            j += 1
            x_val = float(toks[j])
            j += 1
            y_val = float(toks[j])
            j += 1
            z_val = float(toks[j])
            j += 1
            lat_val = float(toks[j])
            j += 1
            lon_val = float(toks[j])
            j += 1
            h_val = float(toks[j])
            j += 1
        except Exception:
            continue
        ns = []
        for _ in range(7):
            if j < len(toks):
                try:
                    ns.append(float(toks[j]))
                except Exception:
                    ns.append(np.nan)
                j += 1
            else:
                ns.append(np.nan)
        if j < len(toks):
            try:
                pdop_val = float(toks[j])
            except Exception:
                pdop_val = np.nan
        else:
            pdop_val = np.nan
        mjd.append(mjd_val)
        sod.append(sod_val)
        x.append(x_val)
        y.append(y_val)
        z.append(z_val)
        lat.append(lat_val)
        lon.append(lon_val)
        h.append(h_val)
        nsat_bundle.append(ns)
        pdop.append(pdop_val)

    out = {
        "days": np.asarray(mjd, dtype=float),
        "times": np.asarray(sod, dtype=float),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
        "lat": np.asarray(lat, dtype=float),
        "lon": np.asarray(lon, dtype=float),
        "elev": np.asarray(h, dtype=float),
        "nsat_bundle": np.asarray(nsat_bundle, dtype=float),
        "pdop": np.asarray(pdop, dtype=float),
    }

    savemat(fout.as_posix(), out, do_compression=True)
    print("parsed rows:", len(out["days"]))
    print("saved to:", fout.resolve().as_posix())

    return out


if __name__ == "__main__":
    GPS_num = 1
    GPS_num_to_name = {1: "1_PortFwd", 2: "2_StbdFwd", 3: "3_StbdAft", 4: "4_PortAft"}

    lat_vals = []
    lon_vals = []
    elev_vals = []
    days_vals = []
    time_vals = []
    PDOP_vals = []
    nsat_bundle = None
    for i in range(6, 14):
        formatted_i = f"{i:02d}"
        input_path = gps_data_path(
            f"GPS_Data/Puerto_Rico/{GPS_num_to_name[GPS_num]}/combined/00{formatted_i}-combined"
        )
        output_path = gps_data_path(
            f"GPS_Data/Puerto_Rico/{GPS_num_to_name[GPS_num]}/combined/00{formatted_i}-combined.mat"
        )
        try:
            output = save_mat(input_path, output_path)
            lat_vals = lat_vals + output["lat"].tolist()
            lon_vals = lon_vals + output["lon"].tolist()
            elev_vals = elev_vals + output["elev"].tolist()
            days_vals = days_vals + output["days"].tolist()
            time_vals = time_vals + output["times"].tolist()
            PDOP_vals = PDOP_vals + output["pdop"].tolist()

            if nsat_bundle is None:
                nsat_bundle = np.asarray(output["nsat_bundle"])
            else:
                nsat_bundle = np.concatenate(
                    (nsat_bundle, np.asarray(output["nsat_bundle"])), axis=0
                )

            # datetimes = (np.asarray(output["days"]) - 59958) * 24 * 3600 + np.
            # # asarray(output["times"])

            # plt.scatter(output["lon"], output["lat"], s=1)
            # plt.title(f"00{formatted_i} Position Plot")
            # plt.xlabel("Longitude (deg)")
            # plt.ylabel("Latitude (deg)")
            # plt.axis("equal")
            # path = gps_data_path(f"Figs/GPS/Puerto_Rico_Segments/
            ##Trajectory_Segment_{i}.png")
            # plt.savefig(path, dpi=300)
            # plt.show()

            # plt.scatter(datetimes, output["elev"])
            # plt.title(f"00{formatted_i} Height Over Time")
            # plt.xlabel("Epoch Index")
            # plt.ylabel("Height (m)")
            # path = gps_data_path(f"Figs/GPS/Puerto_Rico_Segments/
            # #Elevation_Segment_{i}.png")
            # plt.savefig(path, dpi=300)
            # plt.show()
        except Exception as e:
            print(f"❌ Failed to process 00{formatted_i}: {e}")

    days_vals = np.array(days_vals)
    time_vals = np.array(time_vals)

    days = days_vals.flatten() - 59958
    times = time_vals.flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600

    sc = plt.scatter(lon_vals, lat_vals, c=datetimes, s=1, cmap="viridis")
    plt.gca().set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(sc)
    cbar.set_label("Time (hours)")
    plt.xticks(rotation=30)
    plt.title("Trajectory colored by time")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.tight_layout()
    path = gps_data_path(
        f"Figs/GPS/{GPS_num_to_name[GPS_num]}/Puerto_Rico_Trajectory_6.png"
    )
    plt.savefig(path, dpi=300)
    plt.show()

    plt.scatter(datetimes, elev_vals, s=1)
    plt.ylim(-66, -52)

    path = gps_data_path(
        f"Figs/GPS/{GPS_num_to_name[GPS_num]}/Puerto_Rico_Elevation_6.png"
    )

    plt.title(f"Elevation Over Time for {GPS_num_to_name[GPS_num]}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Elevation (m)")
    plt.savefig(path, dpi=300)
    plt.show()

    plt.scatter(datetimes, PDOP_vals, s=1)

    path = gps_data_path(f"Figs/GPS/{GPS_num_to_name[GPS_num]}/Puerto_Rico_PDOP_6.png")
    plt.title(f"PDOP Over Time for {GPS_num_to_name[GPS_num]}")
    plt.xlabel("Time (hours)")
    plt.ylabel("PDOP")
    plt.savefig(path, dpi=300)
    plt.show()

    plt.scatter(datetimes, np.sum(nsat_bundle, axis=1), s=1)
    path = gps_data_path(f"Figs/GPS/{GPS_num_to_name[GPS_num]}/Puerto_Rico_Nsat_6.png")
    plt.title(f"Number of Satellites Over Time for {GPS_num_to_name[GPS_num]}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Number of Satellites")
    plt.savefig(path, dpi=300)
    plt.show()


# Plot trajectory
# Plot PDOP
