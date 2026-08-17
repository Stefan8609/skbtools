from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import savemat
from puertorico_data import gps_data_path


GRS80_A = 6378137.0
GRS80_INV_F = 298.257222101
MJD_EPOCH = date(1858, 11, 17)


def dms_to_decimal(degrees, minutes, seconds):
    sign = -1.0 if degrees < 0.0 else 1.0

    return sign * (
        abs(degrees)
        + minutes / 60.0
        + seconds / 3600.0
    )


def geodetic_to_ecef(lat_deg, lon_deg, height_m):
    flattening = 1.0 / GRS80_INV_F
    eccentricity_squared = flattening * (2.0 - flattening)

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    radius = GRS80_A / np.sqrt(
        1.0 - eccentricity_squared * sin_lat**2
    )

    x = (radius + height_m) * cos_lat * cos_lon
    y = (radius + height_m) * cos_lat * sin_lon
    z = (
        radius * (1.0 - eccentricity_squared) + height_m
    ) * sin_lat

    return x, y, z


def read_csrs_pos(path):
    rows = []

    with path.open("r", encoding="ascii", errors="replace") as file:
        for line in file:
            if not line.startswith("FIN "):
                continue

            fields = line.split()

            if len(fields) < 23:
                continue

            dt = datetime.strptime(
                f"{fields[4]} {fields[5]}",
                "%Y-%m-%d %H:%M:%S.%f",
            ).replace(tzinfo=timezone.utc)

            lat = dms_to_decimal(
                float(fields[16]),
                float(fields[17]),
                float(fields[18]),
            )

            lon = dms_to_decimal(
                float(fields[19]),
                float(fields[20]),
                float(fields[21]),
            )

            rows.append(
                {
                    "time": dt,
                    "nsv": float(fields[6]),
                    "gdop": float(fields[7]),
                    "lat": lat,
                    "lon": lon,
                    "elev": float(fields[22]),
                }
            )

    return rows


def gnss_to_mat(pos_files, output_file):
    rows = []

    for pos_file in pos_files:
        rows.extend(read_csrs_pos(pos_file))

    if not rows:
        raise ValueError("No FIN position rows were read.")

    rows.sort(key=lambda row: row["time"])

    unique_rows = []
    seen_times = set()

    for row in rows:
        time = row["time"]

        if time in seen_times:
            continue

        seen_times.add(time)
        unique_rows.append(row)

    rows = unique_rows

    lat = np.asarray([row["lat"] for row in rows], dtype=np.float64)
    lon_signed = np.asarray([row["lon"] for row in rows], dtype=np.float64)
    lon = np.mod(lon_signed, 360.0)
    elev = np.asarray([row["elev"] for row in rows], dtype=np.float64)

    days = np.asarray(
        [(row["time"].date() - MJD_EPOCH).days for row in rows],
        dtype=np.float64,
    )

    times = np.asarray(
        [
            row["time"].hour * 3600.0
            + row["time"].minute * 60.0
            + row["time"].second
            + row["time"].microsecond / 1_000_000.0
            for row in rows
        ],
        dtype=np.float64,
    )

    nsv = np.asarray([row["nsv"] for row in rows], dtype=np.float64)
    gdop = np.asarray([row["gdop"] for row in rows], dtype=np.float64)

    x, y, z = geodetic_to_ecef(lat, lon_signed, elev)

    savemat(
        output_file,
        {
            "days": days.reshape(1, -1),
            "times": times.reshape(1, -1),
            "lat": lat.reshape(1, -1),
            "lon": lon.reshape(1, -1),
            "elev": elev.reshape(1, -1),
            "x": x.reshape(1, -1),
            "y": y.reshape(1, -1),
            "z": z.reshape(1, -1),
            "nsv": nsv.reshape(1, -1),
            "gdop": gdop.reshape(1, -1),
        },
        do_compression=True,
    )

    print(f"Wrote {output_file}")
    print(f"Epochs: {len(rows):,}")


if __name__ == "__main__":
    base_path = gps_data_path("GPS_Data")
    SITES = ["STPF", "STSF", "STPA", "STSA"]
    DOYS = [15, 16, 17, 18, 19]

    for SITE in SITES:
        raw_path = base_path / "Raw" / SITE

        POS_FILES = [
            raw_path / f"{SITE.lower()}{doy:03d}0.pos"
            for doy in DOYS
        ]

        OUTPUT_FILE = base_path / f"{SITE}-camp_bis.mat"

        gnss_to_mat(POS_FILES, OUTPUT_FILE)