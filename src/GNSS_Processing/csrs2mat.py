from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from pymap3d import Ellipsoid, geodetic2ecef
from scipy.io import savemat

from jacksonville_data import gps_data_path


MJD_EPOCH = date(1858, 11, 17)
SITES = ["JAX1", "JAX2", "JAX3", "JAX4"]
DOYS = [163, 164]


def dms_to_decimal(degrees, minutes, seconds):
    sign = -1.0 if degrees < 0.0 else 1.0

    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


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
                    "sig_n": float(fields[13]),
                    "sig_e": float(fields[14]),
                    "sig_u": float(fields[15]),
                }
            )

    return rows


def gnss_to_mat(pos_files, output_file):
    """Save CSRS-PPP epochs as UTC MJD, seconds of day, and GRS80 ECEF metres."""
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
    sig_n = np.asarray([row["sig_n"] for row in rows], dtype=np.float64)
    sig_e = np.asarray([row["sig_e"] for row in rows], dtype=np.float64)
    sig_u = np.asarray([row["sig_u"] for row in rows], dtype=np.float64)

    x, y, z = geodetic2ecef(lat, lon_signed, elev, ell=Ellipsoid.from_name("grs80"))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

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
            "sig_n": sig_n.reshape(1, -1),
            "sig_e": sig_e.reshape(1, -1),
            "sig_u": sig_u.reshape(1, -1),
        },
        do_compression=True,
    )

    print(f"Wrote {output_file}")
    print(f"Epochs: {len(rows):,}")


if __name__ == "__main__":
    base_path = gps_data_path("GPS_Data")

    for SITE in SITES:
        raw_path = base_path / "Raw" / SITE

        POS_FILES = [raw_path / f"{SITE.lower()}{doy:03d}0.pos" for doy in DOYS]

        OUTPUT_FILE = base_path / f"{SITE}.mat"

        gnss_to_mat(POS_FILES, OUTPUT_FILE)
