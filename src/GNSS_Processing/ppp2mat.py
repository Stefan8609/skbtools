from pathlib import Path
import numpy as np
from scipy.io import savemat
from bermuda_data import gps_data_path


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
    for n in (1, 2, 3, 4):
        save_mat(
            gps_data_path(
                f"GPS_Data/PRIDE_runs/final/ber{n}/2020/168/kin_2020168_ber{n}"
            ),
            gps_data_path(f"GPS_Data/PRIDE_final/Antenna{n}-camp.mat"),
        )
