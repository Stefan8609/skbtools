import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from data import gps_data_path


def load_simple(path_mat):
    m = loadmat(path_mat, squeeze_me=True, struct_as_record=False)

    def a(name):
        if name not in m:
            raise KeyError(f"Missing '{name}' in {path_mat}")
        return np.squeeze(np.asarray(m[name], dtype=float))

    lat = a("lat")
    lon = a("lon")
    ht = a("elev")
    pdop = a("pdop")
    nsb = a("nsat_bundle")

    # Sum bundle -> total nsats per epoch
    if nsb.ndim == 1:
        nsats = nsb
    elif nsb.shape[0] == lat.size:
        nsats = np.nansum(nsb, axis=1)
    else:
        nsats = np.nansum(nsb, axis=0)

    # Optional time (used only for labels); otherwise use sample index
    t = m.get("t", None)
    if t is not None:
        t = np.squeeze(np.asarray(t))
    else:
        t = np.arange(lat.size)

    return dict(lon=lon, lat=lat, ht=ht, pdop=pdop, nsats=nsats, t=t)


def plot_GNSS(path, nthresh=4, pthresh=15.0, downsample = 1, save=False, show=False):
    data = load_simple(path)

    lon = np.asarray(data["lon"], dtype=float)[::downsample]
    lat = np.asarray(data["lat"], dtype=float)[::downsample]
    ht = np.asarray(data["ht"], dtype=float)[::downsample]
    pdop = np.asarray(data["pdop"], dtype=float)[::downsample]
    nsats = np.asarray(data["nsats"], dtype=float)[::downsample]
    t = np.asarray(data["t"])[::downsample]

    N = lon.size
    idx = np.arange(0, N)

    good = (pdop < pthresh) & (pdop != 0) & (nsats > nthresh)
    bad = ~good

    # ---- Larger figure and cleaner layout ----
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3
    )

    ax_map = fig.add_subplot(gs[:, 0])
    ax_ht = fig.add_subplot(gs[0, 1])
    ax_np = fig.add_subplot(gs[1, 1])

    # ---- Map: lon/lat scatter ----
    sz = 15
    cvals = np.linspace(1, 10, idx.size)
    gi = idx[good[idx]]
    bi = idx[~good[idx]]
    sc = ax_map.scatter(
        lon[gi], lat[gi], s=sz, c=cvals[good[idx]], cmap="jet", edgecolors="none"
    )
    ax_map.scatter(lon[bi], lat[bi], s=sz, c=[(0.7, 0.7, 0.7)], edgecolors="none")

    ax_map.set_xlabel("Longitude [deg]")
    ax_map.set_ylabel("Latitude [deg]")
    ax_map.set_title(f"Location of Ship in Geographic Coordinates")
    ax_map.grid(True)

    # Colorbar — larger and further below
    cbar = fig.colorbar(
        sc, ax=ax_map, orientation="horizontal", fraction=0.06, pad=0.15
    )
    if idx.size >= 4:
        ticks_i = [0, idx.size // 3, (2 * idx.size) // 3, idx.size - 1]
    else:
        ticks_i = [0, idx.size - 1]
    cbar.set_ticks(np.interp(ticks_i, np.arange(idx.size), cvals))
    t_thin = t[idx]
    cbar.set_ticklabels([str(t_thin[i]) for i in ticks_i])

    # ---- Height vs time ----
    ht_good = ht.copy()
    ht_good[bad] = np.nan
    ht_bad = ht.copy()
    ht_bad[good] = np.nan

    ax_ht.plot(t[idx], ht_good[idx], lw=1.5, color=(0.4660, 0.6740, 0.1880))
    ax_ht.plot(t[idx], ht_bad[idx], lw=1.0, color=(0.7, 0.7, 0.7))
    ax_ht.set_xlim(t[0], t[-1])
    mu = np.nanmean(ht)
    sd = np.nanstd(ht)
    ymin, ymax = mu - 3 * sd, mu + 3 * sd
    pad = 0.005 * max(1.0, abs(ymin), abs(ymax))
    ax_ht.set_ylim(ymin - pad, ymax + pad)
    ax_ht.set_ylabel("Height [m]")
    ax_ht.set_title(f"Height relative to WGS84")
    ax_ht.grid(True)

    # ---- Satellites & PDOP ----
    ax_left = ax_np
    ax_right = ax_left.twinx()

    ax_left.plot(t, nsats, "b", lw=1.0)
    ns_min = int(np.nanmin(nsats)) if np.isfinite(nsats).any() else 0
    ns_max = int(np.nanmax(nsats)) if np.isfinite(nsats).any() else 1
    ax_left.set_ylim(ns_min - 0.5, ns_max + 0.5)
    ax_left.set_ylabel("Satellites", color="b")
    ax_left.tick_params(axis="y", labelcolor="b")

    ax_right.plot(t, pdop, "r", lw=1.0)
    if np.isfinite(pdop).any():
        ax_right.set_ylim(np.nanmin(pdop) - 0.25, np.nanmax(pdop) + 0.25)
    ax_right.set_ylabel("PDOP", color="r")
    ax_right.tick_params(axis="y", labelcolor="r")

    ax_left.set_xlim(t[0], t[-1])
    ax_left.set_title("Number of Satellites and PDOP")
    ax_left.grid(True)

    # ---- Annotation ----
    fig.suptitle("Ship GNSS Summary", fontsize=14, y=0.95)
    if save == True:
        fig.savefig(gps_data_path(str(path).replace(".mat", "") + ".pdf"), dpi=300)
    if show == True:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    plot_GNSS(
        path=gps_data_path("GPS_Data/Puerto_Rico/4_PortAft/combined/0007-combined.mat"),
        nthresh=4,
        pthresh=15.0,
        show=True,
    )
