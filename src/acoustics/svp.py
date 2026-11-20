import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scipy.interpolate import PchipInterpolator

from puertorico_data import gps_data_path


def depth_to_pressure(z, lat):
    """Pressure (bar) at depth z (m) and latitude lat (deg), Eq. A12."""
    c1 = (5.92 + 5.25 * np.sin(np.abs(lat) * np.pi / 180) ** 2) * 1e-3
    c2 = 2.21e-6
    pressure = ((1 - c1) - np.sqrt((1 - c1) ** 2 - 4 * c2 * z)) / (2 * c2)
    return pressure / 10


def adiabatic_temperature_gradient(S, T, P):
    """UNESCO 1983 adiabatic temperature gradient in deg C / dbar."""
    dS = S - 35.0

    gradient = (
        3.5803e-5
        + (8.5258e-6 + (-6.8360e-8 + 6.6228e-10 * T) * T) * T
        + (1.8932e-6 - 4.2393e-8 * T) * dS
        + (
            1.8741e-8
            + (-6.7795e-10 + (8.7330e-12 - 5.4481e-14 * T) * T) * T
            + (-1.1351e-10 + 2.7759e-12 * T) * dS
        )
        * P
        + (-4.6206e-13 + (1.8676e-14 - 2.1687e-16 * T) * T) * P**2
    )

    return gradient


def insitu_to_potential_temperature(S, T, P, P_ref=0.0):
    """Convert in-situ temperature to potential temperature."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float).copy()
    P = np.asarray(P, dtype=float).copy()

    H = P_ref - P

    XK = H * adiabatic_temperature_gradient(S, T, P)
    T += 0.5 * XK
    Q = XK

    P += 0.5 * H
    XK = H * adiabatic_temperature_gradient(S, T, P)
    T += 0.29289322 * (XK - Q)
    Q = 0.58578644 * XK + 0.121320344 * Q

    XK = H * adiabatic_temperature_gradient(S, T, P)
    T += 1.707106781 * (XK - Q)
    Q = 3.414213562 * XK - 4.121320344 * Q

    P += 0.5 * H
    XK = H * adiabatic_temperature_gradient(S, T, P)

    return T + (XK - 2.0 * Q) / 6.0


def potential_to_insitu_temperature(S, theta, P, tolerance=1e-10, max_iterations=20):
    """Convert potential temperature referenced to 0 dbar to in-situ temperature."""
    S = np.asarray(S, dtype=float)
    theta = np.asarray(theta, dtype=float)
    P = np.asarray(P, dtype=float)

    T = theta.copy()

    for _ in range(max_iterations):
        theta_calculated = insitu_to_potential_temperature(S, T, P)
        correction = theta - theta_calculated
        T += correction

        if np.nanmax(np.abs(correction)) < tolerance:
            break

    return T


def DelGrosso_SV(S, T, P):
    """Del Grosso (1974) sound speed (m/s), Eq. A13.

    S practical salinity, T in-situ temperature (deg C), P pressure (bar).
    """
    P = P * 1.019716213  # bar -> kg/cm^2

    const = {
        "C_T1": 5.012285,
        "C_T2": -0.551184e-1,
        "C_T3": 0.221649e-3,
        "C_S1": 1.329530,
        "C_S2": 0.1288598e-3,
        "C_P1": 0.1560592,
        "C_P2": 0.2449993e-4,
        "C_P3": -0.8833959e-8,
        "C_ST": -0.1275936e-1,
        "C_TP": 0.6353509e-2,
        "C_T2P2": 0.2656174e-7,
        "C_TP2": -0.1593895e-5,
        "C_TP3": 0.5222483e-9,
        "C_T3P": -0.4383615e-6,
        "C_S2P2": -0.1616745e-8,
        "C_ST2": 0.9688441e-4,
        "C_S2TP": 0.4857614e-5,
        "C_STP": -0.3406824e-3,
    }

    Delta_CT = const["C_T1"] * T + const["C_T2"] * T**2 + const["C_T3"] * T**3
    Delta_CS = const["C_S1"] * S + const["C_S2"] * S**2
    Delta_CP = const["C_P1"] * P + const["C_P2"] * P**2 + const["C_P3"] * P**3

    Delta_CSTP = (
        const["C_TP"] * T * P
        + const["C_T3P"] * P * T**3
        + const["C_TP2"] * T * P**2
        + const["C_T2P2"] * T**2 * P**2
        + const["C_TP3"] * T * P**3
        + const["C_ST"] * S * T
        + const["C_ST2"] * S * T**2
        + const["C_STP"] * S * T * P
        + const["C_S2TP"] * S**2 * T * P
        + const["C_S2P2"] * S**2 * P**2
    )

    return 1402.392 + Delta_CT + Delta_CS + Delta_CP + Delta_CSTP


if __name__ == "__main__":
    latitude = 19.000
    longitude = -66.3333

    spacing = 1.0
    target_depth = 3050
    minimum_native_depth = 2865
    extrapolation_fit_depth = 500.0

    profile_path = gps_data_path(
        "Ocean_Data",
        "PuertoRico_SVP_Profile_20230118.nc",
    )

    depth_output = gps_data_path(
        "SVP_Data",
        "depth_PuertoRico_DelGrosso.txt",
    )

    sound_speed_output = gps_data_path(
        "SVP_Data",
        "cz_PuertoRico_DelGrosso.txt",
    )

    # Load selected GLORYS profile
    ds = xr.open_dataset(profile_path)

    depth_native = ds["depth"].squeeze().values
    theta_native = ds["thetao"].squeeze().values
    salinity_native = ds["so"].squeeze().values

    good = (
        np.isfinite(depth_native)
        & np.isfinite(theta_native)
        & np.isfinite(salinity_native)
    )

    depth_native = depth_native[good]
    theta_native = theta_native[good]
    salinity_native = salinity_native[good]

    print("\nSelected GLORYS profile")
    print("-----------------------")
    print(f"Model latitude: {float(ds['latitude'].squeeze()):.4f}")
    print(f"Model longitude: {float(ds['longitude'].squeeze()):.4f}")
    print(f"Native levels: {len(depth_native)}")
    print(f"Native depth range: {depth_native[0]:.1f} to {depth_native[-1]:.1f} m")

    if depth_native[-1] < minimum_native_depth:
        raise ValueError(
            f"Selected GLORYS profile only reaches {depth_native[-1]:.1f} m. "
            "Use a deeper GLORYS cell that reaches at least "
            f"{minimum_native_depth:.0f} m."
        )

    # Add the surface at exactly 0 m.
    if depth_native[0] > 0:
        depth_native = np.insert(depth_native, 0, 0.0)
        theta_native = np.insert(theta_native, 0, theta_native[0])
        salinity_native = np.insert(salinity_native, 0, salinity_native[0])

    # Calculate Del Grosso SVP at the native GLORYS levels.
    pressure_native_bar = depth_to_pressure(depth_native, latitude)
    temperature_native = potential_to_insitu_temperature(
        salinity_native,
        theta_native,
        pressure_native_bar * 10.0,
    )
    cz_native = DelGrosso_SV(salinity_native, temperature_native, pressure_native_bar)

    # Interpolate the resulting SVP onto a 1 m depth grid.
    deepest_native = depth_native[-1]
    depth = np.arange(0.0, np.floor(deepest_native) + spacing, spacing)

    svp_interp = PchipInterpolator(depth_native, cz_native, extrapolate=False)
    cz = svp_interp(depth)

    # Extend the final small section to 5300 m.
    if depth[-1] < target_depth:
        fit_start = depth[-1] - extrapolation_fit_depth
        fit_mask = depth >= fit_start

        slope, _ = np.polyfit(depth[fit_mask], cz[fit_mask], 1)

        new_depth = np.arange(
            depth[-1] + spacing,
            target_depth + 0.5 * spacing,
            spacing,
        )
        new_cz = cz[-1] + slope * (new_depth - depth[-1])

        print("\nDeep extension")
        print("--------------")
        print(f"Profile ends at: {depth[-1]:.1f} m")
        print(f"Extending to: {target_depth:.1f} m")
        print(f"Extension length: {target_depth - depth[-1]:.1f} m")
        print(f"Deep gradient: {slope * 1000:.3f} m/s per km")

        depth = np.concatenate([depth, new_depth])
        cz = np.concatenate([cz, new_cz])

    # Make arrays ready for the Numba ray-tracing code.
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    cz = np.ascontiguousarray(cz, dtype=np.float64)

    # Save.
    depth_output.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(depth_output, depth, fmt="%.6f")
    np.savetxt(sound_speed_output, cz, fmt="%.9f")

    print("\nSaved Puerto Rico SVP")
    print("----------------------")
    print(f"Depth:       {depth_output}")
    print(f"Sound speed: {sound_speed_output}")
    print(f"Points:      {len(depth):,}")
    print(f"Spacing:     {np.median(np.diff(depth)):.1f} m")
    print(f"Depth range: {depth[0]:.1f} to {depth[-1]:.1f} m")
    print(f"SV range:    {cz.min():.3f} to {cz.max():.3f} m/s")

    # Plot native and interpolated SVP.
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.plot(cz, depth, label="1 m Del Grosso SVP")
    ax.scatter(cz_native, depth_native, s=18, label="Native GLORYS levels", zorder=3)

    ax.invert_yaxis()
    ax.set_xlabel("Sound Speed (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Puerto Rico Del Grosso Sound Speed Profile")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
