import numpy as np
import matplotlib.pyplot as plt
from plotting.MCMC_plots import get_init_params_and_prior, cdog_aug_ecef_to_enu
from data import gps_output_path
from pymap3d import geodetic2enu
from geometry.ECEF_Geodetic import ECEF_Geodetic
from plotting.save import save_plot

"""Enable this for paper plots"""
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)


# -------------------------
# Correlation diagnostics
# -------------------------
def check_geometric_correlations(chain, initial_params, dog_index=0, downsample=1):
    tb = chain["time_bias"][::downsample]  # (T, D)
    esv = chain["esv_bias"][::downsample]  # (T, D, S)
    lever_z = chain["lever"][::downsample, 2]  # (T,)
    cdog_aug = chain["CDOG_aug"][::downsample, dog_index, :]  # (T, 3)

    esv_mean = esv[:, dog_index, :].mean(axis=1)  # (T,)
    prior_mean_aug = initial_params["CDOG_aug"][dog_index]
    dog_u = cdog_aug_ecef_to_enu(cdog_aug, prior_mean=prior_mean_aug)[:, 2]

    rho_time_esv = float(np.corrcoef(tb[:, dog_index], esv_mean)[0, 1])
    rho_lever_dog = float(np.corrcoef(lever_z, dog_u)[0, 1])

    print("\n=== Empirical correlation diagnostics ===")
    print(f"DOG index                     : {dog_index}")
    print(f"corr(time_bias, mean(ESV))     = {rho_time_esv:+.3f}")
    print(f"corr(lever_z, DOG_U)           = {rho_lever_dog:+.3f}")

    if abs(rho_time_esv) > 0.8:
        print("Strong timing–ESV degeneracy detected")
    if abs(rho_lever_dog) > 0.8:
        print("Strong lever-Z–DOG-U degeneracy detected")

    tb_j = tb[:, dog_index]
    cov = np.cov(tb_j, esv_mean, ddof=1)
    kappa_hat = float(-cov[0, 1] / cov[1, 1])
    print(f"Estimated kappa (time_bias / ESV) = {kappa_hat:.3f} s/m")

    return {
        "rho_time_esv": rho_time_esv,
        "rho_lever_dog": rho_lever_dog,
        "kappa_hat": kappa_hat,
    }


def check_esv_vertical_correlations(
    chain,
    initial_params,
    dog_index=0,
    downsample=1,
):
    # Downsample
    lever_z = chain["lever"][::downsample, 2]  # (T,)
    esv = chain["esv_bias"][::downsample]  # (T, D, S) ideally
    cdog_aug = chain["CDOG_aug"][::downsample, dog_index, :]  # (T, 3)

    # Normalize esv to (T, D, S)
    T = esv.shape[0]
    if esv.ndim == 1:
        esv = esv.reshape(T, 1, 1)
    elif esv.ndim == 2:
        esv = esv.reshape(T, esv.shape[1], 1)
    elif esv.ndim != 3:
        raise ValueError(f"esv_bias must be 1D/2D/3D; got {esv.shape}")

    if dog_index < 0 or dog_index >= esv.shape[1]:
        raise IndexError(f"dog_index {dog_index} out of range [0, {esv.shape[1] - 1}]")

    # Mean ESV for this DOG across splits (common-mode)
    esv_mean = esv[:, dog_index, :].mean(axis=1)  # (T,)

    # DOG vertical (ENU Up) for this DOG
    prior_mean_aug = initial_params["CDOG_aug"][dog_index]
    dog_u = cdog_aug_ecef_to_enu(cdog_aug, prior_mean=prior_mean_aug)[:, 2]  # (T,)

    # Correlations
    rho_esv_leverz = float(np.corrcoef(esv_mean, lever_z)[0, 1])
    rho_esv_dogu = float(np.corrcoef(esv_mean, dog_u)[0, 1])
    rho_esv_sum = float(np.corrcoef(esv_mean, lever_z + dog_u)[0, 1])
    rho_esv_diff = float(np.corrcoef(esv_mean, lever_z - dog_u)[0, 1])

    m = lever_z - dog_u
    c = esv_mean
    alpha = np.cov(m, c, ddof=1)[0, 1] / np.var(c, ddof=1)

    print("\n=== ESV vs vertical geometry correlations ===")
    print(f"DOG index: {dog_index}")
    print(f"corr(mean(ESV), lever_z)         = {rho_esv_leverz:+.3f}")
    print(f"corr(mean(ESV), DOG_U)           = {rho_esv_dogu:+.3f}")
    print(f"corr(mean(ESV), lever_z + DOG_U) = {rho_esv_sum:+.3f}")
    print(f"corr(mean(ESV), lever_z - DOG_U) = {rho_esv_diff:+.3f}")
    print(f"Estimated alpha (lever_z - DOG_U) / ESV = {alpha:.3f} s")

    return {
        "rho_esv_leverz": rho_esv_leverz,
        "rho_esv_dogu": rho_esv_dogu,
        "rho_esv_sum": rho_esv_sum,
        "rho_esv_diff": rho_esv_diff,
    }


def _as_TDS(arr):
    """Force arr to shape (T, D, S)."""
    arr = np.asarray(arr)
    T = arr.shape[0]
    if arr.ndim == 1:
        return arr.reshape(T, 1, 1)
    if arr.ndim == 2:
        return arr.reshape(T, arr.shape[1], 1)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 1D/2D/3D array; got {arr.shape}")


def plot_posterior_corr_matrix(
    chain,
    initial_params,
    DOG_num=1,
    downsample=1,
    annotate=True,
    save=False,
    show=True,
    chain_name=None,
):
    """
    Correlation matrix for a given DOG.
    """

    # Map physical DOG number -> internal index
    DOG_index_map = {1: 0, 3: 1, 4: 2}
    if DOG_num not in DOG_index_map:
        raise ValueError(f"DOG_num must be one of {list(DOG_index_map.keys())}")
    d = DOG_index_map[DOG_num]

    # Normalize esv to (T,D,S)
    def _as_TDS(arr):
        arr = np.asarray(arr)
        T = arr.shape[0]
        if arr.ndim == 1:
            return arr.reshape(T, 1, 1)
        if arr.ndim == 2:
            return arr.reshape(T, arr.shape[1], 1)
        if arr.ndim == 3:
            return arr
        raise ValueError(f"esv_bias must be 1D/2D/3D; got {arr.shape}")

    # Chain
    lever = np.asarray(chain["lever"][::downsample])
    cdog_aug = np.asarray(chain["CDOG_aug"][::downsample])
    esv = _as_TDS(chain["esv_bias"][::downsample])
    tb = np.asarray(chain["time_bias"][::downsample])

    # Convert selected DOG augment to ENU
    prior_mean_aug = np.asarray(initial_params["CDOG_aug"][d])
    cdog_enu = cdog_aug_ecef_to_enu(
        cdog_aug[:, d, :], prior_mean=prior_mean_aug
    )  # (T,3)

    # Mean ESV + time bias for this DOG
    esv_mean = esv[:, d, :].mean(axis=1)  # (T,)

    if tb.ndim == 1:
        tb_dog = tb
    elif tb.ndim == 2:
        tb_dog = tb[:, d]
    elif tb.ndim == 3:
        tb_dog = tb[:, d, :].mean(axis=1)
    else:
        raise ValueError(f"time_bias must be 1D/2D/3D; got {tb.shape}")

    # Assemble feature matrix
    cols = [
        cdog_enu[:, 0],
        cdog_enu[:, 1],
        cdog_enu[:, 2],
        lever[:, 0],
        lever[:, 1],
        lever[:, 2],
        esv_mean,
        tb_dog,
        lever[:, 2] - cdog_enu[:, 2],
    ]

    labels = [
        f"DOG {DOG_num} E",
        f"DOG {DOG_num} N",
        f"DOG {DOG_num} U",
        "Lever x",
        "Lever y",
        "Lever z",
        f"ESV mean (DOG {DOG_num})",
        f"Time Bias (DOG {DOG_num})",
        f"Lever z − DOG {DOG_num} U",
    ]

    X = np.column_stack(cols)
    X = X[np.isfinite(X).all(axis=1)]
    corr = np.corrcoef(X, rowvar=False)
    P = corr.shape[0]

    corr_plot = corr.copy()
    corr_plot[np.triu_indices(P, k=0)] = np.nan

    corr_plot = corr_plot[1:, :-1]
    labels_y = labels[1:]
    labels_x = labels[:-1]

    P_trim = corr_plot.shape[0]

    # Plot
    fig, ax = plt.subplots(figsize=(0.7 * P_trim + 3, 0.7 * P_trim + 2))

    im = ax.imshow(
        corr_plot,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson correlation")

    ax.set_xticks(np.arange(P_trim))
    ax.set_yticks(np.arange(P_trim))
    ax.set_xticklabels(labels_x, rotation=90, fontsize=9)
    ax.set_yticklabels(labels_y, fontsize=9)

    ax.set_xticks(np.arange(-0.5, P_trim, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, P_trim, 1), minor=True)
    ax.grid(which="minor", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(P_trim):
            for j in range(i + 1):
                r = corr_plot[i, j]
                if np.isfinite(r):
                    ax.text(j, i, f"{r:+.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()

    if save:
        save_plot(
            fig,
            func_name="posterior_correlation_matrix",
            chain_name=chain_name,
            subdir=f"Figs/MCMC/{chain_name}",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return corr, labels


def compare_augments_for_one_dog(
    dog_index,
    aug_expected,
    aug_large,
    aug_tight,
    CDOG_reference=None,
):
    if CDOG_reference is None:
        CDOG_reference = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    DOG_index_num = {0: 1, 1: 3, 2: 4}
    dog_num = DOG_index_num.get(dog_index, dog_index + 1)

    origin_abs = CDOG_reference + aug_expected[dog_index]
    lat0, lon0, h0 = ECEF_Geodetic(np.array([origin_abs]))
    lat0 = float(np.squeeze(lat0))
    lon0 = float(np.squeeze(lon0))
    h0 = float(np.squeeze(h0))

    def _enu_from_aug(aug_vec):
        ecef_abs = CDOG_reference + aug_vec
        lat, lon, h = ECEF_Geodetic(np.array([ecef_abs]))
        lat = float(np.squeeze(lat))
        lon = float(np.squeeze(lon))
        h = float(np.squeeze(h))
        e, n, u = geodetic2enu(lat, lon, h, lat0, lon0, h0)
        return np.array([float(e), float(n), float(u)], dtype=float)

    enu_expected = _enu_from_aug(aug_expected[dog_index])
    enu_large = _enu_from_aug(aug_large[dog_index])
    enu_tight = _enu_from_aug(aug_tight[dog_index])

    print("\n============================================================")
    print(f"Compare augments across runs for DOG {dog_num}")
    print("ENU origin = EXPECTED run location for this DOG")
    print(f"Origin geodetic: lat={lat0:.6f} deg, lon={lon0:.6f} deg, h={h0:.3f} m")
    print("------------------------------------------------------------")
    print(
        f"{'expected':>8}:  E={enu_expected[0]:9.3f} m   "
        "N={enu_expected[1]:9.3f} m   U={enu_expected[2]:9.3f} m"
    )
    print(
        f"{'large':>8}:  E={enu_large[0]:9.3f} m   N={enu_large[1]:9.3f} m   "
        "U={enu_large[2]:9.3f} m"
    )
    print(
        f"{'tight':>8}:  E={enu_tight[0]:9.3f} m   N={enu_tight[1]:9.3f} m   "
        "U={enu_tight[2]:9.3f} m"
    )

    d_large = enu_large - enu_expected
    d_tight = enu_tight - enu_expected
    print(
        f"Δ(large-expected): E={d_large[0]:+.3f} m  N={d_large[1]:+.3f} m"
        "U={d_large[2]:+.3f} m"
    )
    print(
        f"Δ(tight-expected): E={d_tight[0]:+.3f} m  N={d_tight[1]:+.3f} m"
        "U={d_tight[2]:+.3f} m"
    )
    print("============================================================\n")

    return {
        "origin_geodetic": (lat0, lon0, h0),
        "enu_expected": enu_expected,
        "enu_large": enu_large,
        "enu_tight": enu_tight,
        "delta_large": d_large,
        "delta_tight": d_tight,
    }


if __name__ == "__main__":
    file_name = "mcmc_chain_1_22_new_MCMC_long"
    chain_name = "logpost_" + file_name
    chain = np.load(gps_output_path(f"{file_name}.npz"))
    initial_params, prior_scales, proposal_scales = get_init_params_and_prior(chain)

    _ = check_geometric_correlations(chain, initial_params, dog_index=0, downsample=10)
    _ = check_esv_vertical_correlations(
        chain, initial_params, dog_index=1, downsample=10
    )

    _corr, _labels = plot_posterior_corr_matrix(
        chain,
        initial_params,
        downsample=10,
        annotate=True,
        save=True,
        show=True,
        chain_name=chain_name,
    )

    # # Paste your three runs' augments here (ECEF offsets, meters)
    # AUG_expected = np.array(
    #     [
    #         [-396.87920582, 369.53892949, 774.10518354],
    #         [825.97567842, -112.30281457, -733.83150287],
    #         [236.11932934, -1306.79542540, -2190.33287940],
    #     ],
    #     dtype=float,
    # )
    # AUG_large = np.array(
    #     [
    #         [-396.81936608, 369.51687844, 774.16912292],
    #         [826.05852461, -112.39236913, -733.76188301],
    #         [236.18252567, -1306.79921240, -2190.28809100],
    #     ],
    #     dtype=float,
    # )
    # AUG_tight = np.array(
    #     [
    #         [-396.63497693, 368.76390827, 774.66902810],
    #         [826.35431313, -113.16685683, -733.24194315],
    #         [236.01392473, -1306.57786131, -2190.44270239],
    #     ],
    #     dtype=float,
    # )

    # compare_augments_for_one_dog(0, AUG_expected, AUG_large, AUG_tight)  # DOG1
    # compare_augments_for_one_dog(1, AUG_expected, AUG_large, AUG_tight)  # DOG3
    # compare_augments_for_one_dog(2, AUG_expected, AUG_large, AUG_tight)  # DOG4
