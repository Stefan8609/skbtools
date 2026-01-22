import numpy as np
from plotting.MCMC_plots import get_init_params_and_prior, cdog_aug_ecef_to_enu
from data import gps_output_path


def check_geometric_correlations(
    chain,
    initial_params,
    dog_index=0,
    downsample=1,
):
    """
    Empirical correlation diagnostics for GNSS-A degeneracies.

    Checks:
      1) corr(time_bias[j], mean(esv_bias[j,:]))
      2) corr(lever_z, DOG_U)

    Parameters
    ----------
    chain : dict-like
        Loaded npz chain (lever, esv_bias, time_bias, CDOG_aug)
    initial_params : dict
        Output of get_init_params_and_prior(chain)
    dog_index : int
        0 -> DOG1, 1 -> DOG3, 2 -> DOG4
    downsample : int
        Thinning factor
    """

    # ---- Extract chains ----
    tb = chain["time_bias"][::downsample]  # (T, D)
    esv = chain["esv_bias"][::downsample]  # (T, D, S)
    lever_z = chain["lever"][::downsample, 2]  # (T,)
    cdog_aug = chain["CDOG_aug"][::downsample, dog_index, :]  # (T, 3)

    # ---- Mean ESV per DOG ----
    esv_mean = esv[:, dog_index, :].mean(axis=1)  # (T,)

    # ---- DOG vertical (ENU Up) ----
    prior_mean_aug = initial_params["CDOG_aug"][dog_index]
    cdog_enu = cdog_aug_ecef_to_enu(cdog_aug, prior_mean=prior_mean_aug)
    dog_u = cdog_enu[:, 2]  # Up (m)

    # ---- Correlations ----
    def corr(a, b):
        return np.corrcoef(a, b)[0, 1]

    rho_time_esv = corr(tb[:, dog_index], esv_mean)
    rho_lever_dog = corr(lever_z, dog_u)

    print("\n=== Empirical correlation diagnostics ===")
    print(f"DOG index        : {dog_index}")
    print(f"corr(time_bias, mean(ESV)) = {rho_time_esv:+.3f}")
    print(f"corr(lever_z, DOG_U)       = {rho_lever_dog:+.3f}")

    # Simple flags
    if abs(rho_time_esv) > 0.8:
        print("Strong timing–ESV degeneracy detected")
    if abs(rho_lever_dog) > 0.8:
        print("Strong lever-Z–DOG-U degeneracy detected")

    # ---- Estimate kappa ----
    tb_j = tb[:, dog_index]
    esv_j = esv_mean
    cov = np.cov(tb_j, esv_j, ddof=1)
    kappa_hat = -cov[0, 1] / cov[1, 1]
    print(f"Estimated kappa (time_bias / ESV) = {kappa_hat:.3f} s/m")

    return {
        "rho_time_esv": rho_time_esv,
        "rho_lever_dog": rho_lever_dog,
    }


if __name__ == "__main__":
    # Initial Parameters for adding to plot
    file_name = "mcmc_chain_test"
    chain = np.load(gps_output_path(f"{file_name}.npz"))

    initial_params, prior_scales, proposal_scales = get_init_params_and_prior(chain)

    diagnostics = check_geometric_correlations(
        chain, initial_params, dog_index=0, downsample=10
    )
