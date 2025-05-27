import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numba import njit

from Numba_time_bias import calculateTimesRayTracing_Bias_Real
from Numba_xAline import two_pointer_index
from Numba_Geiger import findTransponder

"""Instead of running final bias geiger in the sampler
    - Just compute the residuals from the travel times found with the given parameters.
    - Still need to run alignment, but no need to calculate the jacobian.
"""

@njit
def compute_log_likelihood(
    lever_guess,
    gps1_grid_guess,
    CDOG_augments,
    esv_bias,
    time_bias,
    GPS_Coordinates,
    GPS_data,
    CDOG_guess,
    CDOG_all_data,
    offsets,
    dz_array,
    angle_array,
    esv_matrix
):
    # first: compute transponder positions
    trans_coords = findTransponder(GPS_Coordinates, gps1_grid_guess, lever_guess)

    total_ssq = 0.0
    # loop each DOG
    for j in range(3):
        inv_guess = CDOG_guess + CDOG_augments[j]
        CDOG_data = CDOG_all_data[j]
        try:
            times_guess, esv = calculateTimesRayTracing_Bias_Real(inv_guess, trans_coords, esv_bias[j],
                                                                      dz_array, angle_array, esv_matrix)
            """Note doing GPS_data - time_bias to include time_bias in offset when calculating travel times"""
            CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = (
                two_pointer_index(offsets[j], 0.6, CDOG_data, GPS_data+time_bias[j], times_guess, trans_coords,
                                      esv, True)
                )
        except Exception:
            # if inversion fails, give very low likelihood
            return -np.inf
        resid = (CDOG_full - GPS_full)*1515*100
        total_ssq += np.sqrt(np.nansum(resid**2)/len(resid))
    # assume Gaussian errors with unit variance:
    return -0.5 * total_ssq

# @njit
def mcmc_sampler(
    n_iters,
    initial_lever_base,
    initial_gps_grid,
    initial_CDOG_augments,
    initial_esv_bias,
    initial_time_bias,
    dz_array, angle_array, esv_matrix,
    GPS_Coordinates, GPS_data, CDOG_guess, CDOG_all_data, offsets,
    proposal_scales=None
):
    """
    Metropolis-Hastings MCMC over {lever, gps1_grid, CDOG_augments, esv_bias, time_bias}.
    """
    # default proposal stds
    if proposal_scales is None:
        proposal_scales = {
            'lever':       np.array([0.01, 0.01, 0.1]),
            'gps_grid':    0.005,
            'CDOG_aug':    0.1,
            'esv_bias':    0.001,
            'time_bias':   0.000001,
        }

    # initialize chain
    lever_chain       = np.zeros((n_iters, 3))
    gps_chain         = np.zeros((n_iters, 4, 3))
    cdog_aug_chain    = np.zeros((n_iters, 3, 3))
    ebias_chain       = np.zeros((n_iters,3))
    tbias_chain       = np.zeros((n_iters,3))
    logpost_chain     = np.zeros(n_iters)

    # set initial state
    lever_curr    = initial_lever_base.copy()
    gps_curr      = initial_gps_grid.copy()
    cdog_aug_curr = initial_CDOG_augments.copy()
    ebias_curr    = initial_esv_bias.copy()
    tbias_curr    = initial_time_bias.copy()

    ll_curr = compute_log_likelihood(
        lever_curr, gps_curr, cdog_aug_curr, ebias_curr, tbias_curr,
        GPS_Coordinates, GPS_data, CDOG_guess, CDOG_all_data, offsets,
        dz_array, angle_array, esv_matrix
    )

    for it in range(n_iters):
        # propose new state
        lever_prop    = lever_curr    + np.random.normal(0, proposal_scales['lever'],   3)
        gps_prop      = gps_curr      + np.random.normal(0, proposal_scales['gps_grid'], gps_curr.shape)
        cdog_aug_prop = cdog_aug_curr + np.random.normal(0, proposal_scales['CDOG_aug'], cdog_aug_curr.shape)
        ebias_prop    = ebias_curr    + np.random.normal(0, proposal_scales['esv_bias'], 3)
        tbias_prop    = tbias_curr    + np.random.normal(0, proposal_scales['time_bias'], 3)

        ll_prop = compute_log_likelihood(
            lever_prop, gps_prop, cdog_aug_prop, ebias_prop, tbias_prop,
            GPS_Coordinates, GPS_data, CDOG_guess, CDOG_all_data, offsets,
            dz_array, angle_array, esv_matrix
        )

        delta = ll_prop - ll_curr

        # accept‐reject
        if delta >= 0 or np.log(np.random.rand()) < delta:
            lever_curr, gps_curr, cdog_aug_curr, ebias_curr, tbias_curr = (
                lever_prop, gps_prop, cdog_aug_prop, ebias_prop, tbias_prop
            )
            ll_curr = ll_prop

        # record
        lever_chain[it]    = lever_curr
        gps_chain[it]      = gps_curr
        cdog_aug_chain[it] = cdog_aug_curr
        ebias_chain[it]    = ebias_curr
        tbias_chain[it]    = tbias_curr
        logpost_chain[it]  = ll_curr

        if it % 100 == 0:
            ll_rounded = float(int(ll_curr * 100) / 100.0)  # round to 2 decimal places
            print("Iter", it, ": logpost =", ll_rounded)

    return {
        'lever':       lever_chain,
        'gps1_grid':   gps_chain,
        'CDOG_aug':    cdog_aug_chain,
        'esv_bias':    ebias_chain,
        'time_bias':   tbias_chain,
        'logpost':     logpost_chain
    }


# Example usage:
if __name__ == "__main__":
    # — load your data once —
    esv = sio.loadmat('../../../GPSData/global_table_esv_normal.mat')
    dz_array    = esv['distance'].flatten()
    angle_array = esv['angle'].flatten()
    esv_matrix  = esv['matrice']

    downsample = 50
    data = np.load('../../../GPSData/Processed_GPS_Receivers_DOG_1.npz')
    GPS_Coordinates = data['GPS_Coordinates'][::downsample]
    GPS_data        = data['GPS_data'][::downsample]
    # CDOG_guess      = data['CDOG_guess']
    CDOG_guess = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])

    CDOG_all_data = []
    for i in (1,3,4):
        tmp = sio.loadmat(f'../../../GPSData/DOG{i}-camp.mat')['tags'].astype(float)
        tmp[:,1] /= 1e9
        CDOG_all_data.append(tmp)

    offsets = np.array([1866.0, 3175.0, 1939.0])

    # initial parameters
    init_lever    = np.array([-12.4659, 9.6021, -13.2993])
    init_gps_grid = np.array([[ 0.0,      0.0,      0.0    ],
                              [-2.393414,-4.223503, 0.029415],
                              [-12.095685,-0.945685, 0.004397],
                              [-8.686741, 5.169188,-0.024993]])
    init_aug     = np.array([[-397.63809, 371.47355, 773.26347],
                             [825.31541, -110.93683, -734.15039],
                             [236.27742, -1307.44426, -2189.59746]])
    init_ebias   = np.array([-0.4775, -0.3199, 0.1122])
    init_tbias   = np.array([0.01518602, 0.015779, 0.018898])

    chain = mcmc_sampler(
        n_iters=20000,
        initial_lever_base=init_lever,
        initial_gps_grid=init_gps_grid,
        initial_CDOG_augments=init_aug,
        initial_esv_bias=init_ebias,
        initial_time_bias=init_tbias,
        dz_array=dz_array,
        angle_array=angle_array,
        esv_matrix=esv_matrix,
        GPS_Coordinates=GPS_Coordinates,
        GPS_data=GPS_data,
        CDOG_guess=CDOG_guess,
        CDOG_all_data=CDOG_all_data,
        offsets=offsets
    )

    """Saving and plotting the chain"""
    np.savez(
        'mcmc_chain.npz',
        lever=chain['lever'],
        gps1_grid=chain['gps1_grid'],
        CDOG_aug=chain['CDOG_aug'],
        esv_bias=chain['esv_bias'],
        time_bias=chain['time_bias'],
        logpost=chain['logpost']
    )

    #Trace Plots
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8,10), sharex=True)
    axes[0].plot(chain['lever'][:,0]); axes[0].set_ylabel('lever x')
    axes[1].plot(chain['lever'][:,1]); axes[1].set_ylabel('lever y')
    axes[2].plot(chain['lever'][:,2]); axes[2].set_ylabel('lever z')
    axes[3].plot(chain['esv_bias']);       axes[3].set_ylabel('ESV bias')
    axes[4].plot(chain['time_bias']);      axes[4].set_ylabel('time bias')
    plt.xlabel('Iteration')
    plt.show()

    # Marginal Histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    axes[0].hist(chain['lever'][:, 0], bins=30);
    axes[0].set_title('lever x')
    axes[1].hist(chain['lever'][:, 1], bins=30);
    axes[1].set_title('lever y')
    axes[2].hist(chain['lever'][:, 2], bins=30);
    axes[2].set_title('lever z')
    axes[3].hist(chain['esv_bias'], bins=30);
    axes[3].set_title('ESV bias')
    axes[4].hist(chain['time_bias'], bins=30);
    axes[4].set_title('time bias')
    axes[5].axis('off')
    plt.show()

    # Corner Plot
    import itertools
    pars = {
        'lx': chain['lever'][:, 0],
        'ly': chain['lever'][:, 1],
        'lz': chain['lever'][:, 2],
        'esv': chain['esv_bias'],
        'tmb': chain['time_bias']
    }
    keys = list(pars)
    fig, axes = plt.subplots(len(keys), len(keys), figsize=(12, 12))
    for i, j in itertools.product(range(len(keys)), range(len(keys))):
        if i == j:
            axes[i, j].hist(pars[keys[i]], bins=30)
        else:
            axes[i, j].plot(pars[keys[j]], pars[keys[i]], '.', ms=1, alpha=0.3)
        if i == len(keys) - 1: axes[i, j].set_xlabel(keys[j])
        if j == 0:        axes[i, j].set_ylabel(keys[i])
    plt.tight_layout()
    plt.show()

    # # ACF Plots
    # from statsmodels.graphics.tsaplots import plot_acf
    # for name, vals in pars.items():
    #     plot_acf(vals, lags=50, title=f"ACF of {name}")
    #     plt.show()






