import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.graphics.tsaplots import plot_acf
import itertools


"""
Color code plots with RMSE and also add in the best estimates from previous algorithms (starting positions)
    Add in all of the parameters to the plots
    Separate the time bias and esv biases into their own plots (or normalize them in some way)
"""
def trace_plot(chain, initial_params = None):
    # Trace Plots
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 10), sharex=True)
    axes[0].plot(chain['lever'][:, 0]);
    axes[0].set_ylabel('lever x')
    axes[1].plot(chain['lever'][:, 1]);
    axes[1].set_ylabel('lever y')
    axes[2].plot(chain['lever'][:, 2]);
    axes[2].set_ylabel('lever z')
    axes[3].plot(chain['esv_bias']);
    axes[3].set_ylabel('ESV bias')
    axes[4].plot(chain['time_bias']);
    axes[4].set_ylabel('time bias')
    axes[5].plot(chain['logpost'] * -2);
    axes[5].set_ylabel('RMSE')
    plt.xlabel('Iteration')
    plt.show()

def marginal_hists(chain, initial_params = None):
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

# def corner_plot(chain, initial_params = None):
#     # Corner Plot
#     pars = {
#         'lx': chain['lever'][:, 0],
#         'ly': chain['lever'][:, 1],
#         'lz': chain['lever'][:, 2],
#         'esv': chain['esv_bias'],
#         'tmb': chain['time_bias']
#     }
#     keys = list(pars)
#     fig, axes = plt.subplots(len(keys), len(keys), figsize=(12, 12))
#     for i, j in itertools.product(range(len(keys)), range(len(keys))):
#         if i == j:
#             axes[i, j].hist(pars[keys[i]], bins=30)
#         else:
#             axes[i, j].plot(pars[keys[j]], pars[keys[i]], '.', ms=1, alpha=0.3)
#         if i == len(keys) - 1: axes[i, j].set_xlabel(keys[j])
#         if j == 0:        axes[i, j].set_ylabel(keys[i])
#     plt.tight_layout()
#     plt.show()

def corner_plot(chain, initial_params=None, downsample=1):
    # Extract parameter arrays
    pars = {
        'lx': chain['lever'][::downsample, 0],
        'ly': chain['lever'][::downsample, 1],
        'lz': chain['lever'][::downsample, 2],
        'augx': chain['CDOG_aug'][::downsample, 0, 0],
        'augy': chain['CDOG_aug'][::downsample, 0, 1],
        'augz': chain['CDOG_aug'][::downsample, 0, 2],
    }
    keys = list(pars)
    n = len(keys)

    # Set up figure and axes
    fig, axes = plt.subplots(n, n, figsize=(12, 8))

    # Normalize log-posterior for color mapping
    logpost = chain['logpost'][::downsample]
    norm = mpl.colors.Normalize(vmin=logpost.min(), vmax=logpost.max())
    cmap = plt.get_cmap('viridis')

    # Loop over panels
    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]
        if i == j:
            ax.hist(pars[keys[i]], bins=30, color='gray')
        else:
            sc = ax.scatter(
                pars[keys[j]], pars[keys[i]],
                c=logpost, cmap=cmap, norm=norm,
                s=1, alpha=0.8
            )
        # Labeling
        if i == n - 1:
            ax.set_xlabel(keys[j])
        if j == 0:
            ax.set_ylabel(keys[i])
        # Turn off upper triangle if you only want the lower
        if j > i:
            ax.set_visible(False)

    # Adjust layout and add colorbar
    plt.tight_layout()
    # place colorbar on the right spanning all rows
    cbar = fig.colorbar(sc, ax=axes[:, :], location='right', shrink=0.9)
    cbar.set_label('log posterior')

    plt.show()

def acf_plots(chain, initial_params = None):
    # ACF Plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, key in enumerate(['lever', 'esv_bias', 'time_bias']):
        plot_acf(chain[key], lags=50, ax=axes[i])
        axes[i].set_title(f'ACF of {key}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initial Parameters for adding to plot
    init_lever = np.array([-12.4659, 9.6021, -13.2993])
    init_gps_grid = np.array([[0.0, 0.0, 0.0],
                              [-2.393414, -4.223503, 0.029415],
                              [-12.095685, -0.945685, 0.004397],
                              [-8.686741, 5.169188, -0.024993]])
    init_aug = np.array([[-397.63809, 371.47355, 773.26347],
                         [825.31541, -110.93683, -734.15039],
                         [236.27742, -1307.44426, -2189.59746]])
    init_ebias = np.array([-0.4775, -0.3199, 0.1122])
    init_tbias = np.array([0.01518602, 0.015779, 0.018898])

    chain = np.load('mcmc_chain_slurm.npz')

    # trace_plot(chain)
    # marginal_hists(chain)
    corner_plot(chain, downsample=50)

