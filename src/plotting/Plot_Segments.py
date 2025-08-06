from data import gps_output_path
from GeigerMethod.Synthetic.Numba_Functions.MCMC_best_sample import (
    load_min_logpost_params,
)


def plot_combined_segments(n_splits, path):
    """
    Plot the combined segments from multiple MCMC runs.

    Parameters:
    path (str): Path to the directory containing the split data files.
    """

    for i in range(n_splits):
        sample = load_min_logpost_params(f"{path}/split_{i}.npz")

        # Plotting lever chain
        print(sample["lever"], sample["CDOG_aug"][2])


if __name__ == "__main__":
    path = gps_output_path("7_individual_splits_esv_20250806_143356")
    plot_combined_segments(7, path)
