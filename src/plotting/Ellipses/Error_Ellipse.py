import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def compute_error_ellipse(
    data_xy: np.ndarray, confidence: float = 0.68, zorder: int = 1
):
    """
    Compute an error ellipse patch and the percentage of points within it.

    Parameters
    ----------
    data_xy : (N, 2) array
        2D points for which to compute the error ellipse.
    confidence : float
        Confidence level for the ellipse (e.g., 0.68 for ~68%).

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        Ellipse representing the specified confidence region.
    pct : float
        Percentage of points lying within the ellipse.
    """
    # Compute covariance and mean
    cov = np.cov(data_xy, rowvar=False)
    mean = data_xy.mean(axis=0)

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Compute ellipse parameters
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    radius = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(vals * radius)

    # Create ellipse patch
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor="black",
        fc="none",
        lw=1,
        zorder=zorder,
    )

    # Compute Mahalanobis distances for coverage
    inv_cov = np.linalg.inv(cov)
    diffs = data_xy - mean
    d2 = np.einsum("nj,jk,nk->n", diffs, inv_cov, diffs)
    pct = np.mean(d2 <= radius) * 100

    return ellipse, pct
