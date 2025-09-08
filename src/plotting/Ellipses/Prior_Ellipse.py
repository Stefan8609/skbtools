import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def plot_prior_ellipse(mean=0, cov=None, confidence=0.68, zorder=1, label=None):
    if cov is None:
        cov = np.eye(2)
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Compute ellipse parameters
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    radius = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(vals * radius)

    # Create ellipse patch
    if label is not None:
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="blue",
            fc="none",
            lw=1,
            label=label,
            zorder=zorder,
        )
    else:
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="blue",
            fc="none",
            lw=1,
            zorder=zorder,
        )

    return ellipse
