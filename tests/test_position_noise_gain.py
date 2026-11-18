import numpy as np

from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from plotting.Analysis_Plots.combined_dependence import position_noise_gain

OTHERS = np.array(
    [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.03], [-12.1, -0.95, 0.004], [-8.7, 5.17, 0.025]]
)
LEVER = np.array([-12.5, 9.6, -13.3])


def test_gain_matches_monte_carlo_range_error():
    rng = np.random.default_rng(0)
    pitch = 0.05
    rotation = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    # One epoch, DOG 5000 m below and 1000 m to the side.
    coordinates = (rotation @ OTHERS.T).T[None] + np.array([0.0, 0.0, 5000.0])
    cdog = np.array([1000.0, 0.0, 0.0])
    truth = findTransponder(coordinates, OTHERS, LEVER)[0]
    unit = (truth - cdog) / np.linalg.norm(truth - cdog)

    sigma = 1e-3
    range_errors = np.empty(4000)
    for i in range(len(range_errors)):
        noisy = coordinates + rng.normal(0.0, sigma, coordinates.shape)
        range_errors[i] = unit @ (findTransponder(noisy, OTHERS, LEVER)[0] - truth)

    g = position_noise_gain(coordinates, OTHERS, LEVER, cdog)
    assert np.isclose(g, range_errors.std() / sigma, rtol=0.05)
