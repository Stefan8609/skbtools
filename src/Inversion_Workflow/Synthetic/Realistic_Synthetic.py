"""Realistic synthetic one-way GNSS-A data: the paper's generator plus six knobs.

The paper's synthetics (App. B) have iid noise and a time-invariant sound speed. They
leave about 4.5 cm of range residual against 15-24 cm in the Bermuda data, so they
miss the error that dominates the real data (R3 M3, R4 4). ``realistic_synthetic``
starts from the paper's generator and adds the obstacles a one-way system meets, one
block each, in the order they act on the data:

1. Sound speed: an ESV change dc(t, x), depth-uniform, through the same ESV-bias term
   the inversion estimates. A time series (random walk, or the shape of our own
   estimated common dc(t)), an optional sinusoid, and a linear horizontal gradient.
2. Current: a depth-mean horizontal current, steady plus an optional rotating M2
   tidal part. One-way ranging does not cancel it.
3. Clock and transmit side: offset and linear drift (as in ``bermuda_trajectory``),
   plus a constant transmit-side delay.
4. Vessel GNSS: a common-mode Gauss-Markov error that moves the transducer 1:1, and
   a per-antenna (relative) error that tilts the vessel frame through the 20 m lever.
5. Detection: timing noise and dropped arrivals (none, random, or the real pattern).
6. Geometry: the real Bermuda track, or an idealized circle, grid or cross, with or
   without the lost upper-left quadrant of the GNSS rejection.

Every knob lives in one dict (``ZERO_KNOBS`` lists them with units in the names) and
defaults to off. With every knob off the output equals ``bermuda_trajectory`` bit for
bit, given the same seed.

Conventions
-----------
- Time: GPS seconds, as in ``GPS_data``.
- Frame: local ENU at ``SITE_ORIGIN_ECEF`` (the CDOG reference of the inversion), m.
- Arrival = emission time + travel time + delays + clock offset + drift + noise. A
  delay is positive when the sound arrives late.
- dc is added to the ESV (m/s), as the inversion's ``esv_bias``: positive dc means
  faster sound and earlier arrivals.
"""

from functools import lru_cache

import numpy as np
import scipy.io as sio
from scipy.signal import lfilter

from bermuda_data import gps_data_path
from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias_Real,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Synthetic.Synthetic_Bermuda_Trajectory import (
    bermuda_trajectory,
)

M2_PERIOD_H = 12.4206
SITE_ORIGIN_ECEF = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
# Same values as the defaults inside bermuda_trajectory.
GPS1_TO_OTHERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.39341409, -4.22350344, 0.02941493],
        [-12.09568416, -0.94568462, 0.0043972],
        [-8.68674054, 5.16918806, 0.02499322],
    ]
)
GPS1_TO_TRANSPONDER = np.array([-12.4659, 9.6021, -13.2993])
CDOG_AUGMENTS = {
    1: np.array([-396.80833750, 370.23261256, 774.14404697]),
    3: np.array([826.18512886, -111.85558415, -733.73615228]),
    4: np.array([235.77476616, -1305.38300727, -2190.89445398]),
}

# Common dc(t) estimated from the Bermuda data: posterior mean of the "ar1_common30"
# model (one dc series shared by DOGs 1/3/4 on 30-min windows, zero mean over the
# windows), local chains at commit d2d00cb. Window edges in GPS seconds; values in
# m/s. It contains vessel-side GNSS error as well as ocean (REVISION_REFERENCE K7).
OBSERVED_DC_EDGES_GPS_S = np.array(
    [20116.0, 23716.0] + list(np.arange(25516.0, 68716.1, 1800.0)) + [70400.0]
)
OBSERVED_DC_MPS = 1e-3 * np.array(
    [-10.5, -24.0, -30.2, -13.9, -43.3, -47.7, -68.0, -53.8, -16.1, -5.3, -14.6,
     -24.9, 12.6, 3.6, -16.8, -51.6, -37.7, -10.4, -3.5, -9.1, 51.3, 124.7, 39.1,
     46.3, 63.8, 53.3, 86.6]
)  # fmt: skip

# Real-data alignment, used only to copy which pings each DOG received.
REAL_DOG_OFFSET_S = {1: 1866.016, 3: 3175.017, 4: 1939.0178}
REAL_PAIR_GATE_S = 0.05

# Idealized tracks: speed of the Bermuda survey (median), 1 Hz pings, and a survey
# as long as the Bermuda one.
TRACK_SPEED_MPS = 2.2
CIRCLE_RADIUS_M = 5200.0  # about the water depth
GRID_HALF_WIDTH_M = 7000.0
GRID_LINE_SPACING_M = 2000.0
# The cross is centred on one DOG, so that DOG has a track centred on it and the
# other two (about 2 km away) an offset one.
CROSS_DOG = 3
CROSS_HALF_LENGTH_M = 7000.0

ZERO_KNOBS = {
    # 1. Sound speed: dc added to the ESV, m/s, depth-uniform.
    "dc_series": "none",  # "none" | "random_walk" | "observed"
    "dc_random_walk_mps_per_sqrt_h": 0.0,
    "dc_sinusoid_mps": 0.0,  # amplitude
    "dc_sinusoid_period_h": M2_PERIOD_H,
    "dc_sinusoid_phase_rad": None,  # None: drawn from the seed
    "dc_gradient_mps_per_km": (0.0, 0.0),  # east, north
    # 2. Depth-mean current, m/s, east and north.
    "current_mps": (0.0, 0.0),
    "current_m2_mps": 0.0,  # amplitude of a circular, clockwise M2 tidal current
    "current_m2_phase_rad": None,
    # 3. Clock and transmit side.
    "offset_s": 1991.01236648,
    "drift_rate_s_per_s": 0.0,
    "transmit_delay_s": 0.0,
    # 4. Vessel GNSS.
    "gnss_common_sd_m": (0.0, 0.0, 0.0),  # east, north, up
    "gnss_common_tau_s": 1800.0,
    "gnss_relative": "none",  # "none" | "real_misfit" | "real_misfit_shifted"
    "position_noise_m": 0.0,  # iid per antenna and axis
    # 5. Detection.
    "timing_noise_s": 0.0,
    "dropouts": "none",  # "none" | "random" | "real_pattern"
    "dropout_fraction": 0.15,  # for "random"
    # 6. Geometry.
    "track": "bermuda",  # "bermuda" | "circle" | "grid" | "cross"
    "lost_quadrant": False,
}
KNOB_CHOICES = {
    "dc_series": ("none", "random_walk", "observed"),
    "gnss_relative": ("none", "real_misfit", "real_misfit_shifted"),
    "dropouts": ("none", "random", "real_pattern"),
    "track": ("bermuda", "circle", "grid", "cross"),
}
# Independent random streams, so switching one knob on leaves the others' draws.
STREAM = {
    "dc_random_walk": 1,
    "dc_sinusoid_phase": 2,
    "current_m2_phase": 3,
    "gnss_common": 4,
    "gnss_relative_shift": 5,
    "dropouts": 6,
}


def check_knobs(knobs):
    """Return ZERO_KNOBS updated with ``knobs``; raise on unknown keys or values."""
    unknown = sorted(set(knobs) - set(ZERO_KNOBS))
    if unknown:
        raise ValueError(f"unknown knobs {unknown}; expected keys of ZERO_KNOBS")
    k = {**ZERO_KNOBS, **knobs}
    for name, choices in KNOB_CHOICES.items():
        if k[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {k[name]!r}")
    for name, size in (
        ("dc_gradient_mps_per_km", 2),
        ("current_mps", 2),
        ("gnss_common_sd_m", 3),
    ):
        if np.shape(k[name]) != (size,):
            raise ValueError(f"{name} needs {size} values, got {k[name]!r}")
    for name in (
        "dc_random_walk_mps_per_sqrt_h",
        "timing_noise_s",
        "position_noise_m",
        "current_m2_mps",
        "dc_sinusoid_mps",
    ):
        if k[name] < 0:
            raise ValueError(f"{name} must be >= 0, got {k[name]}")
    if np.any(np.asarray(k["gnss_common_sd_m"]) < 0) or k["gnss_common_tau_s"] <= 0:
        raise ValueError("gnss_common_sd_m must be >= 0 and gnss_common_tau_s > 0")
    if not 0 <= k["dropout_fraction"] < 1:
        raise ValueError(
            f"dropout_fraction must be in [0, 1), got {k['dropout_fraction']}"
        )
    return k


def enu_rotation():
    """3x3 rotation from ECEF vectors to local ENU at ``SITE_ORIGIN_ECEF``."""
    x, y, z = SITE_ORIGIN_ECEF
    lon = np.arctan2(y, x)
    # Geodetic latitude (WGS84, one Bowring step is exact to < 1e-9 rad here).
    e2 = 6.69437999014e-3
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    n = 6378137.0 / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def dc_series_mps(GPS_data, knobs, seed):
    """Common dc(t), m/s: a random walk or the observed series, plus a sinusoid.

    The random walk starts at 0 at the first epoch and has increments of
    ``dc_random_walk_mps_per_sqrt_h`` per sqrt(hour), drawn on a 1 s grid. The
    observed series is piecewise constant on its 30-min windows and held at its end
    values outside them.
    """
    t_h = (GPS_data - GPS_data[0]) / 3600.0
    dc = np.zeros(len(GPS_data))
    if knobs["dc_series"] == "random_walk":
        rng = np.random.default_rng([seed, STREAM["dc_random_walk"]])
        grid_s = np.arange(0.0, GPS_data[-1] - GPS_data[0] + 1.0)
        steps = rng.normal(0.0, np.sqrt(1.0 / 3600.0), len(grid_s))
        steps[0] = 0.0
        walk = knobs["dc_random_walk_mps_per_sqrt_h"] * np.cumsum(steps)
        dc += np.interp(GPS_data - GPS_data[0], grid_s, walk)
    elif knobs["dc_series"] == "observed":
        window = np.searchsorted(OBSERVED_DC_EDGES_GPS_S, GPS_data, side="right") - 1
        dc += OBSERVED_DC_MPS[np.clip(window, 0, len(OBSERVED_DC_MPS) - 1)]
    if knobs["dc_sinusoid_mps"] > 0:
        phase = knobs["dc_sinusoid_phase_rad"]
        if phase is None:
            rng = np.random.default_rng([seed, STREAM["dc_sinusoid_phase"]])
            phase = rng.uniform(0.0, 2 * np.pi)
        dc += knobs["dc_sinusoid_mps"] * np.cos(
            2 * np.pi * t_h / knobs["dc_sinusoid_period_h"] + phase
        )
    return dc


def dc_gradient_mps(transponder_enu, CDOG_enu, gradient_mps_per_km):
    """dc of a linear horizontal gradient, m/s, averaged along each straight ray.

    dc(x) = g . x with x the horizontal ENU position (km) from the site origin. For a
    depth-uniform linear field the average along a straight ray is the value at the
    ray's horizontal midpoint, so DOGs and ship positions in different places see
    different sound speeds.
    """
    midpoint_km = 0.5 * (transponder_enu[:, :2] + CDOG_enu[:2]) / 1000.0
    return midpoint_km @ np.asarray(gradient_mps_per_km, dtype=float)


def current_en_mps(GPS_data, knobs, seed):
    """Depth-mean current (n, 2), m/s: steady plus a circular, clockwise M2 part."""
    current = np.tile(np.asarray(knobs["current_mps"], dtype=float), (len(GPS_data), 1))
    if knobs["current_m2_mps"] > 0:
        phase = knobs["current_m2_phase_rad"]
        if phase is None:
            rng = np.random.default_rng([seed, STREAM["current_m2_phase"]])
            phase = rng.uniform(0.0, 2 * np.pi)
        angle = 2 * np.pi * (GPS_data - GPS_data[0]) / (3600.0 * M2_PERIOD_H) + phase
        current += knobs["current_m2_mps"] * np.column_stack(
            (np.cos(angle), -np.sin(angle))
        )
    return current


def current_delay_s(transponder_enu, CDOG_enu, travel_times, current_mps):
    """Travel-time change from a depth-mean current, s.

    Sound moves with the water, so to first order in u/c,
    dT = -integral(u . s_hat / c^2 ds) = -(u . dx_h) / c_eff^2, with dx_h the
    horizontal vector from the transducer to the DOG and c_eff = R / T the ping's
    effective sound speed (straight-ray approximation, within ~2 % of a ray-traced
    integral down to 25 deg elevation). A current flowing towards the DOG shortens
    the travel time. Unlike two-way ranging, nothing cancels it.
    """
    to_dog = CDOG_enu - transponder_enu
    slant = np.linalg.norm(to_dog, axis=1)
    along = np.sum(current_mps * to_dog[:, :2], axis=1)
    return -along * travel_times**2 / slant**2


def gauss_markov(t_s, sd, tau_s, rng):
    """First-order Gauss-Markov series on ``t_s`` (s), one column per entry of sd.

    Stationary with standard deviation ``sd`` and correlation exp(-|dt| / tau_s).
    Drawn on a 1 s grid from the first time and interpolated, so DOGs that share
    epochs share the series.
    """
    sd = np.asarray(sd, dtype=float)
    grid_s = np.arange(0.0, t_s[-1] - t_s[0] + 1.0)
    phi = np.exp(-1.0 / tau_s)
    innovations = rng.normal(0.0, 1.0, (len(grid_s), len(sd)))
    innovations[1:] *= np.sqrt(1.0 - phi**2)
    series = lfilter([1.0], [1.0, -phi], innovations, axis=0) * sd
    return np.column_stack(
        [np.interp(t_s - t_s[0], grid_s, series[:, k]) for k in range(len(sd))]
    )


@lru_cache(maxsize=4)
def real_vessel(DOG_num):
    """Real Bermuda antennas, their rigid-grid fit, and the body-to-ECEF rotations.

    The rigid fit is the one ``bermuda_trajectory`` uses as the truth: GPS1 at its
    real position and the grid rotated to the real attitude. ``real - rigid`` is the
    real per-antenna (rigid-body) misfit, antenna 4's decimetre jumps included.
    """
    data = np.load(gps_data_path(f"GPS_Data/Processed_GPS_Receivers_DOG_{DOG_num}.npz"))
    real = data["GPS_Coordinates"]
    rotations = np.empty((len(real), 3, 3))
    for i in range(len(real)):
        rotations[i], _ = findRotationAndDisplacement(
            GPS1_TO_OTHERS.T, (real[i] - real[i, 0]).T
        )
    rigid = real[:, :1, :] + np.einsum("nij,kj->nki", rotations, GPS1_TO_OTHERS)
    return real, rigid, rotations, data["GPS_data"]


def real_dropout_mask(DOG_num, dz_array, angle_array, esv_matrix):
    """True at each real Bermuda GPS epoch whose ping the real DOG received.

    A ping counts as received when a real arrival lies within ``REAL_PAIR_GATE_S`` of
    emission + travel time + the DOG's real clock offset, with the travel time of the
    synthetic geometry (rigid real track, synthetic DOG and lever). That geometry is
    within about a metre of the real one, i.e. about 1 ms, far inside the gate.
    """
    tags = sio.loadmat(gps_data_path(f"CDOG_Data/DOG{DOG_num}-camp.mat"))["tags"]
    arrivals = np.sort(tags[:, 0] + tags[:, 1] / 1e9)  # fraction stored in ns
    _, rigid, _, GPS_data = real_vessel(DOG_num)
    transponder = findTransponder(rigid, GPS1_TO_OTHERS, GPS1_TO_TRANSPONDER)
    times, _ = calculateTimesRayTracing_Bias_Real(
        SITE_ORIGIN_ECEF + CDOG_AUGMENTS[DOG_num],
        transponder,
        0.0,
        dz_array,
        angle_array,
        esv_matrix,
    )
    expected = GPS_data + times + REAL_DOG_OFFSET_S[DOG_num]
    index = np.clip(np.searchsorted(arrivals, expected), 1, len(arrivals) - 1)
    gap = np.minimum(
        np.abs(arrivals[index] - expected), np.abs(arrivals[index - 1] - expected)
    )
    return gap < REAL_PAIR_GATE_S


def random_dropout_mask(n_epochs, fraction, rng):
    """Keep-mask with about ``fraction`` dropped: half isolated, half in blocks.

    Blocks are 50-500 epochs long, as dropouts in the real data come in runs.
    """
    keep = rng.random(n_epochs) >= fraction / 2
    while np.sum(~keep) < fraction * n_epochs:
        length = min(int(rng.integers(50, 501)), n_epochs)
        start = int(rng.integers(0, n_epochs - length + 1))
        keep[start : start + length] = False
    return keep


def track_en_m(track, n_epochs, centre_en_m):
    """Ship (GPS1) east, north positions (n, 2), m, and heading (rad from north).

    ``"circle"``: radius ``CIRCLE_RADIUS_M`` around ``centre_en_m``. ``"grid"``:
    east-west lines ``GRID_LINE_SPACING_M`` apart over +-``GRID_HALF_WIDTH_M``.
    ``"cross"``: an east-west and a north-south line, +-``CROSS_HALF_LENGTH_M``
    through ``centre_en_m``, sailed centre-east-west-centre-north-south-centre so
    the ship never leaves the two lines. Sailed at ``TRACK_SPEED_MPS`` from the
    start of the path, repeating it if the survey is longer than the path.
    """
    if track == "circle":
        angle = np.linspace(0.0, 2 * np.pi, 721)
        vertices = CIRCLE_RADIUS_M * np.column_stack((np.sin(angle), np.cos(angle)))
    elif track == "grid":
        lines = np.arange(
            -GRID_HALF_WIDTH_M, GRID_HALF_WIDTH_M + 1.0, GRID_LINE_SPACING_M
        )
        vertices = []
        for k, north in enumerate(lines):
            ends = [-GRID_HALF_WIDTH_M, GRID_HALF_WIDTH_M]
            if k % 2:
                ends = ends[::-1]
            vertices += [(ends[0], north), (ends[1], north)]
        vertices = np.array(vertices)
    elif track == "cross":
        arm = CROSS_HALF_LENGTH_M
        vertices = np.array(
            [(0, 0), (arm, 0), (-arm, 0), (0, 0), (0, arm), (0, -arm), (0, 0)],
            dtype=float,
        )
    else:
        raise ValueError(f"track must be 'circle', 'grid' or 'cross', got {track!r}")
    segment = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    along = np.concatenate(([0.0], np.cumsum(segment)))
    s = (TRACK_SPEED_MPS * np.arange(n_epochs)) % along[-1]
    position = np.column_stack(
        (np.interp(s, along, vertices[:, 0]), np.interp(s, along, vertices[:, 1]))
    )
    step = np.diff(vertices, axis=0)
    heading = np.arctan2(step[:, 0], step[:, 1])
    leg = np.clip(np.searchsorted(along, s, side="right") - 1, 0, len(step) - 1)
    return position + centre_en_m, heading[leg]


def idealized_trajectory(
    track,
    time_noise,
    position_noise,
    esv_bias,
    time_bias,
    dz_array,
    angle_array,
    esv_matrix,
    offset=1991.01236648,
    gps1_to_others=None,
    gps1_to_transponder=None,
    DOG_num=3,
    drift_rate=0.0,
):
    """``bermuda_trajectory`` on an idealized track (same arguments and returns).

    The ship sails ``track`` ("circle", "grid" or "cross", see ``track_en_m``) at
    ``TRACK_SPEED_MPS`` with 1 Hz pings, for as long as the Bermuda survey. The
    circle and grid are centred on the centre of the three DOGs, the cross on DOG
    ``CROSS_DOG`` whichever DOG is generated, so the three DOGs of one survey share
    one track. Attitude: the real Bermuda roll and pitch (epoch by epoch), with the
    yaw turned so the body x axis follows the track heading. GPS1 keeps the
    real height series (heave, tide). The travel times, clock and noise follow
    ``bermuda_trajectory`` line for line.
    """
    if gps1_to_others is None:
        gps1_to_others = GPS1_TO_OTHERS
    if gps1_to_transponder is None:
        gps1_to_transponder = GPS1_TO_TRANSPONDER
    CDOG = SITE_ORIGIN_ECEF + CDOG_AUGMENTS[DOG_num]
    real, _, rotations, real_GPS_data = real_vessel(DOG_num)
    n = int(real_GPS_data[-1] - real_GPS_data[0]) + 1
    GPS_data = real_GPS_data[0] + np.arange(n, dtype=float)

    to_enu = enu_rotation()
    if track == "cross":
        centre_en = (to_enu @ CDOG_AUGMENTS[CROSS_DOG])[:2]
    else:
        centre_en = np.mean([to_enu @ CDOG_AUGMENTS[d] for d in CDOG_AUGMENTS], axis=0)
        centre_en = centre_en[:2]
    position_en, heading = track_en_m(track, n, centre_en)
    real_index = np.arange(n) % len(real)
    gps1_up = ((real[:, 0, :] - SITE_ORIGIN_ECEF) @ to_enu.T)[real_index, 2]
    body_to_enu = np.einsum("ij,njk->nik", to_enu, rotations[real_index])
    # Turn clockwise about the vertical by the difference between the track heading
    # and the heading of the body x axis (both from north).
    turn = heading - np.arctan2(body_to_enu[:, 0, 0], body_to_enu[:, 1, 0])
    yaw = np.zeros((n, 3, 3))
    yaw[:, 0, 0] = yaw[:, 1, 1] = np.cos(turn)
    yaw[:, 0, 1] = np.sin(turn)
    yaw[:, 1, 0] = -np.sin(turn)
    yaw[:, 2, 2] = 1.0
    body_to_enu = np.einsum("nij,njk->nik", yaw, body_to_enu)
    gps1_enu = np.column_stack((position_en, gps1_up))
    antennas_enu = gps1_enu[:, None, :] + np.einsum(
        "nij,kj->nki", body_to_enu, gps1_to_others
    )
    GPS_Coordinates = SITE_ORIGIN_ECEF + antennas_enu @ to_enu

    transponder_coordinates = findTransponder(
        GPS_Coordinates, gps1_to_others, gps1_to_transponder
    )
    synthetic_travel_times, _ = calculateTimesRayTracing_Bias_Real(
        CDOG, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
    )
    synthetic_travel_times = synthetic_travel_times + time_bias
    CDOG_time = (
        GPS_data
        + synthetic_travel_times
        + np.random.normal(0, time_noise, len(GPS_data))
        + offset
        + drift_rate * (GPS_data - 0.5 * (GPS_data[0] + GPS_data[-1]))
    )
    CDOG_remain, CDOG_int = np.modf(CDOG_time)
    CDOG_data = np.stack((CDOG_int, CDOG_remain), axis=0).T
    GPS_Coordinates += np.random.normal(0, position_noise, GPS_Coordinates.shape)
    return CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates


def realistic_synthetic(DOG_num, knobs, seed, dz_array, angle_array, esv_matrix):
    """Synthetic data for one DOG with the obstacles switched on in ``knobs``.

    ``knobs`` overrides entries of ``ZERO_KNOBS``. ``seed`` (int >= 0) fixes every
    random draw: the paper generator's timing and iid antenna noise come from
    ``np.random.seed([seed, DOG_num])``. The ocean, the ship's GNSS error and the
    tidal phases come from streams of ``seed`` alone, so the three DOGs of one
    survey share them. The ESV table is the generation table; dc and currents are
    perturbations about it.

    Returns ``(CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates,
    truth)``. The first five are as in ``bermuda_trajectory``, so every inversion runs
    unchanged. ``truth`` holds the knobs, the true clock offset (Eq. 6 dt, s), and,
    per GPS epoch before dropouts and quadrant loss, each obstacle's travel-time
    contribution (``delay_s``), dc (m/s), the current (m/s, E/N) and the common GNSS
    error (m, ENU).
    """
    k = check_knobs(knobs)
    if DOG_num not in CDOG_AUGMENTS:
        raise ValueError(
            f"DOG_num must be one of {sorted(CDOG_AUGMENTS)}, got {DOG_num}"
        )

    # Geometry, ideal travel times, clock offset and drift, iid timing and antenna
    # noise: the paper's generator (or the same steps on an idealized track).
    np.random.seed([seed, DOG_num])
    common = (k["timing_noise_s"], k["position_noise_m"], 0.0, 0.0)
    tables = (dz_array, angle_array, esv_matrix)
    clock = {"offset": k["offset_s"], "DOG_num": DOG_num}
    clock["drift_rate"] = k["drift_rate_s_per_s"]
    if k["track"] == "bermuda":
        data = bermuda_trajectory(*common, *tables, **clock)
    else:
        data = idealized_trajectory(k["track"], *common, *tables, **clock)
    CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder = data
    n = len(GPS_data)
    to_enu = enu_rotation()
    transponder_enu = (transponder - SITE_ORIGIN_ECEF) @ to_enu.T
    CDOG_enu = to_enu @ (CDOG - SITE_ORIGIN_ECEF)
    travel_times, _ = calculateTimesRayTracing_Bias_Real(
        CDOG, transponder, 0.0, *tables
    )
    delay_s = {}

    # 1. Sound speed: dc(t, x) added to the ESV through the inversion's own model.
    dc = dc_series_mps(GPS_data, k, seed) + dc_gradient_mps(
        transponder_enu, CDOG_enu, k["dc_gradient_mps_per_km"]
    )
    if np.any(dc != 0):
        perturbed, _ = calculateTimesRayTracing_Bias_Real(
            CDOG, transponder, dc, *tables
        )
        delay_s["sound_speed"] = perturbed - travel_times

    # 2. Depth-mean current: one-way ranging does not cancel it.
    current = current_en_mps(GPS_data, k, seed)
    if np.any(current != 0):
        delay_s["current"] = current_delay_s(
            transponder_enu, CDOG_enu, travel_times, current
        )

    # 3. Transmit-side delay: the ping leaves late, so every arrival is late.
    if k["transmit_delay_s"] != 0:
        delay_s["transmit"] = np.full(n, float(k["transmit_delay_s"]))

    if delay_s:
        arrival = CDOG_data[:, 0] + CDOG_data[:, 1] + sum(delay_s.values())
        fraction, integer = np.modf(arrival)
        CDOG_data = np.column_stack((integer, fraction))

    # 4. Vessel GNSS: a common error moves all antennas (and the transducer) 1:1;
    #    a per-antenna error tilts the fitted vessel frame.
    gnss_common_enu = np.zeros((n, 3))
    if np.any(np.asarray(k["gnss_common_sd_m"]) > 0):
        rng = np.random.default_rng([seed, STREAM["gnss_common"]])
        gnss_common_enu = gauss_markov(
            GPS_data, k["gnss_common_sd_m"], k["gnss_common_tau_s"], rng
        )
        GPS_Coordinates = GPS_Coordinates + (gnss_common_enu @ to_enu)[:, None, :]
    if k["gnss_relative"] != "none":
        real, rigid, _, _ = real_vessel(DOG_num)
        lag = 0
        if k["gnss_relative"] == "real_misfit_shifted":
            rng = np.random.default_rng([seed, STREAM["gnss_relative_shift"]])
            lag = int(rng.integers(0, len(real)))
        index = (np.arange(n) + lag) % len(real)
        GPS_Coordinates = GPS_Coordinates + (real - rigid)[index]

    # 5. Detection: dropped arrivals.
    keep = np.ones(n, dtype=bool)
    if k["dropouts"] == "real_pattern":
        received = real_dropout_mask(DOG_num, *tables)
        keep = received[np.arange(n) % len(received)]
    elif k["dropouts"] == "random":
        rng = np.random.default_rng([seed, DOG_num, STREAM["dropouts"]])
        keep = random_dropout_mask(n, k["dropout_fraction"], rng)
    CDOG_data = CDOG_data[keep]

    # 6. GNSS epochs lost in the upper-left (north-west) quadrant of the array, as
    #    in the Bermuda rejection; the DOG still hears those pings.
    lost = np.zeros(n, dtype=bool)
    if k["lost_quadrant"]:
        centre = np.mean([to_enu @ CDOG_AUGMENTS[d] for d in CDOG_AUGMENTS], axis=0)
        ship = transponder_enu[:, :2] - centre[:2]
        lost = (ship[:, 0] < 0) & (ship[:, 1] > 0)

    truth = {
        "seed": seed,
        "DOG_num": DOG_num,
        "knobs": k,
        "CDOG": CDOG,
        "lever": GPS1_TO_TRANSPONDER.copy(),
        "dt_s": k["offset_s"] + k["transmit_delay_s"],
        "drift_rate_s_per_s": k["drift_rate_s_per_s"],
        "GPS_data_all": GPS_data,
        "travel_times_s": travel_times,
        "delay_s": delay_s,
        "dc_mps": dc,
        "current_mps": current,
        "gnss_common_enu_m": gnss_common_enu,
        "keep": keep,
        "lost": lost,
    }
    return (
        CDOG_data,
        CDOG,
        GPS_Coordinates[~lost],
        GPS_data[~lost],
        transponder[~lost],
        truth,
    )
