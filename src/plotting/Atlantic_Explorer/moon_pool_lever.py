"""Moon-pool center of the R/V Atlantic Explorer in the GNSS body frame.

Reproduces the horizontal center of the lever-arm prior used by the Bermuda MCMC
(``MCMC_sampler.PRIOR_LEVER_CENTER_M``). The moon-pool outline was digitized from
the top view of the R/V Atlantic Explorer General Arrangement drawings
(Rev 02-Jun-2014); the pixel coordinates, pixel shifts and scale are the ones
used for the top view in ``Atlantic_Explorer_Plot.py`` (Fig. 8).
"""

import numpy as np

# Digitized moon-pool outline, top view, in drawing pixels.
MOONPOOL_SEGMENTS_PX = [
    ((413.96794872, 366.93099548), (432.05656109, 366.93099548)),
    ((432.05656109, 366.93099548), (432.05656109, 347.07126697)),
    ((432.05656109, 347.07126697), (413.96794872, 347.07126697)),
    ((413.96794872, 347.07126697), (413.96794872, 366.93099548)),
]
MOONPOOL_X_SHIFT_PX = -3
MOONPOOL_Y_SHIFT_PX = 276 + 158.03
TOP_VIEW_SCALE_M_PER_PX = 0.054715

# GPS1 position in the drawing frame (x forward from the stern, y to port, z up
# from the keel), measured from the schematic, in metres.
GPS1_OFFSET = (39.5, 2.2, 14.2)
# Angle from the drawing x-axis to the GNSS body-frame x-axis, in degrees. The
# body x-axis is perpendicular to, and bisects, the line between the two forward
# receivers (the GNSS grid orientation).
ROTATION_DEG = 29.5


def moon_pool_center_body_frame():
    """Return the moon-pool center (x, y) in the GNSS body frame, metres from GPS1.

    The drawing frame is converted as in ``Atlantic_Explorer_Plot.RV_Plot``, where
    a body-frame lever ``l`` is drawn at ``Rz(ROTATION_DEG) @ l + GPS1_OFFSET``;
    here that mapping is inverted.
    """
    corners_px = np.array(
        [point for segment in MOONPOOL_SEGMENTS_PX for point in segment]
    )
    center_px = corners_px.mean(axis=0)
    # Pixel y grows downward in the drawing; flip so that y points to port.
    center_drawing_m = np.array(
        [
            (center_px[0] - MOONPOOL_X_SHIFT_PX) * TOP_VIEW_SCALE_M_PER_PX,
            -(center_px[1] - MOONPOOL_Y_SHIFT_PX) * TOP_VIEW_SCALE_M_PER_PX,
        ]
    )
    relative_to_gps1 = center_drawing_m - np.array(GPS1_OFFSET[:2])

    theta = np.deg2rad(ROTATION_DEG)
    Rz = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return Rz.T @ relative_to_gps1


if __name__ == "__main__":
    print("Moon-pool center, body frame (m from GPS1):", moon_pool_center_body_frame())
