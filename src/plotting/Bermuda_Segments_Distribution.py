import numpy as np
import matplotlib.pyplot as plt
from data import gps_output_path


def px_to_world_segments(
    segments_px, scale, x_shift_px=0, y_shift_px=0, gps1_offset=(0, 0, 0), view="side"
):
    """
    Convert pixel‐coordinates to the same meter‐coordinates as your lever arms.

    - scale: m/pixel
    - x_shift_px, y_shift_px: the same center‐subtractions you used before
    - gps1_offset: (x,y,z) in meters
    - view: 'side' or 'top' → decide which pixel‐axis maps to world‐axis
    """
    ox, oy, oz = gps1_offset
    world = []
    for (x1, y1), (x2, y2) in segments_px:
        # first subtract your pixel‐shifts:
        xp1, yp1 = x1 - x_shift_px, y1 - y_shift_px
        xp2, yp2 = x2 - x_shift_px, y2 - y_shift_px

        # scale into meters:
        xm1, ym1 = xp1 * scale, yp1 * scale
        xm2, ym2 = xp2 * scale, yp2 * scale

        if view == "side":
            # pixel‐Y → world‐Z, pixel‐X → world‐X
            p1 = (xm1 + ox, ym1 + oz)
            p2 = (xm2 + ox, ym2 + oz)
        else:
            # top‐down: pixel‐Y → world‐Y (beam)
            p1 = (xm1 + ox, ym1 + oy)
            p2 = (xm2 + ox, ym2 + oy)
        world.append((p1, p2))
    return world


def plot_segments(segments):
    # Define each segment by its two endpoints
    plt.figure()
    for (x1, y1), (x2, y2) in segments:
        # Draw the line segment
        plt.plot([x1, x2], [y1, y2], color="k", linewidth=2)

    plt.gca().set_aspect("equal", "box")
    plt.gca().invert_yaxis()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Bridge View Points")
    plt.show()


if __name__ == "__main__":
    bridge_segments = [
        ((361.68853695, 255.13511531), (361.68853695, 118.19570136)),
        ((361.68853695, 118.19570136), (412.0361991, 118.19570136)),
        ((412.0361991, 118.19570136), (412.0361991, 134.16699219)),
        ((412.0361991, 134.16699219), (453.99414063, 134.16699219)),
        ((453.99414063, 134.16699219), (453.99414063, 118.30517578)),
        ((453.99414063, 118.30517578), (550.13084465, 118.30517578)),
        ((550.13084465, 118.30517578), (550.13084465, 217.21435547)),
        ((550.13084465, 217.21435547), (425.11539656, 217.21435547)),
        ((425.11539656, 217.21435547), (425.11539656, 255.13511531)),
        ((425.11539656, 255.13511531), (361.68853695, 255.13511531)),
        ((368.54864253, 251.66402715), (368.54864253, 123.49660633)),
        ((368.54864253, 123.49660633), (411.01757812, 123.49660633)),
        ((411.01757812, 123.49660633), (411.01282051, 136.22322775)),
        ((411.01282051, 136.22322775), (457.07006836, 136.09863281)),
        ((457.01809955, 136.14253394), (457.01809955, 123.02941176)),
        ((457.01293945, 123.24609375), (543.94775391, 123.05761719)),
        ((544.12707391, 123.10558069), (543.90422323, 208.97737557)),
        ((544.04931641, 208.38769531), (418.03613281, 208.11523437)),
        ((418.03959276, 207.95927602), (418.03959276, 251.19457014)),
        ((418.04736328, 251.07128906), (368.03431373, 251.04901961)),
    ]

    ship_segments = [
        ((47.18438914, 243.84615385), (47.18438914, 67.09863281)),
        ((53.02148437, 67.09863281), (172.00781250, 51.22558594)),
        ((172.00781250, 51.41308594), (815.26562500, 51.41308594)),
        ((815.26562500, 51.41308594), (1000.39257813, 158.67382812)),
        ((1000.39257813, 158.67382812), (815.26562500, 259.94409180)),
        ((815.26562500, 259.94409180), (175.97265625, 259.94409180)),
        ((175.97265625, 259.94409180), (47.18438914, 243.84615385)),
        ((418.46455505, 89.1821267), (418.46455505, 75.80429864)),
        ((418.46455505, 75.80429864), (432.25527903, 75.80429864)),
        ((432.25527903, 75.80429864), (432.25527903, 89.1821267)),
        ((432.25527903, 89.1821267), (418.46455505, 89.1821267)),
    ]

    bridge_center = 166.0
    ship_center = 158.03
    bridge_x_shift = -174.95701357000002

    top_down_scale = 0.054715

    chain = np.load(gps_output_path("mcmc_chain_adroit_5_test_xy_lever.npz"))
    levers = chain["lever"][::5000]

    segments_bridge = px_to_world_segments(
        bridge_segments,
        scale=top_down_scale,
        x_shift_px=bridge_x_shift,
        y_shift_px=bridge_center,
        gps1_offset=(0, 0, 0),
        view="top",
    )
    segments_ship = px_to_world_segments(
        ship_segments,
        scale=top_down_scale,
        x_shift_px=0,
        y_shift_px=ship_center,
        gps1_offset=(0, 0, 0),
        view="top",
    )
    segments = np.concatenate([segments_bridge, segments_ship])
    plot_segments(segments)

    GPS_1 = [39.5, -2]


"""
TO DO:
    Plot the GPS Grid in relevant places
    Plot the distribution of GPS points
    Find rotated lever and plot distribution
    Change the way that the GPS distribution is found to Procrustes
"""
