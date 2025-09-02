import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data import gps_data_path


def _get_points_matplotlib(image):
    """
    Show `image` at native resolution, let the user zoom/pan, and register
    left‑clicks to collect points. Right‑click (or close the figure) to finish.
    Returns an (N,2) array of (x, y) coords.
    """
    # figure size so that 1 image pixel ≈ 1 screen pixel
    h, w = image.shape[:2]
    dpi = 100
    fig_w, fig_h = w / dpi, h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)

    pts = []
    toolbar = fig.canvas.manager.toolbar

    def _on_click(event):
        # only register clicks in the image axes, and when toolbar.mode=='' (arrow)
        if event.inaxes == ax and toolbar.mode == "":
            # left‑click: record a point
            if event.button == 1:
                pts.append((event.xdata, event.ydata))
                print(f"  • Point {len(pts)}: ({event.xdata:.1f}, {event.ydata:.1f})")
            # right‑click: finish
            elif event.button == 3:
                plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    print("  • Left‑click to add points; right‑click when finished.")
    plt.show()  # blocks until fig is closed
    fig.canvas.mpl_disconnect(cid)

    return np.array(pts, dtype=float)


def find_length(scale=None, image_path=None):
    # Load & verify
    path = gps_data_path(image_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Pick scale bar
    if scale is None:
        print(
            "Zoom/pan as needed, then click the TWO ENDPOINTS of the known scale bar:"
        )
        pts_scale = _get_points_matplotlib(gray)
        if pts_scale.shape[0] < 2:
            print("Aborted.")
            return
        pixel_dist = np.linalg.norm(pts_scale[1] - pts_scale[0])
        true_length_m = 52.197
        scale = true_length_m / pixel_dist
        print(f"\nComputed scale: {scale:.6f} m/pixel")

    # 2) Pick feature
    print(
        "\nZoom/pan as needed, then click the TWO ENDPOINTS of the feature to measure:"
    )
    pts_feat = _get_points_matplotlib(gray)
    if pts_feat.shape[0] < 2:
        print("Aborted.")
        return
    px = np.linalg.norm(pts_feat[1] - pts_feat[0])
    measured_m = px * scale
    print(f"\nMeasured feature length: {measured_m:.4f} m")
    return pts_feat, measured_m


def plot_segments(segments, dashed_segments=None):
    # Define each segment by its two endpoints
    plt.figure()
    for (x1, y1), (x2, y2) in segments:
        # Draw the line segment
        plt.plot([x1, x2], [y1, y2], color="k", linewidth=2, linestyle="-")
    if dashed_segments is not None:
        for (x1, y1), (x2, y2) in dashed_segments:
            # Draw the dashed line segment
            plt.plot([x1, x2], [y1, y2], color="k", linewidth=2, linestyle="--")

    plt.gca().set_aspect("equal", "box")
    plt.gca().invert_yaxis()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Bridge View Points")
    plt.show()


def _shift_segments(segments, y_shift=0, x_shift=0):
    """
    shifts all y’s in segments by subtracting center_y.
    segments: [((x1,y1),(x2,y2)), …]
    """
    return [
        ((x1 - x_shift, y1 - y_shift), (x2 - x_shift, y2 - y_shift))
        for ((x1, y1), (x2, y2)) in segments
    ]


if __name__ == "__main__":
    """
    image from
    https://bios.asu.edu/sites/g/files/litvpz726/files/
    imported-bios/Atlantic_Explorer_General_Arrangement_Drawings_Rev_02-Jun-2014.pdf"""
    image_name = "Side_View"
    image_path = f"Figs/Images/Atlantic_Explorer_Schematic/{image_name}.png"

    if image_name == "Side_View":
        # For the side view, we use a different scale
        scale = 0.07896671
        file = gps_data_path(
            "Figs/Images/Atlantic_Explorer_Schematic/measurement_side.txt"
        )
    else:
        scale = 0.054715
        file = gps_data_path(
            "Figs/Images/Atlantic_Explorer_Schematic/measurement_top.txt"
        )

    points, measurement = find_length(scale=scale, image_path=image_path)

    if measurement is not None:
        with open(file, "a") as f:
            f.write(f"{image_name}: {points}, {measurement:.4f} \n")
    #
    # bridge_center = 166.0
    # ship_center = 158.03
    # moonpool_y_shift = 276 + 158.03
    # bridge_x_shift = -175
    # moonpool_x_shift = -3

    # bridge_segments = [
    #     ((361.68853695, 255.13511531), (361.68853695, 118.19570136)),
    #     ((361.68853695, 118.19570136), (412.0361991, 118.19570136)),
    #     ((412.0361991, 118.19570136), (412.0361991, 134.16699219)),
    #     ((412.0361991, 134.16699219), (453.99414063, 134.16699219)),
    #     ((453.99414063, 134.16699219), (453.99414063, 118.30517578)),
    #     ((453.99414063, 118.30517578), (550.13084465, 118.30517578)),
    #     ((550.13084465, 118.30517578), (550.13084465, 217.21435547)),
    #     ((550.13084465, 217.21435547), (425.11539656, 217.21435547)),
    #     ((425.11539656, 217.21435547), (425.11539656, 255.13511531)),
    #     ((425.11539656, 255.13511531), (361.68853695, 255.13511531)),
    #     ((368.54864253, 251.66402715), (368.54864253, 123.49660633)),
    #     ((368.54864253, 123.49660633), (411.01757812, 123.49660633)),
    #     ((411.01757812, 123.49660633), (411.01282051, 136.22322775)),
    #     ((411.01282051, 136.22322775), (457.07006836, 136.09863281)),
    #     ((457.01809955, 136.14253394), (457.01809955, 123.02941176)),
    #     ((457.01293945, 123.24609375), (543.94775391, 123.05761719)),
    #     ((544.12707391, 123.10558069), (543.90422323, 208.97737557)),
    #     ((544.04931641, 208.38769531), (418.03613281, 208.11523437)),
    #     ((418.03959276, 207.95927602), (418.03959276, 251.19457014)),
    #     ((418.04736328, 251.07128906), (368.03431373, 251.04901961)),
    # ]
    #
    # ship_segments = [
    #     ((47.18438914, 243.84615385), (47.18438914, 67.09863281)),
    #     ((53.02148437, 67.09863281), (172.00781250, 51.22558594)),
    #     ((172.00781250, 51.41308594), (815.26562500, 51.41308594)),
    #     ((815.26562500, 51.41308594), (1000.39257813, 158.67382812)),
    #     ((1000.39257813, 158.67382812), (815.26562500, 259.94409180)),
    #     ((815.26562500, 259.94409180), (175.97265625, 259.94409180)),
    #     ((175.97265625, 259.94409180), (47.18438914, 243.84615385)),
    # ]
    # moonpool_segments = [
    #     ((413.96794872, 366.93099548), (432.05656109, 366.93099548)),
    #     ((432.05656109, 366.93099548), (432.05656109, 347.07126697)),
    #     ((432.05656109, 347.07126697), (413.96794872, 347.07126697)),
    #     ((413.96794872, 347.07126697), (413.96794872, 366.93099548)),
    # ]
    #
    # bridge_shifted = _shift_segments(
    #     bridge_segments, y_shift=bridge_center, x_shift=bridge_x_shift
    # )
    # ship_shifted = _shift_segments(ship_segments, y_shift=ship_center)
    # moonpool_shifted = _shift_segments(
    #     moonpool_segments, y_shift=moonpool_y_shift, x_shift=moonpool_x_shift
    # )
    #
    # # plot them
    # plot_segments(np.concatenate([bridge_shifted, ship_shifted, moonpool_shifted]))

    # side_view_segments = [
    #     ((94.17609, 442.05618), (89.08673, 397.63989)),
    #     ((89.08673, 397.63989), (162.03431, 287.52451)),
    #     ((162.03431, 287.52451), (174.98906, 304.64329)),
    #     ((174.98906, 304.64329), (116.23002, 383.14291)),
    #     ((116.23002, 383.14291), (174.52640, 410.13198)),
    #     ((174.52640, 410.13198), (253.31561, 410.24887)),
    #     ((253.31561, 410.24887), (253.50490, 349.00075)),
    #     ((253.50490, 349.00075), (234.32089, 343.28130)),
    #     ((234.32089, 343.28130), (313.08258, 344.71116)),
    #     ((313.08258, 344.71116), (294.73265, 348.40498)),
    #     ((294.73265, 348.40498), (293.89857, 393.20739)),
    #     ((293.89857, 393.20739), (386.83974, 393.32655)),
    #     ((386.83974, 393.32655), (410.67081, 370.21041)),
    #     ((410.67081, 370.21041), (391.12934, 362.46531)),
    #     ((391.12934, 362.46531), (417.31348, 362.63330)),
    #     ((417.31348, 362.63330), (419.77881, 303.03027)),
    #     ((419.77881, 303.03027), (542.03027, 303.03027)),
    #     ((542.03027, 303.03027), (537.38965, 332.90430)),
    #     ((537.38965, 332.90430), (583.50586, 331.59912)),
    #     ((583.50586, 331.59912), (583.21582, 359.44287)),
    #     ((583.21582, 359.44287), (660.36621, 350.30664)),
    #     ((660.36621, 350.30664), (658.13348, 376.15309)),
    #     ((658.13348, 376.15309), (749.01735, 369.05807)),
    #     ((749.01735, 369.05807), (672.99925, 478.96362)),
    #     ((672.99925, 478.96362), (268.87378, 478.96362)),
    #     ((268.87378, 478.96362), (93.10156, 437.16406)),
    # ]
    #
    # railing_segments = [
    #     ((420.39501953, 302.03784180), (420.39501953, 290.79882812)),
    #     ((420.39501953, 290.79882812), (542.09057617, 290.79882812)),
    #     ((542.09057617, 290.79882812), (542.09057617, 302.03784180)),
    #     ((542.09057617, 302.03784180), (420.39501953, 302.03784180)),
    # ]
    #
    # plot_segments(side_view_segments, railing_segments)
