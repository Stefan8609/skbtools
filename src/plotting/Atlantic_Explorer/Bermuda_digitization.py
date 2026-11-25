import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from bermuda_data import gps_data_path


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
        """Record a left click; finish on a right click."""
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
    """Click points on an image; return them and their length (m) at ``scale`` m/px."""
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
    """Plot line segments ((x1, y1), (x2, y2)), optionally with dashed ones."""
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
    # The digitized segments are stored in Atlantic_Explorer_Plot.py.
