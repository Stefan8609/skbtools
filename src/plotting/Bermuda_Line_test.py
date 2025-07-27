from pdf2image import convert_from_path
from data import gps_data_path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_and_measure_lines(
    pdf_path,
    scale_m_per_px,
    dpi: int = 300,
    canny_thresh: tuple[int, int] = (100, 150),
    hough_params: dict | None = None,
):
    """
    Detects straight lines in each page of the PDF at `pdf_path`, measures
    their lengths (in pixels), then converts to meters using `scale_m_per_px`.

    Parameters
    ----------
    pdf_path : str
        Path to input PDF.
    scale_m_per_px : float
        Scale factor (meters per pixel).
    dpi : int, optional
        Rasterization resolution for PDF→image (default 300).
    canny_thresh : tuple, optional
        Lower/upper thresholds for Canny edge detector (default (100, 150)).
    hough_params : dict, optional
        kwargs for cv2.HoughLinesP; if None, defaults to:
            {
                "rho": 1,
                "theta": np.pi/180,
                "threshold": 80,
                "minLineLength": 75,
                "maxLineGap": 5,
            }
    """
    if hough_params is None:
        hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 80,
            "minLineLength": 75,
            "maxLineGap": 5,
        }
    pages = convert_from_path(pdf_path, dpi=dpi)
    all_results = []
    for _, pil in enumerate(pages):
        # 1) to OpenCV RGB→BGR then gray
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) edge detection
        edges = cv2.Canny(gray, canny_thresh[0], canny_thresh[1])

        # 3) detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=hough_params["rho"],
            theta=hough_params["theta"],
            threshold=hough_params["threshold"],
            minLineLength=hough_params["minLineLength"],
            maxLineGap=hough_params["maxLineGap"],
        )
        page_results = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                px_len = np.hypot(x2 - x1, y2 - y1)
                real_m = px_len * scale_m_per_px
                page_results.append(((x1, y1, x2, y2), px_len, real_m))
        all_results.append(page_results)
    return all_results


def annotate_and_show(page_img, results, figsize=(10, 8)):
    """
    Draws red lines and length labels onto page_img
    (BGR OpenCV image) and shows with Matplotlib.
    """
    disp = page_img.copy()
    for (x1, y1, x2, y2), _, _ in results:
        cv2.line(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        mid = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.putText(
            disp,
            "",
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    # show
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    PDF = gps_data_path(
        "Figs/Images/Atlantic_Explorer_Schematic/Atlantic_Explorer_Schematics.pdf"
    )
    # your measured scale: meters per pixel
    SCALE = 0.054715

    pages = detect_and_measure_lines(PDF, SCALE)
    # e.g. annotate first page
    page_num = 3
    pil0 = convert_from_path(PDF, dpi=300)[page_num]
    cv0 = cv2.cvtColor(np.array(pil0), cv2.COLOR_RGB2BGR)
    annotate_and_show(cv0, pages[page_num])
