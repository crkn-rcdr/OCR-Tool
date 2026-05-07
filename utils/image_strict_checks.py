import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class QualityCheckResult:
    skew_angle_deg: float
    has_skew: bool

    quad_confidence: float
    perspective_asymmetry_ratio: float
    doc_area_ratio: float
    has_warping: bool

    highlight_ratio: float
    shadow_ratio: float
    shadow_highlight_ratio: float
    has_illumination_problem: bool


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate skew angle using PyImageSearch approach (minAreaRect).
    Only returns angle (no correction applied).
    """

    if gray is None or gray.size == 0:
        raise ValueError("Empty grayscale image")

    # Invert image so text becomes white
    inv = cv2.bitwise_not(gray)

    # Binary threshold (Otsu)
    thresh = cv2.threshold(
        inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Extract coordinates of foreground pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return 0.0

    # Compute minimum bounding rectangle angle
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return float(angle)


def check_skew(gray: np.ndarray, threshold: float = 3.0):
    angle = estimate_skew_angle(gray)
    return angle, abs(angle) >= threshold


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points as:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def estimate_warping(gray: np.ndarray):
    """
    Estimate perspective distortion.

    Returns:
        quad_confidence
        asymmetry_ratio
        doc_area_ratio
    """

    if gray is None or gray.size == 0:
        raise ValueError("Empty grayscale image")

    h, w = gray.shape[:2]

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to highlight document region
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    th = 255 - th

    # Morphological closing to connect regions
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, 1.0, 0.0

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]

    area = cv2.contourArea(cnt)
    doc_area_ratio = area / (h * w + 1e-6)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    quad_conf = 0.0
    asym_ratio = 1.0

    if len(approx) == 4 and doc_area_ratio > 0.2:
        pts = approx.reshape(4, 2).astype(np.float32)
        pts = order_points(pts)

        top = np.linalg.norm(pts[1] - pts[0])
        right = np.linalg.norm(pts[2] - pts[1])
        bottom = np.linalg.norm(pts[2] - pts[3])
        left = np.linalg.norm(pts[3] - pts[0])

        horiz_asym = abs(top - bottom) / (max(top, bottom) + 1e-6)
        vert_asym = abs(left - right) / (max(left, right) + 1e-6)

        asym_ratio = max(horiz_asym, vert_asym)

        quad_conf = min(1.0, 0.6 + 0.4 * doc_area_ratio)
    else:
        quad_conf = min(0.34, doc_area_ratio)

    return float(quad_conf), float(asym_ratio), float(doc_area_ratio)


def check_warping(gray: np.ndarray,
                  conf_threshold=0.35,
                  asym_threshold=0.22):

    quad_conf, asym, area = estimate_warping(gray)

    has_warping = (
        quad_conf < conf_threshold
        or asym >= asym_threshold
    )

    return quad_conf, asym, area, has_warping


def estimate_illumination(gray: np.ndarray,
                          bright_thresh=200,
                          dark_thresh=50):
    """
    Detect highlight and shadow regions using thresholding.
    """

    if gray is None or gray.size == 0:
        raise ValueError("Empty grayscale image")

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Bright regions
    bright = cv2.threshold(
        blurred, bright_thresh, 255, cv2.THRESH_BINARY
    )[1]
    bright = cv2.erode(bright, None, iterations=2)
    bright = cv2.dilate(bright, None, iterations=4)

    # Dark regions
    dark = cv2.threshold(
        blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV
    )[1]
    dark = cv2.erode(dark, None, iterations=2)
    dark = cv2.dilate(dark, None, iterations=4)

    highlight_ratio = (bright > 0).mean()
    shadow_ratio = (dark > 0).mean()
    combined_ratio = highlight_ratio + shadow_ratio

    return float(highlight_ratio), float(shadow_ratio), float(combined_ratio)


def check_illumination(gray: np.ndarray, threshold=0.22):
    h, s, sh = estimate_illumination(gray)
    return h, s, sh, sh >= threshold



def analyze_quality(image_bgr: np.ndarray) -> QualityCheckResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    skew_angle, has_skew = check_skew(gray)

    quad_conf, asym, area, has_warping = check_warping(gray)

    highlight, shadow, sh_ratio, illum_problem = check_illumination(gray)

    return QualityCheckResult(
        skew_angle_deg=skew_angle,
        has_skew=has_skew,

        quad_confidence=quad_conf,
        perspective_asymmetry_ratio=asym,
        doc_area_ratio=area,
        has_warping=has_warping,

        highlight_ratio=highlight,
        shadow_ratio=shadow,
        shadow_highlight_ratio=sh_ratio,
        has_illumination_problem=illum_problem,
    )