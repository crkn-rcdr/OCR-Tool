import cv2
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class QualityCheckResult:
    has_skew: bool
    has_illumination_problem: bool
    has_warping: bool


def crop_inner_region(gray: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
    h, w = gray.shape[:2]
    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    if h - 2 * margin_y <= 0 or w - 2 * margin_x <= 0:
        return gray

    return gray[margin_y:h - margin_y, margin_x:w - margin_x]


def normalize_angle_to_horizontal(angle: float) -> float:
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180

    if angle <= -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    return float(angle)


def estimate_skew_angle_from_page(gray: np.ndarray) -> tuple[float, float]:
    roi = crop_inner_region(gray, margin_ratio=0.03)
    h, w = roi.shape[:2]

    blur = cv2.GaussianBlur(roi, (5, 5), 0)

    th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, 0.0

    cnt = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(cnt) / (h * w + 1e-6))

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    best_length = 0.0
    best_angle = 0.0

    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = math.hypot(dx, dy)

        if length <= 0:
            continue

        angle = math.degrees(math.atan2(dy, dx))
        angle = normalize_angle_to_horizontal(angle)

        if length > best_length:
            best_length = length
            best_angle = angle

    return float(best_angle), area_ratio


def estimate_skew_angle_hough(gray: np.ndarray) -> float:
    roi = crop_inner_region(gray, margin_ratio=0.08)
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(60, roi.shape[1] // 8),
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    weights = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            continue

        angle = math.degrees(math.atan2(dy, dx))
        angle = normalize_angle_to_horizontal(angle)

        if abs(angle) <= 15:
            length = math.hypot(dx, dy)
            angles.append(angle)
            weights.append(length)

    if not angles:
        return 0.0

    return float(np.average(angles, weights=weights))


def estimate_skew_angle(gray: np.ndarray) -> float:
    page_angle, page_area_ratio = estimate_skew_angle_from_page(gray)

    if page_area_ratio >= 0.50:
        return page_angle

    return estimate_skew_angle_hough(gray)


def check_skew(gray: np.ndarray, threshold: float = 0.6) -> bool:
    angle = estimate_skew_angle(gray)
    return abs(angle) >= threshold


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def estimate_warping(gray: np.ndarray) -> tuple[float, float]:
    roi = crop_inner_region(gray, margin_ratio=0.03)
    h, w = roi.shape[:2]

    blur = cv2.GaussianBlur(roi, (5, 5), 0)

    th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, 0.0

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        doc_area_ratio = float(area / (h * w + 1e-6))

        if doc_area_ratio < 0.35:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        pts = order_points(pts)

        top = np.linalg.norm(pts[1] - pts[0])
        right = np.linalg.norm(pts[2] - pts[1])
        bottom = np.linalg.norm(pts[2] - pts[3])
        left = np.linalg.norm(pts[3] - pts[0])

        horiz_asym = abs(top - bottom) / (max(top, bottom) + 1e-6)
        vert_asym = abs(left - right) / (max(left, right) + 1e-6)

        asym_ratio = float(max(horiz_asym, vert_asym))

        return asym_ratio, doc_area_ratio

    return 0.0, 0.0


def check_warping(
    gray: np.ndarray,
    asym_threshold: float = 0.55,
    min_area_ratio: float = 0.40,
) -> bool:
    asym_ratio, doc_area_ratio = estimate_warping(gray)

    return (
        doc_area_ratio >= min_area_ratio
        and asym_ratio >= asym_threshold
    )


def estimate_illumination(gray: np.ndarray) -> float:
    roi = crop_inner_region(gray, margin_ratio=0.10)
    h, w = roi.shape[:2]

    max_side = max(h, w)

    if max_side > 900:
        scale = 900.0 / max_side
        roi = cv2.resize(
            roi,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    illum = cv2.GaussianBlur(
        roi,
        (0, 0),
        sigmaX=51,
        sigmaY=51
    ).astype(np.float32)

    median_val = float(np.median(illum))

    if median_val < 1e-6:
        return 0.0

    illum_norm = illum / median_val

    p5, p95 = np.percentile(illum_norm, [5, 95])

    combined_ratio = float((p95 - p5) / (p95 + 1e-6))

    return combined_ratio


def check_illumination(gray: np.ndarray, threshold: float = 0.40) -> bool:
    combined_ratio = estimate_illumination(gray)
    return combined_ratio >= threshold


def analyze_quality(image_bgr: np.ndarray) -> QualityCheckResult:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty input image")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    has_skew = check_skew(gray)
    has_illumination_problem = check_illumination(gray)
    has_warping = check_warping(gray)

    return QualityCheckResult(
        has_skew=has_skew,
        has_illumination_problem=has_illumination_problem,
        has_warping=has_warping,
    )

