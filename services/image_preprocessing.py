"""Optional source-image preprocessing before local OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.ocr_defaults import (
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_PREPROCESSING_PRESET,
    DEFAULT_PREPROCESSING_STEPS,
    PREPROCESSING_PRESET_TO_STEPS,
    PREPROCESSING_PRESET_LABELS,
    PREPROCESSING_STEP_LABELS,
)

PREPROCESSING_ALIASES = {
    "": "none",
    "off": "none",
    "disabled": "none",
    "no": "none",
    "false": "none",
    "none": "none",
    "light clean": "light_clean",
    "light-clean": "light_clean",
    "lightclean": "light_clean",
    "contrast / clahe": "contrast",
    "clahe": "contrast",
    "contrast_clahe": "contrast",
    "background flatten": "background_flatten",
    "background-flatten": "background_flatten",
    "flatten": "background_flatten",
    "adaptive_binary": "sauvola_binary",
    "adaptive-binary": "sauvola_binary",
    "sauvola binary": "sauvola_binary",
    "sauvola-binary": "sauvola_binary",
    "sauvola": "sauvola_binary",
    "deskew + contrast": "deskew_contrast",
    "deskew+contrast": "deskew_contrast",
    "deskew_contrast": "deskew_contrast",
    "deskew-contrast": "deskew_contrast",
    "deskew": "deskew",
}

PREPROCESSING_STEP_ORDER = (
    "light_clean",
    "background_flatten",
    "contrast",
    "deskew",
    "sauvola_binary",
)
MAX_CONSERVATIVE_DESKEW_DEGREES = 5.0
DEFAULT_BLACK_BORDER_THRESHOLD = 80
DEFAULT_BLACK_BORDER_PADDING_PX = 24
DEFAULT_PAGE_PROJECTION_COVERAGE = 0.20


def normalize_preprocessing_preset(value: Any) -> str:
    """Normalize user input into a supported preprocessing preset key."""
    key = str(value or DEFAULT_PREPROCESSING_PRESET).strip().lower()
    key = PREPROCESSING_ALIASES.get(key, key.replace(" ", "_").replace("-", "_"))

    if key in PREPROCESSING_PRESET_LABELS:
        return key

    return DEFAULT_PREPROCESSING_PRESET


def preprocessing_label(preset: Any) -> str:
    """Return the human-readable label for a preprocessing preset."""
    normalized = normalize_preprocessing_preset(preset)
    return PREPROCESSING_PRESET_LABELS.get(normalized, str(preset))


def normalize_preprocessing_step(value: Any) -> list[str]:
    """Normalize one user-supplied step or preset alias into step keys."""
    key = str(value or "").strip().lower()
    key = PREPROCESSING_ALIASES.get(key, key.replace(" ", "_").replace("-", "_"))

    if key in PREPROCESSING_STEP_LABELS:
        return [key]

    if key in PREPROCESSING_PRESET_TO_STEPS:
        return list(PREPROCESSING_PRESET_TO_STEPS[key])

    return []


def normalize_preprocessing_steps(opt: dict | None) -> list[str]:
    """Resolve the effective ordered preprocessing step list from options."""
    opt = opt or {}

    if "preprocessing_steps" in opt:
        selected = opt.get("preprocessing_steps")
        if selected is None:
            raw_steps = []
        elif isinstance(selected, str):
            raw_steps = [
                step.strip()
                for step in selected.replace(";", ",").split(",")
                if step.strip()
            ]
        else:
            raw_steps = list(selected)
    else:
        preset = normalize_preprocessing_preset(opt.get("preprocessing_preset"))
        raw_steps = PREPROCESSING_PRESET_TO_STEPS.get(
            preset,
            DEFAULT_PREPROCESSING_STEPS,
        )

    normalized = []
    for step in raw_steps:
        normalized.extend(normalize_preprocessing_step(step))

    return [
        step
        for step in PREPROCESSING_STEP_ORDER
        if step in set(normalized)
    ]


def preprocessing_steps_label(steps: list[str]) -> str:
    """Format preprocessing steps for logs and UI summaries."""
    if not steps:
        return "None"
    return ", ".join(PREPROCESSING_STEP_LABELS.get(step, step) for step in steps)


def preprocessing_enabled(opt: dict | None) -> bool:
    """Return whether preprocessing or border cropping is enabled."""
    opt = opt or {}
    return (
        bool(normalize_preprocessing_steps(opt))
        or bool(opt.get("ignore_black_border", DEFAULT_IGNORE_BLACK_BORDER))
    )


def detect_microfilm_degradation(gray: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Analyze image contrast, edge gradient, and background unevenness to auto-detect degraded microfilm.

    Historical microfilms typically feature:
    1. Soft, blurry letter strokes (Laplacian variance < 75) and compressed dynamic range with faded gray ink (p5 > 50).
    2. Strong optical vignette / dark border gradient (corner brightness < 35% of page center).
    3. Severe uneven background illumination across page quadrants (quadrant background std > 5.0).
    """
    height, width = gray.shape[:2]
    sample = gray[
        int(height * 0.1) : int(height * 0.9),
        int(width * 0.1) : int(width * 0.9),
    ]
    if sample.size == 0:
        return False, {}

    p5, p95 = np.percentile(sample, [5, 95])
    contrast_range = float(p95 - p5)
    lap_var = float(cv2.Laplacian(sample, cv2.CV_64F).var())
    std = float(sample.std())

    # Check optical vignetting / dark border fade (prominent in microfilm negative reels)
    cw = max(50, int(width * 0.05))
    ch = max(50, int(height * 0.05))
    corners = [
        gray[0:ch, 0:cw],
        gray[0:ch, width - cw : width],
        gray[height - ch : height, 0:cw],
        gray[height - ch : height, width - cw : width],
    ]
    corner_avg = float(np.mean([float(c.mean()) for c in corners]))
    center_area = gray[int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75)]
    center_avg = float(center_area.mean()) if center_area.size > 0 else 128.0
    vignette_ratio = corner_avg / max(1.0, center_avg)

    # Check quadrant background variance
    qh, qw = sample.shape[0] // 2, sample.shape[1] // 2
    quadrants = [
        float(sample[:qh, :qw].mean()),
        float(sample[:qh, qw:].mean()),
        float(sample[qh:, :qw].mean()),
        float(sample[qh:, qw:].mean()),
    ]
    quadrant_std = float(np.std(quadrants))

    # Microfilm signature:
    # 1. Soft stroke edges (lap_var < 75.0) combined with faded ink (p5 > 50.0)
    # 2. Or very low sharpness (lap_var < 60.0) and compressed contrast (range < 140.0)
    # 3. Or severe negative microfilm vignette (vignette_ratio < 0.25) with background gradient
    is_microfilm = bool(
        (lap_var < 75.0 and p5 > 50.0)
        or (lap_var < 60.0 and contrast_range < 140.0)
        or (vignette_ratio < 0.25 and quadrant_std > 5.0)
    )

    details = {
        "laplacian_variance": round(lap_var, 1),
        "percentile_5": round(float(p5), 1),
        "percentile_95": round(float(p95), 1),
        "contrast_range": round(contrast_range, 1),
        "std_dev": round(std, 1),
        "vignette_ratio": round(vignette_ratio, 3),
        "quadrant_std": round(quadrant_std, 2),
        "is_microfilm": is_microfilm,
    }
    return is_microfilm, details


def preprocess_image_for_ocr(
    image_path: Path,
    out_dir: Path,
    opt: dict | None,
) -> tuple[Path, dict[str, Any] | None]:
    """Create a preprocessed OCR input image when a preset is selected.

    The processed image is persisted under the OCR output directory so later
    PDF/ALTO generation can use the same image that Paddle saw.
    """

    preset = normalize_preprocessing_preset((opt or {}).get("preprocessing_preset"))
    preprocessing_steps = normalize_preprocessing_steps(opt)
    ignore_black_border = bool(
        (opt or {}).get("ignore_black_border", DEFAULT_IGNORE_BLACK_BORDER)
    )

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    preprocess_dir = out_dir / "_preprocessed"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image for preprocessing: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height, original_width = gray.shape[:2]
    steps = []
    details = {}

    if ignore_black_border:
        gray, border_details = crop_black_border_area(gray)
        steps.append("crop_black_border")
        details.update(border_details)

    # Auto-deskew: correct slight page skew (0.35 deg to 5.0 deg) to keep line baselines horizontal
    skew_angle = estimate_skew_angle(gray)
    if 0.35 <= abs(skew_angle) <= MAX_CONSERVATIVE_DESKEW_DEGREES:
        gray = rotate_keep_size(gray, skew_angle)
        steps.append("auto_deskew")
        details["auto_deskew_angle_degrees"] = round(float(skew_angle), 4)

    # Auto-suppress heavy table and form ruling lines that slice into text bounding boxes
    gray, lines_suppressed = suppress_ruling_lines_if_present(gray)
    if lines_suppressed:
        steps.append("suppress_ruling_lines")
        details["table_ruling_lines_suppressed"] = True

    # Auto-detect degraded microfilm and dynamically apply background flattening + CLAHE + stroke reconnection
    is_microfilm, microfilm_details = detect_microfilm_degradation(gray)
    details["microfilm_detection"] = microfilm_details
    effective_steps = list(preprocessing_steps)
    if is_microfilm:
        if "background_flatten" not in effective_steps:
            effective_steps.insert(0, "background_flatten")
            details["auto_microfilm_flatten_applied"] = True
        if "contrast" not in effective_steps:
            effective_steps.append("contrast")
            details["auto_microfilm_clahe_applied"] = True

    processed, step_details = apply_preprocessing_steps(gray, effective_steps)
    if is_microfilm:
        processed = reconnect_broken_strokes(processed)
        step_details.setdefault("steps", []).append("stroke_reconnect")
        details["auto_stroke_reconnect_applied"] = True

    # Adaptive micro-print upscaling: if character heights are tiny (< 18px),
    # upscale 1.5x with bicubic interpolation so DBNet & SVTR see optimal training dimensions
    median_h = estimate_median_character_height(gray)
    details["median_character_height_px"] = round(median_h, 1)
    if median_h < 18.0 and max(original_height, original_width) <= 6000:
        new_w = int(round(processed.shape[1] * 1.5))
        new_h = int(round(processed.shape[0] * 1.5))
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        details["coordinate_scale_factor"] = 1.5
        details["micro_print_upscaling_applied"] = True
        step_details.setdefault("steps", []).append("micro_print_upscale")

    if lines_suppressed:
        details["table_unclip_ratio"] = 1.25

    details.update(step_details)
    all_applied_steps = [*steps, *step_details.get("steps", [])]
    details["steps"] = all_applied_steps

    # If no preprocessing transformations were required, pass through the original image
    if not all_applied_steps and not effective_steps and not details.get("micro_print_upscaling_applied"):
        return image_path, None

    step_suffix = "_".join(effective_steps)
    if ignore_black_border and step_suffix:
        suffix = f"black_border_{step_suffix}"
    elif ignore_black_border:
        suffix = "black_border"
    else:
        suffix = step_suffix or "none"
    output_path = preprocess_dir / f"{image_path.stem}__preprocessed_{suffix}.png"
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    if not cv2.imwrite(str(output_path), processed_bgr):
        raise RuntimeError(f"Unable to write preprocessed image: {output_path}")

    height, width = processed.shape[:2]
    metadata = {
        "enabled": True,
        "preset": preset,
        "preprocessing_steps": effective_steps,
        "label": preprocessing_steps_label(effective_steps),
        "ignore_black_border": ignore_black_border,
        "original_input_path": str(image_path),
        "ocr_input_path": str(output_path),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "width": int(width),
        "height": int(height),
        "preserves_dimensions": (height, width) == (original_height, original_width),
        **details,
    }

    return output_path, metadata


def apply_preprocessing_preset(gray: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the named preprocessing preset by expanding it into steps."""
    steps = PREPROCESSING_PRESET_TO_STEPS.get(
        normalize_preprocessing_preset(preset),
        [],
    )
    return apply_preprocessing_steps(gray, steps)


def apply_preprocessing_steps(gray: np.ndarray, steps: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the selected preprocessing steps in the configured order."""
    processed = gray.copy()
    applied_steps = []
    details = {}

    for step in normalize_preprocessing_steps({"preprocessing_steps": steps}):
        if step == "light_clean":
            processed = light_clean(processed)
            applied_steps.append("median_denoise")
        elif step == "background_flatten":
            processed = background_flatten(processed)
            applied_steps.append("background_flatten")
        elif step == "contrast":
            processed = apply_clahe(processed)
            applied_steps.append("clahe")
        elif step == "deskew":
            angle = estimate_skew_angle(processed)
            if angle:
                processed = rotate_keep_size(processed, angle)
            details["deskew_angle_degrees"] = round(float(angle), 4)
            applied_steps.append("conservative_deskew")
        elif step == "sauvola_binary":
            processed = sauvola_threshold(processed)
            applied_steps.append("sauvola_threshold")

    details["steps"] = applied_steps
    return processed, details


def crop_black_border_area(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop a detected black scan border and return geometry offset metadata."""
    bbox = detect_page_bbox_from_black_border(gray)

    if bbox is None:
        return gray.copy(), {
            "black_border_detected": False,
            "black_border_crop_bbox": None,
            "coordinate_offset_x": 0,
            "coordinate_offset_y": 0,
            "use_original_input_for_outputs": False,
        }

    left, top, right, bottom = bbox
    cropped = gray[top:bottom, left:right].copy()

    return cropped, {
        "black_border_detected": True,
        "black_border_crop_bbox": {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        },
        "coordinate_offset_x": int(left),
        "coordinate_offset_y": int(top),
        "use_original_input_for_outputs": True,
    }


def detect_page_bbox_from_black_border(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Estimate the page bounds inside a dark scanner border."""
    height, width = gray.shape[:2]
    bright = gray > DEFAULT_BLACK_BORDER_THRESHOLD
    col_fraction = bright.mean(axis=0)
    row_fraction = bright.mean(axis=1)

    smooth_size = odd_between(min(width, height) // 120, 31, 81)
    col_mask = smooth_projection(col_fraction, smooth_size) > DEFAULT_PAGE_PROJECTION_COVERAGE
    row_mask = smooth_projection(row_fraction, smooth_size) > DEFAULT_PAGE_PROJECTION_COVERAGE

    col_run = largest_true_run(col_mask, min_length=int(width * 0.4))
    row_run = largest_true_run(row_mask, min_length=int(height * 0.4))

    if col_run is None or row_run is None:
        return None

    x0, x1 = col_run
    y0, y1 = row_run

    if (x1 - x0 + 1) * (y1 - y0 + 1) < width * height * 0.25:
        return None

    padding = DEFAULT_BLACK_BORDER_PADDING_PX
    left = max(0, x0 - padding)
    top = max(0, y0 - padding)
    right = min(width, x1 + 1 + padding)
    bottom = min(height, y1 + 1 + padding)

    # Safety safeguard: Verify that cropped margins are genuinely dark scan/carrier borders,
    # preventing internal column gutters, tables, or ruling lines from cutting off text columns.
    if right < width:
        margin_right = gray[:, right:width]
        if margin_right.size > 0 and (margin_right.mean() > 70 or (margin_right > 80).mean() > 0.15):
            outer_w = max(5, int(width * 0.015))
            outer_right = gray[:, width - outer_w:width]
            if outer_right.mean() > 70 or (outer_right > 80).mean() > 0.15:
                right = width
            else:
                col_bright = (gray > 80).mean(axis=0)
                new_right = width
                for x in range(width - 1, right, -1):
                    if col_bright[x] >= 0.20:
                        new_right = min(width, x + padding)
                        break
                right = max(right, new_right)

    if left > 0:
        margin_left = gray[:, 0:left]
        if margin_left.size > 0 and (margin_left.mean() > 70 or (margin_left > 80).mean() > 0.15):
            outer_w = max(5, int(width * 0.015))
            outer_left = gray[:, 0:outer_w]
            if outer_left.mean() > 70 or (outer_left > 80).mean() > 0.15:
                left = 0
            else:
                col_bright = (gray > 80).mean(axis=0)
                new_left = 0
                for x in range(0, left):
                    if col_bright[x] >= 0.20:
                        new_left = max(0, x - padding)
                        break
                left = min(left, new_left)

    if top > 0:
        margin_top = gray[0:top, :]
        if margin_top.size > 0 and (margin_top.mean() > 70 or (margin_top > 80).mean() > 0.15):
            top = 0

    if bottom < height:
        margin_bot = gray[bottom:height, :]
        if margin_bot.size > 0 and (margin_bot.mean() > 70 or (margin_bot > 80).mean() > 0.15):
            bottom = height

    if (right - left) < width * 0.4 or (bottom - top) < height * 0.4:
        return None

    return left, top, right, bottom


def smooth_projection(values: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth a 1D projection profile with a moving-average window."""
    if window_size <= 1:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def largest_true_run(mask: np.ndarray, min_length: int) -> tuple[int, int] | None:
    """Return the longest contiguous true run that meets a minimum length."""
    best = None
    best_length = 0
    start = None

    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            length = index - start
            if length >= min_length and length > best_length:
                best = (start, index - 1)
                best_length = length
            start = None

    if start is not None:
        length = len(mask) - start
        if length >= min_length and length > best_length:
            best = (start, len(mask) - 1)

    return best


def light_clean(gray: np.ndarray) -> np.ndarray:
    """Apply a light median denoise pass suited to readable scans."""
    return cv2.medianBlur(gray, 3)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Boost local contrast with CLAHE while preserving grayscale output."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def reconnect_broken_strokes(gray: np.ndarray) -> np.ndarray:
    """Subtle morphological closing on inverted grayscale to bridge 1-pixel stroke fractures."""
    inv = 255 - gray
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed_inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    return 255 - closed_inv


def background_flatten(gray: np.ndarray) -> np.ndarray:
    """Reduce uneven illumination by dividing out a blurred background map."""
    height, width = gray.shape[:2]
    sigma = max(15.0, min(height, width) / 80.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flattened = cv2.divide(gray, background, scale=255)
    return cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def sauvola_threshold(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale text to a binary image using Sauvola-style thresholding."""
    height, width = gray.shape[:2]
    window = odd_between(int(min(height, width) / 80), minimum=35, maximum=101)

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "niBlackThreshold"):
        try:
            return cv2.ximgproc.niBlackThreshold(
                gray,
                255,
                cv2.THRESH_BINARY,
                window,
                0.34,
                None,
                cv2.ximgproc.BINARIZATION_SAUVOLA,
                128,
            )
        except TypeError:
            return cv2.ximgproc.niBlackThreshold(
                gray,
                255,
                cv2.THRESH_BINARY,
                window,
                0.34,
            )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        window,
        11,
    )


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page skew from dominant near-horizontal text strokes."""
    inverted = cv2.bitwise_not(gray)
    _, threshold = cv2.threshold(
        inverted,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    coords = np.column_stack(np.where(threshold > 0))

    if len(coords) < 500:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.25 or abs(angle) > MAX_CONSERVATIVE_DESKEW_DEGREES:
        return 0.0

    return float(angle)


def rotate_keep_size(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image around center while keeping the original canvas size."""
    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def odd_between(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer into range and force it to an odd value."""
    value = max(minimum, min(maximum, int(value)))
    if value % 2 == 0:
        value += 1
    return min(value, maximum if maximum % 2 == 1 else maximum - 1)



def suppress_ruling_lines_if_present(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Detect and suppress heavy table/form grid lines that fragment text bounding boxes."""
    h, w = gray.shape[:2]
    margin_y = int(h * 0.04)
    margin_x = int(w * 0.04)
    inner = gray[margin_y : h - margin_y, margin_x : w - margin_x]
    if inner.size == 0:
        return gray, False

    thresh = cv2.adaptiveThreshold(
        inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    h_len = max(30, inner.shape[1] // 25)
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))

    v_len = max(30, inner.shape[0] // 30)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))

    # Forms and tables have substantial grid intersections (>= 15 grid points)
    grid_intersections = cv2.bitwise_and(h_lines, v_lines)
    if cv2.countNonZero(grid_intersections) >= 15:
        table_grid = cv2.bitwise_or(h_lines, v_lines)
        dilated = cv2.dilate(table_grid, np.ones((3, 3), np.uint8), iterations=1)
        inner_cleaned = cv2.inpaint(inner, dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        result = gray.copy()
        result[margin_y : h - margin_y, margin_x : w - margin_x] = inner_cleaned
        return result, True

    return gray, False


def estimate_median_character_height(gray: np.ndarray) -> float:
    """Fast estimation of median text character height on the page in pixels."""
    h, w = gray.shape[:2]
    ch = min(1200, h)
    cw = min(1200, w)
    cy, cx = h // 2, w // 2
    crop = gray[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
    if crop.size == 0:
        return 24.0

    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    heights = []
    for i in range(1, min(num_labels, 500)):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_area = stats[i, cv2.CC_STAT_AREA]
        if 8 <= comp_h <= 65 and 4 <= comp_w <= 65 and 20 <= comp_area <= 2200:
            if max(comp_w, comp_h) / max(1, min(comp_w, comp_h)) <= 3.2:
                heights.append(comp_h)

    if len(heights) >= 20:
        return float(np.median(heights))
    return 24.0
