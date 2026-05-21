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
    key = str(value or DEFAULT_PREPROCESSING_PRESET).strip().lower()
    key = PREPROCESSING_ALIASES.get(key, key.replace(" ", "_").replace("-", "_"))

    if key in PREPROCESSING_PRESET_LABELS:
        return key

    return DEFAULT_PREPROCESSING_PRESET


def preprocessing_label(preset: Any) -> str:
    normalized = normalize_preprocessing_preset(preset)
    return PREPROCESSING_PRESET_LABELS.get(normalized, str(preset))


def normalize_preprocessing_step(value: Any) -> list[str]:
    key = str(value or "").strip().lower()
    key = PREPROCESSING_ALIASES.get(key, key.replace(" ", "_").replace("-", "_"))

    if key in PREPROCESSING_STEP_LABELS:
        return [key]

    if key in PREPROCESSING_PRESET_TO_STEPS:
        return list(PREPROCESSING_PRESET_TO_STEPS[key])

    return []


def normalize_preprocessing_steps(opt: dict | None) -> list[str]:
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
    if not steps:
        return "None"
    return ", ".join(PREPROCESSING_STEP_LABELS.get(step, step) for step in steps)


def preprocessing_enabled(opt: dict | None) -> bool:
    opt = opt or {}
    return (
        bool(normalize_preprocessing_steps(opt))
        or bool(opt.get("ignore_black_border", DEFAULT_IGNORE_BLACK_BORDER))
    )


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
    if not preprocessing_steps and not ignore_black_border:
        return image_path, None

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

    processed, step_details = apply_preprocessing_steps(gray, preprocessing_steps)
    details.update(step_details)
    details["steps"] = [*steps, *step_details.get("steps", [])]

    step_suffix = "_".join(preprocessing_steps)
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
        "preprocessing_steps": preprocessing_steps,
        "label": preprocessing_steps_label(preprocessing_steps),
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
    steps = PREPROCESSING_PRESET_TO_STEPS.get(
        normalize_preprocessing_preset(preset),
        [],
    )
    return apply_preprocessing_steps(gray, steps)


def apply_preprocessing_steps(gray: np.ndarray, steps: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
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

    if (right - left) < width * 0.4 or (bottom - top) < height * 0.4:
        return None

    return left, top, right, bottom


def smooth_projection(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def largest_true_run(mask: np.ndarray, min_length: int) -> tuple[int, int] | None:
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
    return cv2.medianBlur(gray, 3)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def background_flatten(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    sigma = max(15.0, min(height, width) / 80.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flattened = cv2.divide(gray, background, scale=255)
    return cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def sauvola_threshold(gray: np.ndarray) -> np.ndarray:
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
    value = max(minimum, min(maximum, int(value)))
    if value % 2 == 0:
        value += 1
    return min(value, maximum if maximum % 2 == 1 else maximum - 1)
