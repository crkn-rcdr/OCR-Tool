from __future__ import annotations

import configparser
import importlib
import json
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2

from services.ocr_defaults import (
    DEFAULT_PERO_CONFIG_PATH,
    DEFAULT_PERO_DEVICE,
    DEFAULT_PERO_GPU_ID,
    DEFAULT_PERO_MIN_LINE_CONFIDENCE,
    DEFAULT_PERO_SOURCE_PATH,
)

PERO_METHOD = "PERO-OCR"

_parser_lock = threading.Lock()
_page_parser: Any | None = None
_parser_params: tuple[Any, ...] | None = None
_page_layout_cls: Any | None = None


def run_pero_ocr_for_image(
    image_path: Path,
    opt: dict,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), 1)
    if image is None:
        raise ValueError(f'Unable to read image "{image_path}"')

    page_parser, page_layout_cls = get_page_parser(opt)
    height, width = image.shape[:2]
    page_layout = page_layout_cls(
        id=image_path.name,
        page_size=(height, width),
    )
    page_layout = page_parser.process_page(image, page_layout)

    if out_dir is not None and opt.get("pero_save_pagexml", False):
        pagexml_path = Path(out_dir) / f"{image_path.stem}.page.xml"
        page_layout.to_pagexml(str(pagexml_path))

    return page_layout_to_ocr_payload(page_layout, image_path, opt)


def write_pero_ocr_json(image_path: Path, out_dir: Path, opt: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run_pero_ocr_for_image(image_path, opt, out_dir=out_dir)
    json_path = out_dir / f"{Path(image_path).stem}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def get_page_parser(opt: dict):
    global _page_parser, _parser_params, _page_layout_cls

    source_path = normalize_path(
        opt.get("pero_source_path") or DEFAULT_PERO_SOURCE_PATH,
        allow_missing=True,
    )
    config_path = normalize_path(
        opt.get("pero_config_path") or DEFAULT_PERO_CONFIG_PATH,
        allow_missing=False,
    )
    device_name = normalize_pero_device(opt.get("pero_device") or DEFAULT_PERO_DEVICE)
    gpu_id = parse_gpu_id(opt.get("pero_gpu_id", DEFAULT_PERO_GPU_ID))
    min_confidence = parse_min_confidence(
        opt.get("pero_min_line_confidence", DEFAULT_PERO_MIN_LINE_CONFIDENCE)
    )
    config_mtime = config_path.stat().st_mtime
    params = (
        str(source_path) if source_path else "",
        str(config_path),
        config_mtime,
        device_name,
        gpu_id,
        min_confidence,
    )

    with _parser_lock:
        if _page_parser is not None and _parser_params == params:
            return _page_parser, _page_layout_cls

        ensure_pero_import(source_path)
        try:
            PageParser = importlib.import_module(
                "pero_ocr.document_ocr.page_parser"
            ).PageParser
            PageLayout = importlib.import_module("pero_ocr.core.layout").PageLayout
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PERO-OCR is importable, but one of its runtime dependencies "
                f"is missing: {exc.name}. Install PERO-OCR dependencies into "
                "the active Python environment first."
            ) from exc

        config = load_pero_config(config_path)
        config["PAGE_PARSER"]["FILTER_CONFIDENT_LINES_THRESHOLD"] = str(
            min_confidence
        )

        device = build_torch_device(device_name, gpu_id)
        parser = PageParser(
            config,
            config_path=str(config_path.parent),
            device=device,
        )

        _page_parser = parser
        _page_layout_cls = PageLayout
        _parser_params = params
        return _page_parser, _page_layout_cls


def ensure_pero_import(source_path: Path | None) -> None:
    try:
        importlib.import_module("pero_ocr")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "pero_ocr":
            raise RuntimeError(
                "PERO-OCR is installed or importable, but one of its dependencies "
                f"is missing: {exc.name}. Install PERO-OCR dependencies first."
            ) from exc

    if source_path is not None and source_path.exists():
        source_path_str = str(source_path)
        if source_path_str not in sys.path:
            sys.path.insert(0, source_path_str)

    try:
        importlib.import_module("pero_ocr")
    except ModuleNotFoundError as exc:
        if exc.name == "pero_ocr":
            raise RuntimeError(
                "Unable to import pero_ocr. Install pero-ocr into this Python "
                "environment or set PERO Python module path to the local "
                "pero-ocr checkout."
            ) from exc
        raise RuntimeError(
            "PERO-OCR source was found, but one of its dependencies is missing: "
            f"{exc.name}. Install PERO-OCR dependencies first."
        ) from exc


def load_pero_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    read_paths = config.read(config_path)
    if not read_paths:
        raise FileNotFoundError(f"PERO config file could not be read: {config_path}")
    if "PAGE_PARSER" not in config:
        raise ValueError(
            "PERO config is missing the required [PAGE_PARSER] section."
        )
    return config


def build_torch_device(device_name: str, gpu_id: int | None):
    torch = importlib.import_module("torch")

    if device_name == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PERO device is set to gpu, but torch.cuda.is_available() is false. "
            "Choose CPU or install/configure CUDA for the active environment."
        )

    if gpu_id is None:
        return torch.device("cuda")

    return torch.device(f"cuda:{gpu_id}")


def normalize_pero_device(value: Any) -> str:
    value = str(value or DEFAULT_PERO_DEVICE).strip().lower()
    if value not in {"gpu", "cpu"}:
        return DEFAULT_PERO_DEVICE
    return value


def parse_gpu_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def parse_min_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_PERO_MIN_LINE_CONFIDENCE
    return parsed


def normalize_path(value: Any, allow_missing: bool) -> Path | None:
    if value is None:
        if allow_missing:
            return None
        raise ValueError("A PERO config path is required.")

    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        if allow_missing:
            return None
        raise ValueError("A PERO config path is required.")

    path = Path(windows_path_to_wsl(raw)).expanduser()
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"PERO path does not exist: {path}")
    return path


def windows_path_to_wsl(value: str) -> str:
    if len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}:
        drive = value[0].lower()
        rest = value[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return value.replace("\\", "/") if value.startswith("\\\\") else value


def page_layout_to_ocr_payload(
    page_layout: Any,
    image_path: Path,
    opt: dict,
) -> dict[str, Any]:
    rec_texts: list[str] = []
    rec_scores: list[float | None] = []
    rec_polys: list[Any] = []
    rec_boxes: list[list[float]] = []
    blocks: list[dict[str, Any]] = []
    line_index = 0

    regions = list(getattr(page_layout, "regions", []) or [])
    for region_order, region in enumerate(regions):
        block_lines = []
        region_polygon = to_plain_geometry(getattr(region, "polygon", None))

        for line_order, line in enumerate(getattr(region, "lines", []) or []):
            text = str(getattr(line, "transcription", "") or "").strip()
            if not text:
                continue

            polygon = to_plain_geometry(getattr(line, "polygon", None))
            if polygon is None:
                polygon = line_fallback_polygon(line)
            if polygon is None:
                continue

            score = getattr(line, "transcription_confidence", None)
            if score is not None:
                score = float(score)

            line_object = {
                "box": deepcopy(polygon),
                "score": score,
                "index": line_index,
                "strings": [
                    {
                        "text": text,
                        "box": deepcopy(polygon),
                        "score": score,
                        "line_index": line_index,
                    }
                ],
            }

            rec_texts.append(text)
            rec_scores.append(score)
            rec_polys.append(deepcopy(polygon))
            rec_boxes.append(geometry_to_bbox(polygon))
            block_lines.append(line_object)
            line_index += 1

        if block_lines:
            block_box = (
                region_polygon
                if region_polygon is not None
                else union_line_boxes(block_lines)
            )
            blocks.append(
                {
                    "box": block_box,
                    "lines": block_lines,
                    "label": "pero_region",
                    "order": region_order,
                }
            )

    return {
        "input_path": str(image_path),
        "fileName": image_path.name,
        "page_index": 0,
        "model_settings": {
            "engine": PERO_METHOD,
            "config_path": str(
                normalize_path(
                    opt.get("pero_config_path") or DEFAULT_PERO_CONFIG_PATH,
                    allow_missing=True,
                )
                or ""
            ),
            "device": normalize_pero_device(
                opt.get("pero_device") or DEFAULT_PERO_DEVICE
            ),
        },
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys,
        "rec_boxes": rec_boxes,
        "dt_polys": deepcopy(rec_polys),
        "ocr_blocks": blocks,
        "source_engine": PERO_METHOD,
    }


def to_plain_geometry(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def line_fallback_polygon(line: Any) -> list[list[float]] | None:
    baseline = to_plain_geometry(getattr(line, "baseline", None))
    heights = to_plain_geometry(getattr(line, "heights", None))
    if not baseline or not heights:
        return None

    x_values = [float(point[0]) for point in baseline]
    y_values = [float(point[1]) for point in baseline]
    if not x_values or not y_values:
        return None

    height_up = float(heights[0]) if len(heights) > 0 else 0.0
    height_down = float(heights[1]) if len(heights) > 1 else height_up
    x0 = min(x_values)
    x1 = max(x_values)
    y0 = min(y_values) - height_up
    y1 = max(y_values) + height_down
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def geometry_to_bbox(geometry: Any) -> list[float]:
    if not geometry:
        return [0.0, 0.0, 0.0, 0.0]

    if (
        isinstance(geometry, list)
        and len(geometry) == 4
        and all(isinstance(value, (int, float)) for value in geometry)
    ):
        return [float(value) for value in geometry]

    points = geometry
    if (
        isinstance(points, list)
        and points
        and isinstance(points[0], (list, tuple))
    ):
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    return [0.0, 0.0, 0.0, 0.0]


def union_line_boxes(lines: list[dict[str, Any]]) -> list[float]:
    boxes = [geometry_to_bbox(line["box"]) for line in lines]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
