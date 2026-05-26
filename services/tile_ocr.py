"""Split large images into OCR tiles and merge tile-level OCR results back together."""

from __future__ import annotations

import json
import threading
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from numbers import Number
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from engine import engine
from services.ocr_defaults import (
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_TILE_OVERLAP_PX,
)
from services.word_box_normalization import normalize_ocr_word_boxes_in_payload

SUPPORTED_TILING_METHODS = {"PP-OCRv5", "PP-StructureV3"}
TILE_GRID_SIZE = 3
DEFAULT_TILE_ROWS = 3
DEFAULT_TILE_COLS = 1
DEFAULT_DOCLAYOUT_MODEL = "PP-DocLayoutV3"
TILE_STRATEGY_GRID = "grid"
TILE_STRATEGY_GRID_3X2 = "grid_3x2"
TILE_STRATEGY_DOCLAYOUT_V3 = "doclayout_v3"
TILE_STRATEGY_GRID_DOCLAYOUT_V3 = "grid_doclayout_v3"
TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3 = "grid_3x2_doclayout_v3"
DEFAULT_TILE_STRATEGY = TILE_STRATEGY_GRID_DOCLAYOUT_V3
GRID_TILE_STRATEGIES = {
    TILE_STRATEGY_GRID,
    TILE_STRATEGY_GRID_3X2,
    TILE_STRATEGY_GRID_DOCLAYOUT_V3,
    TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
}
GRID_DOCLAYOUT_TILE_STRATEGIES = {
    TILE_STRATEGY_GRID_DOCLAYOUT_V3,
    TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
}
SUPPORTED_TILE_STRATEGIES = {
    TILE_STRATEGY_GRID,
    TILE_STRATEGY_GRID_3X2,
    TILE_STRATEGY_DOCLAYOUT_V3,
    TILE_STRATEGY_GRID_DOCLAYOUT_V3,
    TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
}
DOCLAYOUT_TEXT_LABELS = {
    "abstract",
    "aside_text",
    "content",
    "doc_title",
    "footer",
    "footnote",
    "header",
    "paragraph_title",
    "reference",
    "reference_content",
    "table",
    "table_caption",
    "text",
    "vertical_text",
    "vision_footnote",
}
DOCLAYOUT_EXCLUDED_LABELS = {
    "chart",
    "display_formula",
    "figure",
    "figure_title",
    "footer_image",
    "formula_number",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "seal",
}
EXTRA_STRUCTURE_KEYS = ("table_res_list", "seal_res_list", "formula_res_list")
GEOMETRY_KEYS = {
    "bbox",
    "box",
    "boxes",
    "block_bbox",
    "block_polygon_points",
    "coordinate",
    "coordinates",
    "dt_polys",
    "points",
    "polygon",
    "polygon_points",
    "polygons",
    "rec_boxes",
    "rec_polys",
    "text_word_boxes",
}
GEOMETRY_SUFFIXES = (
    "_bbox",
    "_box",
    "_boxes",
    "_coordinate",
    "_coordinates",
    "_polygon_points",
    "_polys",
    "_points",
)


@dataclass(frozen=True)
class TileSpec:
    row: int
    col: int
    left: int
    top: int
    right: int
    bottom: int
    nominal_left: int
    nominal_top: int
    nominal_right: int
    nominal_bottom: int
    path: Path
    label: str | None = None
    score: float | None = None
    order: int | None = None


_doclayout_model: Any | None = None
_doclayout_model_lock = threading.Lock()


def get_effective_method(method: str | None, select: str | None = None) -> str | None:
    """Resolve the actual local OCR method that tiled OCR should use."""
    if select in SUPPORTED_TILING_METHODS:
        return select
    if method in SUPPORTED_TILING_METHODS:
        return method
    return None


def supports_tiled_ocr(method: str | None, select: str | None = None) -> bool:
    """Return whether the selected OCR method can run through the tiling pipeline."""
    return get_effective_method(method, select) is not None


def run_tiled_local_ocr(
    image_path: Path,
    opt: dict,
    select: str | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> dict[str, Any]:
    """Run OCR on a tiled workflow and merge the tile-level results."""
    effective_method = get_effective_method(opt.get("method"), select)
    if effective_method is None:
        raise ValueError(
            "Source-image tiling is only supported for PP-OCRv5, "
            "PP-StructureV3."
        )

    tile_opt = dict(opt)
    tile_opt["tile_source_images"] = False

    grid_rows, grid_cols = resolve_tile_grid(opt, grid_rows, grid_cols)
    tile_strategy = resolve_tile_strategy(opt)
    overlap_value = opt.get("tile_overlap_px")
    grid_overlap_px = max(
        0,
        int(DEFAULT_TILE_OVERLAP_PX if overlap_value is None else overlap_value),
    )
    doclayout_padding_value = opt.get("tile_doclayout_padding_px")
    doclayout_padding_px = max(
        0,
        int(
            DEFAULT_DOCLAYOUT_PADDING_PX
            if doclayout_padding_value is None
            else doclayout_padding_value
        ),
    )
    doclayout_threshold = clamp_float(
        opt.get("tile_doclayout_threshold"),
        DEFAULT_DOCLAYOUT_THRESHOLD,
        0.01,
        1.0,
    )
    doclayout_recovery_enabled = bool(opt.get("tile_doclayout_recovery", True))
    overlap_px = grid_overlap_px

    with tempfile.TemporaryDirectory(prefix="ocr_tiles_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if tile_strategy in GRID_DOCLAYOUT_TILE_STRATEGIES:
            overlap_px = grid_overlap_px
            grid_doclayout_tiles = build_grid_doclayout_tiles(
                image_path,
                temp_dir,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                grid_overlap_px=grid_overlap_px,
                padding_px=doclayout_padding_px,
                threshold=doclayout_threshold,
                include_grid_recovery=doclayout_recovery_enabled,
            )
            if grid_doclayout_tiles:
                tiles = grid_doclayout_tiles
            else:
                tile_strategy = grid_strategy_for_shape(grid_rows, grid_cols)

        if tile_strategy == TILE_STRATEGY_DOCLAYOUT_V3:
            overlap_px = doclayout_padding_px
            doclayout_tiles = build_doclayout_tiles(
                image_path,
                temp_dir,
                padding_px=doclayout_padding_px,
                threshold=doclayout_threshold,
                include_recovery_tile=doclayout_recovery_enabled,
            )
            if doclayout_tiles:
                tiles = doclayout_tiles
                grid_rows = max(1, len(tiles))
                grid_cols = 1
            else:
                tile_strategy = grid_strategy_for_shape(grid_rows, grid_cols)

        if tile_strategy in {TILE_STRATEGY_GRID, TILE_STRATEGY_GRID_3X2}:
            overlap_px = grid_overlap_px
            tiles = build_image_tiles(
                image_path,
                temp_dir,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                overlap_px=overlap_px,
            )

        tile_results = []
        tile_errors = []

        for tile in tiles:
            try:
                tile_json = infer_tile_json(tile.path, tile_opt)
            except Exception as exc:
                tile_errors.append({"tile": tile, "error": str(exc)})
                print(
                    (
                        "[WARN] OCR failed for tile "
                        f"{tile.path.name} ({tile.label or 'tile'}): {exc}"
                    ),
                    flush=True,
                )
                continue

            tile_results.append({"tile": tile, "json": tile_json})

    if not tile_results:
        error_summary = "; ".join(
            f"{item['tile'].path.name}: {item['error']}" for item in tile_errors[:5]
        )
        raise RuntimeError(f"No tiled OCR output was produced. {error_summary}")

    if effective_method == "PP-OCRv5":
        merged = merge_ppocr_tile_results(
            tile_results,
            image_path,
            grid_rows,
            grid_cols,
            overlap_px,
            tile_strategy,
        )
        add_tile_errors_to_metadata(merged, tile_errors)
        return merged

    merged = merge_ppstructure_tile_results(
        tile_results,
        image_path,
        grid_rows,
        grid_cols,
        overlap_px,
        tile_strategy,
    )
    add_tile_errors_to_metadata(merged, tile_errors)
    return merged


def add_tile_errors_to_metadata(
    merged: dict[str, Any],
    tile_errors: list[dict[str, Any]],
) -> None:
    """Attach per-tile OCR errors to the merged tile metadata block."""
    if not tile_errors:
        return

    tile_processing = merged.setdefault("tile_processing", {})
    tile_processing["failed_tiles"] = [
        {
            "row": item["tile"].row,
            "col": item["tile"].col,
            "bbox": [
                item["tile"].left,
                item["tile"].top,
                item["tile"].right,
                item["tile"].bottom,
            ],
            "nominal_bbox": [
                item["tile"].nominal_left,
                item["tile"].nominal_top,
                item["tile"].nominal_right,
                item["tile"].nominal_bottom,
            ],
            "label": item["tile"].label,
            "score": item["tile"].score,
            "order": item["tile"].order,
            "path": item["tile"].path.name,
            "error": item["error"],
        }
        for item in tile_errors
    ]


def resolve_tile_grid(
    opt: dict,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> tuple[int, int]:
    """Resolve the effective tile grid dimensions from OCR options."""
    strategy = resolve_tile_strategy(opt)
    strategy_rows, strategy_cols = tile_grid_for_strategy(strategy)

    if strategy in GRID_TILE_STRATEGIES:
        return strategy_rows, strategy_cols

    rows = grid_rows if grid_rows is not None else opt.get("tile_rows", strategy_rows)
    cols = grid_cols if grid_cols is not None else opt.get("tile_cols", strategy_cols)

    try:
        rows = max(1, int(rows))
    except (TypeError, ValueError):
        rows = strategy_rows

    try:
        cols = max(1, int(cols))
    except (TypeError, ValueError):
        cols = strategy_cols

    return rows, cols


def tile_grid_for_strategy(strategy: str | None) -> tuple[int, int]:
    """Return the default grid dimensions for a named tile strategy."""
    strategy = normalize_tile_strategy(strategy)
    if strategy in {TILE_STRATEGY_GRID_3X2, TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3}:
        return 3, 2
    return DEFAULT_TILE_ROWS, DEFAULT_TILE_COLS


def grid_strategy_for_shape(rows: int, cols: int) -> str:
    """Return the tile strategy label that matches a grid shape."""
    return TILE_STRATEGY_GRID_3X2 if (int(rows), int(cols)) == (3, 2) else TILE_STRATEGY_GRID


def resolve_tile_strategy(opt: dict) -> str:
    """Resolve the effective tile strategy from an OCR options dict."""
    strategy = normalize_tile_strategy(opt.get("tile_strategy"))
    return strategy if strategy in SUPPORTED_TILE_STRATEGIES else TILE_STRATEGY_GRID


def normalize_tile_strategy(value: Any) -> str:
    """Normalize a tile strategy name into a supported key."""
    strategy = str(value or DEFAULT_TILE_STRATEGY)
    strategy = strategy.strip().lower().replace("-", "_").replace(" ", "_")
    strategy_aliases = {
        "pp_doclayoutv3": TILE_STRATEGY_DOCLAYOUT_V3,
        "pp_doclayout_v3": TILE_STRATEGY_DOCLAYOUT_V3,
        "pp_doc_layout_v3": TILE_STRATEGY_DOCLAYOUT_V3,
        "doc_layout_v3": TILE_STRATEGY_DOCLAYOUT_V3,
        "layout_v3": TILE_STRATEGY_DOCLAYOUT_V3,
        "3x1": TILE_STRATEGY_GRID,
        "3x1_grid": TILE_STRATEGY_GRID,
        "fixed_3x1": TILE_STRATEGY_GRID,
        "3x2": TILE_STRATEGY_GRID_3X2,
        "3x2_grid": TILE_STRATEGY_GRID_3X2,
        "fixed_3x2": TILE_STRATEGY_GRID_3X2,
        "3x1_doclayout_v3": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "3x1_doclayout": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "3x1_pp_doclayout_v3": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "3x1_pp_doclayoutv3": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "3x2_doclayout_v3": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
        "3x2_doclayout": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
        "3x2_pp_doclayout_v3": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
        "3x2_pp_doclayoutv3": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
        "grid_doclayout": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "grid_pp_doclayout_v3": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "grid_pp_doclayoutv3": TILE_STRATEGY_GRID_DOCLAYOUT_V3,
        "grid_3x2_doclayout": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
        "grid_3x2_pp_doclayout_v3": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
        "grid_3x2_pp_doclayoutv3": TILE_STRATEGY_GRID_3X2_DOCLAYOUT_V3,
    }
    return strategy_aliases.get(strategy, strategy)


def build_image_tiles(
    image_path: Path,
    temp_dir: Path,
    grid_rows: int = TILE_GRID_SIZE,
    grid_cols: int = TILE_GRID_SIZE,
    overlap_px: int = 0,
) -> list[TileSpec]:
    """Build fixed-grid image tiles and save each cropped tile to disk."""
    tiles = []

    with Image.open(image_path) as image:
        x_edges = [(image.width * index) // grid_cols for index in range(grid_cols + 1)]
        y_edges = [(image.height * index) // grid_rows for index in range(grid_rows + 1)]
        overlap_px = max(0, int(overlap_px))

        for row in range(grid_rows):
            for col in range(grid_cols):
                nominal_left = x_edges[col]
                nominal_right = x_edges[col + 1]
                nominal_top = y_edges[row]
                nominal_bottom = y_edges[row + 1]
                left = max(0, nominal_left - (overlap_px if col > 0 else 0))
                right = min(
                    image.width,
                    nominal_right + (overlap_px if col < grid_cols - 1 else 0),
                )
                top = max(0, nominal_top - (overlap_px if row > 0 else 0))
                bottom = min(
                    image.height,
                    nominal_bottom + (overlap_px if row < grid_rows - 1 else 0),
                )
                tile_path = temp_dir / f"{image_path.stem}__r{row}_c{col}.png"

                image.crop((left, top, right, bottom)).save(tile_path)
                tiles.append(
                    TileSpec(
                        row=row,
                        col=col,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                        nominal_left=nominal_left,
                        nominal_top=nominal_top,
                        nominal_right=nominal_right,
                        nominal_bottom=nominal_bottom,
                        path=tile_path,
                    )
                )

    return tiles


def build_content_box_tiles(
    image_path: Path,
    temp_dir: Path,
    padding_px: int = 8,
) -> list[TileSpec]:
    """Build tiles from detected content boxes on a page image."""
    boxes = detect_content_boxes(image_path)
    tiles = []

    with Image.open(image_path) as image:
        image_width, image_height = image.size
        padding_px = max(0, int(padding_px))

        for index, (x0, y0, x1, y1) in enumerate(boxes):
            nominal_left = clamp_int(x0, 0, image_width - 1)
            nominal_top = clamp_int(y0, 0, image_height - 1)
            nominal_right = clamp_int(x1, nominal_left + 1, image_width)
            nominal_bottom = clamp_int(y1, nominal_top + 1, image_height)
            left = max(0, nominal_left - padding_px)
            top = max(0, nominal_top - padding_px)
            right = min(image_width, nominal_right + padding_px)
            bottom = min(image_height, nominal_bottom + padding_px)
            tile_path = temp_dir / f"{image_path.stem}__content_{index:03d}.png"

            image.crop((left, top, right, bottom)).save(tile_path)
            tiles.append(
                TileSpec(
                    row=index,
                    col=0,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    nominal_left=nominal_left,
                    nominal_top=nominal_top,
                    nominal_right=nominal_right,
                    nominal_bottom=nominal_bottom,
                    path=tile_path,
                )
            )

    return tiles


def build_doclayout_tiles(
    image_path: Path,
    temp_dir: Path,
    padding_px: int = DEFAULT_DOCLAYOUT_PADDING_PX,
    threshold: float = DEFAULT_DOCLAYOUT_THRESHOLD,
    include_recovery_tile: bool = True,
) -> list[TileSpec]:
    """Build tiles from PP-DocLayoutV3 detections on the full source image."""
    boxes = detect_doclayout_boxes(image_path, threshold=threshold)
    tiles = []

    with Image.open(image_path) as image:
        image_width, image_height = image.size
        padding_px = max(0, int(padding_px))

        for index, item in enumerate(boxes):
            x0, y0, x1, y1 = item["bbox"]
            nominal_left = clamp_int(x0, 0, image_width - 1)
            nominal_top = clamp_int(y0, 0, image_height - 1)
            nominal_right = clamp_int(x1, nominal_left + 1, image_width)
            nominal_bottom = clamp_int(y1, nominal_top + 1, image_height)
            left = max(0, nominal_left - padding_px)
            top = max(0, nominal_top - padding_px)
            right = min(image_width, nominal_right + padding_px)
            bottom = min(image_height, nominal_bottom + padding_px)
            tile_path = temp_dir / f"{image_path.stem}__doclayout_{index:03d}.png"

            image.crop((left, top, right, bottom)).save(tile_path)
            tiles.append(
                TileSpec(
                    row=index,
                    col=0,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    nominal_left=nominal_left,
                    nominal_top=nominal_top,
                    nominal_right=nominal_right,
                    nominal_bottom=nominal_bottom,
                    path=tile_path,
                    label=item.get("label"),
                    score=item.get("score"),
                    order=item.get("order"),
                )
            )

        if include_recovery_tile and tiles:
            tiles.append(
                TileSpec(
                    row=len(tiles),
                    col=0,
                    left=0,
                    top=0,
                    right=image_width,
                    bottom=image_height,
                    nominal_left=0,
                    nominal_top=0,
                    nominal_right=image_width,
                    nominal_bottom=image_height,
                    path=image_path,
                    label="layout_recovery",
                    order=len(tiles),
                )
            )

    return tiles


def build_grid_doclayout_tiles(
    image_path: Path,
    temp_dir: Path,
    grid_rows: int = DEFAULT_TILE_ROWS,
    grid_cols: int = DEFAULT_TILE_COLS,
    grid_overlap_px: int = DEFAULT_TILE_OVERLAP_PX,
    padding_px: int = DEFAULT_DOCLAYOUT_PADDING_PX,
    threshold: float = DEFAULT_DOCLAYOUT_THRESHOLD,
    include_grid_recovery: bool = True,
) -> list[TileSpec]:
    """Build fixed grid tiles, then run DocLayout inside each grid tile."""
    grid_source_dir = temp_dir / "grid_doclayout_sources"
    grid_source_dir.mkdir(exist_ok=True)
    source_tiles = build_image_tiles(
        image_path,
        grid_source_dir,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        overlap_px=grid_overlap_px,
    )
    tiles: list[TileSpec] = []

    for source_tile in source_tiles:
        boxes = detect_doclayout_boxes(source_tile.path, threshold=threshold)
        if not boxes:
            tiles.append(
                copy_tile_with_sequence(
                    source_tile,
                    len(tiles),
                    label="grid_fallback",
                )
            )
            continue

        tiles.extend(
            build_doclayout_tiles_from_source_tile(
                source_tile,
                boxes,
                temp_dir,
                start_index=len(tiles),
                padding_px=padding_px,
            )
        )
        if include_grid_recovery:
            tiles.append(
                copy_tile_with_sequence(
                    source_tile,
                    len(tiles),
                    label="grid_recovery",
                )
            )

    return tiles


def build_doclayout_tiles_from_source_tile(
    source_tile: TileSpec,
    boxes: list[dict[str, Any]],
    temp_dir: Path,
    start_index: int,
    padding_px: int,
) -> list[TileSpec]:
    """Build DocLayout child tiles inside one previously generated source tile."""
    tiles: list[TileSpec] = []

    with Image.open(source_tile.path) as image:
        image_width, image_height = image.size
        padding_px = max(0, int(padding_px))

        for item in boxes:
            x0, y0, x1, y1 = item["bbox"]
            nominal_left = clamp_int(x0, 0, image_width - 1)
            nominal_top = clamp_int(y0, 0, image_height - 1)
            nominal_right = clamp_int(x1, nominal_left + 1, image_width)
            nominal_bottom = clamp_int(y1, nominal_top + 1, image_height)

            if not local_box_center_is_in_source_nominal(
                source_tile,
                (nominal_left, nominal_top, nominal_right, nominal_bottom),
            ):
                continue

            left = max(0, nominal_left - padding_px)
            top = max(0, nominal_top - padding_px)
            right = min(image_width, nominal_right + padding_px)
            bottom = min(image_height, nominal_bottom + padding_px)
            tile_index = start_index + len(tiles)
            tile_path = (
                temp_dir
                / f"{source_tile.path.stem}__doclayout_{tile_index:03d}.png"
            )

            image.crop((left, top, right, bottom)).save(tile_path)
            tiles.append(
                TileSpec(
                    row=source_tile.row,
                    col=source_tile.col,
                    left=source_tile.left + left,
                    top=source_tile.top + top,
                    right=source_tile.left + right,
                    bottom=source_tile.top + bottom,
                    nominal_left=source_tile.left + nominal_left,
                    nominal_top=source_tile.top + nominal_top,
                    nominal_right=source_tile.left + nominal_right,
                    nominal_bottom=source_tile.top + nominal_bottom,
                    path=tile_path,
                    label=item.get("label"),
                    score=item.get("score"),
                    order=tile_index,
                )
            )

    if not tiles:
        return [
            copy_tile_with_sequence(
                source_tile,
                start_index,
                label="grid_fallback",
            )
        ]

    return tiles


def local_box_center_is_in_source_nominal(
    source_tile: TileSpec,
    local_box: tuple[int, int, int, int],
) -> bool:
    """Return whether a local box center falls inside the source tile nominal area."""
    center_x = source_tile.left + ((local_box[0] + local_box[2]) / 2.0)
    center_y = source_tile.top + ((local_box[1] + local_box[3]) / 2.0)

    return (
        source_tile.nominal_left <= center_x <= source_tile.nominal_right
        and source_tile.nominal_top <= center_y <= source_tile.nominal_bottom
    )


def copy_tile_with_sequence(
    tile: TileSpec,
    sequence_index: int,
    label: str | None = None,
) -> TileSpec:
    """Copy a generated tile image while assigning a stable output sequence."""
    return TileSpec(
        row=tile.row,
        col=tile.col,
        left=tile.left,
        top=tile.top,
        right=tile.right,
        bottom=tile.bottom,
        nominal_left=tile.nominal_left,
        nominal_top=tile.nominal_top,
        nominal_right=tile.nominal_right,
        nominal_bottom=tile.nominal_bottom,
        path=tile.path,
        label=label or tile.label,
        score=tile.score,
        order=sequence_index,
    )


def detect_doclayout_boxes(
    image_path: Path,
    threshold: float = DEFAULT_DOCLAYOUT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detect document layout boxes for tile generation."""
    try:
        model = get_doclayout_model()
        output = model.predict(
            str(image_path),
            batch_size=1,
            threshold=threshold,
        )
    except Exception as exc:
        print(
            f"[WARN] PP-DocLayoutV3 layout analysis failed for {image_path}: {exc}",
            flush=True,
        )
        return []

    boxes: list[dict[str, Any]] = []
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    for result in output:
        boxes.extend(extract_doclayout_boxes(result, image_width, image_height))

    boxes = dedupe_doclayout_boxes(boxes)
    boxes.sort(
        key=lambda item: (
            item["order"] if item.get("order") is not None else 10_000,
            item["bbox"][1],
            item["bbox"][0],
        )
    )
    return boxes


def get_doclayout_model() -> Any:
    """Return the cached PP-DocLayoutV3 model instance."""
    global _doclayout_model

    with _doclayout_model_lock:
        if _doclayout_model is None:
            from paddlex import create_model

            _doclayout_model = create_model(model_name=DEFAULT_DOCLAYOUT_MODEL)

    return _doclayout_model


def extract_doclayout_boxes(
    result: Any,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Extract normalized DocLayout boxes from a model prediction payload."""
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()

    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]

    raw_boxes = find_doclayout_box_lists(payload)
    boxes: list[dict[str, Any]] = []

    for raw_box in raw_boxes:
        parsed = parse_doclayout_box(raw_box, image_width, image_height)
        if parsed is not None:
            boxes.append(parsed)

    return boxes


def find_doclayout_box_lists(value: Any) -> list[dict[str, Any]]:
    """Find list-shaped DocLayout box collections inside a nested payload."""
    if isinstance(value, dict):
        boxes = value.get("boxes") or value.get("layout_boxes")
        if isinstance(boxes, list) and all(isinstance(item, dict) for item in boxes):
            return boxes

        collected: list[dict[str, Any]] = []
        for child in value.values():
            collected.extend(find_doclayout_box_lists(child))
        return collected

    if isinstance(value, list):
        collected: list[dict[str, Any]] = []
        for item in value:
            collected.extend(find_doclayout_box_lists(item))
        return collected

    return []


def parse_doclayout_box(
    item: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Parse one raw DocLayout detection into normalized metadata."""
    label = str(
        item.get("label")
        or item.get("category_name")
        or item.get("class_name")
        or ""
    ).strip()
    normalized_label = label.lower()

    if normalized_label in DOCLAYOUT_EXCLUDED_LABELS:
        return None
    if DOCLAYOUT_TEXT_LABELS and normalized_label and normalized_label not in DOCLAYOUT_TEXT_LABELS:
        return None

    coordinate = first_present(
        item,
        "coordinate",
        "coordinates",
        "bbox",
        "box",
        "polygon_points",
    )
    bbox = normalize_doclayout_bbox(coordinate, image_width, image_height)
    if bbox is None:
        return None

    min_area = max(64, int(image_width * image_height * 0.00002))
    if content_box_width(bbox) * content_box_height(bbox) < min_area:
        return None

    order = parse_optional_int(
        first_present(item, "order", "read_order", "reading_order", "order_index")
    )
    score = parse_optional_float(first_present(item, "score", "confidence"))

    return {
        "bbox": bbox,
        "label": label or None,
        "score": score,
        "order": order,
    }


def normalize_doclayout_bbox(
    coordinate: Any,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Normalize a DocLayout bounding box into integer image coordinates."""
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 4:
        return None

    if isinstance(coordinate[0], (list, tuple)):
        points = [
            (float(point[0]), float(point[1]))
            for point in coordinate
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not points:
            return None
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        values = [min(x_values), min(y_values), max(x_values), max(y_values)]
    else:
        values = [float(value) for value in coordinate[:4]]

    x0, y0, x1, y1 = values

    if x1 <= x0 or y1 <= y0:
        x1 = x0 + max(0.0, values[2])
        y1 = y0 + max(0.0, values[3])

    left = clamp_int(round(x0), 0, image_width - 1)
    top = clamp_int(round(y0), 0, image_height - 1)
    right = clamp_int(round(x1), left + 1, image_width)
    bottom = clamp_int(round(y1), top + 1, image_height)

    return (left, top, right, bottom)


def dedupe_doclayout_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate highly overlapping DocLayout boxes."""
    kept: list[dict[str, Any]] = []

    for box in sorted(
        boxes,
        key=lambda item: (
            float(item.get("score") or 0.0),
            content_box_width(item["bbox"]) * content_box_height(item["bbox"]),
        ),
        reverse=True,
    ):
        if any(
            content_bbox_overlap_ratio(box["bbox"], existing["bbox"]) > 0.9
            for existing in kept
        ):
            continue
        kept.append(box)

    return kept


def content_bbox_intersection(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    """Return the intersection area between two content boxes."""
    return content_box_horizontal_overlap(left, right) * content_box_vertical_overlap(
        left,
        right,
    )


def content_bbox_overlap_ratio(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return the overlap ratio between two content boxes."""
    intersection = content_bbox_intersection(left, right)
    smaller_area = min(
        content_box_width(left) * content_box_height(left),
        content_box_width(right) * content_box_height(right),
    )
    return intersection / smaller_area if smaller_area > 0 else 0.0


def parse_optional_int(value: Any) -> int | None:
    """Parse an optional integer value and return `None` when unset."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> float | None:
    """Parse an optional float value and return `None` when unset."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_present(item: dict[str, Any], *keys: str) -> Any:
    """Return the first present value from a dict across several keys."""
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def detect_content_boxes(image_path: Path, max_boxes: int = 96) -> list[tuple[int, int, int, int]]:
    """Detect foreground content boxes directly from the source image."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    image_height, image_width = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, foreground = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )
    foreground = remove_rule_lines(foreground, image_width, image_height)

    boxes: list[tuple[int, int, int, int]] = []
    for strength in (1, 2, 3):
        candidate_boxes = find_content_boxes_for_strength(
            foreground,
            image_width,
            image_height,
            strength,
        )
        candidate_boxes = split_content_boxes_on_vertical_whitespace(
            candidate_boxes,
            foreground,
            image_width,
        )
        boxes = group_content_boxes_into_blocks(
            candidate_boxes,
            image_width,
            image_height,
        )
        if 0 < len(boxes) <= max_boxes:
            return boxes

    if len(boxes) > max_boxes:
        return merge_boxes_into_reading_rows(boxes, image_width, image_height)

    return boxes


def remove_rule_lines(foreground: Any, image_width: int, image_height: int) -> Any:
    """Remove large horizontal or vertical rule lines from a foreground mask."""
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, image_width // 8), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(40, image_height // 12)),
    )
    horizontal_lines = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )
    vertical_lines = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )
    rule_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    return cv2.subtract(foreground, rule_lines)


def find_content_boxes_for_strength(
    foreground: Any,
    image_width: int,
    image_height: int,
    strength: int,
) -> list[tuple[int, int, int, int]]:
    """Find connected content boxes using one mask-strength setting."""
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(9, image_width // (95 // strength)),
            max(3, image_height // (380 // strength)),
        ),
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(15, image_width // (70 // strength)),
            max(5, image_height // (260 // strength)),
        ),
    )
    content_mask = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    content_mask = cv2.dilate(content_mask, dilate_kernel, iterations=1)
    contours, _ = cv2.findContours(
        content_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    min_area = max(64, int(image_width * image_height * 0.00025))
    min_width = max(8, image_width // 120)
    min_height = max(8, image_height // 180)
    boxes = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < min_area or width < min_width or height < min_height:
            continue
        boxes.append((x, y, x + width, y + height))

    return sorted(boxes, key=lambda box: (box[1], box[0], box[3], box[2]))


def split_content_boxes_on_vertical_whitespace(
    boxes: list[tuple[int, int, int, int]],
    foreground: Any,
    image_width: int,
) -> list[tuple[int, int, int, int]]:
    """Split content boxes on strong internal vertical whitespace runs."""
    split_boxes: list[tuple[int, int, int, int]] = []

    for box in boxes:
        split_boxes.extend(
            split_content_box_on_vertical_whitespace(box, foreground, image_width)
        )

    return split_boxes


def split_content_box_on_vertical_whitespace(
    box: tuple[int, int, int, int],
    foreground: Any,
    image_width: int,
) -> list[tuple[int, int, int, int]]:
    """Split one content box around strong internal vertical whitespace."""
    x0, y0, x1, y1 = box
    width = content_box_width(box)
    height = content_box_height(box)

    if width < max(260, int(image_width * 0.28)) or height < 45:
        return [box]

    crop = foreground[y0:y1, x0:x1]
    if crop.size == 0:
        return [box]

    density = np.count_nonzero(crop, axis=0).astype(np.float32)
    smooth_width = max(7, (width // 70) | 1)
    smoothed_density = np.convolve(
        density,
        np.ones(smooth_width, dtype=np.float32) / smooth_width,
        mode="same",
    )
    whitespace_threshold = max(1.0, height * 0.1)
    min_gutter_width = max(16, int(width * 0.04))
    gutters = find_vertical_whitespace_runs(
        smoothed_density,
        whitespace_threshold,
        min_gutter_width,
        width,
    )

    if not gutters:
        return [box]

    selected_gutters = choose_vertical_gutters(gutters, min_gutter_width)
    if not selected_gutters:
        return [box]

    min_segment_width = max(24, int(width * 0.08))
    segments: list[tuple[int, int, int, int]] = []
    previous_x = 0

    for gutter_start, gutter_end in [*selected_gutters, (width, width)]:
        if gutter_start - previous_x >= min_segment_width:
            segment = (x0 + previous_x, y0, x0 + gutter_start, y1)
            trimmed = trim_content_box_to_foreground(segment, foreground)
            if (
                trimmed is not None
                and content_box_width(trimmed) >= 8
                and content_box_height(trimmed) >= 8
            ):
                segments.append(trimmed)
        previous_x = gutter_end

    return segments or [box]


def find_vertical_whitespace_runs(
    density: Any,
    threshold: float,
    min_width: int,
    width: int,
) -> list[tuple[int, int]]:
    """Find long whitespace runs inside a content box projection."""
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for x, value in enumerate(density):
        inside_content = width * 0.03 < x < width * 0.97
        if inside_content and value <= threshold:
            if start is None:
                start = x
            continue

        if start is not None and x - start >= min_width:
            runs.append((start, x))
        start = None

    if start is not None and width - start >= min_width:
        runs.append((start, width))

    return runs


def choose_vertical_gutters(
    gutters: list[tuple[int, int]],
    min_gutter_width: int,
) -> list[tuple[int, int]]:
    """Choose the whitespace gutters that are strong enough to split on."""
    selected_gutters: list[tuple[int, int]] = []

    for start, end in sorted(gutters, key=lambda run: run[1] - run[0], reverse=True):
        center = (start + end) // 2
        if all(
            abs(center - ((existing_start + existing_end) // 2)) > min_gutter_width
            for existing_start, existing_end in selected_gutters
        ):
            selected_gutters.append((start, end))

    return sorted(selected_gutters[:3])


def trim_content_box_to_foreground(
    box: tuple[int, int, int, int],
    foreground: Any,
) -> tuple[int, int, int, int] | None:
    """Trim a content box down to its tight foreground bounds."""
    x0, y0, x1, y1 = box
    crop = foreground[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    points = cv2.findNonZero(crop)
    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    return (x0 + x, y0 + y, x0 + x + width, y0 + y + height)


def group_content_boxes_into_blocks(
    boxes: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    """Group detected content boxes into larger reading-order blocks."""
    if not boxes:
        return []

    line_boxes = merge_same_line_content_boxes(boxes, image_width)
    block_boxes = merge_stacked_content_lines(line_boxes, image_width, image_height)
    return sorted(block_boxes, key=lambda box: (box[1], box[0], box[3], box[2]))


def merge_same_line_content_boxes(
    boxes: list[tuple[int, int, int, int]],
    image_width: int,
) -> list[tuple[int, int, int, int]]:
    """Merge neighboring content boxes that belong to the same text line."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    median_height = content_box_median_height(boxes)
    gap_limit = max(18, int(image_width * 0.045), int(median_height * 1.3))
    remaining = boxes[:]
    lines: list[tuple[int, int, int, int]] = []

    while remaining:
        group = [remaining.pop(0)]
        changed = True

        while changed:
            changed = False
            group_box = merge_content_boxes(group)
            keep: list[tuple[int, int, int, int]] = []

            for box in remaining:
                horizontal_gap = max(
                    0,
                    box[0] - group_box[2],
                    group_box[0] - box[2],
                )
                if (
                    content_boxes_are_on_same_line(group_box, box, median_height)
                    and horizontal_gap <= gap_limit
                ):
                    group.append(box)
                    changed = True
                else:
                    keep.append(box)

            remaining = keep

        lines.append(merge_content_boxes(group))

    return sorted(lines, key=lambda box: (box[1], box[0], box[3], box[2]))


def merge_stacked_content_lines(
    lines: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    """Merge vertically stacked content lines that belong to one block."""
    blocks: list[tuple[int, int, int, int]] = []

    for line in sorted(lines, key=lambda box: (box[1], box[0])):
        candidates: list[tuple[float, float, int]] = []

        for index, block in enumerate(blocks):
            if not stacked_content_lines_belong_together(
                block,
                line,
                image_width,
                image_height,
            ):
                continue

            vertical_gap = max(0, line[1] - block[3], block[1] - line[3])
            horizontal_overlap = content_box_horizontal_overlap(block, line)
            overlap_ratio = horizontal_overlap / max(
                1,
                min(content_box_width(block), content_box_width(line)),
            )
            candidates.append((vertical_gap, -overlap_ratio, index))

        if candidates:
            _, _, index = min(candidates)
            blocks[index] = merge_two_content_boxes(blocks[index], line)
        else:
            blocks.append(line)

    return sorted(blocks, key=lambda box: (box[1], box[0], box[3], box[2]))


def content_boxes_are_on_same_line(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    median_height: int,
) -> bool:
    """Return whether two content boxes belong to the same line."""
    left_height = content_box_height(left)
    right_height = content_box_height(right)
    height_ratio = max(left_height, right_height) / max(1, min(left_height, right_height))
    tall_box = max(left_height, right_height) > max(44, int(median_height * 2.3))
    multiline_box = min(left_height, right_height) >= max(92, int(median_height * 3.6))

    if height_ratio > 2.4 or multiline_box:
        return False

    overlap_ratio = content_box_vertical_overlap(left, right) / max(
        1,
        min(left_height, right_height),
    )
    if overlap_ratio >= (0.62 if tall_box else 0.5):
        return True

    center_delta = abs(content_box_center_y(left) - content_box_center_y(right))
    return not tall_box and center_delta <= max(6, median_height * 0.35)


def stacked_content_lines_belong_together(
    block: tuple[int, int, int, int],
    line: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    """Return whether vertically stacked lines should be merged into one block."""
    if line[1] < block[1]:
        block, line = line, block

    vertical_gap = max(0, line[1] - block[3])
    block_width = content_box_width(block)
    line_width = content_box_width(line)
    block_height = content_box_height(block)
    line_height = content_box_height(line)
    gap_limit = max(10, int(min(block_height, line_height) * 0.7), image_height // 120)

    if vertical_gap > gap_limit:
        return False

    horizontal_overlap = content_box_horizontal_overlap(block, line)
    if horizontal_overlap / max(1, min(block_width, line_width)) < 0.52:
        return False

    width_ratio = max(block_width, line_width) / max(1, min(block_width, line_width))
    if width_ratio > 2.25:
        return False

    center_delta = abs(content_box_center_x(block) - content_box_center_x(line))
    if width_ratio > 1.8 and center_delta > max(35, min(block_width, line_width) * 0.5):
        return False

    height_ratio = max(block_height, line_height) / max(1, min(block_height, line_height))
    if height_ratio > 3.0 and vertical_gap > max(4, int(min(block_height, line_height) * 0.25)):
        return False

    merged = merge_two_content_boxes(block, line)
    merged_is_page_spanning = (
        content_box_height(merged) > int(image_height * 0.24)
        and content_box_width(merged) > image_width * 0.44
    )
    return not merged_is_page_spanning


def content_box_median_height(boxes: list[tuple[int, int, int, int]]) -> int:
    """Return the median height across detected content boxes."""
    heights = sorted(content_box_height(box) for box in boxes)
    return max(1, heights[len(heights) // 2])


def content_box_width(box: tuple[int, int, int, int]) -> int:
    """Return the width of a content box."""
    return max(0, box[2] - box[0])


def content_box_height(box: tuple[int, int, int, int]) -> int:
    """Return the height of a content box."""
    return max(0, box[3] - box[1])


def content_box_center_x(box: tuple[int, int, int, int]) -> float:
    """Return the horizontal center of a content box."""
    return (box[0] + box[2]) / 2.0


def content_box_center_y(box: tuple[int, int, int, int]) -> float:
    """Return the vertical center of a content box."""
    return (box[1] + box[3]) / 2.0


def content_box_horizontal_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    """Return the horizontal overlap between content boxes."""
    return max(0, min(left[2], right[2]) - max(left[0], right[0]))


def content_box_vertical_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    """Return the vertical overlap between content boxes."""
    return max(0, min(left[3], right[3]) - max(left[1], right[1]))


def merge_content_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Merge a sequence of content boxes into one bounding box."""
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def merge_two_content_boxes(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Merge two content boxes into one bounding box."""
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def merge_boxes_into_reading_rows(
    boxes: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    """Merge boxes into top-to-bottom reading rows."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    median_height = sorted(box[3] - box[1] for box in boxes)[len(boxes) // 2]
    row_gap = max(12, int(median_height * 1.5), image_height // 80)
    rows: list[list[tuple[int, int, int, int]]] = [[boxes[0]]]
    current_bottom = boxes[0][3]

    for box in boxes[1:]:
        if box[1] - current_bottom <= row_gap:
            rows[-1].append(box)
            current_bottom = max(current_bottom, box[3])
        else:
            rows.append([box])
            current_bottom = box[3]

    row_boxes = []
    for row in rows:
        x0 = max(0, min(box[0] for box in row))
        y0 = max(0, min(box[1] for box in row))
        x1 = min(image_width, max(box[2] for box in row))
        y1 = min(image_height, max(box[3] for box in row))
        row_boxes.append((x0, y0, x1, y1))

    return row_boxes


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    """Clamp an integer into an inclusive range."""
    return max(min_value, min(int(value), max_value))


def clamp_float(
    value: Any,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    """Clamp a float into an inclusive range."""
    try:
        parsed = default if value is None else float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def infer_tile_json(tile_path: Path, opt: dict) -> dict[str, Any]:
    """Infer the JSON payload path associated with a tile image."""
    with tempfile.TemporaryDirectory(prefix="ocr_tile_json_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for result in engine.infer(str(tile_path), opt):
            result.save_to_json(save_path=str(temp_dir))

        json_path = temp_dir / f"{tile_path.stem}_res.json"
        if not json_path.exists():
            candidates = sorted(temp_dir.glob("*.json"))
            if not candidates:
                raise FileNotFoundError(f"No OCR JSON output was produced for tile: {tile_path.name}")
            json_path = candidates[0]

        return json.loads(json_path.read_text(encoding="utf-8"))


def merge_ppocr_tile_results(
    tile_results: list[dict[str, Any]],
    image_path: Path,
    grid_rows: int,
    grid_cols: int,
    overlap_px: int,
    tile_strategy: str,
) -> dict[str, Any]:
    """Merge PP-OCR tile outputs back into one page-level payload."""
    first = tile_results[0]["json"] if tile_results else {}
    entries = collect_ocr_entries(tile_results)
    # Attempt to rejoin tokens that were split at tile seams.
    # Compute vertical seam x positions from the fixed grid so DocLayout subtiles
    # still know where the source image was split.
    seam_positions = grid_vertical_seams(image_path, grid_cols)
    entries = merge_adjacent_word_fragments(entries, seam_x=seam_positions)
    entries = dedupe_overlapping_entries(entries)

    merged = {
        "input_path": str(image_path),
        "page_index": first.get("page_index", 0),
        "model_settings": first.get("model_settings", {}),
        "text_det_params": first.get("text_det_params", {}),
        "text_type": first.get("text_type", "general"),
        "text_rec_score_thresh": first.get("text_rec_score_thresh", 0.0),
        "return_word_box": first.get("return_word_box", False),
        "rec_texts": [entry["text"] for entry in entries],
        "rec_scores": [entry["score"] for entry in entries],
        "rec_polys": [entry["rec_poly"] for entry in entries],
        "rec_boxes": [geometry_to_bbox(entry["rec_poly"]) for entry in entries],
        "dt_polys": [entry["dt_poly"] for entry in entries],
        "tile_processing": build_tile_metadata(
            tile_results,
            image_path,
            grid_rows,
            grid_cols,
            overlap_px,
            tile_strategy,
        ),
    }

    if any("angle" in entry for entry in entries):
        merged["textline_orientation_angles"] = [
            entry.get("angle", -1) for entry in entries
        ]

    if any("text_word_boxes" in entry for entry in entries):
        merged["text_word_boxes"] = [
            entry.get("text_word_boxes", []) for entry in entries
        ]
        merged["text_word"] = [entry.get("text_word", []) for entry in entries]

    normalize_ocr_word_boxes_in_payload(merged)
    return merged


def ocr_line_from_entry(entry: dict[str, Any], line_index: int) -> dict[str, Any]:
    """Build a normalized OCR line record from one merged OCR entry."""
    text = entry.get("text", "")
    score = entry.get("score")
    box = entry.get("rec_poly") or entry.get("dt_poly")
    return {
        "box": box,
        "score": score,
        "index": line_index,
        "strings": [
            {
                "text": text,
                "box": box,
                "score": score,
                "line_index": line_index,
            }
        ],
    }


def union_line_boxes(boxes: list[Any]) -> list[float]:
    """Return the bounding box that encloses several line boxes."""
    rects = [geometry_to_bbox(box) for box in boxes]
    return [
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    ]


def ocr_block_sort_key(block: dict[str, Any]) -> tuple[Any, ...]:
    """Return a reading-order sort key for OCR blocks."""
    order = block.get("order")
    row = block.get("row")
    col = block.get("col")
    y0, x0, y1, x1 = geometry_sort_key(block.get("box"))
    return (
        order if isinstance(order, Number) else 10_000,
        row if isinstance(row, Number) else 10_000,
        col if isinstance(col, Number) else 10_000,
        y0,
        x0,
        y1,
        x1,
    )


def merge_ppstructure_tile_results(
    tile_results: list[dict[str, Any]],
    image_path: Path,
    grid_rows: int,
    grid_cols: int,
    overlap_px: int,
    tile_strategy: str,
) -> dict[str, Any]:
    """Merge PP-Structure tile outputs back into one page-level payload."""
    first = tile_results[0]["json"] if tile_results else {}

    merged = {
        "input_path": str(image_path),
        "page_index": first.get("page_index", 0),
        "model_settings": first.get("model_settings", {}),
        "overall_ocr_res": merge_nested_overall_ocr(tile_results, image_path, grid_cols),
        "parsing_res_list": merge_parsing_results(tile_results),
        "tile_processing": build_tile_metadata(
            tile_results,
            image_path,
            grid_rows,
            grid_cols,
            overlap_px,
            tile_strategy,
        ),
    }

    layout_det_res = merge_layout_detection_results(tile_results, image_path)
    if layout_det_res is not None:
        merged["layout_det_res"] = layout_det_res

    for key in EXTRA_STRUCTURE_KEYS:
        merged_items = []

        for tile_result in tile_results:
            tile = tile_result["tile"]
            raw_items = tile_result["json"].get(key) or []

            for item in raw_items:
                merged_items.append(
                    offset_nested_geometry(deepcopy(item), tile.left, tile.top)
                )

        if merged_items:
            merged[key] = sorted(
                merged_items,
                key=lambda item: geometry_sort_key(find_sort_geometry(item)),
            )

    return merged


def merge_nested_overall_ocr(
    tile_results: list[dict[str, Any]],
    image_path: Path,
    grid_cols: int,
) -> dict[str, Any]:
    """Merge nested `overall_ocr_res` payloads after tiled OCR."""
    first = {}
    entries = []
    max_row, max_col = tile_grid_max_indices(tile_results)

    for tile_result in tile_results:
        tile = tile_result["tile"]
        overall = tile_result["json"].get("overall_ocr_res") or {}
        if not first and overall:
            first = overall
        entries.extend(
            collect_owned_ocr_entries_for_tile(
                overall,
                tile,
                max_row,
                max_col,
            )
        )

    # Merge possible fragments created by tiling, using fixed-grid seam positions.
    seam_positions = grid_vertical_seams(image_path, grid_cols)
    entries = merge_adjacent_word_fragments(entries, seam_x=seam_positions)
    entries = dedupe_overlapping_entries(entries)

    merged = {
        "input_path": str(image_path),
        "page_index": first.get("page_index", 0),
        "model_settings": first.get("model_settings", {}),
        "text_det_params": first.get("text_det_params", {}),
        "text_type": first.get("text_type", "general"),
        "text_rec_score_thresh": first.get("text_rec_score_thresh", 0.0),
        "return_word_box": first.get("return_word_box", False),
        "rec_texts": [entry["text"] for entry in entries],
        "rec_scores": [entry["score"] for entry in entries],
        "rec_polys": [entry["rec_poly"] for entry in entries],
        "rec_boxes": [geometry_to_bbox(entry["rec_poly"]) for entry in entries],
        "dt_polys": [entry["dt_poly"] for entry in entries],
    }

    if any("angle" in entry for entry in entries):
        merged["textline_orientation_angles"] = [
            entry.get("angle", -1) for entry in entries
        ]

    if any("text_word_boxes" in entry for entry in entries):
        merged["text_word_boxes"] = [
            entry.get("text_word_boxes", []) for entry in entries
        ]
        merged["text_word"] = [entry.get("text_word", []) for entry in entries]

    normalize_ocr_word_boxes_in_payload(merged)
    return merged


def grid_vertical_seams(image_path: Path, grid_cols: int) -> list[int]:
    """Return the x positions of vertical seams in a fixed grid tiling layout."""
    if grid_cols <= 1:
        return []

    try:
        with Image.open(image_path) as image:
            width = image.width
    except Exception:
        return []

    return [(width * index) // grid_cols for index in range(1, grid_cols)]


def merge_parsing_results(tile_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge tile-level parsing results into one page-level list."""
    merged_blocks = []

    for tile_result in tile_results:
        tile = tile_result["tile"]
        parsing_res_list = tile_result["json"].get("parsing_res_list") or []

        for block in parsing_res_list:
            merged_blocks.append(
                offset_nested_geometry(deepcopy(block), tile.left, tile.top)
            )

    merged_blocks.sort(key=lambda block: geometry_sort_key(find_sort_geometry(block)))

    for index, block in enumerate(merged_blocks):
        if "block_id" in block:
            block["block_id"] = index
        if "block_order" in block:
            block["block_order"] = index

    return merged_blocks


def merge_layout_detection_results(
    tile_results: list[dict[str, Any]],
    image_path: Path,
) -> dict[str, Any] | None:
    """Merge tile-level layout detections into page-level coordinates."""
    layout_template = None
    merged_boxes = []

    for tile_result in tile_results:
        tile = tile_result["tile"]
        layout_det_res = tile_result["json"].get("layout_det_res")

        if not isinstance(layout_det_res, dict):
            continue

        if layout_template is None:
            layout_template = deepcopy(layout_det_res)
            layout_template["boxes"] = []

        for box in layout_det_res.get("boxes") or []:
            merged_boxes.append(
                offset_nested_geometry(deepcopy(box), tile.left, tile.top)
            )

    if layout_template is None:
        return None

    merged_boxes.sort(key=lambda box: geometry_sort_key(find_sort_geometry(box)))
    layout_template["boxes"] = merged_boxes

    if "input_path" in layout_template:
        layout_template["input_path"] = str(image_path)

    return layout_template


def collect_ocr_entries(tile_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect normalized OCR entries from merged tile payloads."""
    entries = []
    max_row, max_col = tile_grid_max_indices(tile_results)

    for tile_result in tile_results:
        tile = tile_result["tile"]
        entries.extend(
            collect_owned_ocr_entries_for_tile(
                tile_result["json"],
                tile,
                max_row,
                max_col,
            )
        )

    entries.sort(key=lambda entry: geometry_sort_key(entry["rec_poly"]))
    return entries


def tile_grid_max_indices(tile_results: list[dict[str, Any]]) -> tuple[int, int]:
    """Return the maximum row and column indices seen in tile results."""
    max_row = 0
    max_col = 0

    for tile_result in tile_results:
        tile = tile_result["tile"]
        max_row = max(max_row, tile.row)
        max_col = max(max_col, tile.col)

    return max_row, max_col


def collect_owned_ocr_entries_for_tile(
    payload: dict[str, Any],
    tile: TileSpec,
    max_row: int,
    max_col: int,
) -> list[dict[str, Any]]:
    """Collect OCR entries that belong to one tile nominal region."""
    entries = collect_ocr_entries_for_payload(payload, tile.left, tile.top)
    owned_entries = []

    for entry in entries:
        annotate_tile_ownership(entry, tile)
        if entry_center_is_in_nominal_tile(entry, tile, max_row, max_col):
            owned_entries.append(entry)

    return owned_entries


def annotate_tile_ownership(entry: dict[str, Any], tile: TileSpec) -> None:
    """Annotate an OCR entry with its owning tile metadata."""
    entry["_tile_row"] = tile.row
    entry["_tile_col"] = tile.col
    entry["_tile_label"] = tile.label
    entry["_tile_score"] = tile.score
    entry["_tile_order"] = tile.order
    entry["_tile_bbox"] = [tile.left, tile.top, tile.right, tile.bottom]
    entry["_tile_nominal_bbox"] = [
        tile.nominal_left,
        tile.nominal_top,
        tile.nominal_right,
        tile.nominal_bottom,
    ]


def entry_center_is_in_nominal_tile(
    entry: dict[str, Any],
    tile: TileSpec,
    max_row: int,
    max_col: int,
) -> bool:
    """Return whether an OCR entry center lies inside a tile nominal region."""
    bbox = geometry_to_bbox(entry.get("rec_poly"))
    if bbox_area(bbox) <= 0:
        return False

    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    right_inclusive = tile.col == max_col
    bottom_inclusive = tile.row == max_row

    in_x = (
        tile.nominal_left <= center_x <= tile.nominal_right
        if right_inclusive
        else tile.nominal_left <= center_x < tile.nominal_right
    )
    in_y = (
        tile.nominal_top <= center_y <= tile.nominal_bottom
        if bottom_inclusive
        else tile.nominal_top <= center_y < tile.nominal_bottom
    )

    return in_x and in_y


def dedupe_overlapping_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate detections caused by overlap between neighboring tiles."""
    if len(entries) < 2:
        return entries

    ranked_entries = sorted(
        (deepcopy(entry) for entry in entries),
        key=entry_dedupe_preference,
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    kept_by_vertical_bucket: dict[int, list[dict[str, Any]]] = {}

    for entry in ranked_entries:
        candidate_entries = nearby_dedupe_candidates(entry, kept_by_vertical_bucket)
        if any(entries_are_duplicate(entry, existing) for existing in candidate_entries):
            continue
        kept.append(entry)
        bucket = vertical_dedupe_bucket(entry)
        kept_by_vertical_bucket.setdefault(bucket, []).append(entry)

    kept.sort(key=lambda entry: geometry_sort_key(entry["rec_poly"]))
    return kept


def nearby_dedupe_candidates(
    entry: dict[str, Any],
    buckets: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collect nearby OCR entries that may be duplicate fragments."""
    bucket = vertical_dedupe_bucket(entry)
    candidates: list[dict[str, Any]] = []

    for nearby_bucket in range(bucket - 1, bucket + 2):
        candidates.extend(buckets.get(nearby_bucket, []))

    return candidates


def vertical_dedupe_bucket(entry: dict[str, Any]) -> int:
    """Return a coarse vertical bucket used for OCR deduping."""
    bbox = geometry_to_bbox(entry.get("rec_poly"))
    center_y = (bbox[1] + bbox[3]) / 2.0
    return int(center_y // 96)


def entry_dedupe_preference(entry: dict[str, Any]) -> tuple[int, float, float]:
    """Return the sort preference used when keeping one duplicate OCR entry."""
    text_len = len(normalize_for_dedupe(entry.get("text", "")))
    area = bbox_area(geometry_to_bbox(entry.get("rec_poly")))
    score = float(entry.get("score") or 0.0)
    return (text_len, area, score)


def entries_are_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two OCR entries should be treated as duplicates."""
    left_bbox = geometry_to_bbox(left.get("rec_poly"))
    right_bbox = geometry_to_bbox(right.get("rec_poly"))

    if bbox_area(left_bbox) <= 0 or bbox_area(right_bbox) <= 0:
        return False

    overlap_ratio = bbox_overlap_ratio(left_bbox, right_bbox)
    if overlap_ratio < 0.65 and bbox_iou(left_bbox, right_bbox) < 0.45:
        return False

    if text_similarity(left.get("text", ""), right.get("text", "")) >= 0.72:
        return True

    return overlap_ratio >= 0.9 and text_has_contained_duplicate_shape(
        left.get("text", ""),
        right.get("text", ""),
    )


def text_has_contained_duplicate_shape(left: Any, right: Any) -> bool:
    """Catch truncated same-line duplicates from layout/recovery overlap."""
    left_norm = normalize_for_dedupe(left)
    right_norm = normalize_for_dedupe(right)

    if len(left_norm) < 4 or len(right_norm) < 4 or left_norm == right_norm:
        return False

    longer, shorter = (
        (left_norm, right_norm)
        if len(left_norm) >= len(right_norm)
        else (right_norm, left_norm)
    )

    if len(shorter) > len(longer) * 0.85:
        return False

    return (
        truncated_prefix_match(longer, shorter)
        or truncated_suffix_match(longer, shorter)
        or clipped_text_overlap_match(longer, shorter)
    )


def truncated_prefix_match(longer: str, shorter: str) -> bool:
    """Return whether one normalized string is a truncated prefix match."""
    if longer.startswith(shorter):
        return True

    for trim in (1, 2):
        if len(shorter) > trim + 3 and longer.startswith(shorter[:-trim]):
            return True

    return False


def truncated_suffix_match(longer: str, shorter: str) -> bool:
    """Return whether one normalized string is a truncated suffix match."""
    if longer.endswith(shorter):
        return True

    for trim in (1, 2):
        if len(shorter) > trim + 3 and longer.endswith(shorter[trim:]):
            return True

    return False


def clipped_text_overlap_match(longer: str, shorter: str) -> bool:
    """Return whether clipped OCR strings overlap strongly enough to merge."""
    match = SequenceMatcher(None, longer, shorter).find_longest_match(
        0,
        len(longer),
        0,
        len(shorter),
    )
    if match.size < max(4, int(len(shorter) * 0.72)):
        return False

    short_edge = match.b <= 2 or match.b + match.size >= len(shorter) - 2
    long_edge = match.a <= 2 or match.a + match.size >= len(longer) - 2
    return short_edge and long_edge


def text_similarity(left: Any, right: Any) -> float:
    """Return a fuzzy similarity score between two OCR text fragments."""
    left_norm = normalize_for_dedupe(left)
    right_norm = normalize_for_dedupe(right)

    if not left_norm or not right_norm:
        return 0.0

    if left_norm == right_norm:
        return 1.0

    return SequenceMatcher(None, left_norm, right_norm).ratio()


def normalize_for_dedupe(value: Any) -> str:
    """Normalize OCR text for dedupe and seam-merge comparisons."""
    return "".join(char.lower() for char in str(value) if char.isalnum())


def bbox_area(bbox: list[float]) -> float:
    """Return the area of a bounding box."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_intersection(left: list[float], right: list[float]) -> float:
    """Return the intersection area of two bounding boxes."""
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def bbox_iou(left: list[float], right: list[float]) -> float:
    """Return the intersection-over-union score for two bounding boxes."""
    intersection = bbox_intersection(left, right)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_overlap_ratio(left: list[float], right: list[float]) -> float:
    """Return the directional overlap ratio between two bounding boxes."""
    intersection = bbox_intersection(left, right)
    smaller_area = min(bbox_area(left), bbox_area(right))
    return intersection / smaller_area if smaller_area > 0 else 0.0


def merge_adjacent_word_fragments(entries: list[dict[str, Any]], seam_x: list[float] | None = None) -> list[dict[str, Any]]:
    """Try to rejoin word fragments that were split at tile boundaries.

    Heuristic:
    - Group entries into approximate text lines by vertical center.
    - Within each line, sort tokens left-to-right and merge adjacent tokens
      only when one side touches a vertical tile seam.
    - Preserve original geometry when no merge occurred; merged entries are
      emitted as four-point polygons so PP-OCR consumers keep working.

    Returns a new entries list with merged `text` and unified `rec_poly` polygons.
    """
    if not entries:
        return entries

    structured = []
    for e in entries:
        bbox = geometry_to_bbox(e.get("rec_poly", e.get("dt_poly")))
        x0, y0, x1, y1 = bbox
        center_y = (y0 + y1) / 2.0
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        structured.append(
            {
                "entry": e,
                "bbox": bbox,
                "center_y": center_y,
                "width": width,
                "height": height,
            }
        )

    heights = sorted(s["height"] for s in structured)
    median_h = heights[len(heights) // 2] if heights else 12.0
    line_thr = max(5.0, median_h * 0.35)

    structured.sort(key=lambda s: (s["center_y"], s["bbox"][0]))
    lines = []
    cur = [structured[0]]
    cur_y = structured[0]["center_y"]

    for s in structured[1:]:
        if abs(s["center_y"] - cur_y) <= line_thr:
            cur.append(s)
            cur_y = sum(x["center_y"] for x in cur) / len(cur)
        else:
            lines.append(cur)
            cur = [s]
            cur_y = s["center_y"]

    if cur:
        lines.append(cur)

    merged_entries: list[dict[str, Any]] = []

    for line in lines:
        line.sort(key=lambda s: s["bbox"][0])

        char_widths = []
        for s in line:
            text = str(s["entry"].get("text", ""))
            nch = max(1, len(text.replace(" ", "")))
            char_widths.append(s["width"] / nch)
        char_w = sum(char_widths) / len(char_widths) if char_widths else 10.0

        cur_s = line[0]
        cur_original = deepcopy(cur_s["entry"])
        cur_changed = False
        cur_text = str(cur_s["entry"].get("text", ""))
        cur_bbox = list(cur_s["bbox"])
        cur_score = cur_s["entry"].get("score")
        cur_text_words = cur_s["entry"].get("text_word") if "text_word" in cur_s["entry"] else None
        cur_text_word_boxes = cur_s["entry"].get("text_word_boxes") if "text_word_boxes" in cur_s["entry"] else None
        cur_angle = cur_s["entry"].get("angle") if "angle" in cur_s["entry"] else None

        for s in line[1:]:
            nxt_text = str(s["entry"].get("text", ""))
            nxt_bbox = list(s["bbox"])
            gap = nxt_bbox[0] - cur_bbox[2]
            touches_seam = touches_vertical_seam(cur_bbox, nxt_bbox, seam_x, char_w)
            overlap_text = overlapping_fragment_text(cur_text, nxt_text, cur_bbox, nxt_bbox)

            if overlap_text is not None:
                if overlap_text != cur_text:
                    cur_text = overlap_text
                    cur_bbox[2] = max(cur_bbox[2], nxt_bbox[2])
                    cur_bbox[1] = min(cur_bbox[1], nxt_bbox[1])
                    cur_bbox[3] = max(cur_bbox[3], nxt_bbox[3])
                    cur_changed = True
                    cur_text_words = None
                    cur_text_word_boxes = None

                    if cur_score is None:
                        cur_score = s["entry"].get("score")
                    elif s["entry"].get("score") is not None:
                        cur_score = (cur_score + s["entry"].get("score")) / 2.0

                    if cur_angle is None and "angle" in s["entry"]:
                        cur_angle = s["entry"].get("angle")

            elif should_merge_across_seam(gap, char_w, cur_text, nxt_text, touches_seam, seam_x):
                separator = seam_separator(cur_text, nxt_text, gap, char_w)
                cur_text = cur_text + separator + nxt_text
                cur_bbox[2] = max(cur_bbox[2], nxt_bbox[2])
                cur_bbox[1] = min(cur_bbox[1], nxt_bbox[1])
                cur_bbox[3] = max(cur_bbox[3], nxt_bbox[3])
                cur_changed = True

                if cur_score is None:
                    cur_score = s["entry"].get("score")
                elif s["entry"].get("score") is not None:
                    cur_score = (cur_score + s["entry"].get("score")) / 2.0

                if cur_text_words is None and "text_word" in s["entry"]:
                    cur_text_words = s["entry"].get("text_word")
                elif cur_text_words is not None and "text_word" in s["entry"]:
                    cur_text_words = cur_text_words + s["entry"].get("text_word")

                if cur_text_word_boxes is None and "text_word_boxes" in s["entry"]:
                    cur_text_word_boxes = s["entry"].get("text_word_boxes")
                elif cur_text_word_boxes is not None and "text_word_boxes" in s["entry"]:
                    cur_text_word_boxes = cur_text_word_boxes + s["entry"].get("text_word_boxes")

                if cur_angle is None and "angle" in s["entry"]:
                    cur_angle = s["entry"].get("angle")

            else:
                merged_entries.append(
                    finalize_merge_entry(
                        cur_original,
                        cur_changed,
                        cur_text,
                        cur_score,
                        cur_bbox,
                        cur_text_words,
                        cur_text_word_boxes,
                        cur_angle,
                    )
                )

                cur_original = deepcopy(s["entry"])
                cur_changed = False
                cur_text = nxt_text
                cur_bbox = nxt_bbox
                cur_score = s["entry"].get("score")
                cur_text_words = s["entry"].get("text_word") if "text_word" in s["entry"] else None
                cur_text_word_boxes = s["entry"].get("text_word_boxes") if "text_word_boxes" in s["entry"] else None
                cur_angle = s["entry"].get("angle") if "angle" in s["entry"] else None

        merged_entries.append(
            finalize_merge_entry(
                cur_original,
                cur_changed,
                cur_text,
                cur_score,
                cur_bbox,
                cur_text_words,
                cur_text_word_boxes,
                cur_angle,
            )
        )

    merged_entries.sort(key=lambda entry: geometry_sort_key(entry["rec_poly"]))
    return merged_entries


def overlapping_fragment_text(
    left_text: str,
    right_text: str,
    left_bbox: list[float],
    right_bbox: list[float],
) -> str | None:
    """Return the overlapping text fragment shared by adjacent OCR entries."""
    if not left_text or not right_text:
        return None

    overlap_ratio = bbox_overlap_ratio(left_bbox, right_bbox)
    if overlap_ratio < 0.72:
        return None

    left_norm = normalize_for_dedupe(left_text)
    right_norm = normalize_for_dedupe(right_text)
    if len(left_norm) < 4 or len(right_norm) < 4:
        return None

    overlap_len = suffix_prefix_overlap_length(left_norm, right_norm)
    if overlap_len <= 0:
        return None

    required = max(10, int(min(len(left_norm), len(right_norm)) * 0.55))
    if overlap_len < required:
        return None

    suffix_start = raw_index_after_normalized_chars(right_text, overlap_len)
    suffix = right_text[suffix_start:]

    if not suffix.strip():
        return left_text

    return left_text + suffix


def suffix_prefix_overlap_length(left_norm: str, right_norm: str) -> int:
    """Return the normalized suffix/prefix overlap length between two strings."""
    max_len = min(len(left_norm), len(right_norm))

    for size in range(max_len, 3, -1):
        if left_norm[-size:] == right_norm[:size]:
            return size

    return 0


def raw_index_after_normalized_chars(text: str, normalized_count: int) -> int:
    """Map a normalized-character index back into the raw source string."""
    seen = 0

    for index, char in enumerate(text):
        if char.isalnum():
            seen += 1
            if seen >= normalized_count:
                return index + 1

    return len(text)


def touches_vertical_seam(
    left_bbox: list[float],
    right_bbox: list[float],
    seam_x: list[float] | None,
    char_w: float,
) -> bool:
    """Return whether a bounding box touches a vertical tile seam."""
    if seam_x is None:
        return True

    if not seam_x:
        return False

    eps = max(2.0, char_w * 0.35)
    return any(
        abs(left_bbox[2] - sx) <= eps
        or abs(right_bbox[0] - sx) <= eps
        for sx in seam_x
    )


def should_merge_across_seam(
    gap: float,
    char_w: float,
    left_text: str,
    right_text: str,
    touches_seam: bool,
    seam_x: list[float] | None,
) -> bool:
    """Return whether adjacent OCR fragments should merge across a seam."""
    if not left_text or not right_text or left_text.endswith(" "):
        return False

    if gap < 0:
        return False

    if seam_x is not None and not touches_seam:
        return False

    return gap <= max(3.0, char_w * 0.85)


def seam_separator(left_text: str, right_text: str, gap: float, char_w: float) -> str:
    """Choose the separator to use when merging text across a seam."""
    if left_text.endswith((" ", "-", "/", "(", "“", '"')) or right_text.startswith((" ", ".", ",", ":", ";", ")", "”", '"')):
        return ""

    return "" if gap <= max(1.0, char_w * 0.25) else " "


def finalize_merge_entry(
    original_entry: dict[str, Any],
    changed: bool,
    text: str,
    score: Any,
    bbox: list[float],
    text_words: Any,
    text_word_boxes: Any,
    angle: Any,
) -> dict[str, Any]:
    """Finalize one merged OCR entry after seam-aware text combination."""
    if not changed:
        return deepcopy(original_entry)

    poly = bbox_to_polygon(bbox)
    new_entry = {
        "text": text,
        "score": score,
        "rec_poly": poly,
        "dt_poly": poly,
    }

    if angle is not None:
        new_entry["angle"] = angle
    if text_words is not None:
        new_entry["text_word"] = text_words
    if text_word_boxes is not None:
        new_entry["text_word_boxes"] = text_word_boxes

    return new_entry


def collect_ocr_entries_for_payload(
    payload: dict[str, Any],
    offset_x: int,
    offset_y: int,
) -> list[dict[str, Any]]:
    """Collect normalized OCR entries from one payload for sorting or merging."""
    rec_texts = payload.get("rec_texts") or []
    rec_scores = payload.get("rec_scores") or []
    rec_polys = payload.get("rec_polys") or []
    dt_polys = payload.get("dt_polys") or rec_polys
    angles = payload.get("textline_orientation_angles") or []
    text_word_boxes = payload.get("text_word_boxes") or []
    text_word = payload.get("text_word") or []

    entries = []

    for index, text in enumerate(rec_texts):
        if index >= len(rec_polys):
            break

        rec_poly = offset_geometry_value(rec_polys[index], offset_x, offset_y)
        dt_poly_source = dt_polys[index] if index < len(dt_polys) else rec_polys[index]

        entry = {
            "text": text,
            "score": rec_scores[index] if index < len(rec_scores) else None,
            "rec_poly": rec_poly,
            "dt_poly": offset_geometry_value(dt_poly_source, offset_x, offset_y),
        }

        if index < len(angles):
            entry["angle"] = angles[index]

        if index < len(text_word_boxes):
            entry["text_word_boxes"] = offset_geometry_value(
                text_word_boxes[index],
                offset_x,
                offset_y,
            )

        if index < len(text_word):
            entry["text_word"] = text_word[index]

        entries.append(entry)

    return entries


def build_tile_metadata(
    tile_results: list[dict[str, Any]],
    image_path: Path,
    grid_rows: int,
    grid_cols: int,
    overlap_px: int,
    tile_strategy: str,
) -> dict[str, Any]:
    # Backwards compatible: if grid_rows/grid_cols are encoded in the
    # tile_results metadata, prefer them; otherwise infer square grid size.
    # The signature previously accepted a single grid_size; callers now
    # pass rows and cols as separate values in `image_path` parameter
    # positions. To remain compatible, callers should pass two ints after
    # image_path; however, we also support older single-int usage.
    # Determine rows/cols from the tiles list if possible.
    """Build the tile-processing metadata block stored with merged OCR output."""
    rows = None
    cols = None
    for t in tile_results:
        try:
            rows = max(rows or 0, t["tile"].row)
            cols = max(cols or 0, t["tile"].col)
        except Exception:
            pass

    # convert to counts (max index + 1)
    if rows is not None:
        rows = rows + 1
    if cols is not None:
        cols = cols + 1

    # fall back to TILE_GRID_SIZE
    if rows is None:
        rows = TILE_GRID_SIZE
    if cols is None:
        cols = TILE_GRID_SIZE

    return {
        "enabled": True,
        "strategy": tile_strategy,
        "grid_rows": rows,
        "grid_cols": cols,
        "overlap_px": overlap_px,
        "source_image": image_path.name,
        "tiles": [
            {
                "row": tile_result["tile"].row,
                "col": tile_result["tile"].col,
                "bbox": [
                    tile_result["tile"].left,
                    tile_result["tile"].top,
                    tile_result["tile"].right,
                    tile_result["tile"].bottom,
                ],
                "nominal_bbox": [
                    tile_result["tile"].nominal_left,
                    tile_result["tile"].nominal_top,
                    tile_result["tile"].nominal_right,
                    tile_result["tile"].nominal_bottom,
                ],
                "label": tile_result["tile"].label,
                "score": tile_result["tile"].score,
                "order": tile_result["tile"].order,
            }
            for tile_result in tile_results
        ],
    }


def offset_nested_geometry(value: Any, offset_x: int, offset_y: int, key: str | None = None) -> Any:
    """Apply a coordinate offset to nested geometry values."""
    if isinstance(value, dict):
        updated = {}

        for child_key, child_value in value.items():
            if should_offset_key(child_key):
                updated[child_key] = offset_geometry_value(
                    child_value,
                    offset_x,
                    offset_y,
                )
            else:
                updated[child_key] = offset_nested_geometry(
                    child_value,
                    offset_x,
                    offset_y,
                    key=child_key,
                )

        return updated

    if isinstance(value, list):
        if key and should_offset_key(key):
            return offset_geometry_value(value, offset_x, offset_y)

        return [
            offset_nested_geometry(item, offset_x, offset_y, key=key)
            for item in value
        ]

    return value


def should_offset_key(key: str | None) -> bool:
    """Return whether a nested key should receive geometry offsets."""
    if not key:
        return False
    if key in GEOMETRY_KEYS:
        return True
    return key.endswith(GEOMETRY_SUFFIXES)


def offset_geometry_value(value: Any, offset_x: int, offset_y: int) -> Any:
    """Offset one geometry-shaped value by the supplied x and y amounts."""
    if looks_like_bbox(value):
        return [
            value[0] + offset_x,
            value[1] + offset_y,
            value[2] + offset_x,
            value[3] + offset_y,
        ]

    if looks_like_point_sequence(value):
        return [[point[0] + offset_x, point[1] + offset_y] for point in value]

    if isinstance(value, dict):
        return offset_nested_geometry(value, offset_x, offset_y)

    if isinstance(value, list):
        return [offset_geometry_value(item, offset_x, offset_y) for item in value]

    return value


def looks_like_bbox(value: Any) -> bool:
    """Return whether a value looks like a four-number bounding box."""
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, Number) for item in value)
    )


def looks_like_point_sequence(value: Any) -> bool:
    """Return whether a value looks like a polygon point sequence."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(item, Number) for item in point)
            for point in value
        )
    )


def geometry_to_bbox(geometry: Any) -> list[float]:
    """Convert supported geometry shapes into a plain bounding box."""
    bounds = geometry_bounds(geometry)
    if bounds is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = bounds
    return [x0, y0, x1, y1]


def bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    """Convert a bounding box into a four-point polygon."""
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def geometry_bounds(geometry: Any) -> tuple[float, float, float, float] | None:
    """Return the geometric bounds for a supported geometry value."""
    if looks_like_bbox(geometry):
        return float(geometry[0]), float(geometry[1]), float(geometry[2]), float(geometry[3])

    if looks_like_point_sequence(geometry):
        xs = [float(point[0]) for point in geometry]
        ys = [float(point[1]) for point in geometry]
        return min(xs), min(ys), max(xs), max(ys)

    if isinstance(geometry, list):
        nested_bounds = [
            bounds for item in geometry if (bounds := geometry_bounds(item)) is not None
        ]
        if not nested_bounds:
            return None

        return (
            min(bounds[0] for bounds in nested_bounds),
            min(bounds[1] for bounds in nested_bounds),
            max(bounds[2] for bounds in nested_bounds),
            max(bounds[3] for bounds in nested_bounds),
        )

    return None


def geometry_sort_key(geometry: Any) -> tuple[float, float, float, float]:
    """Return a stable reading-order sort key for geometry."""
    bounds = geometry_bounds(geometry)
    if bounds is None:
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    x0, y0, x1, y1 = bounds
    return (y0, x0, y1, x1)


def find_sort_geometry(value: Any) -> Any:
    """Find the most useful geometry field for sorting a payload item."""
    if isinstance(value, dict):
        for key in (
            "block_bbox",
            "coordinate",
            "bbox",
            "box",
            "rec_boxes",
            "rec_polys",
            "polygon_points",
        ):
            if key in value:
                return value[key]
    return None
