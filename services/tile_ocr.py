from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

from PIL import Image

from engine import engine

SUPPORTED_TILING_METHODS = {"PP-OCRv5", "PP-StructureV3"}
TILE_GRID_SIZE = 3
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
    path: Path


def get_effective_method(method: str | None, select: str | None = None) -> str | None:
    if select in SUPPORTED_TILING_METHODS:
        return select
    if method in SUPPORTED_TILING_METHODS:
        return method
    return None


def supports_tiled_ocr(method: str | None, select: str | None = None) -> bool:
    return get_effective_method(method, select) is not None


def run_tiled_local_ocr(
    image_path: Path,
    opt: dict,
    select: str | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> dict[str, Any]:
    effective_method = get_effective_method(opt.get("method"), select)
    if effective_method is None:
        raise ValueError("3x3 tiling is only supported for PP-OCRv5 and PP-StructureV3.")

    tile_opt = dict(opt)
    tile_opt["tile_source_images"] = False

    # Determine grid rows/cols from opt if provided, otherwise fall back
    # to the single grid_size parameter or default TILE_GRID_SIZE.
    if grid_rows is None and grid_cols is None:
        if "tile_rows" in opt or "tile_cols" in opt:
            grid_rows = int(opt.get("tile_rows", TILE_GRID_SIZE))
            grid_cols = int(opt.get("tile_cols", TILE_GRID_SIZE))
        else:
            gs = int(opt.get("tile_grid", TILE_GRID_SIZE))
            grid_rows = grid_rows or gs
            grid_cols = grid_cols or gs

    with tempfile.TemporaryDirectory(prefix="ocr_tiles_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        tiles = build_image_tiles(image_path, temp_dir, grid_rows=grid_rows, grid_cols=grid_cols)
        tile_results = []

        for tile in tiles:
            tile_json = infer_tile_json(tile.path, tile_opt)
            tile_results.append({"tile": tile, "json": tile_json})

    if effective_method == "PP-OCRv5":
        return merge_ppocr_tile_results(tile_results, image_path, grid_rows, grid_cols)

    return merge_ppstructure_tile_results(tile_results, image_path, grid_rows, grid_cols)


def build_image_tiles(
    image_path: Path,
    temp_dir: Path,
    grid_rows: int = TILE_GRID_SIZE,
    grid_cols: int = TILE_GRID_SIZE,
) -> list[TileSpec]:
    tiles = []

    with Image.open(image_path) as image:
        x_edges = [(image.width * index) // grid_cols for index in range(grid_cols + 1)]
        y_edges = [(image.height * index) // grid_rows for index in range(grid_rows + 1)]

        for row in range(grid_rows):
            for col in range(grid_cols):
                left = x_edges[col]
                right = x_edges[col + 1]
                top = y_edges[row]
                bottom = y_edges[row + 1]
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
                        path=tile_path,
                    )
                )

    return tiles


def infer_tile_json(tile_path: Path, opt: dict) -> dict[str, Any]:
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
) -> dict[str, Any]:
    first = tile_results[0]["json"] if tile_results else {}
    entries = collect_ocr_entries(tile_results)
    # Attempt to rejoin tokens that were split at tile seams.
    # Compute vertical seam x positions from tile bboxes (inner vertical borders).
    seam_positions = sorted({
        tr["tile"].right
        for tr in tile_results
        if hasattr(tr["tile"], "col") and tr["tile"].col < (grid_cols - 1)
    })
    entries = merge_adjacent_word_fragments(entries, seam_x=seam_positions)

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
        "tile_processing": build_tile_metadata(tile_results, image_path, grid_rows, grid_cols),
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

    return merged


def merge_ppstructure_tile_results(
    tile_results: list[dict[str, Any]],
    image_path: Path,
    grid_rows: int,
    grid_cols: int,
) -> dict[str, Any]:
    first = tile_results[0]["json"] if tile_results else {}

    merged = {
        "input_path": str(image_path),
        "page_index": first.get("page_index", 0),
        "model_settings": first.get("model_settings", {}),
        "overall_ocr_res": merge_nested_overall_ocr(tile_results, image_path, grid_cols),
        "parsing_res_list": merge_parsing_results(tile_results),
        "tile_processing": build_tile_metadata(tile_results, image_path, grid_rows, grid_cols),
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
    first = {}
    entries = []

    for tile_result in tile_results:
        tile = tile_result["tile"]
        overall = tile_result["json"].get("overall_ocr_res") or {}
        if not first and overall:
            first = overall
        entries.extend(collect_ocr_entries_for_payload(overall, tile.left, tile.top))

    entries.sort(key=lambda entry: geometry_sort_key(entry["rec_poly"]))
    # Merge possible fragments created by tiling, using seam positions
    seam_positions = sorted({
        tr["tile"].right
        for tr in tile_results
        if hasattr(tr["tile"], "col") and tr["tile"].col < (grid_cols - 1)
    })
    entries = merge_adjacent_word_fragments(entries, seam_x=seam_positions)

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

    return merged


def merge_parsing_results(tile_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    entries = []

    for tile_result in tile_results:
        tile = tile_result["tile"]
        entries.extend(
            collect_ocr_entries_for_payload(
                tile_result["json"],
                tile.left,
                tile.top,
            )
        )

    entries.sort(key=lambda entry: geometry_sort_key(entry["rec_poly"]))
    return entries


def merge_adjacent_word_fragments(entries: list[dict[str, Any]], seam_x: list[float] | None = None) -> list[dict[str, Any]]:
    """Try to rejoin word fragments that were split at tile boundaries.

    Heuristic:
    - Group entries into approximate text lines by vertical center.
    - Within each line, sort tokens left-to-right and merge adjacent tokens
      when the horizontal gap is very small relative to the estimated
      character width for the line (likely a tile seam split).

    Returns a new entries list with merged `text` and unified `rec_poly` boxes.
    """
    if not entries:
        return entries

    # Build structured list with bbox and centers
    structured = []
    for e in entries:
        bbox = geometry_to_bbox(e["rec_poly"]) if "rec_poly" in e else geometry_to_bbox(e.get("dt_poly", e.get("rec_poly")))
        x0, y0, x1, y1 = bbox
        center_y = (y0 + y1) / 2.0
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        structured.append({"entry": e, "bbox": bbox, "center_y": center_y, "width": width, "height": height})

    # median height -> line threshold
    heights = sorted(s["height"] for s in structured)
    median_h = heights[len(heights) // 2] if heights else 12.0
    line_thr = max(8.0, median_h * 0.6)

    # group into lines by y center
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
        # sort tokens left-to-right
        line.sort(key=lambda s: s["bbox"][0])

        # estimate char width for the line
        char_widths = []
        for s in line:
            text = str(s["entry"].get("text", ""))
            nch = max(1, len(text.replace(" ", "")))
            char_widths.append(s["width"] / nch)
        char_w = sum(char_widths) / len(char_widths) if char_widths else 10.0

        cur_s = line[0]
        cur_text = str(cur_s["entry"].get("text", ""))
        cur_bbox = list(cur_s["bbox"])
        cur_score = cur_s["entry"].get("score")
        cur_text_words = cur_s["entry"].get("text_word") if "text_word" in cur_s["entry"] else None
        cur_text_word_boxes = cur_s["entry"].get("text_word_boxes") if "text_word_boxes" in cur_s["entry"] else None

        for s in line[1:]:
            nxt_text = str(s["entry"].get("text", ""))
            nxt_bbox = list(s["bbox"])
            gap = nxt_bbox[0] - cur_bbox[2]
            # merge only when gap is very small AND one of the tokens touches a tile seam
            touches_seam = False
            if seam_x:
                eps = max(1.0, char_w * 0.12)
                for sx in seam_x:
                    if abs(cur_bbox[2] - sx) <= eps or abs(nxt_bbox[0] - sx) <= eps:
                        touches_seam = True
                        break

            # only join when gap is very small relative to char width (tight)
            if gap <= max(1.0, char_w * 0.18) and cur_text and nxt_text and not cur_text.endswith(" ") and (seam_x is None or touches_seam):
                # merge without space
                cur_text = cur_text + nxt_text
                cur_bbox[2] = max(cur_bbox[2], nxt_bbox[2])
                cur_bbox[1] = min(cur_bbox[1], nxt_bbox[1])
                cur_bbox[3] = max(cur_bbox[3], nxt_bbox[3])
                # merge scores conservatively
                if cur_score is None:
                    cur_score = s["entry"].get("score")
                elif s["entry"].get("score") is not None:
                    cur_score = (cur_score + s["entry"].get("score")) / 2.0
                # merge text_word arrays if present
                if cur_text_words is None and "text_word" in s["entry"]:
                    cur_text_words = s["entry"].get("text_word")
                elif cur_text_words is not None and "text_word" in s["entry"]:
                    cur_text_words = cur_text_words + s["entry"].get("text_word")

                if cur_text_word_boxes is None and "text_word_boxes" in s["entry"]:
                    cur_text_word_boxes = s["entry"].get("text_word_boxes")
                elif cur_text_word_boxes is not None and "text_word_boxes" in s["entry"]:
                    cur_text_word_boxes = cur_text_word_boxes + s["entry"].get("text_word_boxes")

            else:
                # finalize current
                new_entry = {
                    "text": cur_text,
                    "score": cur_score,
                    "rec_poly": [cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3]],
                    "dt_poly": [cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3]],
                }
                if cur_text_words is not None:
                    new_entry["text_word"] = cur_text_words
                if cur_text_word_boxes is not None:
                    new_entry["text_word_boxes"] = cur_text_word_boxes

                merged_entries.append(new_entry)

                # start new current
                cur_text = nxt_text
                cur_bbox = nxt_bbox
                cur_score = s["entry"].get("score")
                cur_text_words = s["entry"].get("text_word") if "text_word" in s["entry"] else None
                cur_text_word_boxes = s["entry"].get("text_word_boxes") if "text_word_boxes" in s["entry"] else None

        # finalize last in line
        new_entry = {
            "text": cur_text,
            "score": cur_score,
            "rec_poly": [cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3]],
            "dt_poly": [cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3]],
        }
        if cur_text_words is not None:
            new_entry["text_word"] = cur_text_words
        if cur_text_word_boxes is not None:
            new_entry["text_word_boxes"] = cur_text_word_boxes

        merged_entries.append(new_entry)

    # Sort merged entries for downstream consumption
    merged_entries.sort(key=lambda entry: geometry_sort_key(entry["rec_poly"]))
    return merged_entries


def collect_ocr_entries_for_payload(
    payload: dict[str, Any],
    offset_x: int,
    offset_y: int,
) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
    # Backwards compatible: if grid_rows/grid_cols are encoded in the
    # tile_results metadata, prefer them; otherwise infer square grid size.
    # The signature previously accepted a single grid_size; callers now
    # pass rows and cols as separate values in `image_path` parameter
    # positions. To remain compatible, callers should pass two ints after
    # image_path; however, we also support older single-int usage.
    # Determine rows/cols from the tiles list if possible.
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
        "strategy": "grid",
        "grid_rows": rows,
        "grid_cols": cols,
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
            }
            for tile_result in tile_results
        ],
    }


def offset_nested_geometry(value: Any, offset_x: int, offset_y: int, key: str | None = None) -> Any:
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
    if not key:
        return False
    if key in GEOMETRY_KEYS:
        return True
    return key.endswith(GEOMETRY_SUFFIXES)


def offset_geometry_value(value: Any, offset_x: int, offset_y: int) -> Any:
    if looks_like_bbox(value):
        return [
            value[0] + offset_x,
            value[1] + offset_y,
            value[2] + offset_x,
            value[3] + offset_y,
        ]

    if looks_like_point_sequence(value):
        return [[point[0] + offset_x, point[1] + offset_y] for point in value]

    if isinstance(value, list):
        return [offset_geometry_value(item, offset_x, offset_y) for item in value]

    return value


def looks_like_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, Number) for item in value)
    )


def looks_like_point_sequence(value: Any) -> bool:
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
    bounds = geometry_bounds(geometry)
    if bounds is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = bounds
    return [x0, y0, x1, y1]


def geometry_bounds(geometry: Any) -> tuple[float, float, float, float] | None:
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
    bounds = geometry_bounds(geometry)
    if bounds is None:
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    x0, y0, x1, y1 = bounds
    return (y0, x0, y1, x1)


def find_sort_geometry(value: Any) -> Any:
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
