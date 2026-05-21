import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from xml.dom import minidom
from numbers import Number

from PIL import Image

from services.word_box_normalization import merge_apostrophe_word_boxes


ALTO_PROFILE_ABBYY_LIKE = "abbyy_like"
ALTO_PROFILE_LEGACY = "legacy"
ALTO_NS_V3 = "http://www.loc.gov/standards/alto/ns-v3#"
ALTO_NS_V4 = "http://www.loc.gov/standards/alto/ns-v4#"
XLINK_NS = "http://www.w3.org/1999/xlink"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
DOCLAYOUT_TILE_STRATEGIES = {
    "doclayout_v3",
    "grid_doclayout_v3",
    "grid_3x2_doclayout_v3",
}
TITLE_LABELS = {"doc_title", "paragraph_title", "title"}


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image width and height in pixels.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
            - width (int): Image width in pixels.
            - height (int): Image height in pixels.
    """
  
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height


def calculate_textblock(polys) -> Tuple[int, int, int, int]:
    x_axis = []
    y_axis = []

    for poly in polys:

        # CHANGED: numpy -> list
        if hasattr(poly, "tolist"):
            poly = poly.tolist()

        
        if isinstance(poly[0], (int, float)):
            # bbox: [x1,y1,x2,y2]
            x1, y1, x2, y2 = poly
            x_axis.extend([x1, x2])
            y_axis.extend([y1, y2])

        else:
            # polygon: [[x,y],...]
            for x, y in poly:
                x_axis.append(x)
                y_axis.append(y)

    hpos = min(x_axis)
    vpos = min(y_axis)
    width = max(x_axis) - hpos
    height = max(y_axis) - vpos

    return hpos, vpos, width, height


def normalize_box(box):
    if hasattr(box, "tolist"):
        box = box.tolist()
    return box


def box_to_rect(box) -> Tuple[float, float, float, float]:
    box = normalize_box(box)

    if (
        len(box) == 4
        and all(isinstance(v, Number) for v in box)
    ):
        x1, y1, x2, y2 = box
        return float(x1), float(y1), float(x2), float(y2)

    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def make_ocr_item(text, box, score=None, line_index=None):
    return {
        "text": clean_alto_text(text),
        "box": normalize_box(box),
        "score": score,
        "line_index": line_index,
    }


def clean_alto_text(text) -> str:
    return str(text).replace("\n", " ").strip()


def extract_source_ocr_payload(data):
    if "overall_ocr_res" in data:
        return data["overall_ocr_res"], False

    if "rec_texts" in data:
        return data, False

    parsing_res_list = data["result"]["layoutParsingResults"][0]["prunedResult"][
        "parsing_res_list"
    ]
    return parsing_res_list, True


def extract_word_level_lines(ocr_payload):
    rec_texts = ocr_payload.get("rec_texts", [])
    rec_scores = ocr_payload.get("rec_scores", [])
    text_word_lines = ocr_payload.get("text_word") or []
    text_word_box_lines = ocr_payload.get("text_word_boxes") or []
    rec_polys = ocr_payload.get("rec_polys") or []

    if not text_word_lines or not text_word_box_lines:
        return []

    lines = []

    line_count = max(len(rec_texts), len(text_word_lines), len(text_word_box_lines))

    for index in range(line_count):
        words = text_word_lines[index] if index < len(text_word_lines) else []
        boxes = text_word_box_lines[index] if index < len(text_word_box_lines) else []
        line_text = rec_texts[index] if index < len(rec_texts) else None
        line_score = rec_scores[index] if index < len(rec_scores) else None

        if not words or not boxes:
            if line_text and index < len(rec_polys):
                line_box = normalize_box(rec_polys[index])
                lines.append(
                    {
                        "box": line_box,
                        "score": line_score,
                        "strings": [
                            make_ocr_item(
                                line_text,
                                line_box,
                                score=line_score,
                                line_index=index,
                            )
                        ],
                        "index": index,
                    }
                )
            continue

        words, boxes = merge_apostrophe_word_boxes(
            words,
            boxes,
            line_text=line_text,
        )
        string_items = []

        for word_text, word_box in zip(words, boxes):
            if not word_text or not clean_alto_text(word_text):
                continue

            string_items.append(
                make_ocr_item(
                    word_text,
                    word_box,
                    score=line_score,
                    line_index=index,
                )
            )

        if not string_items:
            continue

        string_items.sort(key=lambda item: box_to_rect(item["box"])[0])

        if index < len(rec_polys):
            line_box = rec_polys[index]
        else:
            line_box = calculate_line_box([item["box"] for item in string_items])

        lines.append(
            {
                "box": normalize_box(line_box),
                "score": line_score,
                "strings": string_items,
                "index": index,
            }
        )

    lines = sort_lines_reading_order(lines)
    return lines


def extract_line_level_items(source_payload, select_paddlevl):
    items = []

    if select_paddlevl:
        for block in source_payload:
            text = block.get("block_content", "")
            poly = block.get("block_bbox")

            if not text or not text.strip() or not poly:
                continue

            items.append(make_ocr_item(text, poly))

        return items

    rec_texts = source_payload.get("rec_texts", [])
    rec_scores = source_payload.get("rec_scores", [])
    rec_polys = source_payload.get("rec_polys", [])

    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
        if not text or not text.strip() or not poly or len(poly) < 4:
            continue

        items.append(make_ocr_item(text, poly, score=score))

    return items


def build_alto_lines(data):
    ocr_blocks = data.get("ocr_blocks")
    if isinstance(ocr_blocks, list) and ocr_blocks:
        lines = [
            line
            for block in ocr_blocks
            for line in block.get("lines", [])
            if line.get("strings")
        ]
        return lines, False

    source_payload, select_paddlevl = extract_source_ocr_payload(data)

    if not select_paddlevl:
        word_lines = extract_word_level_lines(source_payload)
        if word_lines:
            return word_lines, select_paddlevl

    # If we don't have explicit word-level arrays, try to build lines
    # from the available `rec_texts` / `rec_polys` tokens by spatial grouping.
    items = extract_line_level_items(source_payload, select_paddlevl)

    def group_words_into_lines(items):
        """Group word-level items into TextLine-like structures.

        Each returned line is a dict: {"box": [x1,y1,x2,y2], "score": None, "strings": [item,...]}
        where each `item` is {'text':..., 'box':..., 'score':...}.
        """

        if not items:
            return []

        # Convert to rects and compute centers
        structured = []
        for it in items:
            x1, y1, x2, y2 = box_to_rect(it["box"])
            center_y = (y1 + y2) / 2
            center_x = (x1 + x2) / 2
            height = max(1.0, y2 - y1)
            structured.append({
                "text": it["text"],
                "box": it["box"],
                "score": it.get("score"),
                "line_index": it.get("line_index"),
                "rect": (x1, y1, x2, y2),
                "center_y": center_y,
                "center_x": center_x,
                "height": height,
            })

        median_height = sorted(s["height"] for s in structured)[len(structured) // 2]
        line_threshold = max(8.0, median_height * 0.6)

        structured.sort(key=lambda s: (s["center_y"], s["rect"][0]))

        lines = []
        current = [structured[0]]
        current_y = structured[0]["center_y"]

        for s in structured[1:]:
            if abs(s["center_y"] - current_y) <= line_threshold:
                current.append(s)
                current_y = sum(x["center_y"] for x in current) / len(current)
            else:
                # finalize current line
                current.sort(key=lambda x: x["rect"][0])
                line_box = calculate_line_box([c["box"] for c in current])
                line_score = average_scores(c.get("score") for c in current)
                strings = [
                    {
                        "text": c["text"],
                        "box": c["box"],
                        "score": c.get("score", line_score),
                        "line_index": c.get("line_index"),
                    }
                    for c in current
                ]
                lines.append({"box": line_box, "score": line_score, "strings": strings})

                current = [s]
                current_y = s["center_y"]

        # finalize last
        if current:
            current.sort(key=lambda x: x["rect"][0])
            line_box = calculate_line_box([c["box"] for c in current])
            line_score = average_scores(c.get("score") for c in current)
            strings = [
                {
                    "text": c["text"],
                    "box": c["box"],
                    "score": c.get("score", line_score),
                    "line_index": c.get("line_index"),
                }
                for c in current
            ]
            lines.append({"box": line_box, "score": line_score, "strings": strings})

        return sort_lines_reading_order(lines)

    lines = group_words_into_lines(items)
    return lines, select_paddlevl


def calculate_line_box(boxes):
    hpos, vpos, width, height = calculate_textblock(boxes)
    return [hpos, vpos, hpos + width, vpos + height]


def average_scores(scores) -> float | None:
    values = [float(score) for score in scores if isinstance(score, Number)]
    if not values:
        return None
    return sum(values) / len(values)


def box_sort_key(box):
    x1, y1, x2, y2 = box_to_rect(box)
    return (y1, x1, y2, x2)


def sort_lines_reading_order(lines):
    if len(lines) < 2:
        return list(lines)

    structured = []
    for line in lines:
        x1, y1, x2, y2 = box_to_rect(line["box"])
        structured.append(
            {
                "line": line,
                "rect": (x1, y1, x2, y2),
                "center_y": (y1 + y2) / 2.0,
                "height": max(1.0, y2 - y1),
            }
        )

    heights = sorted(item["height"] for item in structured)
    median_h = heights[len(heights) // 2] if heights else 12.0
    line_threshold = max(5.0, median_h * 0.35)

    structured.sort(key=lambda item: (item["center_y"], item["rect"][0]))
    rows = []
    current = [structured[0]]
    current_y = structured[0]["center_y"]

    for item in structured[1:]:
        if abs(item["center_y"] - current_y) <= line_threshold:
            current.append(item)
            current_y = sum(entry["center_y"] for entry in current) / len(current)
        else:
            rows.append(current)
            current = [item]
            current_y = item["center_y"]

    if current:
        rows.append(current)

    ordered = []
    for row in rows:
        row.sort(key=lambda item: item["rect"][0])
        ordered.extend(item["line"] for item in row)

    return ordered


def box_to_alto_attrs(box, scale):
    x1, y1, x2, y2 = box_to_rect(box)
    return {
        "HPOS": format_alto_number(x1 * scale),
        "VPOS": format_alto_number(y1 * scale),
        "WIDTH": format_alto_number((x2 - x1) * scale),
        "HEIGHT": format_alto_number((y2 - y1) * scale),
    }


def format_alto_number(value) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_alto_confidence(value) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def detect_columns(lines, page_width, min_valley_width_ratio=0.03):
    """Detect vertical column separators using a horizontal projection.

    Returns a list of (x0, x1) column ranges in pixel coordinates. If no
    clear separators are found, returns a single column covering the page.
    """
    if not lines or page_width <= 0:
        return [(0, page_width)]

    # bin the page width into coarse bins to build a projection profile.
    page_width = int(page_width)
    bin_size = max(8, page_width // 200)
    num_bins = (page_width + bin_size - 1) // bin_size

    coverage = [0] * num_bins

    for ln in lines:
        x1, y1, x2, y2 = box_to_rect(ln["box"])
        b0 = max(0, int(x1 // bin_size))
        b1 = min(num_bins - 1, int(x2 // bin_size))
        for b in range(b0, b1 + 1):
            coverage[b] += 1

    max_cov = max(coverage) if coverage else 0
    valley_thr = max(1, int(max_cov * 0.05))

    # find contiguous low-coverage (valley) ranges
    valleys = []
    in_valley = False
    vstart = None
    for i, c in enumerate(coverage):
        if c <= valley_thr:
            if not in_valley:
                in_valley = True
                vstart = i
        else:
            if in_valley:
                valleys.append((vstart, i - 1))
                in_valley = False
    if in_valley:
        valleys.append((vstart, num_bins - 1))

    # filter narrow valleys
    min_valley_bins = max(1, int(min_valley_width_ratio * num_bins))
    separations = [v for v in valleys if (v[1] - v[0] + 1) >= min_valley_bins]

    if not separations:
        return [(0, page_width)]

    cols = []
    left = 0
    for v in separations:
        sep_start = v[0] * bin_size
        sep_end = min(page_width, (v[1] + 1) * bin_size)
        if sep_start - left > bin_size:
            cols.append((left, min(sep_start, page_width)))
        left = sep_end

    if page_width - left > bin_size:
        cols.append((left, page_width))

    # If no columns detected (all valleys at edges), fall back to whole page
    if not cols:
        return [(0, page_width)]

    return cols


def group_lines_into_blocks(lines, select_paddlevl, page_width=None):
    """Group sorted `lines` into higher-level text blocks.

    - If `select_paddlevl` is True (PaddleVL), each line represents a layout
      block already, so return one block per line.
    - Otherwise, cluster consecutive lines into blocks based on vertical
      proximity using a median line-height heuristic.

    Returns a list of blocks: {"box": [x1,y1,x2,y2], "lines": [...]}
    """
    if not lines:
        return []

    if select_paddlevl:
        blocks = []
        for ln in lines:
            blocks.append({"box": ln["box"], "lines": [ln]})
        return blocks

    # Non-PaddleVL: cluster by vertical gaps
    # If page_width not provided, estimate from line boxes
    if page_width is None:
        page_width = int(max((box_to_rect(l["box"])[2] for l in lines), default=0))

    # Detect columns and cluster lines within each column separately.
    columns = detect_columns(lines, page_width)

    blocks = []

    for col_start, col_end in sorted(columns, key=lambda c: c[0]):
        # select lines whose horizontal center falls into this column
        col_lines = []
        for ln in lines:
            x1, y1, x2, y2 = box_to_rect(ln["box"])
            center_x = (x1 + x2) / 2.0
            # include right edge in last column
            if (center_x >= col_start and center_x < col_end) or (
                col_end == page_width and center_x <= col_end
            ):
                col_lines.append(ln)

        if not col_lines:
            continue

        rects = [box_to_rect(l["box"]) for l in col_lines]
        heights = [r[3] - r[1] for r in rects]
        median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
        gap_threshold = max(8.0, median_h * 0.8)

        sorted_lines = sort_lines_reading_order(col_lines)

        current = [sorted_lines[0]]
        _, cur_y1, _, cur_y2 = box_to_rect(sorted_lines[0]["box"])

        for ln in sorted_lines[1:]:
            x1, y1, x2, y2 = box_to_rect(ln["box"])

            if y1 - cur_y2 <= gap_threshold:
                current.append(ln)
                cur_y2 = max(cur_y2, y2)
            else:
                boxes = [l["box"] for l in current]
                block_box = calculate_line_box(boxes)
                blocks.append({"box": block_box, "lines": current})

                current = [ln]
                cur_y2 = y2

        if current:
            boxes = [l["box"] for l in current]
            block_box = calculate_line_box(boxes)
            blocks.append({"box": block_box, "lines": current})

    # Sort blocks left-to-right then top-to-bottom for reading order.
    blocks.sort(key=lambda b: (box_to_rect(b["box"])[0], box_to_rect(b["box"])[1]))
    return blocks


def build_alto_blocks(data, lines, select_paddlevl, page_width):
    ocr_blocks = data.get("ocr_blocks")
    if isinstance(ocr_blocks, list) and ocr_blocks:
        return [
            block
            for block in ocr_blocks
            if block.get("lines") and block.get("box")
        ]

    tile_blocks = build_blocks_from_doclayout_tiles(data, lines)
    if tile_blocks:
        return tile_blocks

    return group_lines_into_blocks(
        lines,
        select_paddlevl,
        page_width=page_width,
    )


def infer_page_width_from_lines(lines):
    right_edges = []

    for line in lines or []:
        try:
            right_edges.append(box_to_rect(line["box"])[2])
        except Exception:
            continue

    return int(max(right_edges, default=0))


def build_ordered_blocks(data, page_width=None):
    """Build OCR blocks in the same reading order used for ALTO output."""
    lines, select_paddlevl = build_alto_lines(data)

    if page_width is None:
        page_width = infer_page_width_from_lines(lines)

    blocks = build_alto_blocks(
        data,
        lines,
        select_paddlevl,
        page_width=page_width,
    )

    if blocks:
        return blocks

    boxes = [
        string_item["box"]
        for line in lines
        for string_item in line.get("strings", [])
    ]
    fallback_box = calculate_line_box(boxes) if boxes else [0, 0, page_width or 0, 0]

    return [{"box": fallback_box, "lines": lines}]


def line_to_text(line):
    """Reconstruct readable text from a normalized OCR line."""
    string_items = [
        item
        for item in line.get("strings", [])
        if clean_alto_text(item.get("text", ""))
    ]

    parts = []
    previous_item = None

    for item in string_items:
        if previous_item is not None and should_insert_sp(previous_item, item):
            parts.append(" ")

        parts.append(clean_alto_text(item.get("text", "")))

        if item.get("add_hyp"):
            parts.append("-")

        previous_item = item

    return "".join(parts).strip()


def ordered_text_lines_from_data(data, page_width=None):
    """Return text lines in the ALTO/DocLayout reading order."""
    ordered_lines = []

    for block in build_ordered_blocks(data, page_width=page_width):
        for line in block.get("lines") or []:
            text = line_to_text(line)
            if text:
                ordered_lines.append(text)

    return ordered_lines


def build_blocks_from_doclayout_tiles(data, lines):
    tile_processing = data.get("tile_processing") or {}
    strategy = tile_processing.get("strategy")
    if strategy not in DOCLAYOUT_TILE_STRATEGIES:
        return []

    tiles = normalize_doclayout_tiles(tile_processing.get("tiles") or [])
    if not tiles:
        return []

    line_assignments = {tile["index"]: [] for tile in tiles}
    unassigned_lines = []

    for line in lines:
        tile = find_best_tile_for_line(line, tiles)
        if tile is None:
            unassigned_lines.append(line)
        else:
            line_assignments[tile["index"]].append(line)

    blocks = []
    for tile in sorted(tiles, key=tile_sort_key):
        tile_lines = line_assignments.get(tile["index"]) or []
        if not tile_lines:
            continue

        tile_lines = sort_lines_reading_order(tile_lines)
        blocks.append(
            {
                "box": calculate_line_box([line["box"] for line in tile_lines]),
                "lines": tile_lines,
                "tile": tile,
                "label": tile.get("label"),
            }
        )

    if unassigned_lines:
        blocks.extend(
            group_lines_into_blocks(
                unassigned_lines,
                select_paddlevl=False,
                page_width=None,
            )
        )

    return blocks


def normalize_doclayout_tiles(raw_tiles):
    tiles = []

    for index, tile in enumerate(raw_tiles):
        bbox = normalize_rect(tile.get("bbox") or tile.get("nominal_bbox"))
        if bbox is None:
            continue

        nominal_bbox = normalize_rect(tile.get("nominal_bbox")) or bbox
        tiles.append(
            {
                "index": index,
                "bbox": bbox,
                "nominal_bbox": nominal_bbox,
                "label": tile.get("label"),
                "order": tile.get("order"),
                "row": tile.get("row"),
                "col": tile.get("col"),
                "score": tile.get("score"),
            }
        )

    return tiles


def normalize_rect(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def tile_sort_key(tile):
    order = tile.get("order")
    row = tile.get("row")
    col = tile.get("col")
    bbox = tile["bbox"]

    return (
        numeric_or_default(order, tile["index"]),
        numeric_or_default(row, tile["index"]),
        numeric_or_default(col, 0),
        bbox[1],
        bbox[0],
    )


def numeric_or_default(value, default):
    return value if isinstance(value, Number) else default


def find_best_tile_for_line(line, tiles):
    line_rect = list(box_to_rect(line["box"]))
    center_x = (line_rect[0] + line_rect[2]) / 2.0
    center_y = (line_rect[1] + line_rect[3]) / 2.0
    line_area = max(1.0, rect_area(line_rect))
    best_tile = None
    best_score = None

    for tile in tiles:
        nominal_bbox = tile["nominal_bbox"]
        bbox = tile["bbox"]
        overlap = rect_overlap_area(line_rect, bbox)
        overlap_ratio = overlap / line_area

        if rect_contains_point(nominal_bbox, center_x, center_y):
            tier = 3
            area = rect_area(nominal_bbox)
        elif rect_contains_point(bbox, center_x, center_y):
            tier = 2
            area = rect_area(bbox)
        elif overlap > 0:
            tier = 1
            area = rect_area(bbox)
        else:
            continue

        score = (tier, overlap_ratio, -area, -tile["index"])
        if best_score is None or score > best_score:
            best_score = score
            best_tile = tile

    return best_tile


def rect_contains_point(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def rect_area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def rect_overlap_area(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return rect_area([x1, y1, x2, y2])


def annotate_block_hyphenation(blocks):
    for block in blocks:
        lines = block.get("lines") or []
        for index, line in enumerate(lines[:-1]):
            last_item = last_string_item(line)
            first_next_item = first_string_item(lines[index + 1])
            if not last_item or not first_next_item:
                continue

            last_text = clean_alto_text(last_item.get("text", ""))
            next_text = clean_alto_text(first_next_item.get("text", ""))
            if not is_hyphenated_line_break(last_text, next_text):
                continue

            prefix = last_text.rstrip("-\u00ad")
            subs_content = f"{prefix}{next_text}"
            last_item["text"] = prefix
            last_item["subs_type"] = "HypPart1"
            last_item["subs_content"] = subs_content
            last_item["add_hyp"] = True
            first_next_item["subs_type"] = "HypPart2"
            first_next_item["subs_content"] = subs_content


def first_string_item(line):
    for item in line.get("strings") or []:
        if clean_alto_text(item.get("text", "")):
            return item
    return None


def last_string_item(line):
    for item in reversed(line.get("strings") or []):
        if clean_alto_text(item.get("text", "")):
            return item
    return None


def is_hyphenated_line_break(last_text, next_text):
    if not last_text.endswith(("-", "\u00ad")):
        return False

    prefix = last_text.rstrip("-\u00ad")
    return (
        len(prefix) >= 2
        and any(char.isalpha() for char in prefix)
        and len(next_text) >= 2
        and next_text[0].islower()
    )


def should_insert_sp(left_item, right_item):
    left = clean_alto_text(left_item.get("text", ""))
    right = clean_alto_text(right_item.get("text", ""))
    if not left or not right or left_item.get("add_hyp"):
        return False

    if right[0] in ".,;:!?)]}%\u201d\u2019":
        return False
    if left[-1] in "([{\u201c\u2018":
        return False
    if left in {"$", "£", "#"}:
        return False

    return True


def sp_attrs_between(left_item, right_item, line, scale):
    left = list(box_to_rect(left_item["box"]))
    right = list(box_to_rect(right_item["box"]))
    line_rect = list(box_to_rect(line["box"]))
    line_height = max(1.0, line_rect[3] - line_rect[1])
    gap = right[0] - left[2]

    if gap <= 0:
        gap = max(1.0, min(line_height * 0.35, 12.0))

    y1 = min(left[1], right[1], line_rect[1])
    y2 = max(left[3], right[3], line_rect[3])
    return box_to_alto_attrs([left[2], y1, left[2] + gap, y2], scale)


def resolve_alto_profile():
    profile = os.environ.get("OCR_ALTO_PROFILE", ALTO_PROFILE_ABBYY_LIKE)
    profile = profile.strip().lower().replace("-", "_").replace(" ", "_")
    if profile in {"legacy", "v3", "inch1200"}:
        return ALTO_PROFILE_LEGACY
    return ALTO_PROFILE_ABBYY_LIKE


def alto_scale_and_unit(profile, dpi):
    if profile == ALTO_PROFILE_LEGACY:
        return 1200.0 / dpi, "inch1200"
    return 1.0, "pixel"


def create_alto_root(profile):
    if profile == ALTO_PROFILE_LEGACY:
        return ET.Element("alto", xmlns=ALTO_NS_V3)

    return ET.Element(
        "alto",
        {
            "xmlns": ALTO_NS_V4,
            "xmlns:xlink": XLINK_NS,
            "xmlns:xsi": XSI_NS,
            "xsi:schemaLocation": (
                f"{ALTO_NS_V4} http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
            ),
        },
    )


def add_processing_description(alto, profile, file_name, measurement_unit):
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = measurement_unit

    source_info = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source_info, "fileName").text = file_name

    if profile == ALTO_PROFILE_LEGACY:
        ocr_proc = ET.SubElement(desc, "OCRProcessing", ID="ID0")
        step = ET.SubElement(ocr_proc, "ocrProcessingStep")
    else:
        step = ET.SubElement(desc, "Processing", ID="IdOcr")

    ET.SubElement(step, "processingDateTime").text = datetime.now().isoformat()

    software = ET.SubElement(step, "processingSoftware")
    ET.SubElement(software, "softwareName").text = "PaddleOCR"
    ET.SubElement(software, "softwareVersion").text = "3.1.0"


def add_alto_styles(alto):
    styles = ET.SubElement(alto, "Styles")
    ET.SubElement(styles, "TextStyle", ID="font_regular", FONTFAMILY="Unknown")
    ET.SubElement(
        styles,
        "TextStyle",
        ID="font_title",
        FONTFAMILY="Unknown",
        FONTSTYLE="bold",
    )


def block_style_id(block):
    return "font_title" if block.get("label") in TITLE_LABELS else "font_regular"


def generate_alto(
    json_file: Path,
    image: Path,
    dpi: int,
    output_dir: Path = Path("."),
    lang: str | None = None,
    method: str | None = None,
) -> Path:
    """
    Generate an ALTO XML file from OCR JSON and an image file.

    The input JSON is expected to have the following structure:
        {
            "code": 100,
            "data": [
                {
                    "text": "...",
                    "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                    "score": 0.98
                },
                ...
            ],
            "fileName": "example.jpg"
        }

    The function:
        1. Reads OCR results from JSON
        2. Reads image width and height
        3. Uses Abbyy-like pixel ALTO by default, or legacy inch1200 if requested
        4. Builds an ALTO XML structure
        5. Saves the XML file locally

    Args:
        json_file: Path to the OCR JSON file.
        image: Path to the source image.
        dpi: Image DPI used when legacy inch1200 output is requested.
        output_dir: Directory where the XML file will be saved.

    Returns:
        Path to the generated XML file.

    Raises:
        SystemExit: If OCR data is missing or saving fails.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines, select_paddlevl = build_alto_lines(data)

    # Use fileName from JSON if available; otherwise fall back to image name.
    file_name = data.get("fileName", image.name)

    # Get original image dimensions in pixels.
    width, height = get_image_dimensions(image)

    profile = resolve_alto_profile()
    scale, measurement_unit = alto_scale_and_unit(profile, dpi)

    # Extract all OCR boxes for calculating the global text block.
    boxes = [
        string_item["box"]
        for line in lines
        for string_item in line["strings"]
    ]
    page_width = width * scale
    page_height = height * scale

    # Register xlink namespace for the xlink:type attribute.
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    # Create the ALTO root element.
    alto = create_alto_root(profile)

    # ------------------------------------------------------------------
    # Description section
    # ------------------------------------------------------------------
    add_processing_description(alto, profile, file_name, measurement_unit)
    add_alto_styles(alto)

    # ------------------------------------------------------------------
    # Layout section
    # ------------------------------------------------------------------
    layout = ET.SubElement(alto, "Layout")

    page = ET.SubElement(
        layout,
        "Page",
        {
            "ID": "Page1",
            "WIDTH": format_alto_number(page_width),
            "HEIGHT": format_alto_number(page_height),
            "PHYSICAL_IMG_NR": "1",
        },
    )

    # PrintSpace is set to cover the whole page.
    print_space = ET.SubElement(
        page,
        "PrintSpace",
        {
            "HPOS": "0",
            "VPOS": "0",
            "WIDTH": format_alto_number(page_width),
            "HEIGHT": format_alto_number(page_height),
        },
    )

    # Group lines into TextBlocks. DocLayout-tiled JSON preserves its detected
    # content regions here so ALTO consumers see article-like blocks.
    blocks = build_alto_blocks(data, lines, select_paddlevl, width)

    if not blocks:
        # Fallback: if we have individual string boxes, use their overall box;
        # otherwise create a single block covering the whole page.
        if boxes:
            blocks = [{"box": calculate_line_box(boxes), "lines": lines}]
        else:
            blocks = [{"box": [0, 0, width, height], "lines": lines}]

    annotate_block_hyphenation(blocks)

    # Create a TextBlock element for each block and add TextLine/String children.
    for block_index, block in enumerate(blocks, start=1):
        tb_attrs = box_to_alto_attrs(block["box"], scale)
        tb_attrs["ID"] = f"Page1_Block{block_index}"
        tb_attrs["STYLEREFS"] = block_style_id(block)
        if lang:
            tb_attrs["LANG"] = lang
        if profile == ALTO_PROFILE_LEGACY:
            tb_attrs[f"{{{XLINK_NS}}}type"] = "simple"

        text_block = ET.SubElement(print_space, "TextBlock", tb_attrs)

        for line in block["lines"]:
            text_line = ET.SubElement(
                text_block,
                "TextLine",
                box_to_alto_attrs(line["box"], scale),
            )

            string_items = [
                item
                for item in line["strings"]
                if clean_alto_text(item.get("text", ""))
            ]

            for string_index, string_item in enumerate(string_items):
                attrs = {
                    "CONTENT": clean_alto_text(string_item["text"]),
                    **box_to_alto_attrs(string_item["box"], scale),
                }

                if lang and not select_paddlevl:
                    attrs["LANG"] = lang

                wc = string_item.get("score")
                if wc is not None:
                    attrs["WC"] = format_alto_confidence(wc)

                subs_type = string_item.get("subs_type")
                subs_content = string_item.get("subs_content")
                if subs_type and subs_content:
                    attrs["SUBS_TYPE"] = subs_type
                    attrs["SUBS_CONTENT"] = subs_content

                ET.SubElement(text_line, "String", attrs)

                if string_item.get("add_hyp"):
                    ET.SubElement(text_line, "HYP", {"CONTENT": "\u00ad"})

                if string_index < len(string_items) - 1:
                    next_item = string_items[string_index + 1]
                    if should_insert_sp(string_item, next_item):
                        ET.SubElement(
                            text_line,
                            "SP",
                            sp_attrs_between(string_item, next_item, line, scale),
                        )

    # Convert XML tree to formatted XML bytes.
    xml_bytes = minidom.parseString(
        ET.tostring(alto, encoding="utf-8")
    ).toprettyxml(indent="  ", encoding="utf-8")

    # Create output directory if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the image file name stem as the XML file name.
    output_path = output_dir / f"{Path(file_name).stem}.xml"

    try:
        with open(output_path, "wb") as f:
            f.write(xml_bytes)

        print(f"ALTO XML saved to: {output_path}")
        return output_path

    except Exception as e:
        raise RuntimeError(f"Error saving ALTO XML locally: {e}") from e
