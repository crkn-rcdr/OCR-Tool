import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from xml.dom import minidom
from numbers import Number

from PIL import Image


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


def make_ocr_item(text, box, score=None):
    return {
        "text": text,
        "box": normalize_box(box),
        "score": score,
    }


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
    rec_scores = ocr_payload.get("rec_scores", [])
    text_word_lines = ocr_payload.get("text_word") or []
    text_word_box_lines = ocr_payload.get("text_word_boxes") or []
    rec_polys = ocr_payload.get("rec_polys") or []

    if not text_word_lines or not text_word_box_lines:
        return []

    lines = []

    for index, (words, boxes) in enumerate(zip(text_word_lines, text_word_box_lines)):
        if not words or not boxes:
            continue

        string_items = []

        for word_text, word_box in zip(words, boxes):
            if not word_text or not str(word_text).strip():
                continue

            string_items.append(make_ocr_item(word_text, word_box))

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
                "score": rec_scores[index] if index < len(rec_scores) else None,
                "strings": string_items,
            }
        )

    lines.sort(key=lambda line: box_sort_key(line["box"]))
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
                strings = [{"text": c["text"], "box": c["box"], "score": c.get("score")} for c in current]
                lines.append({"box": line_box, "score": None, "strings": strings})

                current = [s]
                current_y = s["center_y"]

        # finalize last
        if current:
            current.sort(key=lambda x: x["rect"][0])
            line_box = calculate_line_box([c["box"] for c in current])
            strings = [{"text": c["text"], "box": c["box"], "score": c.get("score")} for c in current]
            lines.append({"box": line_box, "score": None, "strings": strings})

        # Sort lines top-to-bottom
        lines.sort(key=lambda line: box_sort_key(line["box"]))
        return lines

    lines = group_words_into_lines(items)
    return lines, select_paddlevl


def calculate_line_box(boxes):
    hpos, vpos, width, height = calculate_textblock(boxes)
    return [hpos, vpos, hpos + width, vpos + height]


def box_sort_key(box):
    x1, y1, x2, y2 = box_to_rect(box)
    return (y1, x1, y2, x2)


def box_to_alto_attrs(box, scale):
    x1, y1, x2, y2 = box_to_rect(box)
    return {
        "HPOS": str(x1 * scale),
        "VPOS": str(y1 * scale),
        "WIDTH": str((x2 - x1) * scale),
        "HEIGHT": str((y2 - y1) * scale),
    }


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

        sorted_lines = sorted(col_lines, key=lambda l: box_to_rect(l["box"])[1])

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
        3. Converts pixel coordinates to ALTO "inch1200" coordinates
        4. Builds an ALTO XML structure
        5. Saves the XML file locally

    Args:
        json_file: Path to the OCR JSON file.
        image: Path to the source image.
        dpi: Image DPI used to convert pixels to inch1200 units.
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

    # ALTO uses "inch1200" as the measurement unit here.
    # If image coordinates are in pixels, convert pixels to inch1200
    # using the image DPI.
    scale = 1200.0 / dpi

    # Extract all OCR boxes for calculating the global text block.
    boxes = [
        string_item["box"]
        for line in lines
        for string_item in line["strings"]
    ]
    if boxes:
        # Calculate the overall text block that contains all OCR strings.
        hpos, vpos, block_width, block_height = calculate_textblock(boxes)

        # Convert page and text block coordinates from pixels to inch1200.
        hpos *= scale
        vpos *= scale
        block_width *= scale
        block_height *= scale
    else:
        hpos = 0
        vpos = 0
        block_width = 0
        block_height = 0

    page_width = width * scale
    page_height = height * scale

    # Register xlink namespace for the xlink:type attribute.
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    # Create the ALTO root element.
    alto = ET.Element(
        "alto",
        xmlns="http://www.loc.gov/standards/alto/ns-v3#",
    )

    # ------------------------------------------------------------------
    # Description section
    # ------------------------------------------------------------------
    desc = ET.SubElement(alto, "Description")
    ET.SubElement(desc, "MeasurementUnit").text = "inch1200"

    source_info = ET.SubElement(desc, "sourceImageInformation")
    ET.SubElement(source_info, "fileName").text = file_name

    ocr_proc = ET.SubElement(desc, "OCRProcessing", ID="ID0")
    step = ET.SubElement(ocr_proc, "ocrProcessingStep")
    ET.SubElement(step, "processingDateTime").text = datetime.now().isoformat()

    software = ET.SubElement(step, "processingSoftware")
    ET.SubElement(software, "softwareName").text = "PaddleOCR"
    ET.SubElement(software, "softwareVersion").text = "3.1.0"

    # ------------------------------------------------------------------
    # Layout section
    # ------------------------------------------------------------------
    layout = ET.SubElement(alto, "Layout")

    page = ET.SubElement(
        layout,
        "Page",
        {
            "ID": "Page1",
            "WIDTH": str(page_width),
            "HEIGHT": str(page_height),
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
            "WIDTH": str(page_width),
            "HEIGHT": str(page_height),
        },
    )

    # Group lines into TextBlocks (use layout blocks from PaddleVL when available).
    blocks = group_lines_into_blocks(lines, select_paddlevl)

    if not blocks:
        # Fallback: if we have individual string boxes, use their overall box;
        # otherwise create a single block covering the whole page.
        if boxes:
            blocks = [{"box": calculate_line_box(boxes), "lines": lines}]
        else:
            blocks = [{"box": [0, 0, width, height], "lines": lines}]

    # Create a TextBlock element for each block and add TextLine/String children.
    for block in blocks:
        tb_attrs = box_to_alto_attrs(block["box"], scale)
        tb_attrs["{http://www.w3.org/1999/xlink}type"] = "simple"

        text_block = ET.SubElement(print_space, "TextBlock", tb_attrs)

        for line in block["lines"]:
            text_line = ET.SubElement(
                text_block,
                "TextLine",
                box_to_alto_attrs(line["box"], scale),
            )

            for string_item in line["strings"]:
                attrs = {
                    "CONTENT": string_item["text"],
                    **box_to_alto_attrs(string_item["box"], scale),
                }

                if lang and not select_paddlevl:
                    attrs["LANG"] = lang

                wc = string_item.get("score")
                if wc is not None:
                    attrs["WC"] = str(wc)

                ET.SubElement(text_line, "String", attrs)

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
