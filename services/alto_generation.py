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

    items = extract_line_level_items(source_payload, select_paddlevl)
    lines = []

    for item in items:
        lines.append(
            {
                "box": item["box"],
                "score": item.get("score"),
                "strings": [item],
            }
        )

    lines.sort(key=lambda line: box_sort_key(line["box"]))
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

    # Create one TextBlock that contains all OCR strings.
    text_block = ET.SubElement(
        print_space,
        "TextBlock",
        {
            "HPOS": str(hpos),
            "VPOS": str(vpos),
            "WIDTH": str(block_width),
            "HEIGHT": str(block_height),
            "{http://www.w3.org/1999/xlink}type": "simple",
            
        },
    )

    # ------------------------------------------------------------------
    # TextLine and String elements
    # ------------------------------------------------------------------
    for line in lines:
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
