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


def generate_alto(
    json_file: Path,
    image: Path,
    dpi: int,
    output_dir: Path = Path("."),
    lang: str | None = None
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

    items = []

    # OCR items are stored in the "data" field.
    if "overall_ocr_res" in data:
        ocr = data["overall_ocr_res"]

        rec_texts = ocr.get("rec_texts", [])
        rec_scores = ocr.get("rec_scores", [])
        rec_polys = ocr.get("rec_polys", [])

        if not (len(rec_texts) == len(rec_scores) == len(rec_polys)):
            raise ValueError("OCR arrays length mismatch in overall_ocr_res.")

        items = []

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            if not text or not text.strip():
                continue

            if not poly or len(poly) < 4:
                continue

            items.append(
                {
                    "text": text,
                    "box": poly,
                    "score": score,
                }
            )
    elif "rec_texts" in data:
        rec_texts = data.get("rec_texts")
        rec_scores = data.get("rec_scores")
        rec_polys = data.get("rec_polys")

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            items.append(
                {
                    "text": text,
                    "box": poly,
                    "score": score,
                }
            )

    else:
        parsing_list =  data["result"]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"]
        items = []

        for block in  parsing_list:
            text = block.get("block_content","")
            poly = block.get("block_bbox")

            items.append(
                {
                    "text": text,
                    "box": poly, 
                    
                }
                )
     

    # Use fileName from JSON if available; otherwise fall back to image name.
    file_name = data.get("fileName", image.name)

    if not items:
        raise ValueError("No OCR items found in JSON data.")

    # Get original image dimensions in pixels.
    width, height = get_image_dimensions(image)

    # ALTO uses "inch1200" as the measurement unit here.
    # If image coordinates are in pixels, convert pixels to inch1200
    # using the image DPI.
    scale = 1200.0 / dpi

    # Extract all OCR boxes for calculating the global text block.
    boxes = [item["box"] for item in items]

    # Calculate the overall text block that contains all OCR strings.
    hpos, vpos, block_width, block_height = calculate_textblock(boxes)

    # Convert page and text block coordinates from pixels to inch1200.
    hpos *= scale
    vpos *= scale
    block_width *= scale
    block_height *= scale
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

    # Current version puts all strings into one TextLine.
    # This is structurally valid, though not ideal for multi-line layout.
    text_line = ET.SubElement(
        text_block,
        "TextLine",
        {
            "HPOS": str(hpos),
            "VPOS": str(vpos),
            "WIDTH": str(block_width),
            "HEIGHT": str(block_height),
        },
    )

    # ------------------------------------------------------------------
    # String elements
    # ------------------------------------------------------------------
    for item in items:
        text = item["text"]
        wc = item.get("score")
        box = item["box"]

        # CHANGED: numpy -> list
        if hasattr(box, "tolist"):
            box = box.tolist()

        # PaddleVL block_bbox: [x1, y1, x2, y2]
        if (
            len(box) == 4
            and all(isinstance(v, Number) for v in box)
            ):
            x1, y1, x2, y2 = box

            string_hpos = x1 * scale
            string_vpos = y1 * scale
            string_width = (x2 - x1) * scale
            string_height = (y2 - y1) * scale

        # PaddleOCR polygon: [[x,y], [x,y], [x,y], [x,y]]
        else:
            # Extract all x and y coordinates from the box.
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]

            # Convert the box into a simple bounding rectangle.
            string_hpos = min(xs) * scale
            string_vpos = min(ys) * scale
            string_width = (max(xs) - min(xs)) * scale
            string_height = (max(ys) - min(ys)) * scale

        attrs = {
            "CONTENT": text,
            "HPOS": str(string_hpos),
            "VPOS": str(string_vpos),
            "WIDTH": str(string_width),
            "HEIGHT": str(string_height),
        }

        if lang:
            attrs["LANG"] = lang
            
        if wc is not None:
            attrs["WC"] = str(wc)

        ET.SubElement(
            text_line,
            "String",
            attrs
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