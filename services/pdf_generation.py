import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from services.alto_generation import generate_alto
from services.ocr_logger import OCRLoggerManger

BASE_DIR = Path(__file__).resolve().parent

LATIN_FONT_PATH = BASE_DIR / "fonts" / "NotoSans-Regular.ttf"
CJK_FONT_PATH = BASE_DIR / "fonts" / "NotoSansSC-Regular.ttf"

LATIN_FONT_ALIAS = "LATIN"
CJK_FONT_ALIAS = "CJK"


@dataclass
class OCRItem:
    """Represent one normalized OCR text item and its geometry."""

    text: str
    poly: list
    rect: fitz.Rect
    center_x: float
    center_y: float
    width: float
    height: float


def polygon_to_rect(poly):
    """Convert a polygon point list into a PyMuPDF rectangle."""
    if hasattr(poly, "tolist"):
        poly = poly.tolist()

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def bbox_to_rect(bbox):
    """Convert bbox or polygon to fitz.Rect.

    Supports:
    - [x0, y0, x1, y1]
    - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()

    # Polygon format.
    if (
        isinstance(bbox, list)
        and len(bbox) > 0
        and isinstance(bbox[0], (list, tuple))
    ):
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]

        return fitz.Rect(
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    # Standard bbox format.
    x0, y0, x1, y1 = bbox

    return fitz.Rect(
        float(x0),
        float(y0),
        float(x1),
        float(y1),
    )


def normalize_text(text):
    """Normalize OCR text by removing line breaks and repeated spaces."""
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_text_keep_newlines(text):
    """Normalize text but keep paragraph line breaks."""
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()

def contains_cjk(text):
    """Return True if text contains Chinese, Japanese, or Korean characters."""
    if not text:
        return False

    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def choose_font_alias(text):
    """Choose the PDF font alias according to whether text contains CJK."""
    return CJK_FONT_ALIAS if contains_cjk(text) else LATIN_FONT_ALIAS


def is_mostly_cjk(text):
    """Return True if at least half of the non-space characters are CJK."""
    if not text:
        return False

    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False

    cjk_count = sum(1 for ch in chars if contains_cjk(ch))

    return cjk_count / len(chars) >= 0.5


def is_vertical_cjk_rect(rect, text=None):
    """Return True if a CJK OCR box looks like vertical text.

    Historical Chinese newspapers are often vertically laid out. A vertical
    text region is usually much taller than it is wide. If such a region is
    rendered with horizontal character splitting, the characters may appear
    compressed or clustered.

    This function is intentionally heuristic. It only switches to vertical
    rendering when the rectangle is clearly tall and narrow.
    """
    if rect.width <= 0 or rect.height <= 0:
        return False

    if text is not None and not is_mostly_cjk(text):
        return False

    return rect.height > rect.width * 1.8


def collect_json_files(output_dir, recursive=True):
    """Yield JSON files from a file path or directory."""
    if output_dir.is_file():
        if output_dir.suffix.lower() == ".json":
            yield output_dir
        return

    pattern = output_dir.rglob("*.json") if recursive else output_dir.glob("*.json")

    for path in pattern:
        yield path


def find_image_for_json(json_path, input_dir):
    """Find the source JPG image that corresponds to an OCR JSON file."""
    image_name = json_path.stem.replace("_res", "") + ".jpg"

    if input_dir.is_file():
        return input_dir if input_dir.name == image_name else None

    matches = list(input_dir.rglob(image_name))

    return matches[0] if matches else None


def extract_ocr_data(data):
    """Extract OCR texts and geometry from supported PaddleOCR JSON formats."""
    texts = []
    polys = []

    if "overall_ocr_res" in data:
        ocr_res = data["overall_ocr_res"]
        texts = ocr_res["rec_texts"]
        polys = ocr_res["rec_polys"]

    elif "result" in data:
        parsing_res_list = data["result"]["layoutParsingResults"][0][
            "prunedResult"
        ]["parsing_res_list"]

        for block in parsing_res_list:
            text = block.get("block_content")

            # Prefer polygon points when PaddleVL provides them. They preserve
            # irregular or non-rectangular layout regions better than bbox.
            poly = block.get("block_polygon_points") or block.get("block_bbox")

            texts.append(text)
            polys.append(poly)

    elif "rec_texts" in data:
        texts = data.get("rec_texts")
        polys = data.get("rec_polys")

    if len(texts) != len(polys):
        raise ValueError(
            f"rec_texts count ({len(texts)}) does not match "
            f"rec_polys count ({len(polys)})"
        )

    return texts, polys


def build_items(texts, polys, method, select):
    """Build OCR items from OCR text and geometry."""
    items = []

    is_paddlevl = method == "PaddleVL" or select == "PaddleVL"

    for text, poly in zip(texts, polys):
        if is_paddlevl:
            text = normalize_text_keep_newlines(text)
        else:
            text = normalize_text(text)

        if not text:
            continue

        if method in ["PP-StructureV3", "PP-OCRv5"] or select == "PP-StructureV3":
            rect = polygon_to_rect(poly)
        elif is_paddlevl:
            rect = bbox_to_rect(poly)
        else:
            continue

        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            continue

        items.append(
            OCRItem(
                text=text,
                poly=poly.tolist() if hasattr(poly, "tolist") else poly,
                rect=rect,
                center_x=(rect.x0 + rect.x1) / 2,
                center_y=(rect.y0 + rect.y1) / 2,
                width=rect.width,
                height=rect.height,
            )
        )

    return items


def sort_items_reading_order(items):
    """Sort OCR items into a simple top-to-bottom, left-to-right order."""
    if not items:
        return []

    median_height = sorted(item.height for item in items)[len(items) // 2]
    line_threshold = max(8.0, median_height * 0.6)

    items = sorted(items, key=lambda item: (item.center_y, item.rect.x0))

    lines = []
    current_line = [items[0]]
    current_y = items[0].center_y

    for item in items[1:]:
        if abs(item.center_y - current_y) <= line_threshold:
            current_line.append(item)
            current_y = sum(x.center_y for x in current_line) / len(current_line)
        else:
            lines.append(sorted(current_line, key=lambda x: x.rect.x0))
            current_line = [item]
            current_y = item.center_y

    if current_line:
        lines.append(sorted(current_line, key=lambda x: x.rect.x0))

    lines.sort(key=lambda line: min(item.rect.y0 for item in line))

    ordered = []
    for line in lines:
        ordered.extend(line)

    return ordered


def group_items_into_lines(items):
    """Group OCR items into approximate horizontal text lines."""
    if not items:
        return []

    median_height = sorted(item.height for item in items)[len(items) // 2]
    line_threshold = max(8.0, median_height * 0.6)

    items = sorted(items, key=lambda item: (item.center_y, item.rect.x0))

    lines = []
    current_line = [items[0]]
    current_y = items[0].center_y

    for item in items[1:]:
        if abs(item.center_y - current_y) <= line_threshold:
            current_line.append(item)
            current_y = sum(x.center_y for x in current_line) / len(current_line)
        else:
            current_line.sort(key=lambda x: x.rect.x0)
            lines.append(current_line)
            current_line = [item]
            current_y = item.center_y

    if current_line:
        current_line.sort(key=lambda x: x.rect.x0)
        lines.append(current_line)

    lines.sort(key=lambda line: min(item.rect.y0 for item in line))

    return lines


def merge_latin_line_items(line_items):
    """Merge Latin OCR items in one line into a single text block."""
    if not line_items:
        return None

    merged_parts = []
    prev_item = None

    for item in line_items:
        if prev_item is None:
            merged_parts.append(item.text)
            prev_item = item
            continue

        gap = item.rect.x0 - prev_item.rect.x1
        avg_height = (prev_item.height + item.height) / 2

        if gap > avg_height * 0.35:
            merged_parts.append(" ")

        merged_parts.append(item.text)
        prev_item = item

    merged_text = "".join(merged_parts).strip()
    if not merged_text:
        return None

    x0 = min(item.rect.x0 for item in line_items)
    y0 = min(item.rect.y0 for item in line_items)
    x1 = max(item.rect.x1 for item in line_items)
    y1 = max(item.rect.y1 for item in line_items)

    rect = fitz.Rect(x0, y0, x1, y1)
    pad_x = max(1.0, rect.height * 0.08)
    pad_y = max(0.5, rect.height * 0.12)

    rect = fitz.Rect(
        rect.x0 - pad_x,
        rect.y0 - pad_y,
        rect.x1 + pad_x,
        rect.y1 + pad_y,
    )

    return {
        "mode": "latin_line",
        "text": merged_text,
        "rect": rect,
        "font_alias": LATIN_FONT_ALIAS,
    }


def make_cjk_blocks_from_items(line_items):
    """Create CJK text blocks from OCR items.

    CJK text usually has no spaces between words. Rendering a long CJK string
    as one textbox often gives poor PDF search highlighting and selection
    geometry. Therefore CJK blocks are later rendered character by character.
    """
    blocks = []

    for item in line_items:
        text = normalize_text(item.text)
        if not text:
            continue

        rect = fitz.Rect(item.rect)
        pad_x = max(0.5, rect.height * 0.05)
        pad_y = max(0.5, rect.height * 0.10)

        rect = fitz.Rect(
            rect.x0 - pad_x,
            rect.y0 - pad_y,
            rect.x1 + pad_x,
            rect.y1 + pad_y,
        )

        blocks.append(
            {
                "mode": "cjk_chars",
                "text": text,
                "rect": rect,
                "font_alias": CJK_FONT_ALIAS,
            }
        )

    return blocks


def build_text_blocks(items, method=None, select=None):
    """Build text blocks for PDF writing.

    PaddleVL returns paragraph-level layout blocks instead of
    line-level OCR boxes. Therefore, PaddleVL content should
    be written as paragraph text.
    """
    if method == "PaddleVL" or select == "PaddleVL":
        blocks = []

        for item in items:
            blocks.append(
                {
                    "mode": "paragraph",
                    "text": item.text,
                    "rect": item.rect,
                    "font_alias": choose_font_alias(item.text),
                }
            )

        return blocks

    ordered_items = sort_items_reading_order(items)
    lines = group_items_into_lines(ordered_items)

    blocks = []

    for line_items in lines:
        joined_text = "".join(item.text for item in line_items)

        if is_mostly_cjk(joined_text):
            blocks.extend(make_cjk_blocks_from_items(line_items))
        else:
            merged = merge_latin_line_items(line_items)

            if merged:
                blocks.append(merged)

    return blocks


def fit_textbox(
    page,
    rect,
    text,
    fontname,
    font_obj,
    min_fontsize=1.0,
    max_fontsize_factor=0.78,
):
    """Insert OCR text into a PDF rectangle without dropping content.

    If the text cannot fit into the rectangle, the function keeps reducing
    the font size. As a final fallback, it still writes the full text using
    insert_text so searchable OCR content is never lost.
    """
    text = normalize_text(text)

    if not text:
        return False

    rect = fitz.Rect(rect)

    if rect.width <= 1 or rect.height <= 1:
        return False

    fontsize = max(min_fontsize, rect.height * max_fontsize_factor)

    text_width = font_obj.text_length(text, fontsize=fontsize)

    if text_width > 0:
        scale = min(1.0, rect.width / text_width)
        fontsize *= scale

    fontsize = min(fontsize, rect.height * 0.95)
    fontsize = max(min_fontsize, fontsize)

    for _ in range(30):
        rc = page.insert_textbox(
            rect,
            text,
            fontname=fontname,
            fontsize=fontsize,
            render_mode=0,
            overlay=True,
            align=0,
            color=(1, 0, 0)
        )

        if rc is not None and rc >= 0:
            return True

        fontsize *= 0.9

        if fontsize < min_fontsize:
            break

    # Final fallback: always preserve OCR text.
    page.insert_text(
        point=fitz.Point(rect.x0, rect.y0 + min_fontsize),
        text=text,
        fontname=fontname,
        fontsize=min_fontsize,
        render_mode=0,
        overlay=True,
        color=(1, 0, 0)
    )

    return True

def wrap_text_to_width(text, font_obj, fontsize, max_width):
    """Wrap text into lines based on actual font width."""
    lines = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # CJK text usually has no spaces, so wrap by character.
        if contains_cjk(paragraph):
            current_line = ""

            for char in paragraph:
                test_line = current_line + char
                width = font_obj.text_length(test_line, fontsize=fontsize)

                if width <= max_width or not current_line:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = char

            if current_line:
                lines.append(current_line)

            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            test_line = word if not current_line else f"{current_line} {word}"
            width = font_obj.text_length(test_line, fontsize=fontsize)

            if width <= max_width or not current_line:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

    return lines


def fit_paragraph_textbox(
    page,
    rect,
    text,
    fontname,
    font_obj,
    min_fontsize=1.0,
    max_fontsize=16.0,
):
    """Write all PaddleVL paragraph text without dropping content.

    PaddleVL provides paragraph-level boxes instead of OCR line boxes.
    This function splits text into estimated virtual lines and writes
    every line as invisible searchable text.
    """
    text = normalize_text_keep_newlines(text)

    if not text:
        return False

    rect = fitz.Rect(rect)

    if rect.width <= 1 or rect.height <= 1:
        return False

    pad_x = max(0.5, rect.width * 0.005)
    pad_y = max(0.5, rect.height * 0.025)

    write_rect = fitz.Rect(
        rect.x0 + pad_x,
        rect.y0 + pad_y,
        rect.x1 - pad_x,
        rect.y1 - pad_y,
    )

    if write_rect.width <= 1 or write_rect.height <= 1:
        return False

    fontsize = min(max_fontsize, max(min_fontsize, rect.height * 0.065))

    lines = []

    for _ in range(40):
        lines = wrap_text_to_width(
            text=text,
            font_obj=font_obj,
            fontsize=fontsize,
            max_width=write_rect.width,
        )

        if not lines:
            return False

        line_height = fontsize * 1.08
        total_height = len(lines) * line_height

        if total_height <= write_rect.height:
            break

        fontsize *= 0.9

        if fontsize <= min_fontsize:
            fontsize = min_fontsize

            lines = wrap_text_to_width(
                text=text,
                font_obj=font_obj,
                fontsize=fontsize,
                max_width=write_rect.width,
            )

            break

    if not lines:
        return False

    # Compress vertically instead of dropping text.
    if len(lines) > 1:
        line_height = min(
            fontsize * 1.08,
            write_rect.height / len(lines),
        )
    else:
        line_height = fontsize * 1.08

    fontsize = min(fontsize, max(0.5, line_height * 0.95))

    y = write_rect.y0 + fontsize
    written_count = 0

    for line in lines:
        if y > write_rect.y1:
            y = write_rect.y1

        page.insert_text(
            point=fitz.Point(write_rect.x0, y),
            text=line,
            fontname=fontname,
            fontsize=fontsize,
            render_mode=0,
            overlay=True,
            color=(1, 0, 0)
        )

        written_count += 1
        y += line_height

    return written_count == len(lines)

def measure_char_weights(text, font_obj, fontsize):
    """Measure approximate visual width for each character."""
    weights = []

    for ch in text:
        if ch.isspace():
            weights.append(0.0)
            continue

        width = font_obj.text_length(ch, fontsize=fontsize)
        if width <= 0:
            width = fontsize * 0.6

        weights.append(max(width, fontsize * 0.15))

    return weights


def split_cjk_text_to_weighted_char_boxes(text, rect, font_obj):
    """Split horizontal CJK text into character boxes.

    Why split CJK characters:
    Chinese, Japanese, and Korean text often has no spaces. If an entire CJK
    string is rendered as one PDF textbox, PDF viewers may produce inaccurate
    search highlights and text selection. Character-level boxes give the PDF
    viewer smaller and more stable geometry.
    """
    chars = list(text)
    visible_chars = [ch for ch in chars if not ch.isspace()]

    if not visible_chars:
        return []

    rect = fitz.Rect(rect)
    probe_fontsize = max(6.0, rect.height * 0.82)
    weights = measure_char_weights(chars, font_obj, probe_fontsize)

    visible_weights = [
        width for ch, width in zip(chars, weights) if not ch.isspace()
    ]
    total_weight = sum(visible_weights)

    if total_weight <= 0:
        equal_width = rect.width / len(visible_chars)
        boxes = []
        idx = 0

        for ch in chars:
            if ch.isspace():
                continue

            x0 = rect.x0 + idx * equal_width
            x1 = rect.x0 + (idx + 1) * equal_width
            boxes.append((ch, fitz.Rect(x0, rect.y0, x1, rect.y1)))
            idx += 1

        return boxes

    boxes = []
    current_x = rect.x0
    visible_index = 0

    for ch, width in zip(chars, weights):
        if ch.isspace():
            continue

        width_ratio = visible_weights[visible_index] / total_weight
        box_width = rect.width * width_ratio

        x0 = current_x
        x1 = current_x + box_width

        boxes.append((ch, fitz.Rect(x0, rect.y0, x1, rect.y1)))

        current_x = x1
        visible_index += 1

    if boxes:
        last_ch, last_rect = boxes[-1]
        boxes[-1] = (
            last_ch,
            fitz.Rect(last_rect.x0, last_rect.y0, rect.x1, last_rect.y1),
        )

    return boxes


def split_cjk_text_to_vertical_char_boxes(text, rect):
    """Split vertical CJK text into stacked character boxes.

    Historical Chinese newspapers often arrange text vertically. For tall
    and narrow OCR boxes, the characters should be distributed from top to
    bottom instead of left to right.
    """
    chars = [ch for ch in text if not ch.isspace()]

    if not chars:
        return []

    rect = fitz.Rect(rect)
    char_height = rect.height / len(chars)

    boxes = []

    for idx, ch in enumerate(chars):
        y0 = rect.y0 + idx * char_height
        y1 = rect.y0 + (idx + 1) * char_height

        boxes.append((ch, fitz.Rect(rect.x0, y0, rect.x1, y1)))

    return boxes


def write_cjk_vertical_chars(page, rect, text, fontname, font_obj):
    """Render vertical CJK text character by character."""
    char_boxes = split_cjk_text_to_vertical_char_boxes(text, rect)

    if not char_boxes:
        return False

    ok_count = 0

    for ch, char_rect in char_boxes:
        pad_x = max(0.25, char_rect.width * 0.08)
        pad_y = max(0.25, char_rect.height * 0.03)

        padded_rect = fitz.Rect(
            char_rect.x0 - pad_x,
            char_rect.y0 - pad_y,
            char_rect.x1 + pad_x,
            char_rect.y1 + pad_y,
        )

        ok = fit_textbox(
            page=page,
            rect=padded_rect,
            text=ch,
            fontname=fontname,
            font_obj=font_obj,
            min_fontsize=4.0,
            max_fontsize_factor=0.86,
        )

        if ok:
            ok_count += 1

    return ok_count > 0


def write_cjk_chars(page, rect, text, fontname, font_obj):
    """Render CJK text using horizontal or vertical character splitting."""
    if is_vertical_cjk_rect(rect, text):
        return write_cjk_vertical_chars(
            page=page,
            rect=rect,
            text=text,
            fontname=fontname,
            font_obj=font_obj,
        )

    char_boxes = split_cjk_text_to_weighted_char_boxes(text, rect, font_obj)

    if not char_boxes:
        return False

    ok_count = 0

    for ch, char_rect in char_boxes:
        pad_x = max(0.25, char_rect.height * 0.03)
        pad_y = max(0.25, char_rect.height * 0.05)

        padded_rect = fitz.Rect(
            char_rect.x0 - pad_x,
            char_rect.y0 - pad_y,
            char_rect.x1 + pad_x,
            char_rect.y1 + pad_y,
        )

        ok = fit_textbox(
            page=page,
            rect=padded_rect,
            text=ch,
            fontname=fontname,
            font_obj=font_obj,
            min_fontsize=4.0,
            max_fontsize_factor=0.86,
        )

        if ok:
            ok_count += 1

    return ok_count > 0


def write_block(page, block, font_objects):
    """Write one OCR block to the PDF page."""
    mode = block["mode"]
    text = block["text"]
    rect = block["rect"]
    font_alias = block["font_alias"]
    font_obj = font_objects[font_alias]

    if mode == "latin_line":
        return fit_textbox(
            page=page,
            rect=rect,
            text=text,
            fontname=font_alias,
            font_obj=font_obj,
            min_fontsize=4.0,
            max_fontsize_factor=0.78,
        )

    if mode == "cjk_chars":
        return write_cjk_chars(
            page=page,
            rect=rect,
            text=text,
            fontname=font_alias,
            font_obj=font_obj,
        )

    if mode == "paragraph":
        return fit_paragraph_textbox(
            page=page,
            rect=rect,
            text=text,
            fontname=font_alias,
            font_obj=font_obj,
        )

    return False


def build_pdf(json_path, input_dir, method, select):
    """Build one searchable PDF and one ALTO XML file from one OCR JSON."""
    json_path = Path(json_path)

    result = {
        "json": str(json_path),
        "pdf_success": False,
        "pdf_error": None,
        "pdf_traceback": None,
        "pdf_output": None,
        "alto_success": False,
        "alto_error": None,
        "alto_traceback": None,
        "alto_output": None,
        "image_used": None,
    }

    image_path = None
    doc = None

    try:
        image_path = find_image_for_json(json_path, input_dir)

        if image_path is None:
            raise FileNotFoundError(f"Image not found for JSON: {json_path.name}")

        result["image_used"] = str(image_path)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        texts, polys = extract_ocr_data(data)
        items = build_items(texts, polys, method, select)
        blocks = build_text_blocks(items, method, select)

        with Image.open(image_path) as img:
            width, height = img.size

        doc = fitz.open()
        page = doc.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(image_path))

        page.insert_font(fontname=LATIN_FONT_ALIAS, fontfile=str(LATIN_FONT_PATH))
        page.insert_font(fontname=CJK_FONT_ALIAS, fontfile=str(CJK_FONT_PATH))

        font_objects = {
            LATIN_FONT_ALIAS: fitz.Font(fontfile=str(LATIN_FONT_PATH)),
            CJK_FONT_ALIAS: fitz.Font(fontfile=str(CJK_FONT_PATH)),
        }

        failed_blocks = 0

        for block in blocks:
            ok = write_block(page, block, font_objects)
            if not ok:
                failed_blocks += 1

        output_pdf = json_path.parent / f"{json_path.stem.replace('_res', '')}.pdf"

        doc.save(
            str(output_pdf),
            deflate=True,
            garbage=3,
            clean=True,
        )
        doc.close()
        doc = None

        if not output_pdf.exists():
            raise FileNotFoundError(f"PDF was not created: {output_pdf}")

        if failed_blocks > 0:
            print(f"WARN: {json_path.name} had {failed_blocks} failed text blocks")

        result["pdf_success"] = True
        result["pdf_output"] = str(output_pdf)

    except Exception as exc:
        result["pdf_error"] = str(exc)
        result["pdf_traceback"] = traceback.format_exc()

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    try:
        if image_path is None:
            raise FileNotFoundError("Image missing, skip ALTO generation")

        generate_alto(
            json_file=json_path,
            image=image_path,
            dpi=300,
            output_dir=json_path.parent,
        )

        output_xml = json_path.parent / f"{json_path.stem}.xml"
        result["alto_success"] = True
        result["alto_output"] = str(output_xml)

    except Exception as exc:
        result["alto_error"] = str(exc)
        result["alto_traceback"] = traceback.format_exc()

        print("ALTO ERROR:", exc)
        print(result["alto_traceback"])

    return result


def main(output_dir, input_dir, method, select=None, log_dir="logs", recursive=True):
    """Process all OCR JSON files and generate PDFs plus ALTO XML files."""
    summary_logger, error_logger, summary_log_path, error_log_path = (
        OCRLoggerManger.setup(method)
    )

    output_dir = Path(output_dir)
    input_dir = Path(input_dir)

    if not output_dir.exists():
        error_logger.error(f"OCR output directory not found: {output_dir}")
        raise FileNotFoundError(f"OCR output directory not found: {output_dir}")

    worker_fn = partial(
        build_pdf,
        input_dir=input_dir,
        method=method,
        select=select,
    )

    pdf_success = 0
    pdf_fail = 0
    alto_success = 0
    alto_fail = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        json_files = collect_json_files(output_dir, recursive=recursive)

        for result in pool.map(worker_fn, json_files):
            if result["pdf_success"]:
                pdf_success += 1
            else:
                pdf_fail += 1
                error_logger.error(f"PDF FAIL: {result['json']}")
                error_logger.error(f"Image used: {result['image_used']}")
                error_logger.error(f"Error: {result['pdf_error']}")

                if result["pdf_traceback"]:
                    error_logger.error(result["pdf_traceback"])

            if result["alto_success"]:
                alto_success += 1
            else:
                alto_fail += 1
                error_logger.error(f"ALTO FAIL: {result['json']}")
                error_logger.error(f"Error: {result['alto_error']}")

                if result["alto_traceback"]:
                    error_logger.error(result["alto_traceback"])

    summary_logger.info("========== PDF AND ALTO SUMMARY ==========")
    summary_logger.info(f"PDF success: {pdf_success}, fail: {pdf_fail}")
    summary_logger.info(f"ALTO success: {alto_success}, fail: {alto_fail}")
    summary_logger.info(f"Summary log path: {summary_log_path}")
    summary_logger.info(f"Error log path: {error_log_path}")


if __name__ == "__main__":
    main()