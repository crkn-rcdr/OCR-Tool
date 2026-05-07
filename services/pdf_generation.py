import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import  ThreadPoolExecutor
from functools import partial

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
    text: str
    poly: list
    rect: fitz.Rect
    center_x: float
    center_y: float
    width: float
    height: float


def polygon_to_rect(poly):
    if hasattr(poly, "tolist"):
        poly = poly.tolist()

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

def bbox_to_rect(bbox):
    x0, y0, x1, y1 = bbox
    return fitz.Rect(float(x0), float(y0), float(x1), float(y1))

def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_cjk(text):
    if not text:
        return False
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def choose_font_alias(text):
    return CJK_FONT_ALIAS if contains_cjk(text) else LATIN_FONT_ALIAS


def is_mostly_cjk(text):
    if not text:
        return False

    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False

    cjk_count = sum(1 for ch in chars if contains_cjk(ch))
    return cjk_count / len(chars) >= 0.5

def collect_json_files(output_dir,recursive = True):
    if output_dir.is_file():
        if output_dir.suffix.lower() == ".json":
            yield output_dir
        return

    pattern = output_dir.rglob("*.json") if recursive else output_dir.glob("*.json")

    for p in pattern:
        yield p

def find_image_for_json(json_path,input_dir):
    image_name = json_path.stem.replace("_res", "") + ".jpg"

    if input_dir.is_file():
        return input_dir if input_dir.name == image_name else None

    matches = list(input_dir.rglob(image_name))
    return matches[0] if matches else None

def extract_ocr_data(data):
    if "overall_ocr_res" in data:
        ocr_res = data["overall_ocr_res"]
        texts = ocr_res["rec_texts"]
        polys = ocr_res["rec_polys"]
        
    elif "result" in data:
        texts = []
        polys = []

        parsing_res_list = data["result"]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"]
        for block in parsing_res_list:
            text = block.get("block_content")
            poly = block.get("block_bbox")
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


def build_items(texts, polys,method,select):
    items = []
  
    for text, poly in zip(texts, polys):
        text = normalize_text(text)
        if not text:
            continue

        if method in ["PP-StructureV3","PP-OCRv5"] or select == "PP-StructureV3":
            rect = polygon_to_rect(poly)
        elif method == "PaddleVL" or select == "PaddleVL":
            rect = bbox_to_rect(poly)

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
    if not items:
        return []

    median_height = sorted(i.height for i in items)[len(items) // 2]
    line_threshold = max(8.0, median_height * 0.6)

    items = sorted(items, key=lambda i: (i.center_y, i.rect.x0))

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

    lines.sort(key=lambda line: min(i.rect.y0 for i in line))

    ordered = []
    for line in lines:
        ordered.extend(line)

    return ordered


def group_items_into_lines(items):
    if not items:
        return []

    median_height = sorted(i.height for i in items)[len(items) // 2]
    line_threshold = max(8.0, median_height * 0.6)

    items = sorted(items, key=lambda i: (i.center_y, i.rect.x0))

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

    lines.sort(key=lambda line: min(i.rect.y0 for i in line))
    return lines


def merge_latin_line_items(line_items):
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


def build_text_blocks(items):
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


def fit_textbox(page, rect, text, fontname, font_obj, min_fontsize=4.0, max_fontsize_factor=0.78):
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

    for _ in range(16):
        rc = page.insert_textbox(
            rect,
            text,
            fontname=fontname,
            fontsize=fontsize,
            render_mode=0,
            overlay=True,
            align=0,
        )
        if rc is not None and rc >= 0:
            return True
        fontsize *= 0.94

    return False


def measure_char_weights(text, font_obj, fontsize):
    weights = []
    for ch in text:
        if ch.isspace():
            weights.append(0.0)
            continue

        w = font_obj.text_length(ch, fontsize=fontsize)
        if w <= 0:
            w = fontsize * 0.6

        weights.append(max(w, fontsize * 0.15))

    return weights


def split_cjk_text_to_weighted_char_boxes(text, rect, font_obj):
    chars = list(text)
    visible_chars = [ch for ch in chars if not ch.isspace()]
    if not visible_chars:
        return []

    probe_fontsize = max(6.0, rect.height * 0.82)
    weights = measure_char_weights(chars, font_obj, probe_fontsize)

    visible_weights = [w for ch, w in zip(chars, weights) if not ch.isspace()]
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
    for ch, w in zip(chars, weights):
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
        boxes[-1] = (last_ch, fitz.Rect(last_rect.x0, last_rect.y0, rect.x1, last_rect.y1))

    return boxes


def write_cjk_chars(page, rect, text, fontname, font_obj):
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

    return False


def build_pdf(json_path,input_dir,method,select):
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
        image_path = find_image_for_json(json_path,input_dir)
      
        if image_path is None:
            raise FileNotFoundError(f"Image not found for JSON: {json_path.name}")

        result["image_used"] = str(image_path)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        texts, polys = extract_ocr_data(data)
        items = build_items(texts, polys,method,select)
        blocks = build_text_blocks(items)

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
    


def main(output_dir, input_dir,method,select=None,log_dir="logs",recursive=True):
 
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
       
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)

    if not output_dir.exists():
        error_logger.error(f"OCR output directory not found: {output_dir}")
        raise FileNotFoundError(f"OCR output directory not found: {output_dir}")

    worker_fn = partial(build_pdf,input_dir=input_dir,method=method,select=select)

    pdf_success = 0
    pdf_fail = 0
    alto_success = 0
    alto_fail = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        json_files = collect_json_files(output_dir,recursive=recursive)

        for result in pool.map(worker_fn,json_files):
            
            # -------- PDF --------
            if result["pdf_success"]:
                pdf_success += 1
                
            else:
                pdf_fail += 1
                error_logger.error(f"PDF FAIL: {result['json']}")
                error_logger.error(f"Image used: {result['image_used']}")
                error_logger.error(f"Error: {result['pdf_error']}")

                if result["pdf_traceback"]:
                    error_logger.error(result["pdf_traceback"])

            # -------- ALTO --------
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