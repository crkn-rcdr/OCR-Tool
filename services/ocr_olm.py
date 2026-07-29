"""Run standalone olmOCR jobs through an OpenAI-compatible vision endpoint."""

import base64
import json
import os
import re
import threading
import time
from pathlib import Path

import requests
from PIL import Image

from services.constants import IMAGE_EXTENSIONS
from services.ocr_defaults import DEFAULT_OLMOCR_API_URL, DEFAULT_OLMOCR_MODEL
from services.ocr_logger import OCRLoggerManger, log_run_configuration
from services.pdf_generation import main as generate_pdf_and_alto
from services.txt_generation import generate_txt_file

import services.ocr_pp as ocr_pp


RUN_LOCK = threading.Lock()


def completion_url(server_url: str) -> str:
    """Return the OpenAI-compatible chat completions URL."""
    server_url = str(server_url or DEFAULT_OLMOCR_API_URL).rstrip("/")
    if server_url.endswith("/chat/completions"):
        return server_url
    return f"{server_url}/chat/completions"


def image_to_base64(image_path: Path) -> str:
    """Read an image file as a base64 data payload."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def image_mime_type(image_path: Path) -> str:
    """Return the image MIME type expected in OpenAI-compatible image URLs."""
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "image/png"


def extract_message_text(data: dict) -> str:
    """Extract text from common OpenAI-compatible response shapes."""
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def clean_olmocr_text(text: str) -> str:
    """Normalize wrapper markup without changing OCR content."""
    value = str(text or "").strip()
    fenced = re.search(
        r"```(?:text|markdown|md)?\s*(.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = fenced.group(1).strip()
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def text_lines(text: str) -> list[str]:
    """Split OCR text into non-empty lines."""
    lines = []
    for raw_line in clean_olmocr_text(text).split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return lines


def estimated_line_boxes(lines: list[str], width: int, height: int) -> list[list[int]]:
    """Create line-level boxes when the model returns text without geometry."""
    if not lines:
        return []

    margin_x = max(16, int(width * 0.04))
    margin_y = max(16, int(height * 0.04))
    usable_height = max(1, height - (margin_y * 2))
    line_height = max(12, int(usable_height / max(len(lines), 1)))
    box_height = max(10, int(line_height * 0.72))

    boxes = []
    for index, _line in enumerate(lines):
        y0 = margin_y + (index * line_height)
        y1 = min(height - margin_y, y0 + box_height)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        boxes.append([margin_x, y0, max(margin_x + 1, width - margin_x), y1])
    return boxes


def request_olmocr_page(image_path: Path, opt: dict) -> str:
    """Send one page image to olmOCR and return recognized text."""
    url = completion_url(opt.get("olmocr_url") or DEFAULT_OLMOCR_API_URL)
    model = str(opt.get("olmocr_model") or DEFAULT_OLMOCR_MODEL)
    mime_type = image_mime_type(image_path)
    prompt = (
        "Transcribe this newspaper page as OCR text. Return only the text in "
        "natural reading order. Preserve French accents, punctuation, line "
        "breaks, headings, and visible table text. Do not describe the image."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{image_to_base64(image_path)}"
                            )
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": int(opt.get("olmocr_max_tokens") or 8192),
    }
    headers = {"Content-Type": "application/json"}
    api_key = opt.get("olmocr_api_key") or os.getenv("OLMOCR_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(10, 600),
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the olmOCR service at {url}. "
            "Set OLMOCR_API_URL to an OpenAI-compatible vision endpoint."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    return clean_olmocr_text(extract_message_text(response.json()))


def build_olmocr_payload(image_path: Path, text: str, opt: dict) -> dict:
    """Build repo-compatible OCR JSON with estimated line geometry."""
    lines = text_lines(text)
    with Image.open(image_path) as img:
        width, height = img.size

    boxes = estimated_line_boxes(lines, width, height)
    return {
        "input_path": str(image_path),
        "fileName": image_path.name,
        "rec_texts": lines,
        "rec_scores": [1.0 for _ in lines],
        "rec_polys": boxes,
        "olmocr": {
            "api_url": str(opt.get("olmocr_url") or DEFAULT_OLMOCR_API_URL),
            "model": str(opt.get("olmocr_model") or DEFAULT_OLMOCR_MODEL),
            "geometry": "estimated_line_boxes",
            "pre_process_enabled": False,
            "post_process_enabled": False,
        },
    }


def iter_image_files(input_path: Path):
    """Yield supported image files from a file or recursive folder input."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    for image_path in input_path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def run_olmocr_job(input_path, output_root, opt, select=None):
    """Run standalone olmOCR and generate JSON plus TXT or PDF/ALTO outputs."""
    ocr_pp.STOP_FLAG = False
    input_path = Path(input_path)
    output_root = Path(output_root)
    generate_text_file = opt.get("generate_text_file", False)
    lang = opt.get("lang")
    method = opt.get("method") or select or "OLMOCR"
    run_opt = dict(opt)
    run_opt.update(
        {
            "method": method,
            "clean_punctuation_issues": False,
            "post_ocr_vl_enabled": False,
            "preprocessing_steps": [],
            "ignore_black_border": False,
            "tile_source_images": False,
        }
    )

    summary_logger, error_logger, _summary_log_path, _error_log_path = (
        OCRLoggerManger.setup(method)
    )
    log_run_configuration(
        summary_logger,
        run_opt,
        select=select,
        input_path=input_path,
        output_root=output_root,
        stage="olmOCR",
    )

    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

    success_count = 0
    failure_count = 0
    messages = []
    start_time = time.perf_counter()

    try:
        if not input_path.exists():
            yield f"Input path does not exist: {input_path}"
            return

        input_base = input_path if input_path.is_dir() else input_path.parent
        output_base = output_root / input_path.name if input_path.is_dir() else output_root

        for image_path in iter_image_files(input_path):
            if ocr_pp.STOP_FLAG:
                return

            try:
                rel = image_path.relative_to(input_base)
                out_dir = output_base / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                text = request_olmocr_page(image_path, run_opt)
                data = build_olmocr_payload(image_path, text, run_opt)
                json_path = out_dir / f"{image_path.stem}.json"
                json_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                success_count += 1
                messages.append(f"{image_path.name} created and saved in {out_dir}")
                yield "\n".join(messages)

                if generate_text_file:
                    txt_path = generate_txt_file(json_path)
                    messages.append(f"Text file created: {txt_path}")
                    yield "\n".join(messages)
                else:
                    generate_pdf_and_alto(
                        json_path,
                        image_path,
                        method,
                        lang,
                        select,
                        run_options=run_opt,
                    )
                    yield "\n".join(messages)

            except Exception as exc:
                error_logger.error(
                    f"Error processing {image_path.name}: {exc}",
                    exc_info=True,
                )
                failure_count += 1
                messages.append(f"Error processing {image_path.name}: {exc}")
                yield "\n".join(messages)

        if success_count + failure_count == 0:
            yield "No supported image files found in the selected folder."
            return

        total_time = time.perf_counter() - start_time
        total_files = success_count + failure_count
        avg_time = total_time / total_files if total_files else 0.0
        summary_logger.info(f"========== {method} SUMMARY ==========")
        summary_logger.info(f"Total Successful JSON files: {success_count}")
        summary_logger.info(f"Total Failed JSON files: {failure_count}")
        summary_logger.info(f"Total Processing Time: {total_time:.2f} seconds")
        summary_logger.info(f"Average Time per File: {avg_time:.2f} seconds")

        if failure_count:
            yield "OCR partially completed. Some errors occurred during processing."
        elif generate_text_file:
            yield "OCR Completed, please check json and text files at your output folder."
        else:
            yield "OCR Completed, please check pdf and alto files at your output folder"

    finally:
        RUN_LOCK.release()
