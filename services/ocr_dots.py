"""Run standalone Dots OCR jobs through an OpenAI-compatible vision endpoint or local runner."""

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
from services.ocr_defaults import DEFAULT_DOTSOCR_API_URL, DEFAULT_DOTSOCR_MODEL
from services.ocr_logger import OCRLoggerManger, log_run_configuration
from services.pdf_generation import build_pdf
from services.txt_generation import generate_txt_file

import services.ocr_pp as ocr_pp


RUN_LOCK = threading.Lock()


def completion_url(server_url: str) -> str:
    """Return the OpenAI-compatible chat completions URL."""
    server_url = str(server_url or DEFAULT_DOTSOCR_API_URL).rstrip("/")
    if server_url.endswith("/chat/completions"):
        return server_url
    return f"{server_url}/chat/completions"


_OPENAI_MODEL_CACHE: dict[str, str] = {}


def resolve_openai_model(
    base_url: str,
    requested_model: str,
    headers: dict | None = None,
) -> str:
    """Resolve the active model on an OpenAI-compatible server if requested model is unavailable."""
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/chat/completions"):
        clean_base = clean_base[:-len("/chat/completions")]

    if clean_base in _OPENAI_MODEL_CACHE:
        return _OPENAI_MODEL_CACHE[clean_base]

    try:
        models_url = f"{clean_base}/models"
        resp = requests.get(models_url, headers=headers or {}, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            available = [m.get("id") for m in data.get("data", []) if m.get("id")]
            if requested_model in available:
                _OPENAI_MODEL_CACHE[clean_base] = requested_model
                return requested_model
            if available:
                chosen = available[0]
                _OPENAI_MODEL_CACHE[clean_base] = chosen
                return chosen
    except Exception:
        pass
    return requested_model


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


def clean_dotsocr_text(text: str) -> str:
    """Normalize wrapper markup without changing OCR content."""
    value = str(text or "").strip()
    fenced = re.match(
        r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$",
        value,
        flags=re.DOTALL,
    )
    if fenced:
        value = fenced.group(1).strip()
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def text_lines(text: str) -> list[str]:
    """Split OCR text into non-empty lines."""
    lines = []
    for raw_line in clean_dotsocr_text(text).split("\n"):
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


def request_dotsocr_page(image_path: Path, opt: dict) -> str:
    """Send one page image to Dots OCR and return recognized text."""
    url = completion_url(opt.get("dotsocr_url") or DEFAULT_DOTSOCR_API_URL)
    model = str(opt.get("dotsocr_model") or DEFAULT_DOTSOCR_MODEL)
    mime_type = image_mime_type(image_path)
    prompt = (
        "Transcribe this historical document/newspaper page as accurate OCR text. "
        "Preserve French accents, historical ligatures, punctuation, line breaks, "
        "multi-column reading order, headings, and tables. Return only the transcription."
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
        "max_tokens": int(opt.get("dotsocr_max_tokens") or 8192),
    }
    headers = {"Content-Type": "application/json"}
    api_key = opt.get("dotsocr_api_key") or os.getenv("DOTSOCR_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model = resolve_openai_model(
        url,
        str(opt.get("dotsocr_model") or DEFAULT_DOTSOCR_MODEL),
        headers=headers,
    )
    payload["model"] = model

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(10, 600),
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the Dots OCR service at {url}. "
            "Set DOTSOCR_API_URL to an OpenAI-compatible vision endpoint (e.g., vLLM or SGLang)."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    return clean_dotsocr_text(extract_message_text(response.json()))


def build_dotsocr_payload(image_path: Path, text: str, opt: dict) -> dict:
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
        "dotsocr": {
            "api_url": str(opt.get("dotsocr_url") or DEFAULT_DOTSOCR_API_URL),
            "model": str(opt.get("dotsocr_model") or DEFAULT_DOTSOCR_MODEL),
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


def run_dotsocr_job(input_path, output_root, opt, select=None):
    """Run standalone Dots OCR and generate JSON plus TXT or PDF/ALTO outputs."""
    ocr_pp.STOP_FLAG = False
    input_path = Path(input_path)
    output_root = Path(output_root)
    generate_text_file = opt.get("generate_text_file", False)
    lang = opt.get("lang")
    method = opt.get("method") or select or "DotsOCR"
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
        stage="DotsOCR",
    )

    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

    success_count = 0
    failure_count = 0
    messages = []
    per_page_timings: dict[str, float] = {}
    start_time = time.perf_counter()

    try:
        if not input_path.exists():
            error_msg = f"Input path does not exist: {input_path}"
            error_logger.error(error_msg)
            yield error_msg
            return

        target_images = list(iter_image_files(input_path))
        if not target_images:
            msg = f"No supported images found in: {input_path}"
            error_logger.warning(msg)
            yield msg
            return

        yield f"Found {len(target_images)} images to process using Dots OCR."

        for image_path in target_images:
            if ocr_pp.STOP_FLAG:
                yield "OCR processing stopped by user."
                break

            out_dir = (
                output_root
                if input_path.is_file()
                else output_root / image_path.parent.relative_to(input_path)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem
            json_file = out_dir / f"{stem}.json"
            t0 = time.perf_counter()

            try:
                recognized_text = request_dotsocr_page(image_path, run_opt)
                payload = build_dotsocr_payload(image_path, recognized_text, run_opt)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

                pdf_res = build_pdf(
                    json_file,
                    image_path.parent,
                    method="Dots OCR",
                    lang=lang,
                    select="Dots OCR",
                )
                if pdf_res.get("pdf_success") and pdf_res.get("alto_success"):
                    yield f"Generated PDF/ALTO: {stem}.pdf"
                else:
                    yield f"Generated PDF/ALTO (warnings logged): {stem}.pdf"

                if generate_text_file:
                    generate_txt_file(json_file, out_dir / f"{stem}.txt")
                    yield f"Generated TXT: {stem}.txt"

                success_count += 1
                elapsed = time.perf_counter() - t0
                per_page_timings[stem] = round(elapsed, 2)
                summary_logger.info(
                    f"Successfully processed {image_path.name} in {elapsed:.2f}s"
                )

            except Exception as exc:
                failure_count += 1
                error_logger.error(
                    f"Failed to process {image_path.name}: {exc}", exc_info=True
                )
                yield f"Error processing {image_path.name}: {exc}"

        total_time = time.perf_counter() - start_time
        if run_opt.get("generate_results_dashboard", True) and success_count > 0:
            try:
                from services.results_dashboard import build_ocr_run_dashboard
                dash_path = build_ocr_run_dashboard(
                    output_root=output_root,
                    input_root=input_path,
                    engine_name="Dots OCR",
                    total_time_sec=total_time,
                    per_page_timings=per_page_timings,
                    auto_open=True,
                    dashboard_folder_name="HTML_Results_Dashboard",
                )
                if dash_path:
                    yield f"HTML Results Dashboard generated: {dash_path.resolve()}"
            except Exception as exc:
                error_logger.warning(f"Error generating HTML results dashboard: {exc}")

    finally:
        total_time = time.perf_counter() - start_time
        RUN_LOCK.release()
        yield (
            f"Dots OCR run finished: {success_count} succeeded, {failure_count} failed "
            f"in {total_time:.2f}s."
        )
