"""Run standalone Indra OCR (Fast CPU ONNX OCR) with automatic broadsheet tiling."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
from PIL import Image

from services.constants import IMAGE_EXTENSIONS
from services.ocr_defaults import DEFAULT_INDRA_THREADS
from services.ocr_logger import OCRLoggerManger, log_run_configuration
from services.pdf_generation import build_pdf
from services.tile_ocr import (
    build_image_tiles,
    merge_ppocr_tile_results,
    should_use_3x2_tiling,
)
from services.txt_generation import generate_txt_file

import services.ocr_pp as ocr_pp

RUN_LOCK = threading.Lock()
_CONVERTER_CACHE: dict = {}


def get_indra_converter(threads: Optional[int] = None, math: bool = False):
    """Lazy initialize and cache the Indra OCR Converter."""
    threads = threads or DEFAULT_INDRA_THREADS
    cache_key = (threads, math)
    if cache_key in _CONVERTER_CACHE:
        return _CONVERTER_CACHE[cache_key]

    import indra_ocr

    conv = indra_ocr.load(threads=threads, math=math)
    _CONVERTER_CACHE[cache_key] = conv
    return conv


def iter_image_files(input_path: Path):
    """Yield supported image files from a file or recursive folder input."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    for image_path in sorted(input_path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def build_indra_payload(image_path: Path, page, opt: dict) -> dict:
    """Build repository-compatible OCR JSON payload with line polygons and metadata."""
    with Image.open(image_path) as img:
        width, height = img.size

    rec_texts = []
    rec_scores = []
    rec_polys = []
    rec_boxes = []

    for ln in page.lines:
        text = str(ln.text or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = [float(v) for v in ln.box]
        # Clockwise 4-point polygon: top-left, top-right, bottom-right, bottom-left
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        rec_texts.append(text)
        rec_scores.append(round(float(ln.conf), 4))
        rec_polys.append(poly)
        rec_boxes.append([round(v, 1) for v in (x0, y0, x1, y1)])

    return {
        "input_path": str(image_path),
        "fileName": image_path.name,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys,
        "rec_boxes": rec_boxes,
        "markdown": getattr(page, "markdown", ""),
        "indra": {
            "version": "0.3.1",
            "source": getattr(page, "source", "ocr"),
            "note": getattr(page, "note", ""),
            "rotation": getattr(page, "rotation", 0),
            "skew": getattr(page, "skew", 0.0),
            "timings": getattr(page, "timings", {}),
            "blocks": getattr(page, "blocks", []),
            "width": width,
            "height": height,
        },
    }


def process_image_with_indra(image_path: Path, conv, opt: dict) -> tuple[dict, str]:
    """Process one page with Indra OCR, using 3x2 quadrant tiling on large broadsheets."""
    tile_strategy = opt.get("tile_strategy", "auto")
    use_tiling = (
        tile_strategy in {"grid_3x2", "grid"}
        or (tile_strategy == "auto" and should_use_3x2_tiling(image_path))
    )

    if use_tiling:
        with tempfile.TemporaryDirectory() as td:
            tiles = build_image_tiles(
                image_path,
                Path(td),
                grid_rows=3,
                grid_cols=2,
                overlap_px=96,
            )
            tile_results = []
            for tile in tiles:
                # Read tile as grayscale; use turn=False on cropped quadrants to avoid orientation confusion
                gray = cv2.imread(str(tile.path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                tile_page = conv.ocr(gray, turn=False)
                tile_payload = build_indra_payload(tile.path, tile_page, opt)
                tile_results.append({"tile": tile, "json": tile_payload})

            merged = merge_ppocr_tile_results(
                tile_results=tile_results,
                image_path=image_path,
                grid_rows=3,
                grid_cols=2,
                overlap_px=96,
                tile_strategy="grid_3x2",
            )
            merged["markdown"] = "\n\n".join(
                tr["json"].get("markdown", "")
                for tr in tile_results
                if tr["json"].get("markdown")
            )
            return merged, "auto-detected 3x2 quadrant tiling"

    # Standard full-image single pass
    pages = list(conv.convert(image_path))
    if not pages:
        raise RuntimeError(f"Indra OCR produced no pages for {image_path.name}")
    _, page = pages[0]

    if getattr(page, "source", None) == "error":
        raise RuntimeError(f"Indra OCR error: {page.note}")

    payload = build_indra_payload(image_path, page, opt)
    return payload, "full-image single pass"


def run_indra_job(input_path, output_root, opt, select=None) -> Generator[str, None, None]:
    """Run standalone Indra OCR and generate JSON plus TXT or PDF/ALTO outputs."""
    ocr_pp.STOP_FLAG = False
    input_path = Path(input_path)
    output_root = Path(output_root)
    generate_text_file = opt.get("generate_text_file", False)
    lang = opt.get("lang", "en")
    method = "IndraOCR"
    run_opt = dict(opt)
    run_opt.update(
        {
            "method": method,
            "clean_punctuation_issues": False,
            "post_ocr_vl_enabled": False,
            "preprocessing_steps": [],
            "ignore_black_border": False,
        }
    )

    summary_logger, error_logger, _summary_log_path, _error_log_path = (
        OCRLoggerManger.setup(method)
    )
    log_run_configuration(
        summary_logger,
        run_opt,
        select=select or method,
        input_path=input_path,
        output_root=output_root,
        stage="IndraOCR",
    )

    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

    success_count = 0
    failure_count = 0
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

        yield f"Initializing Indra OCR engine (threads: {DEFAULT_INDRA_THREADS})..."
        conv = get_indra_converter()

        yield f"Found {len(target_images)} images to process using Indra OCR."

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
                payload, mode_desc = process_image_with_indra(image_path, conv, run_opt)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

                if generate_text_file:
                    generate_txt_file(json_file, out_dir / f"{stem}.txt")
                    yield f"Generated TXT: {stem}.txt ({len(payload['rec_texts'])} lines, {mode_desc} in {time.perf_counter()-t0:.2f}s)"
                else:
                    pdf_res = build_pdf(
                        json_file,
                        image_path.parent,
                        method=method,
                        lang=lang,
                        select=method,
                    )
                    if not pdf_res.get("pdf_success"):
                        error_logger.error(f"PDF generation failed for {image_path.name}: {pdf_res.get('pdf_error')}")
                    if not pdf_res.get("alto_success"):
                        error_logger.error(f"ALTO generation failed for {image_path.name}: {pdf_res.get('alto_error')}")

                    yield f"Generated PDF/ALTO: {stem}.pdf ({len(payload['rec_texts'])} lines, {mode_desc} in {time.perf_counter()-t0:.2f}s)"

                success_count += 1
                elapsed = time.perf_counter() - t0
                summary_logger.info(
                    f"Successfully processed {image_path.name} ({len(payload['rec_texts'])} lines, {mode_desc}) in {elapsed:.2f}s"
                )

            except Exception as exc:
                failure_count += 1
                error_logger.error(
                    f"Failed to process {image_path.name}: {exc}", exc_info=True
                )
                yield f"Error processing {image_path.name}: {exc}"

        # If processing a folder and benchmark ALTO files exist, trigger auto-comparison
        if input_path.is_dir() and not generate_text_file and success_count > 0:
            benchmark_has_eval_files = any(
                f.suffix.lower() in {".xml", ".pdf"} for f in input_path.glob("*") if f.is_file()
            )
            if benchmark_has_eval_files:
                try:
                    from tools.compare_alto import compare_directory_batch
                    compare_root = Path(r"C:\CRKN Projects\paddle_pipeline\compare")
                    compared = compare_directory_batch(
                        pipeline_dir=output_root,
                        benchmark_dir=input_path,
                        output_dir=compare_root,
                        collection_name=f"IndraOCR/{input_path.name}",
                        render_html=True,
                        render_image=False,
                    )
                    if compared > 0:
                        yield f"Auto-comparison updated for Indra OCR vs ABBYY ({compared} pages)"
                except Exception as exc:
                    error_logger.warning(f"Auto-comparison error for Indra OCR: {exc}")

    finally:
        total_time = time.perf_counter() - start_time
        RUN_LOCK.release()
        yield (
            f"Indra OCR run finished: {success_count} succeeded, {failure_count} failed "
            f"in {total_time:.2f}s."
        )
