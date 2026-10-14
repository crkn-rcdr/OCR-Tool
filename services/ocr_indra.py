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

    local_model_dir = Path(__file__).resolve().parent.parent / "models" / "indra"
    model_path = None
    if local_model_dir.exists() and (local_model_dir / "det.onnx").exists():
        model_path = local_model_dir
        os.environ.setdefault("INDRA_OCR_MODEL_DIR", str(local_model_dir))

    conv = indra_ocr.load(path=model_path, threads=threads, math=math)
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


def should_indra_tile(image_path: Path, tile_strategy: str = "auto") -> bool:
    """Determine whether to use 3x2 quadrant tiling for Indra OCR.

    Broadsheet newspapers contain micro-print multi-column classifieds where 3x2 tiling
    yields major recall benefits (+12.7%). Standard books, annuals, and microfilm pages
    run natively in single-pass mode in ~0.4s and capture more lines without tiling overhead.
    """
    if tile_strategy in {"grid_3x2", "grid"}:
        return True
    if tile_strategy in {"none", "no_tiles", "notiles"}:
        return False
    path_str = str(image_path).replace("\\", "/").lower()
    if "newspaper" in path_str:
        return True
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        return (w * h) >= 24_000_000 and min(w, h) >= 3800
    except Exception:
        return False


def process_image_with_indra(image_path: Path, conv, opt: dict) -> tuple[dict, str]:
    """Process one page with Indra OCR, using 3x2 quadrant tiling on large broadsheets."""
    tile_strategy = opt.get("tile_strategy", "auto")
    use_tiling = should_indra_tile(image_path, tile_strategy)

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


_worker_conv = None


def _init_worker(model_dir_str: Optional[str] = None, threads_per_worker: int = 1):
    global _worker_conv
    import indra_ocr
    model_path = Path(model_dir_str) if model_dir_str else None
    _worker_conv = indra_ocr.load(path=model_path, threads=threads_per_worker)


def _process_page_worker_task(task_args: tuple) -> dict:
    image_path, input_path, output_root, run_opt, generate_text_file, lang, method = task_args
    global _worker_conv
    if _worker_conv is None:
        import indra_ocr
        local_model_dir = Path(__file__).resolve().parent.parent / "models" / "indra"
        m_path = local_model_dir if local_model_dir.exists() else None
        _worker_conv = indra_ocr.load(path=m_path, threads=1)

    try:
        rel_dir = image_path.parent.relative_to(input_path) if input_path.is_dir() else Path("")
    except ValueError:
        rel_dir = Path("")

    out_dir = output_root / rel_dir if input_path.is_dir() else output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    json_file = out_dir / f"{stem}.json"
    t0 = time.perf_counter()

    try:
        payload, mode_desc = process_image_with_indra(image_path, _worker_conv, run_opt)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        if generate_text_file:
            txt_path = out_dir / f"{stem}.txt"
            generate_txt_file(json_file, txt_path)
            alto_success = False
            pdf_success = False
        else:
            pdf_res = build_pdf(
                json_file,
                image_path.parent,
                method=method,
                lang=lang,
                select=method,
            )
            alto_success = bool(pdf_res.get("alto_success"))
            pdf_success = bool(pdf_res.get("pdf_success"))

        rec_count = len(payload.get("rec_texts", []))
        elapsed = time.perf_counter() - t0
        return {
            "success": True,
            "stem": stem,
            "rel_dir": str(rel_dir),
            "file_name": image_path.name,
            "rec_count": rec_count,
            "mode_desc": mode_desc,
            "elapsed": elapsed,
            "alto_success": alto_success,
            "pdf_success": pdf_success,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "stem": stem,
            "rel_dir": str(rel_dir),
            "file_name": image_path.name,
            "elapsed": elapsed,
            "error": str(exc),
        }


def run_indra_job(input_path, output_root, opt, select=None) -> Generator[str, None, None]:
    """Run standalone Indra OCR with multi-process concurrency and real-time GUI streaming."""
    import concurrent.futures

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

        num_images = len(target_images)
        workers = min(4, os.cpu_count() or 4) if num_images > 1 else 1

        local_model_dir = Path(__file__).resolve().parent.parent / "models" / "indra"
        m_dir_str = str(local_model_dir) if local_model_dir.exists() else None

        per_page_timings: dict[str, float] = {}
        yield f"Found {num_images} images. Launching Indra OCR engine ({workers} parallel worker processes)..."

        if num_images == 1:
            conv = get_indra_converter()
            image_path = target_images[0]
            out_dir = output_root
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem
            json_file = out_dir / f"{stem}.json"
            t0 = time.perf_counter()
            payload, mode_desc = process_image_with_indra(image_path, conv, run_opt)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            elapsed_page = time.perf_counter() - t0
            per_page_timings[stem] = round(elapsed_page, 2)
            if generate_text_file:
                generate_txt_file(json_file, out_dir / f"{stem}.txt")
                yield f"Generated TXT: {stem}.txt ({len(payload['rec_texts'])} lines in {elapsed_page:.2f}s)"
            else:
                build_pdf(json_file, image_path.parent, method=method, lang=lang, select=method)
                yield f"Generated PDF/ALTO: {stem}.pdf ({len(payload['rec_texts'])} lines in {elapsed_page:.2f}s)"
            success_count = 1
        else:
            task_args = [
                (img, input_path, output_root, run_opt, generate_text_file, lang, method)
                for img in target_images
            ]

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(m_dir_str, 1),
            ) as executor:
                futures = {executor.submit(_process_page_worker_task, t): t[0] for t in task_args}
                completed = 0
                for fut in concurrent.futures.as_completed(futures):
                    if ocr_pp.STOP_FLAG:
                        executor.shutdown(wait=False, cancel_futures=True)
                        yield "OCR processing stopped by user."
                        break

                    completed += 1
                    res = fut.result()
                    overall_elapsed = time.perf_counter() - start_time
                    rate = completed / overall_elapsed if overall_elapsed > 0 else 0.0

                    if res["success"]:
                        success_count += 1
                        per_page_timings[Path(res["file_name"]).stem] = round(res["elapsed"], 2)
                        status = "ALTO [OK]" if res.get("alto_success") else "PDF [OK]"
                        yield f"[{completed}/{num_images}] {res['file_name']} -> {status} ({res['rec_count']} lines in {res['elapsed']:.2f}s | Avg: {rate:.1f} pgs/s)"
                        summary_logger.info(f"Processed {res['file_name']} in {res['elapsed']:.2f}s")
                    else:
                        failure_count += 1
                        yield f"[{completed}/{num_images}] ERROR on {res['file_name']}: {res.get('error')}"
                        error_logger.error(f"Error on {res['file_name']}: {res.get('error')}")

        # If processing a folder and benchmark ALTO files exist, trigger auto-comparison
        if input_path.is_dir() and not generate_text_file and success_count > 0:
            benchmark_has_eval_files = any(
                f.suffix.lower() in {".xml", ".pdf"} for f in input_path.glob("*") if f.is_file()
            )
            if benchmark_has_eval_files:
                try:
                    from tools.compare_alto import compare_directory_batch
                    compare_root = Path(r"C:\CRKN Projects\paddle_pipeline\compare_indra")
                    compared = compare_directory_batch(
                        pipeline_dir=output_root,
                        benchmark_dir=input_path,
                        output_dir=compare_root,
                        collection_name=f"IndraOCR/{input_path.name}",
                        render_html=True,
                        render_image=False,
                        pipeline_name="Indra OCR",
                        pipeline_short="Indra",
                    )
                    if compared > 0:
                        yield f"Auto-comparison updated for Indra OCR vs ABBYY ({compared} pages)"
                except Exception as exc:
                    error_logger.warning(f"Auto-comparison error for Indra OCR: {exc}")

        total_time = time.perf_counter() - start_time
        if run_opt.get("generate_results_dashboard", True) and success_count > 0:
            try:
                from services.results_dashboard import build_ocr_run_dashboard
                dash_path = build_ocr_run_dashboard(
                    output_dir=output_root,
                    source_images_dir=input_path,
                    engine_name="Indra OCR",
                    total_time_sec=total_time,
                    per_page_timings=per_page_timings,
                    auto_open=True,
                )
                if dash_path:
                    yield f"HTML Results Dashboard generated: {dash_path.resolve()}"
            except Exception as exc:
                error_logger.warning(f"Error generating HTML results dashboard: {exc}")

    finally:
        total_time = time.perf_counter() - start_time
        RUN_LOCK.release()
        yield (
            f"Indra OCR run finished: {success_count} succeeded, {failure_count} failed "
            f"in {total_time:.2f}s."
        )
