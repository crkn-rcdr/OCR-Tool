"""Run local Paddle OCR jobs, including tiling, normalization, and outputs."""

import copy
import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from engine import engine
import traceback
import time

from services.constants import IMAGE_EXTENSIONS
from services.pdf_generation import build_pdf, main as generate_pdf_and_alto
from services.ocr_logger import OCRLoggerManger, log_run_configuration
from services.txt_generation import generate_txt_file 
from services.image_preprocessing import preprocess_image_for_ocr
from services.tile_ocr import (
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_TILE_OVERLAP_PX,
    DEFAULT_TILE_STRATEGY,
    offset_nested_geometry,
    scale_nested_geometry,
    tile_grid_for_strategy,
    run_tiled_local_ocr,
    supports_tiled_ocr,
    should_use_3x2_tiling,
    normalize_tile_strategy,
)
from services.post_ocr_vl import apply_post_ocr_vl_check, supports_post_ocr_vl
from services.punctuation_cleanup import clean_ocr_json_text_in_place
from services.word_box_normalization import normalize_ocr_word_boxes_in_payload
from services.ocr_defaults import DEFAULT_TEXT_DET_MAX_SIDE_LIMIT, SAFE_FALLBACK_TEXT_DET_MAX_SIDE_LIMIT, SAFE_FALLBACK_TEXT_RECOGNITION_BATCH_SIZE
from services.language_detection import resolve_effective_language

PADDLE_OCR_METHODS = {"PP-OCRv6"}

# Thread synchronization lock used to ensure that
# only one OCR job can run at the same time.
#
# This prevents:
# - Multiple OCR tasks from running concurrently
# - GPU/CPU resource conflicts
# - Duplicate file writing
# - Race conditions during OCR processing
#
# Example:
# If one OCR task is already running,
# another task attempting to start will be rejected.
RUN_LOCK = threading.Lock()

# Global flag to signal stopping the OCR process
STOP_FLAG = False

GPU_RECOVERY_ERROR_MARKERS = (
    "cudnn_status_not_supported",
    "cudnn error(3000)",
    "cuda error(2)",
    "cudaerrormemoryallocation",
    "out of memory",
)


def describe_tiling(opt):
    """Return a readable description of the active tiling strategy."""
    strategy = normalize_tile_strategy(opt.get("tile_strategy", DEFAULT_TILE_STRATEGY))
    if strategy == "none" or not opt.get("tile_source_images", False):
        return "full-image pass (no tiles)"
    if strategy == "auto":
        return "auto-detected 3x2 quadrant tiling"

    rows, cols = tile_grid_for_strategy(strategy)
    grid_label = f"{rows}x{cols}"

    if strategy == "doclayout_v3":
        padding_value = opt.get("tile_doclayout_padding_px")
        padding = int(
            DEFAULT_DOCLAYOUT_PADDING_PX
            if padding_value is None
            else padding_value
        )
        threshold = float(opt.get("tile_doclayout_threshold", DEFAULT_DOCLAYOUT_THRESHOLD))
        padding_label = f", {padding}px padding" if padding > 0 else ""
        return f"PP-DocLayoutV3 tiled OCR{padding_label}, threshold {threshold:g}"

    if strategy in {"grid_doclayout_v3", "grid_3x2_doclayout_v3"}:
        padding_value = opt.get("tile_doclayout_padding_px")
        padding = int(
            DEFAULT_DOCLAYOUT_PADDING_PX
            if padding_value is None
            else padding_value
        )
        overlap_value = opt.get("tile_overlap_px")
        overlap = int(
            DEFAULT_TILE_OVERLAP_PX
            if overlap_value is None
            else overlap_value
        )
        threshold = float(opt.get("tile_doclayout_threshold", DEFAULT_DOCLAYOUT_THRESHOLD))
        overlap_label = f", {overlap}px band overlap" if overlap > 0 else ""
        padding_label = f", {padding}px layout padding" if padding > 0 else ""
        return (
            f"{grid_label} + PP-DocLayoutV3 tiled OCR"
            f"{overlap_label}{padding_label}, threshold {threshold:g}"
        )

    overlap_value = opt.get("tile_overlap_px")
    overlap = int(
        DEFAULT_TILE_OVERLAP_PX
        if overlap_value is None
        else overlap_value
    )
    overlap_label = f", {overlap}px overlap" if overlap > 0 else ""
    return f"{grid_label} tiled OCR{overlap_label}"


def stop_ocr():
    """
    Stop the currently running OCR job.

    Sets the global STOP_FLAG to True so that the OCR loop
    can terminate gracefully.
    """
    global STOP_FLAG
    STOP_FLAG = True


def is_recoverable_gpu_ocr_error(exc: Exception, opt: dict) -> bool:
    """Return whether an OCR exception should trigger a safer retry path."""
    device = str(opt.get("device") or "").strip().lower()
    if device.startswith("cpu"):
        return False

    message = str(exc).lower()
    return any(marker in message for marker in GPU_RECOVERY_ERROR_MARKERS)


def release_paddle_gpu_memory() -> None:
    """Best-effort cleanup after a failed GPU OCR attempt."""
    gc.collect()

    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def build_safe_ppocr_retry_options(opt: dict) -> list[tuple[dict, str]]:
    """Build safer retry configurations for PaddleOCR GPU/CPU recovery."""
    if opt.get("method") not in PADDLE_OCR_METHODS:
        return []

    base = copy.deepcopy(opt)
    safe_side_limit = SAFE_FALLBACK_TEXT_DET_MAX_SIDE_LIMIT
    base_limit = int(base.get("text_det_limit_side_len") or safe_side_limit)
    base_max_side = int(base.get("text_det_max_side_limit") or safe_side_limit)
    base_rec_batch = int(base.get("text_recognition_batch_size") or 1)

    attempts: list[tuple[dict, str]] = []

    reduced_gpu = copy.deepcopy(base)
    reduced_gpu["device"] = "gpu"
    reduced_gpu["text_det_limit_side_len"] = min(base_limit, safe_side_limit)
    reduced_gpu["text_det_max_side_limit"] = min(base_max_side, safe_side_limit)
    reduced_gpu["text_recognition_batch_size"] = min(base_rec_batch, SAFE_FALLBACK_TEXT_RECOGNITION_BATCH_SIZE)
    if reduced_gpu != base:
        attempts.append(
            (
                reduced_gpu,
                "completed after retrying on GPU with reduced detector limits",
            )
        )

    cpu_fallback = copy.deepcopy(reduced_gpu)
    cpu_fallback["device"] = "cpu"
    cpu_fallback["text_recognition_batch_size"] = min(
        int(cpu_fallback.get("text_recognition_batch_size") or 1),
        4,
    )
    attempts.append(
        (
            cpu_fallback,
            "completed using CPU fallback after GPU memory/cuDNN failure",
        )
    )

    return attempts


def cleanup_partial_ocr_json_outputs(out_dir: Path, stem: str) -> None:
    """Remove partial OCR JSON outputs before a retry attempt."""
    for candidate in out_dir.glob(f"{stem}*.json"):
        try:
            candidate.unlink()
        except OSError:
            pass


def save_local_ocr_results_with_recovery(
    image_path: Path,
    out_dir: Path,
    opt: dict,
) -> str | None:
    """Run local OCR and retry with safer settings on recoverable GPU failures."""
    cleanup_partial_ocr_json_outputs(out_dir, image_path.stem)

    attempt_queue: list[tuple[dict, str | None]] = [(opt, None)]
    attempt_queue.extend(build_safe_ppocr_retry_options(opt))
    allow_retries = True
    last_exc: Exception | None = None

    for attempt_opt, success_note in attempt_queue:
        cleanup_partial_ocr_json_outputs(out_dir, image_path.stem)
        try:
            output = engine.infer(str(image_path), attempt_opt)
            for res in output:
                res.save_to_json(save_path=str(out_dir))
            return success_note
        except Exception as exc:
            last_exc = exc
            if allow_retries and is_recoverable_gpu_ocr_error(exc, attempt_opt):
                allow_retries = False
                release_paddle_gpu_memory()
                continue
            if success_note is not None:
                release_paddle_gpu_memory()
                continue
            raise

    if last_exc is not None:
        raise last_exc

    raise RuntimeError(f"OCR did not produce output for image: {image_path.name}")


def run_ocr_job(input_path, output_root, opt,select=None):
    """
    Run OCR processing on either a single image or a folder of images.

    This function:
    - Performs OCR using the configured OCR engine
    - Saves OCR JSON results
    - Optionally generates TXT files
    - Generates PDF and ALTO XML outputs
    - Logs summary and error information

    Args:
        input_path (Path):
            Input image file or folder containing images.

        output_root (Path):
            Root output directory.

        opt (dict):
            OCR configuration options.

        select (str, optional):
            Optional effective OCR engine label for delegated callers.

    Yields:
        str:
            Progress and status messages during OCR execution.
    """

    method = opt.get("method")
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
    log_run_configuration(
        summary_logger,
        opt,
        select=select,
        input_path=input_path,
        output_root=output_root,
        stage="OCR",
    )
    generate_text_file = opt.get("generate_text_file", False)
    lang = opt.get("lang", "None")
    tile_source_images = opt.get("tile_source_images", False)
   
    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

    try:
        global STOP_FLAG
        STOP_FLAG = False

        output_root.mkdir(parents=True, exist_ok=True)

        success_count = 0
        failure_count = 0
        messages = []
        per_page_timings: dict[str, float] = {}

        start_time = time.perf_counter()

        if input_path.is_file():
            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                yield f"Input path is a file but not a supported image type: {input_path}"
                return

            input_base = input_path.parent
            output_base = output_root

            try:
                rel = input_path.relative_to(input_base)

                out_dir = output_base / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                effective_lang = resolve_effective_language(lang, input_path)
                file_opt = dict(opt, lang=effective_lang)

                t_page = time.perf_counter()
                final_json, recovery_note = run_ocr_for_image(
                    image_path=input_path,
                    out_dir=out_dir,
                    opt=file_opt,
                    select=select,
                )
                per_page_timings[input_path.stem] = round(time.perf_counter() - t_page, 2)

                success_count += 1
                if recovery_note:
                    messages.append(f"{input_path.name} {recovery_note}")
                if tile_source_images and supports_tiled_ocr(method, select):
                    messages.append(
                        f"{input_path.name} created from {describe_tiling(file_opt)} and saved in {out_dir}"
                    )
                else:
                    messages.append(f"{input_path.name} created and saved in {out_dir}")

                # Once single-file OCR is complete, PDF/ALTO will be generated immediately.
                generate_pdf_and_alto(
                    final_json,
                    input_path,
                    method,
                    effective_lang,
                    select,
                    run_options=file_opt,
                )
                yield "\n".join(messages)

                if generate_text_file:
                    txt_path = generate_txt_file(final_json)
                    messages.append(f"Text file created: {txt_path}")
                    yield "\n".join(messages)

            except Exception as e:
                error_logger.error(
                    f"Error processing {input_path.name}: {str(e)}",
                    exc_info=True
                )
                failure_count += 1
                messages.append(f"Error processing {input_path.name}: {str(e)}")
                yield "\n".join(messages)

                summary_logger.info(f"{input_path.name} failed during PDF/ALTO generation")

        elif input_path.is_dir():
            input_base = input_path
            out_root_norm = output_root.name.lower().replace("-", " ").replace("_", " ")
            in_path_norm = input_path.name.lower().replace("-", " ").replace("_", " ")
            if out_root_norm == in_path_norm or out_root_norm.startswith(in_path_norm):
                output_base = output_root
            else:
                output_base = output_root / input_path.name

            # Group by the subfolder where the images are located.
            folders = {}

            for img in input_path.rglob("*"):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    folders.setdefault(img.parent, []).append(img)

            import re

            def natural_sort_key(s):
                return [
                    int(text) if text.isdigit() else text.lower()
                    for text in re.split(r"(\d+)", str(s))
                ]

            COLLECTION_PRIORITY = {"newspapers": 0, "annuals": 1, "heritage": 2}

            def folder_sort_key(folder_path):
                path_str = str(folder_path).lower().replace("\\", "/")
                priority = 99
                for name, p in COLLECTION_PRIORITY.items():
                    if f"/{name}/" in path_str or path_str.endswith(f"/{name}") or f"\\{name}\\" in str(folder_path).lower():
                        priority = p
                        break
                return (priority, natural_sort_key(folder_path))

            sorted_folders = sorted(
                folders.items(), key=lambda item: folder_sort_key(item[0])
            )

            # OCR for each subfolder with overlapping prefetch & background PDF/ALTO generation
            for folder, images in sorted_folders:
                images = sorted(images, key=lambda p: natural_sort_key(p.name))
                rel_folder = folder.relative_to(input_base)
                if folder.parent.name.lower() in {"newspapers", "annuals", "heritage"}:
                    if output_root.name.lower() == folder.parent.name.lower():
                        out_dir = output_root / folder.name
                    else:
                        out_dir = output_root / folder.parent.name / folder.name
                else:
                    out_dir = output_base / rel_folder
                out_dir.mkdir(parents=True, exist_ok=True)

                effective_lang = resolve_effective_language(lang, folder)
                folder_opt = dict(opt, lang=effective_lang)
                if not str(folder_opt.get("device") or "").lower().startswith("cpu"):
                    folder_opt["text_recognition_batch_size"] = max(32, int(folder_opt.get("text_recognition_batch_size") or 32))

                folder_success = 0
                folder_failure = 0
                folder_pdf_success = 0
                folder_pdf_fail = 0
                folder_alto_success = 0
                folder_alto_fail = 0

                messages.append(f"Start processing folder: {folder} (Language: {effective_lang})")
                yield "\n".join(messages)

                # Initialize background workers for this folder:
                # 1. prefetch_pool: 4 concurrent CPU workers for image reading & preprocessing across a sliding window
                # 2. post_pool: 4 concurrent CPU workers for PDF & ALTO generation overlapping next images' OCR
                PREFETCH_WINDOW = 4
                with ThreadPoolExecutor(max_workers=4) as post_pool, ThreadPoolExecutor(max_workers=4) as prefetch_pool:
                    pending_post = {}
                    prefetch_futures = {}

                    # Pre-populate prefetch window
                    for p_idx in range(min(PREFETCH_WINDOW, len(images))):
                        p_img = images[p_idx]
                        prefetch_futures[p_img] = prefetch_pool.submit(
                            preprocess_image_for_ocr, p_img, out_dir, folder_opt
                        )

                    for img_idx, img in enumerate(images):
                        if STOP_FLAG:
                            return

                        # Maintain 4-worker sliding lookahead window
                        next_submit_idx = img_idx + PREFETCH_WINDOW
                        if next_submit_idx < len(images):
                            next_img = images[next_submit_idx]
                            prefetch_futures[next_img] = prefetch_pool.submit(
                                preprocess_image_for_ocr, next_img, out_dir, folder_opt
                            )

                        preprocessed_res = None
                        fut = prefetch_futures.pop(img, None)
                        if fut is not None:
                            try:
                                preprocessed_res = fut.result()
                            except Exception as exc:
                                error_logger.warning(
                                    f"Prefetch warning for {img.name}: {exc}; falling back to sync preprocessing"
                                )
                                preprocessed_res = None

                        try:
                            t_page = time.perf_counter()
                            final_json, recovery_note = run_ocr_for_image(
                                image_path=img,
                                out_dir=out_dir,
                                opt=folder_opt,
                                select=select,
                                preprocessed_result=preprocessed_res,
                            )
                            per_page_timings[img.stem] = round(time.perf_counter() - t_page, 2)

                            success_count += 1
                            folder_success += 1

                            if recovery_note:
                                messages.append(f"{img.name} {recovery_note}")
                            effective_strat = normalize_tile_strategy(folder_opt.get("tile_strategy", DEFAULT_TILE_STRATEGY))
                            if effective_strat == "auto":
                                used_tiling = should_use_3x2_tiling(img)
                                t_desc = "auto-detected 3x2 quadrant tiling" if used_tiling else "auto-detected full-image pass"
                                messages.append(f"{img.name} created ({t_desc}) and saved in {out_dir}")
                            elif tile_source_images and supports_tiled_ocr(method, select):
                                messages.append(
                                    f"{img.name} created from {describe_tiling(folder_opt)} and saved in {out_dir}"
                                )
                            else:
                                messages.append(f"{img.name} created and saved in {out_dir}")

                            # Immediately submit background PDF/ALTO and optional TXT generation
                            fut_pdf = post_pool.submit(
                                build_pdf,
                                final_json,
                                folder,
                                method,
                                effective_lang,
                                select,
                            )
                            pending_post[fut_pdf] = ("pdf", img.name, final_json)

                            if generate_text_file:
                                fut_txt = post_pool.submit(generate_txt_file, final_json)
                                pending_post[fut_txt] = ("txt", img.name, final_json)

                            # Drain any post-tasks that completed while OCR was running
                            for done_fut in [f for f in pending_post if f.done()]:
                                task_type, item_name, _ = pending_post.pop(done_fut)
                                try:
                                    res = done_fut.result()
                                    if task_type == "pdf":
                                        if res.get("pdf_success"):
                                            folder_pdf_success += 1
                                        else:
                                            folder_pdf_fail += 1
                                            error_logger.error(f"PDF FAIL for {item_name}: {res.get('pdf_error')}")
                                        if res.get("alto_success"):
                                            folder_alto_success += 1
                                        else:
                                            folder_alto_fail += 1
                                            error_logger.error(f"ALTO FAIL for {item_name}: {res.get('alto_error')}")
                                        messages.append(f"PDF/ALTO generated: {item_name}")
                                    elif task_type == "txt":
                                        messages.append(f"Text file created: {Path(res).name}")
                                except Exception as exc:
                                    error_logger.error(f"Post-processing error for {item_name}: {exc}")

                            yield "\n".join(messages)

                        except Exception as e:
                            error_logger.error(
                                f"Error ocr processing {img.name}: {str(e)}",
                                exc_info=True
                            )
                            summary_logger.info(f"ERROR: OCR processing failed for {img.name}")

                            failure_count += 1
                            folder_failure += 1

                            messages.append(f"Error processing {img.name}: {str(e)}")
                            yield "\n".join(messages)

                    # Drain all remaining post-processing tasks for this folder
                    for fut in as_completed(list(pending_post.keys())):
                        task_type, item_name, _ = pending_post.pop(fut)
                        try:
                            res = fut.result()
                            if task_type == "pdf":
                                if res.get("pdf_success"):
                                    folder_pdf_success += 1
                                else:
                                    folder_pdf_fail += 1
                                    error_logger.error(f"PDF FAIL for {item_name}: {res.get('pdf_error')}")
                                if res.get("alto_success"):
                                    folder_alto_success += 1
                                else:
                                    folder_alto_fail += 1
                                    error_logger.error(f"ALTO FAIL for {item_name}: {res.get('alto_error')}")
                                messages.append(f"PDF/ALTO generated: {item_name}")
                            elif task_type == "txt":
                                messages.append(f"Text file created: {Path(res).name}")
                        except Exception as exc:
                            error_logger.error(f"Post-processing error for {item_name}: {exc}")

                    if folder_success > 0:
                        summary_logger.info(f"==========PDF AND ALTO SUMMARY FOR FOLDER: {folder.name} ==========")
                        summary_logger.info(f"PDF success: {folder_pdf_success}, fail: {folder_pdf_fail}")
                        summary_logger.info(f"ALTO success: {folder_alto_success}, fail: {folder_alto_fail}")
                        messages.append(
                            f"PDF/ALTO completed for folder: {folder.name} (PDF: {folder_pdf_success}/{folder_success}, ALTO: {folder_alto_success}/{folder_success})"
                        )
                        if generate_text_file:
                            messages.append(f"Text generation completed for folder: {folder.name}")

                        # Auto-compare against benchmark if benchmark ALTO/PDF files exist in input folder
                            benchmark_has_eval_files = any(
                                f.suffix.lower() in {".xml", ".pdf"} for f in folder.glob("*") if f.is_file()
                            )
                            if benchmark_has_eval_files and folder_alto_success > 0:
                                try:
                                    from tools.compare_alto import compare_directory_batch
                                    compare_root = Path(r"C:\CRKN Projects\paddle_pipeline\compare")
                                    if folder.parent.name.lower() in {"newspapers", "annuals", "heritage"}:
                                        coll_name = f"{folder.parent.name}/{folder.name}"
                                    else:
                                        coll_name = str(rel_folder).replace("\\", "/")
                                    compared_pages = compare_directory_batch(
                                        pipeline_dir=out_dir,
                                        benchmark_dir=folder,
                                        output_dir=compare_root,
                                        collection_name=coll_name,
                                        render_html=True,
                                        render_image=False,
                                    )
                                    if compared_pages > 0:
                                        messages.append(
                                            f"Auto-comparison updated for {folder.name} ({compared_pages} pages -> compare/index.html)"
                                        )
                                except Exception as exc:
                                    error_logger.warning(f"Auto-comparison error for {folder.name}: {exc}")
                        yield "\n".join(messages)

        else:
            yield f"Input path is neither a file nor a folder: {input_path}"
            return
    
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_files = success_count + failure_count

        avg_time = total_time / total_files if total_files > 0 else 0

        summary_logger.info(f"========== {method} SUMMARY ==========")
        summary_logger.info(f"Total Successful JSON files: {success_count}")
        summary_logger.info(f"Total Failed JSON files: {failure_count}")
        summary_logger.info(f"Total Processing Time: {total_time:.2f} seconds")
        summary_logger.info(f"Average Time per File: {avg_time:.2f} seconds")

        if opt.get("generate_results_dashboard", True) and success_count > 0:
            try:
                from services.results_dashboard import build_ocr_run_dashboard
                dash_path = build_ocr_run_dashboard(
                    output_root=output_root,
                    input_root=input_path,
                    engine_name=method or "PaddleOCR",
                    total_time_sec=total_time,
                    per_page_timings=per_page_timings,
                    auto_open=True,
                    dashboard_folder_name="HTML_Results_Dashboard",
                )
                if dash_path:
                    messages.append(f"\nHTML Results Dashboard generated: {dash_path.resolve()}")
            except Exception as e:
                error_logger.warning(f"Error generating HTML results dashboard: {e}")

        if failure_count > 0:
            yield "\n".join(messages) + "\n\nOCR partially completed. Some errors occurred during processing. Please review the error report."
        else:
            yield "\n".join(messages) + "\n\nOCR Completed, please check generated PDF, ALTO XML, and JSON files in your output folder."

    finally:
        RUN_LOCK.release()


def run_ocr_for_image(image_path, out_dir, opt, select=None, preprocessed_result=None):
    """
    Run OCR for a single image and save the final JSON output.
    """
    final_json = out_dir / f"{image_path.stem}.json"
    if preprocessed_result is not None:
        ocr_image_path, preprocess_metadata = preprocessed_result
    else:
        ocr_image_path, preprocess_metadata = preprocess_image_for_ocr(
            image_path=image_path,
            out_dir=out_dir,
            opt=opt,
        )

    current_opt = dict(opt)
    if preprocess_metadata and preprocess_metadata.get("table_unclip_ratio"):
        current_opt["text_det_unclip_ratio"] = float(preprocess_metadata["table_unclip_ratio"])

    effective_strat = normalize_tile_strategy(current_opt.get("tile_strategy", DEFAULT_TILE_STRATEGY))
    if effective_strat == "auto":
        use_3x2 = should_use_3x2_tiling(ocr_image_path)
        if use_3x2:
            current_opt["tile_source_images"] = True
            current_opt["tile_strategy"] = "grid_3x2"
            current_opt["tile_rows"] = 3
            current_opt["tile_cols"] = 2
            current_opt["tile_overlap_px"] = int(current_opt.get("tile_overlap_px", DEFAULT_TILE_OVERLAP_PX))
        else:
            current_opt["tile_source_images"] = False
            current_opt["tile_strategy"] = "none"
            current_opt["tile_rows"] = 1
            current_opt["tile_cols"] = 1
    elif effective_strat == "none":
        current_opt["tile_source_images"] = False

    if current_opt.get("tile_source_images", False) and supports_tiled_ocr(current_opt.get("method"), select):
        merged_json = run_tiled_local_ocr(ocr_image_path, current_opt, select=select)
        normalize_ocr_word_boxes_in_payload(merged_json)
        annotate_preprocessing_payload(merged_json, preprocess_metadata)
        final_json.write_text(
            json.dumps(merged_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        maybe_apply_post_ocr_vl_check(final_json, out_dir, current_opt, select)
        maybe_apply_output_text_cleanup(final_json, current_opt)
        return final_json, None

    recovery_note = save_local_ocr_results_with_recovery(ocr_image_path, out_dir, current_opt)

    json_file = out_dir / f"{ocr_image_path.stem}_res.json"
    if not json_file.exists():
        candidates = sorted(out_dir.glob(f"{ocr_image_path.stem}*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No OCR JSON output was produced for image: {ocr_image_path.name}"
            )
        json_file = candidates[0]

    json_file.replace(final_json)
    normalize_saved_ocr_json(final_json, preprocess_metadata=preprocess_metadata)
    maybe_apply_post_ocr_vl_check(final_json, out_dir, opt, select)
    maybe_apply_output_text_cleanup(final_json, opt)

    return final_json, recovery_note


def maybe_apply_post_ocr_vl_check(
    json_path: Path,
    out_dir: Path,
    opt: dict,
    select: str | None = None,
) -> None:
    """Run the optional post-OCR VL audit when the current method supports it."""
    if not opt.get("post_ocr_vl_enabled", False):
        return
    if not supports_post_ocr_vl(opt.get("method"), select):
        return
    apply_post_ocr_vl_check(json_path, out_dir, opt)


def maybe_apply_output_text_cleanup(json_path: Path, opt: dict) -> None:
    """Apply conservative punctuation cleanup to a saved OCR JSON payload."""
    if not opt.get("clean_punctuation_issues", False):
        return
    clean_ocr_json_text_in_place(json_path)


def normalize_saved_ocr_json(
    json_path: Path,
    preprocess_metadata: dict | None = None,
) -> None:
    """Normalize a saved OCR JSON file in place after local inference."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    changed = normalize_ocr_word_boxes_in_payload(payload)
    changed = annotate_preprocessing_payload(payload, preprocess_metadata) or changed

    if changed:
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def annotate_preprocessing_payload(
    payload: dict,
    preprocess_metadata: dict | None,
) -> bool:
    """Attach preprocessing metadata to an OCR payload when available."""
    if not preprocess_metadata:
        return False

    metadata = dict(preprocess_metadata)

    scale_factor = float(metadata.get("coordinate_scale_factor") or 1.0)
    if scale_factor != 1.0 and scale_factor > 0:
        inv_scale = 1.0 / scale_factor
        scaled_payload = scale_nested_geometry(payload, inv_scale, inv_scale)
        payload.clear()
        payload.update(scaled_payload)

    offset_x = int(metadata.get("coordinate_offset_x") or 0)
    offset_y = int(metadata.get("coordinate_offset_y") or 0)
    if offset_x or offset_y:
        offset_payload = offset_nested_geometry(payload, offset_x, offset_y)
        payload.clear()
        payload.update(offset_payload)

    payload["source_preprocessing"] = metadata
    payload["original_input_path"] = metadata.get("original_input_path")
    output_input_path = (
        metadata.get("original_input_path")
        if metadata.get("use_original_input_for_outputs")
        else metadata.get("ocr_input_path")
    )
    payload["input_path"] = output_input_path

    overall_ocr_res = payload.get("overall_ocr_res")
    if isinstance(overall_ocr_res, dict):
        overall_ocr_res["input_path"] = output_input_path

    return True
