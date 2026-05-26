"""Run PaddleOCR-VL API jobs and convert responses into repo output formats."""

import json
import base64
import os
import threading
from pathlib import Path
import time

import requests

from services.constants import IMAGE_EXTENSIONS
from services.pdf_generation import main as generate_pdf_and_alto
from services.ocr_logger import OCRLoggerManger, log_run_configuration
from services.txt_generation import generate_txt_file 
from services.punctuation_cleanup import clean_ocr_json_text_in_place

import services.ocr_pp as ocr_pp

# Thread synchronization lock used to prevent
# multiple PaddleVL OCR jobs from running simultaneously.
#
# Why this is needed:
# - Prevents concurrent OCR requests to the PaddleVL service
# - Avoids GPU/CPU resource conflicts
# - Prevents race conditions when writing output files
# - Ensures stable OCR processing
#
# Example:
# If one PaddleVL OCR task is already running,
# another task attempting to start will be rejected
# until the current task finishes.
RUN_LOCK = threading.Lock()

BASE_URL = os.getenv("PADDLE_VL_API_URL", "http://127.0.0.1:8080").rstrip("/")

PADDLEX_OPTION_KEYS = {
    "use_doc_orientation_classify": "useDocOrientationClassify",
    "use_doc_unwarping": "useDocUnwarping",
    "use_layout_detection": "useLayoutDetection",
    "use_chart_recognition": "useChartRecognition",
    "use_seal_recognition": "useSealRecognition",
    "use_ocr_for_image_block": "useOcrForImageBlock",
    "layout_threshold": "layoutThreshold",
    "layout_nms": "layoutNms",
    "layout_unclip_ratio": "layoutUnclipRatio",
    "layout_merge_bboxes_mode": "layoutMergeBboxesMode",
    "layout_shape_mode": "layoutShapeMode",
    "prompt_label": "promptLabel",
    "format_block_content": "formatBlockContent",
    "repetition_penalty": "repetitionPenalty",
    "temperature": "temperature",
    "top_p": "topP",
    "min_pixels": "minPixels",
    "max_pixels": "maxPixels",
    "max_new_tokens": "maxNewTokens",
    "merge_layout_blocks": "mergeLayoutBlocks",
    "markdown_ignore_labels": "markdownIgnoreLabels",
    "vlm_extra_args": "vlmExtraArgs",
    "prettify_markdown": "prettifyMarkdown",
    "show_formula_number": "showFormulaNumber",
    "restructure_pages": "restructurePages",
    "merge_tables": "mergeTables",
    "relevel_titles": "relevelTitles",
    "visualize": "visualize",
    "log_id": "logId",
}

ALLOWED_PADDLEX_KEYS = {
    "useDocOrientationClassify",
    "useDocUnwarping",
    "useLayoutDetection",
    "useChartRecognition",
    "useSealRecognition",
    "useOcrForImageBlock",
    "layoutThreshold",
    "layoutNms",
    "layoutUnclipRatio",
    "layoutMergeBboxesMode",
    "layoutShapeMode",
    "promptLabel",
    "formatBlockContent",
    "repetitionPenalty",
    "temperature",
    "topP",
    "minPixels",
    "maxPixels",
    "maxNewTokens",
    "mergeLayoutBlocks",
    "markdownIgnoreLabels",
    "vlmExtraArgs",
    "prettifyMarkdown",
    "showFormulaNumber",
    "restructurePages",
    "mergeTables",
    "relevelTitles",
    "visualize",
    "logId",
    "outputFormats",
}


def _build_paddlevl_payload(opt: dict) -> dict:
    """Map repo option names into the PaddleX request payload format."""
    payload = {}

    for key, value in opt.items():
        if key in {"method", "lang", "generate_text_file"} or value is None:
            continue

        mapped_key = PADDLEX_OPTION_KEYS.get(key)
        if mapped_key is None and key in ALLOWED_PADDLEX_KEYS:
            mapped_key = key

        if mapped_key is None:
            continue

        payload[mapped_key] = value

    return payload


def _request_layout_parsing(payload: dict) -> dict:
    """Call the PaddleOCR-VL layout API and return the parsed JSON response."""
    try:
        response = requests.post(
            BASE_URL + "/layout-parsing",
            json=payload,
            timeout=(10, 600),
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the PaddleOCR-VL service at {BASE_URL}. "
            "Start it with `docker compose up -d` and wait for the API "
            "healthcheck to pass."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    return response.json()

def run_paddlevl_job(input_path, output_root, opt,select=None):
    """
    Run OCR processing using the PaddleVL API service.

    Args:
        input_path (Path):
            Input image file or folder.

        output_root (Path):
            Output directory for OCR results.

        opt (dict):
            OCR configuration options.

        select (str, optional):
            OCR engine identifier.

    Yields:
        str:
            OCR progress and status messages.
    """
    ocr_pp.STOP_FLAG = False
    base_payload = _build_paddlevl_payload(opt)
    generate_text_file = opt.get("generate_text_file", False)
    lang = opt.get("lang")
    method = opt.get("method")
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
    log_run_configuration(
        summary_logger,
        opt,
        select=select,
        input_path=input_path,
        output_root=output_root,
        stage="PaddleVL OCR",
    )
    
    success_count = 0
    failure_count = 0
    messages = []

    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

    start_time = time.perf_counter()

    try:
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

                with open(input_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("ascii")

                    payload = {
                        **base_payload,
                        "file": image_data,
                        "fileType": 1,
                    }

                    data = _request_layout_parsing(payload)
                    json_path = out_dir / f"{input_path.stem}.json"  

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    maybe_apply_output_text_cleanup(json_path, opt)

                    success_count += 1
                    messages.append(f"{input_path.name} created and saved in {out_dir}")
                    yield "\n".join(messages)
        
                    if generate_text_file:
                        txt_path = generate_txt_file(json_path)
                        messages.append(f"Text file created: {txt_path}")
                        yield "\n".join(messages)
                    else:
                        generate_pdf_and_alto(
                            json_path,
                            input_path,
                            method,
                            lang,
                            select,
                            run_options=opt,
                        )
                        yield "\n".join(messages)
           
            except Exception as e: 
                error_logger.error(
                    f"Error processing {input_path.name}:{str(e)}",
                    exc_info=True
                )
                failure_count += 1  
                messages.append(f"Error processing {input_path.name}: {str(e)}")
                yield "\n".join(messages)

        elif input_path.is_dir():
            input_base = input_path
            output_base = output_root / input_path.name

            for img in input_path.rglob("*"):
                if not (img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS):
                    continue

                if ocr_pp.STOP_FLAG:
                    return
                
                try:
                    rel = img.relative_to(input_base)
                    out_dir = output_base / rel.parent
                    out_dir.mkdir(parents=True, exist_ok=True)

                    with open(img, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("ascii")

                        payload = {
                            **base_payload,
                            "file": image_data,
                            "fileType": 1,
                        }

                        data = _request_layout_parsing(payload)
                        json_path = out_dir / f"{img.stem}.json"  

                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        maybe_apply_output_text_cleanup(json_path, opt)

                        success_count += 1
                        messages.append(f"{img.name} created and saved in {out_dir}")
                        yield "\n".join(messages)

                        if generate_text_file:
                            txt_path = generate_txt_file(json_path)
                            messages.append(f"Text file created: {txt_path}")
                            yield "\n".join(messages)
                        else:
                            generate_pdf_and_alto(
                                json_path,
                                img,
                                method,
                                lang,
                                select,
                                run_options=opt,
                            )
                            yield "\n".join(messages)

                except Exception as e: 
                    error_logger.error(
                        f"ERROR processing {img.name}:{str(e)}",
                        exc_info=True
                    )
                    summary_logger.info(f"ERROR:{img.name} during OCR request or PDF/ALTO generation")

                    failure_count += 1  
                    messages.append(f"Error processing {img.name}: {str(e)}")
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

        if generate_text_file and failure_count == 0:
            yield "OCR Completed, please check json and text files at your output folder."

        elif not generate_text_file and failure_count > 0:
             yield "OCR partially completed. Some errors occurred during processing. Please review the error report and check the generated PDF and ALTO files in your output folder."
            
        elif not generate_text_file and failure_count == 0:
            yield "OCR Completed, please check pdf and alto files at your output folder"

    finally:
        RUN_LOCK.release()


def maybe_apply_output_text_cleanup(json_path: Path, opt: dict) -> None:
    """Apply optional punctuation cleanup to a saved PaddleVL JSON payload."""
    if not opt.get("clean_punctuation_issues", False):
        return
    clean_ocr_json_text_in_place(json_path)
