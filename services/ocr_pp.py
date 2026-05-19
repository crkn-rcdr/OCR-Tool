import json
import threading
from pathlib import Path
from engine import engine
import traceback
import time

from services.constants import IMAGE_EXTENSIONS
from services.pdf_generation import main as generate_pdf_and_alto
from services.ocr_logger import OCRLoggerManger
from services.txt_generation import generate_txt_file 
from services.tile_ocr import (
    DEFAULT_CONTENT_BOX_PADDING_PX,
    run_tiled_local_ocr,
    supports_tiled_ocr,
)

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


def describe_tiling(opt):
    strategy = opt.get("tile_strategy", "grid")

    if strategy == "content_boxes":
        padding_value = opt.get("tile_content_padding_px")
        padding = int(
            DEFAULT_CONTENT_BOX_PADDING_PX
            if padding_value is None
            else padding_value
        )
        padding_label = f", {padding}px padding" if padding > 0 else ""
        return f"content-box tiled OCR{padding_label}"

    overlap = int(opt.get("tile_overlap_px", 0) or 0)
    overlap_label = f", {overlap}px overlap" if overlap > 0 else ""
    return f"3x1 tiled OCR{overlap_label}"


def stop_ocr():
    """
    Stop the currently running OCR job.

    Sets the global STOP_FLAG to True so that the OCR loop
    can terminate gracefully.
    """
    global STOP_FLAG
    STOP_FLAG = True
   

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
            OCR engine name (e.g., PaddleVL or PP-StructureV3).

    Yields:
        str:
            Progress and status messages during OCR execution.
    """

    method = opt.get("method")
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
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

                final_json = run_ocr_for_image(
                    image_path=input_path,
                    out_dir=out_dir,
                    opt=opt,
                    select=select,
                )

                success_count += 1
                if tile_source_images and supports_tiled_ocr(method, select):
                    messages.append(
                        f"{input_path.name} created from {describe_tiling(opt)} and saved in {out_dir}"
                    )
                else:
                    messages.append(f"{input_path.name} created and saved in {out_dir}")

                if generate_text_file:
                    txt_path = generate_txt_file(final_json)
                    messages.append(f"Text file created: {txt_path}")
                    yield "\n".join(messages)

                else:
                    # Once single-file OCR is complete, PDF/ALTO will be generated immediately.
                    generate_pdf_and_alto(final_json, input_path, method, lang,select)
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
            output_base = output_root / input_path.name

            # Group by the subfolder where the images are located.
            folders = {}

            for img in input_path.rglob("*"):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    folders.setdefault(img.parent, []).append(img)

            # OCR for each subfolder
            for folder, images in folders.items():
                rel_folder = folder.relative_to(input_base)
                out_dir = output_base / rel_folder
                out_dir.mkdir(parents=True, exist_ok=True)

                folder_success = 0
                folder_failure = 0

                messages.append(f"Start processing folder: {folder}")
                yield "\n".join(messages)

                for img in images:
                    if STOP_FLAG:
                        return

                    try:
                        run_ocr_for_image(
                            image_path=img,
                            out_dir=out_dir,
                            opt=opt,
                            select=select,
                        )

                        success_count += 1
                        folder_success += 1

                        if tile_source_images and supports_tiled_ocr(method, select):
                            messages.append(
                                f"{img.name} created from {describe_tiling(opt)} and saved in {out_dir}"
                            )
                        else:
                            messages.append(f"{img.name} created and saved in {out_dir}")
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

                # Once the OCR process in the current subfolder is complete, immediately generate PDF/ALTO.
                if folder_success > 0:
                    try:
                        if generate_text_file:
                            for json_file in out_dir.glob("*.json"):
                                txt_path = generate_txt_file(json_file)
                                messages.append(f"Text file created: {txt_path}")
                                yield "\n".join(messages)
                        else:
                            generate_pdf_and_alto(out_dir, folder, method, lang,select)
                            messages.append(f"PDF/ALTO generated for folder: {folder.name}")
                            yield "\n".join(messages)

                    except Exception as e:
                        error_logger.error(
                            f"Error generating PDF/ALTO for folder {folder}: {str(e)}",
                            exc_info=True
                        )
                        messages.append(
                            f"Error generating PDF/ALTO for folder {folder.name}: {str(e)}"
                        )
                        yield "\n".join(messages)

                        summary_logger.info(f"ERROR:{folder.name} failed during PDF/ALTO generation")

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


def run_ocr_for_image(image_path, out_dir, opt, select=None):
    """
    Run OCR for a single image and save the final JSON output.
    """
    final_json = out_dir / f"{image_path.stem}.json"

    if opt.get("tile_source_images", False) and supports_tiled_ocr(opt.get("method"), select):
        merged_json = run_tiled_local_ocr(image_path, opt, select=select)
        final_json.write_text(
            json.dumps(merged_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return final_json

    output = engine.infer(str(image_path), opt)

    for res in output:
        res.save_to_json(save_path=str(out_dir))

    json_file = out_dir / f"{image_path.stem}_res.json"
    json_file.rename(final_json)

    return final_json
