import json
import threading
from pathlib import Path
from engine import engine
import traceback

from services.constants import IMAGE_EXTENSIONS
from services.pdf_generation import main as generate_pdf_and_alto
from services.ocr_logger import OCRLoggerManger
from services.txt_generation import generate_txt_file 

RUN_LOCK = threading.Lock()
# Global flag to signal stopping the OCR process
STOP_FLAG = False

def stop_ocr():
    global STOP_FLAG
    STOP_FLAG = True
   

def run_ocr_job(input_path, output_root, opt,select=None):

    method = opt.get("method")
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
    generate_text_file = opt.get("generate_text_file", False)
    lang = opt.get("lang", "None")
   
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

                output = engine.infer(str(input_path), opt)

                for res in output:
                    res.save_to_json(save_path=str(out_dir))

                json_file = out_dir / f"{input_path.stem}_res.json"
                final_json = out_dir / f"{input_path.stem}.json"  
                json_file.rename(final_json)  

                success_count += 1
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

                summary_logger.info(f"{folder.name} failed during PDF/ALTO generation")

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
                        output = engine.infer(str(img), opt)

                        for res in output:
                            res.save_to_json(save_path=str(out_dir))

                        json_file = out_dir / f"{img.stem}_res.json"
                        json_file.rename(out_dir / f"{img.stem}.json")

                        success_count += 1
                        folder_success += 1

                        messages.append(f"{img.name} created and saved in {out_dir}")
                        yield "\n".join(messages)

                    except Exception as e:
                        error_logger.error(
                            f"Error processing {img.name}: {str(e)}",
                            exc_info=True
                        )
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

                        summary_logger.info(f"{folder.name} failed during PDF/ALTO generation")

        else:
            yield f"Input path is neither a file nor a folder: {input_path}"
            return

        summary_logger.info(f"========== {method} SUMMARY ==========")
        summary_logger.info(f"Total Successful JSON files: {success_count}")
        summary_logger.info(f"Total Failed JSON files: {failure_count}")

        if generate_text_file:
            yield "OCR Completed, please check json and text files at your output folder"
        else:
            
            yield "OCR Completed, please check pdf and alto files at your output folder"

    finally:
        RUN_LOCK.release()