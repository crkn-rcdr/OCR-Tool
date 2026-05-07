import json
import threading
from pathlib import Path
from engine import engine
import traceback

from services.constants import IMAGE_EXTENSIONS
from services.pdf_generation import main as generate_pdf_and_alto
from services.ocr_logger import OCRLoggerManger

RUN_LOCK = threading.Lock()

def run_ocr_job(input_path, output_root, opt,select=None):
    
    method = opt.get("method")
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
    
    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

    try:
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

                # Running pp-structureV3 inference on the single file input
                output = engine.infer(str(input_path), opt)

                for res in output:
                    res.save_to_json(save_path=str(out_dir))

                json_file = out_dir / f"{input_path.stem}_res.json"
                json_file.rename(out_dir / f"{input_path.stem}.json")

                success_count += 1
                messages.append(f"{input_path.name} created and saved in {out_dir}")

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
                try:
                    rel = img.relative_to(input_base)

                    out_dir = output_base / rel.parent
                    out_dir.mkdir(parents=True, exist_ok=True)

                    output = engine.infer(str(img), opt)

                    for res in output:
                        res.save_to_json(save_path=str(out_dir))
                    
                    json_file = out_dir / f"{img.stem}_res.json"
                    json_file.rename(out_dir / f"{img.stem}.json")

                    success_count += 1
                    messages.append(f"{img.name} created and saved in {out_dir}")
                    yield "\n".join(messages)

                except Exception as e:
                    error_logger.error(
                        f"Error processing {img.name}:{str(e)}",
                        exc_info=True
                    )
                    failure_count += 1  
                    messages.append(f"Error processing {img.name}: {str(e)}")
                    yield "\n".join(messages)   
        else:
            yield f"Input path is neither a file nor a folder: {input_path}"
            return

        summary_logger.info(f"========== {method} SUMMARY ==========")
        summary_logger.info(f"Total Successful JSON files: {success_count}")
        summary_logger.info(f"Total Failed JSON files: {failure_count}")

        generate_pdf_and_alto(output_base, input_path, method,select)


        yield "OCR job completed successfully."

    finally:
        RUN_LOCK.release()
            
       
      

        
