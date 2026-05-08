import json
import base64
import threading
from pathlib import Path

import requests

from services.constants import IMAGE_EXTENSIONS
from services.pdf_generation import main as generate_pdf_and_alto
from services.ocr_logger import OCRLoggerManger

RUN_LOCK = threading.Lock()
BASE_URL = "http://127.0.0.1:8080"

def run_paddlevl_job(input_path, output_root, opt,select=None):
    payload = {}
   
    for key,value in opt.items():
        if key != "method" and value is not None:
            payload[key] = value

    method = opt.get("method")
    summary_logger, error_logger, summary_log_path, error_log_path = OCRLoggerManger.setup(method)
    
    success_count = 0
    failure_count = 0
    messages = []

    if not RUN_LOCK.acquire(blocking=False):
        yield "Another OCR job is currently running. Please wait until it finishes."
        return

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

                    payload.update({
                        "file": image_data,
                        "fileType": 1
                    })
                    
                    response = requests.post(BASE_URL + "/layout-parsing", json=payload) 
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

                    data = response.json()
                    json_path = out_dir / f"{input_path.stem}.json"  

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    success_count += 1
                    messages.append(f"{input_path.name} created and saved in {out_dir}")
                    yield "\n".join(messages)

                    generate_pdf_and_alto(json_path, input_path, method, select)
           
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

                    with open(img, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("ascii")

                        payload.update({
                            "file": image_data,
                            "fileType": 1
                        })
                     
                        response = requests.post(BASE_URL + "/layout-parsing", json=payload) 
                        if response.status_code != 200:
                            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

                        data = response.json()
                        json_path = out_dir / f"{img.stem}.json"  

                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        success_count += 1
                        messages.append(f"{img.name} created and saved in {out_dir}")
                        yield "\n".join(messages)

                        generate_pdf_and_alto(json_path, img, method, select)

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
        
        summary_logger.info("========== PP-StructureV3 SUMMARY ==========")
        summary_logger.info(f"Total Successful JSON files: {success_count}")
        summary_logger.info(f"Total Failed JSON files: {failure_count}")

        #generate_pdf_and_alto(output_base, input_path, method,select)
      

        yield "OCR job completed successfully."

    finally:
        RUN_LOCK.release()