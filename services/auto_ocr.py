import cv2
from utils.image_checks import analyze_quality
from services.ocr_vl import run_paddlevl_job
from services.ocr_pp import run_ocr_job
from services.constants import IMAGE_EXTENSIONS
from pathlib import Path

def iter_image_files(input_path):
    """
    Recursively iterate through all supported image files.

    Args:
        input_path (Path):
            Root directory containing image files or subfolders.

    Yields:
        Path:
            Path object for each supported image file.
    """
    for file in input_path.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            yield file
        elif file.is_dir():
            yield from iter_image_files(file)

def run_auto_job(image_input_folder, output_root, opt):
    """
    Automatically select the OCR engine based on image quality analysis.

    The function scans all supported image files in the input folder,
    analyzes image quality issues such as skew, warping, and illumination,
    and selects the appropriate OCR engine:

    - PaddleVL:
        Used for images with quality issues.
    - PP-StructureV3:
        Used for clean and well-structured images.

    OCR results are saved while preserving the original folder structure.

    Args:
        image_input_folder (str | Path):
            Input folder containing images.

        output_root (str | Path):
            Root folder for OCR output results.

        opt:
            OCR configuration/options object passed to OCR services.

    Yields:
        str:
            Progress and status messages during processing.
    """
    input_path = Path(image_input_folder)
    output_root = Path(output_root)
  
    output_base = output_root / input_path.name
    output_base.mkdir(parents=True, exist_ok=True)

    messages = []
    found = False

    for file in iter_image_files(input_path):
        found = True
        select = "PP-StructureV3"

        img = cv2.imread(str(file))
        quality_result = analyze_quality(img)

        relative_path = file.parent.relative_to(input_path)

        if relative_path == Path("."):
            file_output_folder = output_base 
        else:
            file_output_folder = output_base / relative_path 

        file_output_folder.mkdir(parents=True, exist_ok=True)

        if quality_result.has_skew or quality_result.has_warping or quality_result.has_illumination_problem:
            select = "PaddleVL"
            messages.append(f"{file.name} using PaddleVL for OCR.")
            yield "\n".join(messages)
            for msg in run_paddlevl_job(file, file_output_folder, opt,select):
                messages.append(msg)
                yield "\n".join(messages)
           
        else:
            messages.append(f"{file.name} using Paddle Structure for OCR.")
            yield "\n".join(messages)
            for msg in run_ocr_job(file,file_output_folder, opt,select):
                messages.append(msg)
                yield "\n".join(messages)

    if not found:
        yield "No supported image files found in the selected folder. "
           