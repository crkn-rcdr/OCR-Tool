"""Generate plain-text outputs from normalized OCR JSON payloads."""

from pathlib import Path
import json

from services.alto_generation import ordered_text_lines_from_data
from services.punctuation_cleanup import clean_ocr_line_list


def extract_fallback_texts(data):
    """Extract best-effort text lines from several supported OCR JSON layouts."""
    if "overall_ocr_res" in data:
        ocr_res = data["overall_ocr_res"]
        return ocr_res["rec_texts"]

    if "rec_texts" in data:
        return data["rec_texts"]

    if "result" in data:
        parsing_res_list = data["result"]["layoutParsingResults"][0]["prunedResult"][
            "parsing_res_list"
        ]

        return [block.get("block_content") for block in parsing_res_list]

    return []


def generate_txt_file(json_path:Path):
    """
    Generate a TXT file from OCR JSON results.

    Supports multiple OCR JSON formats and extracts
    recognized text content into a plain text file.

    Args:
        json_path (Path):
            Path to the OCR JSON file.

    Returns:
        Path:
            Path to the generated TXT file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "overall_ocr_res" in data or "rec_texts" in data:
        texts = ordered_text_lines_from_data(data)
        if not texts:
            texts = extract_fallback_texts(data)
    else:
        texts = extract_fallback_texts(data)

    texts = clean_ocr_line_list(texts)

    txt_path = json_path.with_suffix(".txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"{text}\n")
    
    return txt_path
