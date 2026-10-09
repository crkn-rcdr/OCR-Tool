#!/usr/bin/env python3
r"""Run and benchmark the full CRKN PaddleOCR pipeline against ABBYY FineReader baseline.

Executes PP-OCRv6 with GPU batch recognition (RTX 5090), asynchronous prefetch,
multithreaded PDF/ALTO output generation, and PaddleVL post-OCR word verification.
Auto-updates the comparative QA dashboard at c:\CRKN Projects\paddle_pipeline\compare\index.html.

Usage:
  # Run entire benchmark (Newspapers -> Annuals -> Heritage):
  python scripts/run_pdf_test_benchmark.py

  # Run only Newspapers collection:
  python scripts/run_pdf_test_benchmark.py --collection Newspapers

  # Quick smoke test (e.g. first 2 pages per folder):
  python scripts/run_pdf_test_benchmark.py --collection Newspapers --limit-pages 2
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ocr_pp import run_ocr_job
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BATCH_SIZE,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_CONCURRENCY,
    DEFAULT_POST_OCR_VL_ENABLED,
    DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_PREPROCESSING_STEPS,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
)

DEFAULT_INPUT_DIR = Path(r"C:\Users\blapierre\Documents\PDF Test")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\blapierre\Desktop\PDF Test - Paddle OCR then Dots")


def parse_args():
    parser = argparse.ArgumentParser(description="CRKN OCR Benchmark Runner")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional collection subfolder name to process (e.g. 'Newspapers', 'Annuals', 'Heritage')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="PaddleOCR language code (default: None / auto)",
    )
    return parser.parse_args()


def build_pipeline_options():
    return {
        "method": "PP-OCRv6",
        "ocr_version": DEFAULT_OCR_VERSION,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "generate_text_file": False,
        "clean_punctuation_issues": bool(DEFAULT_CLEAN_PUNCTUATION_ISSUES),
        "preprocessing_steps": list(DEFAULT_PREPROCESSING_STEPS),
        "ignore_black_border": bool(DEFAULT_IGNORE_BLACK_BORDER),
        "tile_source_images": False,
        "text_det_thresh": float(DEFAULT_TEXT_DET_THRESH),
        "text_det_limit_type": DEFAULT_TEXT_DET_LIMIT_TYPE,
        "text_det_limit_side_len": int(DEFAULT_TEXT_DET_LIMIT_SIDE_LEN),
        "text_det_max_side_limit": int(DEFAULT_TEXT_DET_MAX_SIDE_LIMIT),
        "text_det_box_thresh": float(DEFAULT_TEXT_DET_BOX_THRESH),
        "text_det_unclip_ratio": float(DEFAULT_TEXT_DET_UNCLIP_RATIO),
        "text_rec_score_thresh": float(DEFAULT_TEXT_REC_SCORE_THRESH),
        "text_recognition_batch_size": int(DEFAULT_TEXT_RECOGNITION_BATCH_SIZE),
        "post_ocr_vl_enabled": bool(DEFAULT_POST_OCR_VL_ENABLED),
        "post_ocr_vl_backend": "dots_ocr",
        "post_ocr_vl_candidate_mode": DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
        "post_ocr_vl_max_words_enabled": False,
        "post_ocr_vl_max_words": 0,
        "post_ocr_vl_padding_px": 5,
        "post_ocr_vl_low_confidence_threshold": float(DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD),
        "post_ocr_vl_apply_corrections": bool(DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS),
        "post_ocr_vl_save_tiles": False,
        "post_ocr_vl_concurrency": int(DEFAULT_POST_OCR_VL_CONCURRENCY),
        "post_ocr_vl_batch_size": int(DEFAULT_POST_OCR_VL_BATCH_SIZE),
    }


def main():
    args = parse_args()
    input_path = args.input
    output_root = args.output

    if args.collection:
        input_path = input_path / args.collection

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(" CRKN DIGITAL COLLECTIONS - PADDLEOCR ULTRA-FAST BENCHMARK RUNNER")
    print("=" * 70)
    print(f" • Input Path:  {input_path}")
    print(f" • Output Root: {output_root}")
    print(f" • Hardware:    NVIDIA RTX 5090 (32GB VRAM)")
    print(f" • Local OCR:   PP-OCRv6 (batch_size=64, limit=9984)")
    print(f" • Post-OCR VL: Active (Dots OCR / vLLM, concurrency=16, batch=32)")
    print(f" • Preprocess:  Black-border cropping + Background flattening")
    print(f" • Overlap:     CPU prefetch + async multithreaded PDF/ALTO generation")
    print(f" • QA Dashboard: Auto-updating at c:\\CRKN Projects\\paddle_pipeline\\compare\\index.html")
    print("=" * 70)

    opt = build_pipeline_options()
    if args.language:
        opt["lang"] = args.language

    start_time = time.perf_counter()
    prev_message_count = 0

    try:
        iterator = run_ocr_job(input_path, output_root, opt)
        for raw_output in iterator:
            lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
            if len(lines) > prev_message_count:
                for line in lines[prev_message_count:]:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {line}", flush=True)
                prev_message_count = len(lines)

    except KeyboardInterrupt:
        print("\n[INFO] Benchmark interrupted by user.")
        return 130
    except Exception as exc:
        print(f"\n[FATAL ERROR] Pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print(f" BENCHMARK RUN COMPLETED in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)
    print(f" Output files: {output_root}")
    print(" Comparison Dashboard: c:\\CRKN Projects\\paddle_pipeline\\compare\\index.html")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
