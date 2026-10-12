"""Full OCR execution and ABBYY benchmarking script for CRKN test set.

Usage:
    python tools/run_full_benchmark.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.ocr_pp import run_ocr_job
from tools.compare_alto import compare_directory_batch, sync_to_evaluations_repo
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_POST_OCR_VL_BATCH_SIZE,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_CONCURRENCY,
    DEFAULT_POST_OCR_VL_DOTSOCR_MODEL,
    DEFAULT_POST_OCR_VL_DOTSOCR_URL,
    DEFAULT_POST_OCR_VL_ENABLED,
    DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_POST_OCR_VL_MAX_WORDS,
    DEFAULT_POST_OCR_VL_PADDING_PX,
    DEFAULT_POST_OCR_VL_SAVE_TILES,
    DEFAULT_PREPROCESSING_STEPS,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
    DEFAULT_TILE_OVERLAP_PX,
)

DEFAULT_INPUT_DIR = Path(r"C:\Users\blapierre\Documents\PDF Test\Newspapers")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\blapierre\Desktop\PDF Test - Final\Newspapers")
DEFAULT_COMPARE_DIR = root_dir / "compare"


def build_options() -> dict:
    return {
        "method": "PP-OCRv6",
        "ocr_version": DEFAULT_OCR_VERSION,
        "lang": "auto",  # Automatically resolves Traditional Chinese, French, English
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_det_thresh": float(DEFAULT_TEXT_DET_THRESH),
        "text_det_limit_type": DEFAULT_TEXT_DET_LIMIT_TYPE,
        "text_det_limit_side_len": int(DEFAULT_TEXT_DET_LIMIT_SIDE_LEN),
        "text_det_max_side_limit": int(DEFAULT_TEXT_DET_MAX_SIDE_LIMIT),
        "text_det_box_thresh": float(DEFAULT_TEXT_DET_BOX_THRESH),
        "text_det_unclip_ratio": float(DEFAULT_TEXT_DET_UNCLIP_RATIO),
        "text_rec_score_thresh": float(DEFAULT_TEXT_REC_SCORE_THRESH),
        "text_recognition_batch_size": int(DEFAULT_TEXT_RECOGNITION_BATCH_SIZE),
        "generate_text_file": False,
        "clean_punctuation_issues": bool(DEFAULT_CLEAN_PUNCTUATION_ISSUES),
        "preprocessing_steps": list(DEFAULT_PREPROCESSING_STEPS),
        "ignore_black_border": bool(DEFAULT_IGNORE_BLACK_BORDER),
        "tile_source_images": True,
        "tile_strategy": "auto",
        "tile_rows": 3,
        "tile_cols": 2,
        "tile_overlap_px": int(DEFAULT_TILE_OVERLAP_PX),
        "tile_doclayout_padding_px": int(DEFAULT_DOCLAYOUT_PADDING_PX),
        "tile_doclayout_threshold": float(DEFAULT_DOCLAYOUT_THRESHOLD),
        "tile_doclayout_recovery": False,
        # Post-OCR VL configuration
        "post_ocr_vl_enabled": True,
        "post_ocr_vl_backend": "dotsocr",
        "post_ocr_vl_candidate_mode": DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
        "post_ocr_vl_max_words_enabled": False,
        "post_ocr_vl_max_words": int(DEFAULT_POST_OCR_VL_MAX_WORDS),
        "post_ocr_vl_padding_px": int(DEFAULT_POST_OCR_VL_PADDING_PX),
        "post_ocr_vl_low_confidence_threshold": float(DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD),
        "post_ocr_vl_apply_corrections": True,
        "post_ocr_vl_save_tiles": bool(DEFAULT_POST_OCR_VL_SAVE_TILES),
        "post_ocr_vl_dotsocr_url": DEFAULT_POST_OCR_VL_DOTSOCR_URL,
        "post_ocr_vl_dotsocr_model": DEFAULT_POST_OCR_VL_DOTSOCR_MODEL,
        "post_ocr_vl_concurrency": int(DEFAULT_POST_OCR_VL_CONCURRENCY),
        "post_ocr_vl_batch_size": int(DEFAULT_POST_OCR_VL_BATCH_SIZE),
    }


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_INPUT_DIR
    if input_path == Path(r"C:\Users\blapierre\Documents\PDF Test"):
        output_path = Path(r"C:\Users\blapierre\Desktop\PDF Test - Final")
    else:
        output_path = DEFAULT_OUTPUT_DIR
    compare_root = DEFAULT_COMPARE_DIR

    print("==================================================================")
    print("      CRKN FULL OCR PIPELINE EXECUTION & ABBYY BENCHMARK          ")
    print("==================================================================")
    print(f"Input Directory : {input_path}")
    print(f"Output Directory: {output_path}")
    print(f"Compare Output  : {compare_root}")
    print("==================================================================")

    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)

    opt = build_options()

    print("\n--- Phase 1: Running OCR on Full Collection ---")
    start_time = time.perf_counter()

    for message in run_ocr_job(input_path, output_path, opt):
        lines = message.splitlines()
        if lines:
            print(f"  {lines[-1]}")

    elapsed = time.perf_counter() - start_time
    print(f"\n[Phase 1 Complete] Full OCR run finished in {elapsed:.1f}s")

    print("\n--- Phase 2: Updating ABBYY Benchmark & Comparison Dashboard ---")
    matched_count = 0

    for sub in sorted(output_path.rglob("*")):
        if sub.is_dir() and any(sub.glob("*.xml")):
            rel = sub.relative_to(output_path)
            bench_dir = input_path / rel
            if bench_dir.exists() and any(bench_dir.glob("*.xml")):
                if output_path.name.lower() == "newspapers":
                    coll_name = f"Newspapers/{str(rel).replace('\\', '/')}"
                else:
                    coll_name = str(rel).replace("\\", "/")
                print(f"  Comparing collection: {coll_name}...")
                count = compare_directory_batch(
                    pipeline_dir=sub,
                    benchmark_dir=bench_dir,
                    output_dir=compare_root,
                    collection_name=coll_name,
                    render_html=True,
                    render_image=False,
                )
                matched_count += count

    print("\n==================================================================")
    print("                      ALL PHASES COMPLETE                         ")
    print("==================================================================")
    print(f"OCR Outputs Saved To: {output_path}")
    print(f"Comparison Dashboard: {compare_root / 'index.html'}")
    print(f"Summary Metrics CSV : {compare_root / 'comparison_summary.csv'}")
    print(f"Summary Metrics JSON: {compare_root / 'comparison_summary.json'}")
    print("==================================================================")

    print("\n--- Phase 3: Syncing to OCR-Evaluations Repo & GitHub Pages ---")
    sync_to_evaluations_repo(compare_root, auto_push=True)


if __name__ == "__main__":
    main()

