"""Headless CLI batch runner for large unattended OCR runs across heritage collections.

Features:
- Recursive subfolder / reel scanning (e.g. LAC reels, Peel newspapers, CIHM monographs).
- Resume / skip-existing support for interrupted multi-hour batches.
- Rich CLI progress bar via tqdm with live ETA and throughput stats.
- Output delivery to Searchable PDF + ALTO XML or TXT.
- Graceful interrupt handling (SIGINT / Ctrl+C) with progress preservation.
- Full manifest and error telemetry saved to batch_run_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from services.constants import IMAGE_EXTENSIONS
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_OLMOCR_API_URL,
    DEFAULT_OLMOCR_MODEL,
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BATCH_SIZE,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_CONCURRENCY,
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
from services.pdf_generation import main as generate_pdf_and_alto
from services.tile_ocr import tile_grid_for_strategy
from services.txt_generation import generate_txt_file
import services.ocr_pp as ocr_pp

BEST_PREPROCESSING_STEPS = list(DEFAULT_PREPROCESSING_STEPS)
BEST_TILE_STRATEGY = "auto"
PADDLE_OCR_METHODS = {"PP-OCRv6"}

logger = logging.getLogger("batch_runner")


class BatchRunner:
    def __init__(self, opt: dict[str, Any], resume: bool = False, quiet: bool = False):
        self.opt = opt
        self.resume = resume
        self.quiet = quiet
        self.interrupted = False
        self._setup_signals()

    def _setup_signals(self):
        def _handle_sigint(signum, frame):
            print("\n[!] SIGINT received. Stopping batch gracefully after current page...")
            self.interrupted = True
            ocr_pp.stop_ocr()

        signal.signal(signal.SIGINT, _handle_sigint)

    def is_page_already_done(self, out_dir: Path, stem: str, generate_txt: bool) -> bool:
        """Check if output files already exist for a given image stem."""
        json_file = out_dir / f"{stem}.json"
        if not json_file.exists() or json_file.stat().st_size == 0:
            return False

        if generate_txt:
            txt_file = out_dir / f"{stem}.txt"
            return txt_file.exists() and txt_file.stat().st_size > 0
        else:
            pdf_file = out_dir / f"{stem}.pdf"
            xml_file = out_dir / f"{stem}.xml"
            return (pdf_file.exists() and pdf_file.stat().st_size > 0) or (
                xml_file.exists() and xml_file.stat().st_size > 0
            )

    def run(self, input_path: Path, output_root: Path) -> int:
        start_time = time.perf_counter()
        start_dt = datetime.now(timezone.utc).isoformat()

        output_root.mkdir(parents=True, exist_ok=True)
        generate_txt = bool(self.opt.get("generate_text_file"))
        method = self.opt.get("method", "PP-OCRv6")
        lang = self.opt.get("lang")

        # Discover all target images
        if input_path.is_file():
            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                print(f"Error: Unsupported image extension for file: {input_path}", file=sys.stderr)
                return 1
            image_files = [input_path]
            input_base = input_path.parent
        elif input_path.is_dir():
            image_files = sorted(
                [
                    p
                    for p in input_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )
            input_base = input_path
        else:
            print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
            return 1

        if not image_files:
            print(f"No supported images found in: {input_path}")
            return 0

        print(f"============================================================")
        print(f" CRKN Headless Batch Runner")
        print(f" Method:       {method}")
        print(f" Input:        {input_path} ({len(image_files)} pages)")
        print(f" Output Root:  {output_root}")
        print(f" Mode:         {'TXT only' if generate_txt else 'Searchable PDF + ALTO XML'}")
        print(f" Pre-process:  {bool(self.opt.get('preprocessing_steps'))}")
        print(f" Post-process: {bool(self.opt.get('post_ocr_vl_enabled'))} (workers: {self.opt.get('post_ocr_vl_concurrency')}, batch: {self.opt.get('post_ocr_vl_batch_size')})")
        print(f" Resume mode:  {self.resume}")
        print(f"============================================================\n")

        records: list[dict[str, Any]] = []
        success_count = 0
        skipped_count = 0
        failure_count = 0

        pbar = tqdm(image_files, desc="Batch Progress", unit="page", disable=self.quiet)

        for img in pbar:
            if self.interrupted:
                break

            rel_path = img.relative_to(input_base)
            out_dir = output_root / rel_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = img.stem

            # Check resume state
            if self.resume and self.is_page_already_done(out_dir, stem, generate_txt):
                skipped_count += 1
                records.append({
                    "file": str(rel_path),
                    "status": "skipped",
                    "reason": "existing_output",
                    "duration_seconds": 0.0,
                })
                pbar.set_postfix(
                    ok=success_count,
                    skip=skipped_count,
                    fail=failure_count,
                    current=img.name[:16],
                )
                continue

            page_start = time.perf_counter()
            pbar.set_postfix(
                ok=success_count,
                skip=skipped_count,
                fail=failure_count,
                current=img.name[:16],
            )

            try:
                final_json, recovery_note = ocr_pp.run_ocr_for_image(
                    image_path=img,
                    out_dir=out_dir,
                    opt=self.opt,
                    select=method,
                )

                if generate_txt:
                    generate_txt_file(final_json)
                else:
                    generate_pdf_and_alto(
                        final_json,
                        img,
                        method,
                        lang,
                        select=method,
                        run_options=self.opt,
                    )

                page_dur = time.perf_counter() - page_start
                success_count += 1
                records.append({
                    "file": str(rel_path),
                    "status": "success",
                    "duration_seconds": round(page_dur, 2),
                    "recovery_note": recovery_note,
                })

            except Exception as exc:
                page_dur = time.perf_counter() - page_start
                failure_count += 1
                err_msg = str(exc)
                records.append({
                    "file": str(rel_path),
                    "status": "failed",
                    "duration_seconds": round(page_dur, 2),
                    "error": err_msg,
                })
                if not self.quiet:
                    tqdm.write(f"\n[ERROR] Failed {img.name}: {err_msg}")

            pbar.set_postfix(
                ok=success_count,
                skip=skipped_count,
                fail=failure_count,
                current=img.name[:16],
            )

        pbar.close()

        total_elapsed = time.perf_counter() - start_time
        processed_total = success_count + failure_count
        pages_per_min = (
            (processed_total / (total_elapsed / 60.0)) if total_elapsed > 0 else 0.0
        )

        manifest = {
            "batch_runner_version": "1.0",
            "start_time": start_dt,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": round(total_elapsed, 2),
            "pages_per_minute": round(pages_per_min, 2),
            "total_images": len(image_files),
            "success_count": success_count,
            "skipped_count": skipped_count,
            "failure_count": failure_count,
            "interrupted": self.interrupted,
            "configuration": {
                "method": method,
                "language": lang,
                "generate_text_file": generate_txt,
                "preprocessing_steps": self.opt.get("preprocessing_steps"),
                "post_ocr_vl_enabled": self.opt.get("post_ocr_vl_enabled"),
                "post_ocr_vl_concurrency": self.opt.get("post_ocr_vl_concurrency"),
                "post_ocr_vl_batch_size": self.opt.get("post_ocr_vl_batch_size"),
            },
            "records": records,
        }

        manifest_path = output_root / "batch_run_summary.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\n============================================================")
        print(" Batch Run Completed")
        print(f" Total Elapsed:    {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f" Processed:        {success_count} succeeded, {failure_count} failed")
        print(f" Skipped:          {skipped_count} (existing outputs)")
        print(f" Throughput:       {pages_per_min:.2f} pages/min")
        print(f" Summary Manifest: {manifest_path}")
        print("============================================================\n")

        if failure_count > 0:
            return 1
        return 0


def build_batch_options(args: argparse.Namespace) -> dict[str, Any]:
    tile_strategy = args.tile_strategy
    tile_rows, tile_cols = tile_grid_for_strategy(tile_strategy)
    process_enabled = args.method in PADDLE_OCR_METHODS
    pre_process_enabled = args.pre_process and process_enabled
    post_process_enabled = args.post_process and process_enabled

    opt = {
        "method": args.method,
        "ocr_version": DEFAULT_OCR_VERSION,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "generate_text_file": bool(args.format == "txt"),
        "clean_punctuation_issues": post_process_enabled and bool(DEFAULT_CLEAN_PUNCTUATION_ISSUES),
        "preprocessing_steps": list(BEST_PREPROCESSING_STEPS) if pre_process_enabled else [],
        "ignore_black_border": pre_process_enabled and bool(DEFAULT_IGNORE_BLACK_BORDER),
        "tile_source_images": process_enabled,
        "tile_strategy": tile_strategy,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_overlap_px": int(DEFAULT_TILE_OVERLAP_PX),
        "tile_doclayout_padding_px": int(DEFAULT_DOCLAYOUT_PADDING_PX),
        "tile_doclayout_threshold": float(DEFAULT_DOCLAYOUT_THRESHOLD),
        "tile_doclayout_recovery": True,
        "use_layout_detection": args.method == "PaddleVL",
        "olmocr_url": args.olmocr_url,
        "olmocr_model": DEFAULT_OLMOCR_MODEL,
        "olmocr_max_tokens": 8192,
        "lang": args.lang,
        "text_det_thresh": float(DEFAULT_TEXT_DET_THRESH),
        "text_det_limit_type": DEFAULT_TEXT_DET_LIMIT_TYPE,
        "text_det_limit_side_len": int(DEFAULT_TEXT_DET_LIMIT_SIDE_LEN),
        "text_det_max_side_limit": int(DEFAULT_TEXT_DET_MAX_SIDE_LIMIT),
        "text_det_box_thresh": float(DEFAULT_TEXT_DET_BOX_THRESH),
        "text_det_unclip_ratio": float(DEFAULT_TEXT_DET_UNCLIP_RATIO),
        "text_rec_score_thresh": float(DEFAULT_TEXT_REC_SCORE_THRESH),
        "text_recognition_batch_size": int(DEFAULT_TEXT_RECOGNITION_BATCH_SIZE),
        "post_ocr_vl_enabled": post_process_enabled and bool(DEFAULT_POST_OCR_VL_ENABLED),
        "post_ocr_vl_candidate_mode": DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
        "post_ocr_vl_max_words_enabled": False,
        "post_ocr_vl_max_words": int(DEFAULT_POST_OCR_VL_MAX_WORDS),
        "post_ocr_vl_padding_px": int(DEFAULT_POST_OCR_VL_PADDING_PX),
        "post_ocr_vl_low_confidence_threshold": float(DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD),
        "post_ocr_vl_apply_corrections": bool(DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS),
        "post_ocr_vl_save_tiles": bool(DEFAULT_POST_OCR_VL_SAVE_TILES),
        "post_ocr_vl_paddlevl_url": args.paddlevl_url,
        "post_ocr_vl_concurrency": max(1, args.concurrency),
        "post_ocr_vl_batch_size": max(1, args.batch_size),
    }
    return opt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless CLI batch runner for large unattended OCR runs."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input folder containing images (or single image file).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Root output folder for generated OCR deliverables.",
    )
    parser.add_argument(
        "--method",
        choices=["PP-OCRv6", "PaddleVL", "OLMOCR"],
        default="PP-OCRv6",
        help="OCR engine method (default: PP-OCRv6).",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Language code for OCR recognition (e.g. 'en', 'fr', 'chinese_cht').",
    )
    parser.add_argument(
        "--format",
        choices=["pdf_alto", "txt"],
        default="pdf_alto",
        help="Output format: 'pdf_alto' for Searchable PDF + ALTO XML, or 'txt' for text only.",
    )
    parser.add_argument(
        "--pre-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable border crop and background flattening (PP-OCRv6 only).",
    )
    parser.add_argument(
        "--post-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VLM-based word verification and punctuation cleanup.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Skip images that already have complete output in the target destination.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_POST_OCR_VL_CONCURRENCY,
        help=f"Post-OCR VL concurrency worker count (default: {DEFAULT_POST_OCR_VL_CONCURRENCY}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_POST_OCR_VL_BATCH_SIZE,
        help=f"Post-OCR VL candidate batch size (default: {DEFAULT_POST_OCR_VL_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--tile-strategy",
        default=BEST_TILE_STRATEGY,
        help=f"Tiling strategy for large broadsheets (default: {BEST_TILE_STRATEGY}).",
    )
    parser.add_argument(
        "--paddlevl-url",
        default=DEFAULT_PADDLE_VL_API_URL,
        help="PaddleOCR-VL service URL.",
    )
    parser.add_argument(
        "--olmocr-url",
        default=DEFAULT_OLMOCR_API_URL,
        help="olmOCR OpenAI-compatible API URL.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential console logs.",
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    options = build_batch_options(args)
    runner = BatchRunner(opt=options, resume=args.resume, quiet=args.quiet)
    return runner.run(input_path=input_path, output_root=output_root)


if __name__ == "__main__":
    sys.exit(main())
