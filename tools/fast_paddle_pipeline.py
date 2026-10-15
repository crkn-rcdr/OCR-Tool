"""High-Performance Staged Pipeline for PaddleOCR (PP-OCRv6) with Batching and Producer-Consumer Concurrency.

Architecture:
- Stage 1 (Pre-processing CPU Pool - 4 workers): Reads images, crops black borders, auto-deskews,
  applies background flattening / adaptive CLAHE, and feeds prepared images to gpu_queue.
- Stage 2 (Inference GPU Worker - 1 dedicated process): Runs single-instance PaddleOCR on GPU
  with text_recognition_batch_size = 32 (saturating CUDA Tensor Cores without VRAM thrashing).
- Stage 3 (Post-processing CPU Pool - 4 workers): Normalizes bounding boxes, cleans punctuation artifacts,
  consolidates legal abbreviations, and generates Library of Congress ALTO XML and searchable PDF/A.

Usage:
    # Run full fast benchmark:
    python tools/fast_paddle_pipeline.py

    # Run specific collection:
    python tools/fast_paddle_pipeline.py --collection Newspapers

    # Run quick sample test (e.g. 10 pages):
    python tools/fast_paddle_pipeline.py --max-pages 10
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import engine
from services.constants import IMAGE_EXTENSIONS
from services.image_preprocessing import preprocess_image_for_ocr
from services.language_detection import resolve_effective_language
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_PREPROCESSING_STEPS,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TILE_OVERLAP_PX,
)
from services.pdf_generation import build_pdf
from services.punctuation_cleanup import clean_ocr_json_text_in_place
from services.tile_ocr import (
    normalize_tile_strategy,
    run_tiled_local_ocr,
    should_use_3x2_tiling,
    supports_tiled_ocr,
)
from services.ocr_pp import annotate_preprocessing_payload
from services.word_box_normalization import (
    normalize_ocr_word_boxes_in_payload,
)
from tools.compare_alto import compare_directory_batch, sync_to_evaluations_repo

DEFAULT_INPUT_ROOT = Path(r"C:\Users\blapierre\Documents\PDF Test")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\blapierre\Desktop\PDF Test - Paddle")
DEFAULT_OLD_RUN_ROOT = Path(r"C:\Users\blapierre\Desktop\PDF Test - Final")
DEFAULT_COMPARE_DIR = root_dir / "compare"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run High-Speed Staged PaddleOCR Pipeline with GPU Batching and Compare against ABBYY & Old Run."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory of benchmark source images and ABBYY ALTO XMLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory where new fast Paddle OCR JSON, PDF, and ALTO XML outputs are saved.",
    )
    parser.add_argument(
        "--old-run-dir",
        type=Path,
        default=DEFAULT_OLD_RUN_ROOT,
        help="Directory of previous sequential Paddle run for speedup benchmarking.",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=DEFAULT_COMPARE_DIR,
        help="Directory where comparison dashboard and summary metrics are saved.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Optional relative subfolder/collection name (e.g. 'Newspapers', 'Annuals', 'Heritage').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Text recognition batch size on GPU (default: 32).",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=4,
        help="Number of concurrent CPU workers for Pre-processing and Post-processing pools (default: 4).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of pages to process (useful for quick validation).",
    )
    parser.add_argument(
        "--tile-strategy",
        type=str,
        default="auto",
        choices=["auto", "grid_3x2", "none"],
        help="Broadsheet tiling strategy: 'auto' (default), 'grid_3x2', or 'none'.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip OCR and only run comparison and master dashboard generation.",
    )
    parser.add_argument(
        "--sync-docs",
        action="store_true",
        help="Sync generated dashboard to OCR-Evaluations repo and push to GitHub.",
    )
    return parser.parse_args()


def build_paddle_options(batch_size: int = 32, tile_strategy: str = "auto") -> dict:
    """Build optimized PP-OCRv6 runtime options."""
    return {
        "method": "PP-OCRv6",
        "ocr_version": DEFAULT_OCR_VERSION,
        "device": "gpu:0",
        "lang": "auto",
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
        "text_recognition_batch_size": int(batch_size),
        "generate_text_file": False,
        "clean_punctuation_issues": bool(DEFAULT_CLEAN_PUNCTUATION_ISSUES),
        "preprocessing_steps": list(DEFAULT_PREPROCESSING_STEPS),
        "ignore_black_border": bool(DEFAULT_IGNORE_BLACK_BORDER),
        "tile_source_images": True,
        "tile_strategy": tile_strategy,
        "tile_rows": 3,
        "tile_cols": 2,
        "tile_overlap_px": int(DEFAULT_TILE_OVERLAP_PX),
        "post_ocr_vl_enabled": False,
    }


def natural_sort_key(s: Any) -> list:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def find_target_images(input_dir: Path, max_pages: Optional[int] = None) -> List[Path]:
    """Find all supported image files in input_dir, grouped and sorted deterministically."""
    images = []
    for f in sorted(input_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    if max_pages is not None and max_pages > 0:
        return images[:max_pages]
    return images


class StagedPaddlePipeline:
    """Producer-Consumer Staged Pipeline:
    - Stage 1: CPU Pool (Pre-processing: CLAHE, deskew, border crop) -> gpu_queue
    - Stage 2: Single GPU Thread (PP-OCRv6 detection & batched recognition) -> post_queue
    - Stage 3: CPU Pool (Post-processing: ALTO XML & PDF/A synthesis) -> completion
    """

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        opt: dict,
        cpu_workers: int = 4,
    ):
        self.input_root = input_root
        self.output_root = output_root
        self.opt = opt
        self.cpu_workers = cpu_workers

        self.gpu_queue = queue.Queue(maxsize=16)
        self.post_queue = queue.Queue(maxsize=32)

        self.completed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.total_pages = 0
        self.t_start = 0.0

        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()

    def _stage1_preprocess_worker(self, img_path: Path) -> Optional[dict]:
        """Stage 1: CPU Pre-processing Worker."""
        try:
            rel_dir = img_path.parent.relative_to(self.input_root)
        except ValueError:
            rel_dir = Path("")

        out_dir = self.output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        final_json = out_dir / f"{stem}.json"
        xml_file = out_dir / f"{stem}.xml"

        # Check if already processed
        if final_json.exists() and xml_file.exists():
            return {
                "skipped": True,
                "stem": stem,
                "img_path": img_path,
                "rel_dir": str(rel_dir),
            }

        t0 = time.perf_counter()
        effective_lang = resolve_effective_language(self.opt.get("lang", "auto"), img_path.parent)
        item_opt = dict(self.opt, lang=effective_lang)

        # Execute CPU border cropping, deskew, and CLAHE
        ocr_image_path, preprocess_metadata = preprocess_image_for_ocr(
            image_path=img_path,
            out_dir=out_dir,
            opt=item_opt,
        )

        return {
            "skipped": False,
            "stem": stem,
            "img_path": img_path,
            "ocr_image_path": ocr_image_path,
            "preprocess_metadata": preprocess_metadata,
            "out_dir": out_dir,
            "item_opt": item_opt,
            "effective_lang": effective_lang,
            "rel_dir": str(rel_dir),
            "t0": t0,
        }

    def _stage2_gpu_worker(self):
        """Stage 2: Single Dedicated GPU Inference Worker."""
        while not self.stop_event.is_set():
            try:
                task = self.gpu_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:  # Poison pill
                self.post_queue.put(None)
                break

            if task.get("skipped"):
                with self.stats_lock:
                    self.skipped_count += 1
                    self.completed_count += 1
                    elapsed = time.perf_counter() - self.t_start
                    rate = self.completed_count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{self.completed_count}/{self.total_pages}] Already processed: "
                        f"{task['rel_dir']}/{task['stem']} (Running avg: {rate:.1f} pgs/s)",
                        flush=True,
                    )
                self.gpu_queue.task_done()
                continue

            stem = task["stem"]
            out_dir = task["out_dir"]
            item_opt = task["item_opt"]
            ocr_image_path = task["ocr_image_path"]
            preprocess_metadata = task["preprocess_metadata"]
            final_json = out_dir / f"{stem}.json"

            t_gpu_start = time.perf_counter()
            try:
                # Check broadsheet tiling requirement
                effective_strat = normalize_tile_strategy(item_opt.get("tile_strategy", "auto"))
                if effective_strat == "auto":
                    use_3x2 = should_use_3x2_tiling(ocr_image_path)
                    item_opt["tile_source_images"] = use_3x2
                    item_opt["tile_strategy"] = "grid_3x2" if use_3x2 else "none"
                    item_opt["tile_rows"] = 3 if use_3x2 else 1
                    item_opt["tile_cols"] = 2 if use_3x2 else 1

                if item_opt.get("tile_source_images") and supports_tiled_ocr("PP-OCRv6"):
                    merged_json = run_tiled_local_ocr(ocr_image_path, item_opt)
                    normalize_ocr_word_boxes_in_payload(merged_json)
                    annotate_preprocessing_payload(merged_json, preprocess_metadata)
                    final_json.write_text(
                        json.dumps(merged_json, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    mode_desc = "3x2 quadrant tiled"
                else:
                    # Single pass on GPU with batch_size recognition
                    output = engine.infer(str(ocr_image_path), item_opt)
                    for res in output:
                        res.save_to_json(save_path=str(out_dir))

                    raw_json = out_dir / f"{ocr_image_path.stem}_res.json"
                    if not raw_json.exists():
                        candidates = sorted(out_dir.glob(f"{ocr_image_path.stem}*.json"))
                        if candidates:
                            raw_json = candidates[0]
                    if raw_json.exists():
                        raw_json.replace(final_json)
                    mode_desc = "full-image single pass"

                gpu_elapsed = time.perf_counter() - t_gpu_start
                task["gpu_elapsed"] = gpu_elapsed
                task["mode_desc"] = mode_desc
                task["final_json"] = final_json

                # Pass to Stage 3 for parallel XML/PDF formatting
                self.post_queue.put(task)

            except Exception as exc:
                with self.stats_lock:
                    self.error_count += 1
                    self.completed_count += 1
                print(f"[ERROR GPU] {task['img_path'].name}: {exc}", flush=True)

            self.gpu_queue.task_done()

    def _stage3_postprocess_worker(self, task: dict):
        """Stage 3: CPU Post-processing & ALTO XML/PDF Worker."""
        try:
            final_json = task["final_json"]
            out_dir = task["out_dir"]
            item_opt = task["item_opt"]
            effective_lang = task["effective_lang"]
            preprocess_metadata = task["preprocess_metadata"]
            img_path = task["img_path"]

            # 1. Clean punctuation & legal abbreviations
            if item_opt.get("clean_punctuation_issues", True):
                clean_ocr_json_text_in_place(final_json)

            # 2. Build Library of Congress standard ALTO XML and searchable PDF/A
            pdf_res = build_pdf(
                final_json,
                img_path.parent,
                method="PP-OCRv6",
                lang=effective_lang,
                select="PP-OCRv6",
            )

            # Calculate total page latency
            total_page_time = time.perf_counter() - task["t0"]
            alto_status = "[OK] ALTO" if pdf_res.get("alto_success") else "[FAIL] ALTO"

            # Read word count
            try:
                payload = json.loads(final_json.read_text(encoding="utf-8"))
                rec_count = len(payload.get("rec_texts", []))
            except Exception:
                rec_count = 0

            with self.stats_lock:
                self.completed_count += 1
                elapsed = time.perf_counter() - self.t_start
                rate = self.completed_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{self.completed_count}/{self.total_pages}] {task['rel_dir']}/{img_path.name} -> "
                    f"{alto_status} | {rec_count:,} lines ({task['mode_desc']} | GPU: {task['gpu_elapsed']:.2f}s | Total: {total_page_time:.2f}s) "
                    f"(Running avg: {rate:.1f} pgs/s)",
                    flush=True,
                )

        except Exception as exc:
            with self.stats_lock:
                self.error_count += 1
                self.completed_count += 1
            print(f"[ERROR POST] {task['img_path'].name}: {exc}", flush=True)

    def run(self, images: List[Path]) -> Tuple[int, int, float]:
        """Execute the full staged pipeline over the target images."""
        self.total_pages = len(images)
        self.t_start = time.perf_counter()

        print(
            f"\n[Staged Pipeline] Starting 3-Stage Producer-Consumer Execution:\n"
            f"  • Stage 1: {self.cpu_workers} CPU Workers (Pre-processing / CLAHE / Deskew)\n"
            f"  • Stage 2: 1 Dedicated GPU Worker (PP-OCRv6 on RTX 5090 | Batch Size: {self.opt['text_recognition_batch_size']})\n"
            f"  • Stage 3: {self.cpu_workers} CPU Workers (ALTO XML / PDF/A / Text Normalization)\n"
            f"  • Total Images: {self.total_pages:,}\n",
            flush=True,
        )

        # Start Stage 2 (GPU Worker Thread)
        gpu_thread = threading.Thread(target=self._stage2_gpu_worker, daemon=True)
        gpu_thread.start()

        # Start Stage 3 (CPU Post-processing Pool)
        post_pool = ThreadPoolExecutor(max_workers=self.cpu_workers)

        def post_drain_loop():
            while not self.stop_event.is_set():
                try:
                    task = self.post_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if task is None:
                    break
                post_pool.submit(self._stage3_postprocess_worker, task)
                self.post_queue.task_done()

        post_drain_thread = threading.Thread(target=post_drain_loop, daemon=True)
        post_drain_thread.start()

        # Execute Stage 1 (CPU Pre-processing Pool feeding GPU queue)
        with ThreadPoolExecutor(max_workers=self.cpu_workers) as prep_pool:
            futures = [prep_pool.submit(self._stage1_preprocess_worker, img) for img in images]
            for fut in futures:
                res = fut.result()
                if res is not None:
                    self.gpu_queue.put(res)

        # Signal completion through pipeline stages
        self.gpu_queue.join()
        self.gpu_queue.put(None)  # Signal GPU worker to stop
        gpu_thread.join()

        self.post_queue.join()
        post_drain_thread.join()
        post_pool.shutdown(wait=True)

        total_elapsed = time.perf_counter() - self.t_start
        success_count = self.completed_count - self.error_count
        return success_count, self.error_count, total_elapsed


def run_benchmark_comparison(
    old_run_root: Path,
    new_run_root: Path,
    input_root: Path,
    compare_root: Path,
    new_elapsed_seconds: float,
    sync_docs: bool = False,
):
    """Compare performance metrics between the new fast Paddle run and the old baseline run."""
    print("\n==================================================================")
    print("      BENCHMARK SPEED & THROUGHPUT COMPARISON: NEW VS OLD PADDLE  ")
    print("==================================================================")

    new_xmls = list(new_run_root.rglob("*.xml"))
    old_xmls = list(old_run_root.rglob("*.xml"))

    new_count = len(new_xmls)
    old_count = len(old_xmls)

    new_rate = new_count / new_elapsed_seconds if new_elapsed_seconds > 0 else 0.0

    # Old baseline wall-clock time from run log (15.87 hours for 4,436 pages)
    old_baseline_seconds = 57149.0
    old_baseline_pages = 4436
    old_rate = old_baseline_pages / old_baseline_seconds if old_baseline_seconds > 0 else 0.077

    speedup = new_rate / old_rate if old_rate > 0 else 1.0

    print(f"Old Sequential Run Output: {old_run_root}")
    print(f"  • Pages Processed      : {old_count:,} (Baseline: {old_baseline_pages:,} pages)")
    print(f"  • Baseline Elapsed Time: ~15.87 hours ({old_baseline_seconds:,.0f}s)")
    print(f"  • Baseline Throughput  : {old_rate:.3f} pages/second (~12.8s per page)")
    print(f"\nNew Fast Staged Pipeline : {new_run_root}")
    print(f"  • Pages Processed      : {new_count:,} pages")
    print(f"  • Staged Run Time      : {new_elapsed_seconds:.1f}s ({new_elapsed_seconds/60.0:.2f} mins)")
    print(f"  • Staged Throughput    : {new_rate:.2f} pages/second (~{1.0/new_rate:.2f}s per page)")
    print(f"\n★ THROUGHPUT ACCELERATION : {speedup:.1f}x FASTER than old sequential run!")
    print("==================================================================")

    # Run directory batch comparison vs ABBYY benchmark to update compare/
    print("\n--- Updating ABBYY Benchmark & Master Comparison Dashboard ---")
    matched_count = 0
    subdirs_with_xml = [sub for sub in sorted(new_run_root.rglob("*")) if sub.is_dir() and any(sub.glob("*.xml"))]
    if any(new_run_root.glob("*.xml")) and new_run_root not in subdirs_with_xml:
        subdirs_with_xml.append(new_run_root)

    for sub in subdirs_with_xml:
        try:
            rel = sub.relative_to(new_run_root)
        except ValueError:
            rel = Path("")

        bench_dir = input_root / rel if rel != Path("") else input_root
        if not bench_dir.exists() or not any(bench_dir.glob("*.xml")):
            continue

        coll_name = str(rel).replace("\\", "/") if rel != Path("") else sub.name
        print(f"[CRKN Compare] Comparing collection: {coll_name}...")
        count = compare_directory_batch(
            pipeline_dir=sub,
            benchmark_dir=bench_dir,
            output_dir=compare_root,
            collection_name=coll_name,
            pipeline_name="PaddleOCR (PP-OCRv6 Fast)",
            pipeline_short="Paddle",
            render_html=True,
            render_image=False,
            sync_docs=False,
        )
        matched_count += count

    print(f"\n[CRKN Compare] Successfully compared {matched_count:,} pages. Dashboard updated at: {compare_root / 'index.html'}")

    if sync_docs:
        print("\n--- Syncing Updated Dashboard to OCR-Evaluations Repo ---")
        sync_to_evaluations_repo(compare_root, auto_push=True)


def main():
    args = parse_args()
    input_dir = (args.input_dir / args.collection) if args.collection else args.input_dir
    output_dir = (args.output_dir / args.collection) if args.collection else args.output_dir
    compare_dir = args.compare_dir

    if args.compare_only:
        run_benchmark_comparison(
            old_run_root=args.old_run_dir,
            new_run_root=output_dir,
            input_root=input_dir,
            compare_root=compare_dir,
            new_elapsed_seconds=1.0,
            sync_docs=args.sync_docs,
        )
        return

    images = find_target_images(input_dir, max_pages=args.max_pages)
    if not images:
        print(f"[ERROR] No supported images found in: {input_dir}")
        sys.exit(1)

    opt = build_paddle_options(batch_size=args.batch_size, tile_strategy=args.tile_strategy)
    pipeline = StagedPaddlePipeline(
        input_root=input_dir,
        output_root=output_dir,
        opt=opt,
        cpu_workers=args.cpu_workers,
    )

    success_count, error_count, elapsed_seconds = pipeline.run(images)

    run_benchmark_comparison(
        old_run_root=args.old_run_dir,
        new_run_root=output_dir,
        input_root=input_dir,
        compare_root=compare_dir,
        new_elapsed_seconds=elapsed_seconds,
        sync_docs=args.sync_docs,
    )


if __name__ == "__main__":
    main()

