"""Full OCR execution and ABBYY benchmarking script for Indra OCR on the CRKN test set.

Replicates the full comparison suite workflow (similar to run_full_benchmark.py + compare_alto.py)
for Indra OCR vs. ABBYY FineReader, outputting results to compare_indra/.

Usage:
    # Run full OCR + comparison for Newspapers:
    python tools/compare_indra.py --collection Newspapers

    # Run full OCR + comparison for a single issue (e.g. 8 pages):
    python tools/compare_indra.py --collection Newspapers/ome.00003_19290103

    # Run quick sample test (e.g. max 2 pages per collection):
    python tools/compare_indra.py --collection Newspapers --max-pages 2

    # Re-run dashboard / comparison only without re-running OCR:
    python tools/compare_indra.py --compare-only

    # Run entire benchmark archive:
    python tools/compare_indra.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.constants import IMAGE_EXTENSIONS
from services.ocr_defaults import DEFAULT_INDRA_THREADS
from services.ocr_indra import get_indra_converter, process_image_with_indra
from services.pdf_generation import build_pdf
from tools.compare_alto import compare_directory_batch, sync_to_evaluations_repo

DEFAULT_INPUT_ROOT = Path(r"C:\Users\blapierre\Documents\PDF Test")
DEFAULT_OUTPUT_ROOT = root_dir / "output" / "indra"
DEFAULT_COMPARE_DIR = root_dir / "compare_indra"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Indra OCR benchmark and generate side-by-side comparison dashboard against ABBYY."
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
        help="Root directory where Indra OCR JSON, PDF, and ALTO XML outputs are saved.",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=DEFAULT_COMPARE_DIR,
        help="Directory where the comparison dashboard, viewers, and summaries are generated.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Optional relative subfolder/collection name (e.g., 'Newspapers' or 'Newspapers/ome.00003_19290103').",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of pages to process per collection (useful for quick validation).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_INDRA_THREADS,
        help=f"Number of CPU threads for Indra OCR (default: {DEFAULT_INDRA_THREADS}).",
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
        help="Skip OCR phase and only run ALTO comparison and dashboard generation over existing outputs.",
    )
    parser.add_argument(
        "--sync-docs",
        action="store_true",
        help="Sync generated compare_indra dashboard to OCR-Evaluations repo if present.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.25,
        help="Minimum Intersection-over-Union bounding box threshold for word alignment (default: 0.25).",
    )
    return parser.parse_args()


def find_target_images(input_dir: Path, max_pages: Optional[int] = None) -> List[Path]:
    """Find all supported image files in input_dir, grouped and sorted deterministically."""
    images = []
    for f in sorted(input_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    if max_pages is not None and max_pages > 0:
        return images[:max_pages]
    return images


def run_ocr_phase(
    images: List[Path],
    input_root: Path,
    output_root: Path,
    threads: int,
    tile_strategy: str = "auto",
) -> tuple[int, int]:
    """Execute Indra OCR on list of images and produce ALTO XML + PDF files."""
    print(f"\n[Indra OCR] Initializing Indra Converter (threads={threads})...")
    conv = get_indra_converter(threads=threads)
    opt = {"tile_strategy": tile_strategy, "lang": "auto"}

    success_count = 0
    fail_count = 0
    total_imgs = len(images)

    for idx, img_path in enumerate(images, 1):
        try:
            rel_dir = img_path.parent.relative_to(input_root)
        except ValueError:
            rel_dir = Path("")

        target_out_dir = output_root / rel_dir
        target_out_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        json_file = target_out_dir / f"{stem}.json"
        xml_file = target_out_dir / f"{stem}.xml"

        # Check if already processed
        if json_file.exists() and xml_file.exists():
            print(f"[{idx}/{total_imgs}] Already processed: {stem} (Skipping OCR)")
            success_count += 1
            continue

        print(f"[{idx}/{total_imgs}] Processing: {rel_dir}/{img_path.name}...")
        t0 = time.perf_counter()

        try:
            payload, mode_desc = process_image_with_indra(img_path, conv, opt)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            # Generate ALTO XML and PDF
            pdf_res = build_pdf(
                json_file,
                img_path.parent,
                method="IndraOCR",
                lang="en",
                select="IndraOCR",
            )

            rec_count = len(payload.get("rec_texts", []))
            elapsed = time.perf_counter() - t0
            alto_status = "[OK] ALTO" if pdf_res.get("alto_success") else "[FAIL] ALTO"
            print(f"       -> {alto_status} | {rec_count:,} text lines ({mode_desc}) in {elapsed:.2f}s")
            success_count += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"       -> ERROR on {img_path.name} ({elapsed:.2f}s): {exc}")
            fail_count += 1

    return success_count, fail_count


def run_comparison_phase(
    output_root: Path,
    input_root: Path,
    compare_root: Path,
    target_subfolder: str = "",
    min_iou: float = 0.25,
    sync_docs: bool = False,
) -> int:
    """Run directory comparison between Indra OCR ALTO outputs and ABBYY ALTO benchmark."""
    scan_dir = (output_root / target_subfolder).resolve() if target_subfolder else output_root
    print("\n==================================================================")
    print("      RUNNING ALTO XML COMPARISON & MASTER DASHBOARD GENERATION    ")
    print("==================================================================")
    print(f"Indra Pipeline Outputs : {scan_dir}")
    print(f"ABBYY Benchmark Source : {input_root}")
    print(f"Dashboard Destination  : {compare_root}")
    print("==================================================================")

    matched_count = 0
    subdirs_with_xml = []

    for sub in sorted(scan_dir.rglob("*")):
        if sub.is_dir() and any(sub.glob("*.xml")):
            subdirs_with_xml.append(sub)

    # Check scan_dir itself if it directly contains XML files
    if any(scan_dir.glob("*.xml")) and scan_dir not in subdirs_with_xml:
        subdirs_with_xml.append(scan_dir)

    if not subdirs_with_xml:
        print(f"[WARN] No ALTO XML files found in {scan_dir}")
        return 0

    for sub in subdirs_with_xml:
        try:
            rel = sub.relative_to(output_root)
        except ValueError:
            rel = Path("")

        bench_dir = input_root / rel if rel != Path("") else input_root
        if not bench_dir.exists() or not any(bench_dir.glob("*.xml")):
            print(f"[SKIP] No benchmark ALTO XMLs in matching folder: {bench_dir}")
            continue

        coll_name = str(rel).replace("\\", "/") if rel != Path("") else sub.name
        print(f"\n[CRKN Compare] Comparing collection: {coll_name}...")
        count = compare_directory_batch(
            pipeline_dir=sub,
            benchmark_dir=bench_dir,
            output_dir=compare_root,
            collection_name=coll_name,
            pipeline_name="Indra OCR",
            pipeline_short="Indra",
            render_html=True,
            render_image=False,
            min_iou=min_iou,
            sync_docs=False,  # We handle sync once at the end
        )
        matched_count += count

    if sync_docs and matched_count > 0:
        print("\n--- Syncing to OCR-Evaluations Repo & GitHub Pages ---")
        sync_to_evaluations_repo(compare_root, auto_push=True)

    return matched_count


def main():
    args = parse_args()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    compare_root = args.compare_dir.resolve()

    if args.collection:
        target_input = (input_root / args.collection).resolve()
        target_output = (output_root / args.collection).resolve()
    else:
        target_input = input_root
        target_output = output_root

    print("==================================================================")
    print("        CRKN INDRA OCR PIPELINE EXECUTION & ABBYY BENCHMARK       ")
    print("==================================================================")
    print(f"Target Input Path : {target_input}")
    print(f"Target Output Path: {target_output}")
    print(f"Compare Dashboard : {compare_root}")
    print(f"Tiling Strategy   : {args.tile_strategy}")
    print(f"CPU Threads       : {args.threads}")
    print(f"Compare Only Mode : {args.compare_only}")
    print("==================================================================")

    if not target_input.exists():
        print(f"ERROR: Target input path does not exist: {target_input}")
        sys.exit(1)

    target_output.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: OCR
    if not args.compare_only:
        images = find_target_images(target_input, max_pages=args.max_pages)
        if not images:
            print(f"No supported images found in: {target_input}")
            sys.exit(0)

        print(f"\nFound {len(images)} images to process using Indra OCR.")
        t_start = time.perf_counter()
        success, fails = run_ocr_phase(
            images=images,
            input_root=input_root,
            output_root=output_root,
            threads=args.threads,
            tile_strategy=args.tile_strategy,
        )
        t_total = time.perf_counter() - t_start
        print(f"\n[Phase 1 Finished] {success} succeeded, {fails} failed in {t_total:.1f}s")
    else:
        print("\n[Phase 1 Skipped] --compare-only specified.")

    # Phase 2: Comparison & Dashboard Generation
    matched = run_comparison_phase(
        output_root=output_root,
        input_root=input_root,
        compare_root=compare_root,
        target_subfolder=args.collection,
        min_iou=args.min_iou,
        sync_docs=args.sync_docs,
    )

    print("\n==================================================================")
    print("                   INDRA BENCHMARK COMPLETE                       ")
    print("==================================================================")
    print(f"Total Pages Compared: {matched}")
    print(f"Master Dashboard    : {compare_root / 'index.html'}")
    print(f"Metrics CSV         : {compare_root / 'comparison_summary.csv'}")
    print(f"Metrics JSON        : {compare_root / 'comparison_summary.json'}")
    print("==================================================================")


if __name__ == "__main__":
    main()
