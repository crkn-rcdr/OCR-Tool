"""Comprehensive comparison of Tiling Strategies on Historical Newspapers.

Compares:
1. No Tiles (Full-image high-resolution pass)
2. 3x1 Grid (3 horizontal strip tiles, 96px overlap)
3. 3x2 Grid (6 quadrant tiles, 96px overlap)

Across a statistically representative sample of 7 newspaper publications:
- 2 English broadsheets (ome.00003_19290103.1, ome.00003_19291219.1)
- 2 French newspapers (qmbn.00001_19550107.1, qmbn.00001_19560615.1)
- 3 Chinese broadsheets (sfu.00001_19180302.1, sfu.00001_19410110.1, sfu.00001_19800811.1)

Evaluates:
- Total words recognized (Pipeline vs ABBYY Benchmark)
- Word recall delta (%)
- Concordance rate (Clean % and Exact %)
- Paddle Lexical QA Win Rate (%)
- Spatial match rate (%) & Mean IoU
- Execution time per page & total throughput
"""

import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.ocr_pp import run_ocr_for_image
from services.pdf_generation import build_pdf
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOCLAYOUT_THRESHOLD,
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
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
    DEFAULT_TILE_OVERLAP_PX,
)
from tools.compare_alto import parse_alto_xml, compare_pages

BENCHMARK_BASE = Path(r"C:\Users\blapierre\Documents\PDF Test\Newspapers")
OUTPUT_BASE = root_dir / "tests" / "tiling_comparison"

TEST_PAGES = [
    {
        "collection": "ome.00003_19290103",
        "stem": "ome.00003_19290103.1",
        "lang": "en",
        "language_name": "English",
        "pub": "The Omemee Mirror (1929)",
    },
    {
        "collection": "ome.00003_19291219",
        "stem": "ome.00003_19291219.1",
        "lang": "en",
        "language_name": "English",
        "pub": "The Omemee Mirror (1929)",
    },
    {
        "collection": "qmbn.00001_19550107",
        "stem": "qmbn.00001_19550107.1",
        "lang": "fr",
        "language_name": "French",
        "pub": "Le Canadien Réformé (1955)",
    },
    {
        "collection": "qmbn.00001_19560615",
        "stem": "qmbn.00001_19560615.1",
        "lang": "fr",
        "language_name": "French",
        "pub": "Le Canadien Réformé (1956)",
    },
    {
        "collection": "sfu.00001_19180302",
        "stem": "sfu.00001_19180302.1",
        "lang": "ch",
        "language_name": "Chinese",
        "pub": "The Chinese Times 大漢公報 (1918)",
    },
    {
        "collection": "sfu.00001_19410110",
        "stem": "sfu.00001_19410110.1",
        "lang": "ch",
        "language_name": "Chinese",
        "pub": "The Chinese Times 大漢公報 (1941)",
    },
    {
        "collection": "sfu.00001_19800811",
        "stem": "sfu.00001_19800811.1",
        "lang": "ch",
        "language_name": "Chinese",
        "pub": "The Chinese Times 大漢公報 (1980)",
    },
]

STRATEGIES = [
    {
        "id": "no_tiles",
        "label": "No Tiles (Full-Image)",
        "tile_source_images": False,
        "tile_strategy": "none",
        "tile_rows": 1,
        "tile_cols": 1,
    },
    {
        "id": "grid_3x1",
        "label": "3x1 Grid (3 horizontal tiles)",
        "tile_source_images": True,
        "tile_strategy": "grid",
        "tile_rows": 3,
        "tile_cols": 1,
    },
    {
        "id": "grid_3x2",
        "label": "3x2 Grid (6 quadrant tiles)",
        "tile_source_images": True,
        "tile_strategy": "grid_3x2",
        "tile_rows": 3,
        "tile_cols": 2,
    },
]


def build_base_options() -> Dict[str, Any]:
    return {
        "method": "PP-OCRv6",
        "ocr_version": DEFAULT_OCR_VERSION,
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
        "tile_overlap_px": int(DEFAULT_TILE_OVERLAP_PX),
        "tile_doclayout_padding_px": int(DEFAULT_DOCLAYOUT_PADDING_PX),
        "tile_doclayout_threshold": float(DEFAULT_DOCLAYOUT_THRESHOLD),
        "tile_doclayout_recovery": False,
        "post_ocr_vl_enabled": False,  # Isolate tiling geometry without LLM calls
    }


def run_experiment() -> Dict[str, Any]:
    print("=" * 78)
    print("   CRKN TILING BENCHMARK EXPERIMENT: NO TILES vs 3x1 GRID vs 3x2 GRID   ")
    print("=" * 78)
    print(f"Dataset Sample: {len(TEST_PAGES)} historical newspaper broadsheet pages")
    print(f"Publications  : 2 English, 2 French, 3 Chinese (100% collection coverage)")
    print("Strategies    : No Tiles, 3x1 Grid (3 tiles), 3x2 Grid (6 tiles)")
    print("=" * 78)

    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    base_opt = build_base_options()

    for strat in STRATEGIES:
        strat_id = strat["id"]
        strat_label = strat["label"]
        print(f"\n>>> Running Strategy: {strat_label} ({strat_id}) <<<")
        strat_dir = OUTPUT_BASE / strat_id
        strat_dir.mkdir(parents=True, exist_ok=True)

        opt = copy.deepcopy(base_opt)
        opt["tile_source_images"] = strat["tile_source_images"]
        opt["tile_strategy"] = strat["tile_strategy"]
        opt["tile_rows"] = strat["tile_rows"]
        opt["tile_cols"] = strat["tile_cols"]

        for item in TEST_PAGES:
            stem = item["stem"]
            collection = item["collection"]
            lang = item["lang"]
            opt["lang"] = lang

            in_img = BENCHMARK_BASE / collection / f"{stem}.jpg"
            bench_xml = BENCHMARK_BASE / collection / f"{stem}.xml"

            if not in_img.exists() or not bench_xml.exists():
                print(f"  [SKIP] Missing image or benchmark XML for {stem}")
                continue

            page_out_dir = strat_dir / collection
            page_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Processing {stem} ({item['language_name']})...", end="", flush=True)
            t0 = time.perf_counter()

            # 1. Run OCR
            final_json, _ = run_ocr_for_image(
                image_path=in_img,
                out_dir=page_out_dir,
                opt=opt,
            )

            # 2. Generate ALTO XML & PDF
            build_res = build_pdf(
                json_path=final_json,
                input_dir=in_img.parent,
                method=opt["method"],
                lang=lang,
                select=None,
            )
            alto_xml = page_out_dir / f"{stem}.xml"
            elapsed = time.perf_counter() - t0

            if not alto_xml.exists():
                print(f" FAILED (no ALTO generated in {elapsed:.1f}s)")
                continue

            # 3. Compare with ABBYY Benchmark
            p_pages = parse_alto_xml(alto_xml, "PaddleOCR")
            b_pages = parse_alto_xml(bench_xml, "ABBYY")

            if not p_pages or not b_pages:
                print(f" FAILED (parse error in {elapsed:.1f}s)")
                continue

            comp = compare_pages(p_pages[0], b_pages[0], stem, min_iou=0.25)

            rec = {
                "stem": stem,
                "collection": collection,
                "language": item["language_name"],
                "strategy": strat_id,
                "time_sec": round(elapsed, 2),
                "pipeline_words": comp.pipeline_total_words,
                "benchmark_words": comp.benchmark_total_words,
                "word_delta": comp.pipeline_total_words - comp.benchmark_total_words,
                "word_delta_pct": round(
                    (comp.pipeline_total_words - comp.benchmark_total_words)
                    / max(1, comp.benchmark_total_words)
                    * 100.0,
                    2,
                ),
                "matched_boxes": len(comp.matched_pairs),
                "spatial_match_pct": round(comp.spatial_match_rate_pipeline, 2),
                "clean_concordance_pct": round(comp.clean_text_concordance_rate, 2),
                "exact_concordance_pct": round(comp.text_concordance_rate, 2),
                "mean_sim": round(comp.mean_char_similarity, 4),
                "mean_iou": round(comp.mean_iou, 4),
                "paddle_wins": comp.paddle_lexical_wins,
                "abbyy_wins": comp.abbyy_lexical_wins,
                "both_valid": comp.both_valid_count,
                "neither_valid": comp.neither_valid_count,
                "paddle_win_rate_pct": round(comp.lexical_win_rate_pct, 2),
                "discrepancies": len(comp.discrepancies),
            }
            results[strat_id].append(rec)
            print(
                f" DONE in {elapsed:.1f}s | Words: {rec['pipeline_words']} (ABBYY: {rec['benchmark_words']}) | "
                f"Conc: {rec['clean_concordance_pct']}% | P-Win: {rec['paddle_win_rate_pct']}%"
            )

    return results


def print_comparison_table(results: Dict[str, List[Dict[str, Any]]]):
    print("\n" + "=" * 92)
    print("                      AGGREGATE EXPERIMENT RESULTS                      ")
    print("=" * 92)

    headers = [
        "Strategy",
        "Total Words",
        "vs ABBYY",
        "Clean Conc %",
        "P-Win Rate %",
        "Spatial Match %",
        "Total Time",
        "Avg s/Page",
    ]
    print(f"{headers[0]:<24} {headers[1]:<12} {headers[2]:<10} {headers[3]:<14} {headers[4]:<14} {headers[5]:<16} {headers[6]:<12} {headers[7]:<10}")
    print("-" * 115)

    summary_by_strat = {}

    for strat in STRATEGIES:
        sid = strat["id"]
        recs = results.get(sid, [])
        if not recs:
            continue

        tot_p_words = sum(r["pipeline_words"] for r in recs)
        tot_b_words = sum(r["benchmark_words"] for r in recs)
        delta_pct = (tot_p_words - tot_b_words) / max(1, tot_b_words) * 100.0
        avg_clean_conc = sum(r["clean_concordance_pct"] for r in recs) / len(recs)
        tot_p_wins = sum(r["paddle_wins"] for r in recs)
        tot_b_wins = sum(r["abbyy_wins"] for r in recs)
        decisive = tot_p_wins + tot_b_wins
        win_rate = (tot_p_wins / decisive * 100.0) if decisive > 0 else 50.0
        avg_spatial = sum(r["spatial_match_pct"] for r in recs) / len(recs)
        tot_time = sum(r["time_sec"] for r in recs)
        avg_time = tot_time / len(recs)

        summary_by_strat[sid] = {
            "label": strat["label"],
            "total_words": tot_p_words,
            "benchmark_words": tot_b_words,
            "delta_pct": delta_pct,
            "avg_clean_conc": avg_clean_conc,
            "win_rate": win_rate,
            "avg_spatial": avg_spatial,
            "total_time": tot_time,
            "avg_time": avg_time,
        }

        sign = "+" if delta_pct >= 0 else ""
        print(
            f"{strat['label']:<24} {tot_p_words:<12,d} {sign}{delta_pct:<9.1f}% {avg_clean_conc:<13.1f}% {win_rate:<13.1f}% {avg_spatial:<15.1f}% {tot_time:<11.1f}s {avg_time:<9.1f}s"
        )

    print("=" * 115)

    # Breakdown by language
    print("\n--- Language-Specific Word Count Breakdown ---")
    for lang in ["English", "French", "Chinese"]:
        print(f"\n[{lang} Pages]")
        print(f"{'Strategy':<24} {'Words Recognized':<18} {'Concordance %':<16} {'Paddle Win %':<16} {'Avg Time':<12}")
        print("-" * 88)
        for strat in STRATEGIES:
            sid = strat["id"]
            recs = [r for r in results.get(sid, []) if r["language"] == lang]
            if not recs:
                continue
            words = sum(r["pipeline_words"] for r in recs)
            conc = sum(r["clean_concordance_pct"] for r in recs) / len(recs)
            pw = sum(r["paddle_wins"] for r in recs)
            bw = sum(r["abbyy_wins"] for r in recs)
            wr = (pw / (pw + bw) * 100.0) if (pw + bw) > 0 else 50.0
            avg_t = sum(r["time_sec"] for r in recs) / len(recs)
            print(f"{strat['label']:<24} {words:<18,d} {conc:<15.1f}% {wr:<15.1f}% {avg_t:<11.1f}s")

    # Save to JSON
    json_path = OUTPUT_BASE / "tiling_comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_by_strat, "details": results}, f, indent=2)
    print(f"\n[Saved Full Results to: {json_path}]")


if __name__ == "__main__":
    results = run_experiment()
    print_comparison_table(results)

