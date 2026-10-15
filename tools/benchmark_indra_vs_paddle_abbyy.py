"""Three-way benchmark comparing Indra OCR (with 3x2 Broadsheet Tiling) vs PaddleOCR (PP-OCRv6) vs ABBYY.

Evaluates:
  1. Inference Speed / Latency (ms per page)
  2. Word Extraction Recall (Words extracted per page)
  3. Clean & Exact Concordance vs ABBYY
  4. Spatial Alignment / Mean IoU
  5. Lexical Validity & QA Win Rates
  6. Head-to-head Indra vs Paddle metrics
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.ocr_indra import get_indra_converter, process_image_with_indra
from services.pdf_generation import build_pdf
from tools.compare_alto import parse_alto_xml, compare_pages

BENCHMARK_BASE = Path(r"W:\Brittny\PDF Test")
PADDLE_BASE = Path(r"C:\Users\blapierre\Desktop\PDF Test - Final")
OUTPUT_BASE = root_dir / "scratch" / "indra_benchmark_tiled"

SAMPLE_PAGES = [
    # Category 1: Annuals (Commercial publishing, crisp type, tables)
    ("Annuals", "oocihm.72118", "oocihm.72118.1.jpg"),
    ("Annuals", "oocihm.72118", "oocihm.72118.5.jpg"),
    ("Annuals", "oocihm.72118", "oocihm.72118.10.jpg"),
    ("Annuals", "oocihm.8_06619_19270902", "oocihm.8_06619_19270902.5.jpg"),
    ("Annuals", "sru.00004_196107", "sru.00004_196107.12.jpg"),

    # Category 2: Heritage (Archival microfilm, 18th/19th century manuscripts & pamphlets, bleed-through)
    ("Heritage", "oocihm.lac_reel_c6063", "oocihm.lac_reel_c6063.10.jpg"),
    ("Heritage", "oocihm.lac_reel_c6063", "oocihm.lac_reel_c6063.25.jpg"),
    ("Heritage", "oocihm.lac_reel_c6063", "oocihm.lac_reel_c6063.50.jpg"),
    ("Heritage", "oocihm.lac_reel_t16535", "oocihm.lac_reel_t16535.15.jpg"),
    ("Heritage", "oocihm.lac_reel_c10143", "oocihm.lac_reel_c10143.20.jpg"),

    # Category 3: Newspapers (Large multi-column broadsheets, fine micro-print, headlines)
    ("Newspapers", "sfu.00001_19180302", "sfu.00001_19180302.1.jpg"),
    ("Newspapers", "sfu.00001_19180302", "sfu.00001_19180302.3.jpg"),
    ("Newspapers", "ome.00003_19290103", "ome.00003_19290103.2.jpg"),
    ("Newspapers", "ome.00003_19291219", "ome.00003_19291219.2.jpg"),
    ("Newspapers", "qmbn.00001_19550107", "qmbn.00001_19550107.1.jpg"),
]


def run_benchmark():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    conv = get_indra_converter()

    results = []

    print("=" * 80)
    print("STARTING 3-WAY BENCHMARK: INDRA OCR (TILED) vs PADDLEOCR vs ABBYY")
    print("=" * 80)

    for cat, folder, img_name in SAMPLE_PAGES:
        img_path = BENCHMARK_BASE / cat / folder / img_name
        stem = Path(img_name).stem
        abbyy_xml = BENCHMARK_BASE / cat / folder / f"{stem}.xml"
        paddle_xml = PADDLE_BASE / cat / folder / f"{stem}.xml"

        if not img_path.exists():
            print(f"Skipping {img_name}: image not found at {img_path}")
            continue

        print(f"\n[{cat}] Processing {folder} / {img_name}...")

        # 1. Run Indra OCR with broadsheet auto-tiling
        t0 = time.perf_counter()
        indra_payload, mode_desc = process_image_with_indra(
            img_path, conv, {"tile_strategy": "auto", "lang": "en"}
        )
        indra_elapsed_ms = (time.perf_counter() - t0) * 1000

        # Save Indra JSON & ALTO XML
        out_cat_dir = OUTPUT_BASE / cat / folder
        out_cat_dir.mkdir(parents=True, exist_ok=True)
        indra_json = out_cat_dir / f"{stem}.json"
        with open(indra_json, "w", encoding="utf-8") as f:
            json.dump(indra_payload, f, indent=2, ensure_ascii=False)

        build_pdf(
            json_path=indra_json,
            input_dir=img_path.parent,
            method="IndraOCR",
            lang="en",
            select="IndraOCR",
        )
        indra_xml = out_cat_dir / f"{stem}.xml"

        # 2. Parse ALTO XML files
        abbyy_pdata = parse_alto_xml(abbyy_xml, "ABBYY")[0] if abbyy_xml.exists() else None
        paddle_pdata = parse_alto_xml(paddle_xml, "PaddleOCR")[0] if paddle_xml.exists() else None
        indra_pdata = parse_alto_xml(indra_xml, "IndraOCR")[0] if indra_xml.exists() else None

        if not abbyy_pdata or not indra_pdata:
            print(f"   Missing ALTO XML for comparison on {img_name}")
            continue

        # 3. Compare Indra vs ABBYY
        res_indra_abbyy = compare_pages(indra_pdata, abbyy_pdata, item_name=stem)

        # 4. Compare Paddle vs ABBYY
        res_paddle_abbyy = (
            compare_pages(paddle_pdata, abbyy_pdata, item_name=stem)
            if paddle_pdata
            else None
        )

        # 5. Head-to-Head: Indra vs Paddle
        res_indra_paddle = (
            compare_pages(indra_pdata, paddle_pdata, item_name=stem)
            if paddle_pdata
            else None
        )

        stat = {
            "category": cat,
            "folder": folder,
            "image": img_name,
            "mode": mode_desc,
            "indra_time_ms": round(indra_elapsed_ms, 1),
            # Word volumes
            "words_abbyy": len(abbyy_pdata.boxes),
            "words_paddle": len(paddle_pdata.boxes) if paddle_pdata else 0,
            "words_indra": len(indra_pdata.boxes),
            # Concordance vs ABBYY
            "indra_exact_match": round(res_indra_abbyy.text_concordance_rate, 1),
            "indra_clean_match": round(res_indra_abbyy.clean_text_concordance_rate, 1),
            "paddle_exact_match": round(res_paddle_abbyy.text_concordance_rate, 1) if res_paddle_abbyy else 0.0,
            "paddle_clean_match": round(res_paddle_abbyy.clean_text_concordance_rate, 1) if res_paddle_abbyy else 0.0,
            # Spatial Alignment
            "indra_mean_iou": round(res_indra_abbyy.mean_iou, 3),
            "paddle_mean_iou": round(res_paddle_abbyy.mean_iou, 3) if res_paddle_abbyy else 0.0,
            "indra_spatial_match_rate": round(res_indra_abbyy.spatial_match_rate_benchmark, 1),
            "paddle_spatial_match_rate": round(res_paddle_abbyy.spatial_match_rate_benchmark, 1) if res_paddle_abbyy else 0.0,
            # Lexical QA Validity vs ABBYY
            "indra_lexical_wins": res_indra_abbyy.paddle_lexical_wins,
            "abbyy_lexical_wins_vs_indra": res_indra_abbyy.abbyy_lexical_wins,
            "indra_win_rate_vs_abbyy": round(res_indra_abbyy.lexical_win_rate_pct, 1),
            "paddle_win_rate_vs_abbyy": round(res_paddle_abbyy.lexical_win_rate_pct, 1) if res_paddle_abbyy else 0.0,
            "indra_valid_words": res_indra_abbyy.paddle_valid_words,
            "paddle_valid_words": res_paddle_abbyy.paddle_valid_words if res_paddle_abbyy else 0,
            "abbyy_valid_words": res_indra_abbyy.abbyy_valid_words,
            # Indra vs Paddle Direct
            "indra_vs_paddle_clean": round(res_indra_paddle.clean_text_concordance_rate, 1) if res_indra_paddle else 0.0,
            "indra_win_rate_vs_paddle": round(res_indra_paddle.lexical_win_rate_pct, 1) if res_indra_paddle else 0.0,
        }
        results.append(stat)

        p_clean = stat["paddle_clean_match"]
        i_clean = stat["indra_clean_match"]
        p_win = stat["paddle_win_rate_vs_abbyy"]
        i_win = stat["indra_win_rate_vs_abbyy"]
        print(
            f"   [{mode_desc}] Time: {indra_elapsed_ms:.1f}ms | Words: Indra={stat['words_indra']} | Paddle={stat['words_paddle']} | ABBYY={stat['words_abbyy']}"
        )
        print(
            f"   Clean Concordance vs ABBYY: Indra={i_clean}% (Paddle={p_clean}%) | Spatial Alignment: Indra={stat['indra_spatial_match_rate']}% (Paddle={stat['paddle_spatial_match_rate']}%)"
        )
        print(
            f"   QA Win Rate vs ABBYY: Indra={i_win}% (Paddle={p_win}%)"
        )

    # Save summary JSON
    summary_path = OUTPUT_BASE / "indra_tiled_vs_paddle_abbyy_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Macro rollups by Category
    print("\n" + "=" * 80)
    print("MACRO BENCHMARK ROLLUP BY CATEGORY")
    print("=" * 80)

    categories = ["Annuals", "Heritage", "Newspapers"]
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        if not cat_items:
            continue
        avg_ms = sum(r["indra_time_ms"] for r in cat_items) / len(cat_items)
        tot_abbyy_w = sum(r["words_abbyy"] for r in cat_items)
        tot_paddle_w = sum(r["words_paddle"] for r in cat_items)
        tot_indra_w = sum(r["words_indra"] for r in cat_items)
        avg_indra_clean = sum(r["indra_clean_match"] for r in cat_items) / len(cat_items)
        avg_paddle_clean = sum(r["paddle_clean_match"] for r in cat_items) / len(cat_items)
        avg_indra_spatial = sum(r["indra_spatial_match_rate"] for r in cat_items) / len(cat_items)
        avg_paddle_spatial = sum(r["paddle_spatial_match_rate"] for r in cat_items) / len(cat_items)
        avg_indra_win = sum(r["indra_win_rate_vs_abbyy"] for r in cat_items) / len(cat_items)
        avg_paddle_win = sum(r["paddle_win_rate_vs_abbyy"] for r in cat_items) / len(cat_items)

        print(f"\n--- Category: {cat} ({len(cat_items)} sample pages) ---")
        print(f"  CPU Latency: {avg_ms:.1f} ms/page ({1000/avg_ms:.1f} pages/sec)")
        print(f"  Words Extracted: Indra={tot_indra_w:,} | Paddle={tot_paddle_w:,} | ABBYY={tot_abbyy_w:,}")
        print(f"  Word Recall vs ABBYY: Indra={((tot_indra_w/tot_abbyy_w)-1)*100:+.1f}% | Paddle={((tot_paddle_w/tot_abbyy_w)-1)*100:+.1f}%")
        print(f"  Clean Concordance: Indra={avg_indra_clean:.1f}% | Paddle={avg_paddle_clean:.1f}%")
        print(f"  Spatial Match Rate: Indra={avg_indra_spatial:.1f}% | Paddle={avg_paddle_spatial:.1f}%")
        print(f"  QA Win Rate vs ABBYY: Indra={avg_indra_win:.1f}% | Paddle={avg_paddle_win:.1f}%")

    tot_time = sum(r["indra_time_ms"] for r in results)
    tot_abbyy_w = sum(r["words_abbyy"] for r in results)
    tot_paddle_w = sum(r["words_paddle"] for r in results)
    tot_indra_w = sum(r["words_indra"] for r in results)
    avg_indra_clean = sum(r["indra_clean_match"] for r in results) / len(results)
    avg_paddle_clean = sum(r["paddle_clean_match"] for r in results) / len(results)
    avg_indra_spatial = sum(r["indra_spatial_match_rate"] for r in results) / len(results)
    avg_paddle_spatial = sum(r["paddle_spatial_match_rate"] for r in results) / len(results)
    avg_indra_win = sum(r["indra_win_rate_vs_abbyy"] for r in results) / len(results)
    avg_paddle_win = sum(r["paddle_win_rate_vs_abbyy"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print(f"OVERALL ARCHIVE BENCHMARK ({len(results)} sample pages)")
    print(f"  Average Indra CPU Latency: {tot_time/len(results):.1f} ms/page ({len(results)/(tot_time/1000):.1f} pages/sec)")
    print(f"  Total Words: Indra={tot_indra_w:,} | Paddle={tot_paddle_w:,} | ABBYY={tot_abbyy_w:,}")
    print(f"  Word Recall vs ABBYY: Indra={((tot_indra_w/tot_abbyy_w)-1)*100:+.1f}% | Paddle={((tot_paddle_w/tot_abbyy_w)-1)*100:+.1f}%")
    print(f"  Clean Concordance vs ABBYY: Indra={avg_indra_clean:.1f}% | Paddle={avg_paddle_clean:.1f}%")
    print(f"  Spatial Alignment: Indra={avg_indra_spatial:.1f}% | Paddle={avg_paddle_spatial:.1f}%")
    print(f"  Lexical QA Win Rate vs ABBYY: Indra={avg_indra_win:.1f}% | Paddle={avg_paddle_win:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
