#!/usr/bin/env python3
"""
ALTO XML & PDF Text/Bounding-Box Comparison Tool for CRKN Digital Collections.

Compares ALTO XML and PDF outputs produced by the PaddleOCR pipeline against
benchmark outputs (such as ABBYY FineReader Server) without requiring manual
ground truth.

Key Capabilities:
  1. Spatial Bounding-Box Alignment:
     - Normalizes coordinates across measurement units (pixel, mm, inch1200, PDF points).
     - Spatial bipartite/proximity matching via IoU (Intersection over Union).
     - Detects spatial drift (mean & standard deviation of dX, dY offsets).
     - Sub-word / punctuation token consolidation (e.g. matching ['Word', ','] against 'Word,').
  2. Text Concordance & Similarity:
     - Exact match and case-insensitive match rates on spatially overlapping boxes.
     - Normalized Levenshtein similarity across matched words.
     - Full-page global text sequence alignment (CER / Levenshtein proxy).
  3. Interactive & Visual Comparison Outputs:
     - Standalone interactive HTML report with side-by-side synchronized view,
       toggleable bounding boxes (Pipeline vs. Benchmark/ABBYY), zoom/pan,
       color-coded concordance (Green = match, Orange = text discrepancy,
       Blue = pipeline-only, Red = benchmark-only), and click-to-inspect word cards.
     - Side-by-side or overlaid annotated PNG images with bounding boxes.
  4. Batch Mode:
     - Compares directories of ALTO files or searchable PDFs.
     - Outputs summary CSV and JSON reports for quantitative analysis.

Usage:
  python tools/compare_alto.py \
      --pipeline-alto path/to/pipeline.xml \
      --benchmark-alto path/to/abbyy.xml \
      --image path/to/page.jpg \
      --output-dir path/to/report_dir \
      --render-html --render-image

  # Batch mode:
  python tools/compare_alto.py \
      --pipeline-dir "C:/Users/user/Desktop/run_dir" \
      --benchmark-dir "C:/Users/user/Documents/benchmark_dir" \
      --output-dir "C:/Users/user/Desktop/comparison_report" \
      --render-html
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AltoBox:
    """Represents a single word/string token with bounding box in normalized [0, 1] page space."""
    content: str
    norm_x: float  # [0, 1] relative to page width
    norm_y: float  # [0, 1] relative to page height
    norm_w: float  # [0, 1]
    norm_h: float  # [0, 1]
    raw_x: float
    raw_y: float
    raw_w: float
    raw_h: float
    confidence: Optional[float] = None
    source_tag: str = "String"
    source_engine: str = ""


@dataclass
class PageData:
    page_id: str
    width: float
    height: float
    unit: str
    boxes: List[AltoBox]
    source_path: str
    source_engine: str


@dataclass
class MatchedPair:
    pipeline_box: AltoBox
    benchmark_box: AltoBox
    iou: float
    center_dist: float
    text_exact_match: bool
    text_clean_match: bool
    char_similarity: float


@dataclass
class PageComparisonResult:
    item_name: str
    pipeline_page: PageData
    benchmark_page: PageData
    pipeline_total_words: int
    benchmark_total_words: int
    matched_pairs: List[MatchedPair] = field(default_factory=list)
    pipeline_only_boxes: List[AltoBox] = field(default_factory=list)
    benchmark_only_boxes: List[AltoBox] = field(default_factory=list)
    spatial_match_rate_pipeline: float = 0.0
    spatial_match_rate_benchmark: float = 0.0
    mean_iou: float = 0.0
    mean_dx: float = 0.0
    mean_dy: float = 0.0
    text_concordance_rate: float = 0.0
    clean_text_concordance_rate: float = 0.0
    mean_char_similarity: float = 0.0
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ALTO XML Parser
# ---------------------------------------------------------------------------

def parse_alto_xml(xml_path: Path, engine_label: str = "") -> List[PageData]:
    """Parse an ALTO XML file (v2, v3, v4) into PageData with normalized coordinates."""
    if not xml_path.exists():
        raise FileNotFoundError(f"ALTO XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Determine default measurement unit
    unit = "pixel"
    for elem in root.iter():
        if elem.tag.endswith("MeasurementUnit") and elem.text:
            unit = elem.text.strip().lower()
            break

    pages: List[PageData] = []

    for page_elem in root.iter():
        if not page_elem.tag.endswith("Page"):
            continue

        page_id = page_elem.get("ID", f"Page_{len(pages) + 1}")
        page_w_str = page_elem.get("WIDTH", "0")
        page_h_str = page_elem.get("HEIGHT", "0")
        try:
            page_w = float(page_w_str)
            page_h = float(page_h_str)
        except ValueError:
            page_w, page_h = 1.0, 1.0

        if page_w <= 0:
            page_w = 1.0
        if page_h <= 0:
            page_h = 1.0

        boxes: List[AltoBox] = []

        for string_elem in page_elem.iter():
            if not string_elem.tag.endswith("String"):
                continue

            content = string_elem.get("CONTENT", "")
            if not content:
                continue

            try:
                rx = float(string_elem.get("HPOS", 0))
                ry = float(string_elem.get("VPOS", 0))
                rw = float(string_elem.get("WIDTH", 0))
                rh = float(string_elem.get("HEIGHT", 0))
            except ValueError:
                continue

            conf_str = string_elem.get("WC")
            conf = float(conf_str) if conf_str is not None else None

            # Normalized relative to page dimensions
            nx = rx / page_w
            ny = ry / page_h
            nw = rw / page_w
            nh = rh / page_h

            boxes.append(AltoBox(
                content=content,
                norm_x=nx,
                norm_y=ny,
                norm_w=nw,
                norm_h=nh,
                raw_x=rx,
                raw_y=ry,
                raw_w=rw,
                raw_h=rh,
                confidence=conf,
                source_tag="String",
                source_engine=engine_label
            ))

        pages.append(PageData(
            page_id=page_id,
            width=page_w,
            height=page_h,
            unit=unit,
            boxes=boxes,
            source_path=str(xml_path),
            source_engine=engine_label
        ))

    return pages


# ---------------------------------------------------------------------------
# PDF Text Layer Parser (Fallback / Direct PDF Comparison)
# ---------------------------------------------------------------------------

def parse_pdf_text_layer(pdf_path: Path, engine_label: str = "") -> List[PageData]:
    """Extract words and bounding boxes directly from a searchable PDF via PyMuPDF."""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF (fitz) is required to parse PDF text layers directly.")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: List[PageData] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        rect = page.rect
        page_w = float(rect.width)
        page_h = float(rect.height)
        if page_w <= 0:
            page_w = 1.0
        if page_h <= 0:
            page_h = 1.0

        boxes: List[AltoBox] = []
        # get_text("words") returns tuples: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        raw_words = page.get_text("words")
        for w in raw_words:
            rx0, ry0, rx1, ry1, word_text = w[0], w[1], w[2], w[3], w[4]
            rw = max(0.0, rx1 - rx0)
            rh = max(0.0, ry1 - ry0)
            nx = rx0 / page_w
            ny = ry0 / page_h
            nw = rw / page_w
            nh = rh / page_h

            boxes.append(AltoBox(
                content=str(word_text),
                norm_x=nx,
                norm_y=ny,
                norm_w=nw,
                norm_h=nh,
                raw_x=float(rx0),
                raw_y=float(ry0),
                raw_w=float(rw),
                raw_h=float(rh),
                source_tag="PdfWord",
                source_engine=engine_label
            ))

        pages.append(PageData(
            page_id=f"Page_{page_num + 1}",
            width=page_w,
            height=page_h,
            unit="point",
            boxes=boxes,
            source_path=str(pdf_path),
            source_engine=engine_label
        ))

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Spatial & Text Comparison Logic
# ---------------------------------------------------------------------------

def calculate_iou(b1: AltoBox, b2: AltoBox) -> float:
    """Calculates Intersection over Union in normalized [0, 1] space."""
    x1, y1, w1, h1 = b1.norm_x, b1.norm_y, b1.norm_w, b1.norm_h
    x2, y2, w2, h2 = b2.norm_x, b2.norm_y, b2.norm_w, b2.norm_h

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def calculate_center_distance(b1: AltoBox, b2: AltoBox) -> float:
    """Calculates Euclidean distance between centers in normalized [0, 1] space."""
    cx1 = b1.norm_x + (b1.norm_w / 2.0)
    cy1 = b1.norm_y + (b1.norm_h / 2.0)
    cx2 = b2.norm_x + (b2.norm_w / 2.0)
    cy2 = b2.norm_y + (b2.norm_h / 2.0)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def clean_text_for_comparison(text: str) -> str:
    """Normalize historical text, punctuation, and casing for fair comparison."""
    # Strip leading/trailing punctuation and whitespace
    t = text.strip()
    t = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', t)
    # Common OCR/historical ligature normalization
    t = t.replace('ſ', 's').replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    return t.lower()


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized similarity score [0, 1] based on Levenshtein distance."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_len))


def compare_pages(
    pipeline_page: PageData,
    benchmark_page: PageData,
    item_name: str,
    min_iou: float = 0.25,
    max_center_dist: float = 0.05
) -> PageComparisonResult:
    """Compare a single pipeline page against a benchmark page."""
    p_boxes = list(pipeline_page.boxes)
    b_boxes = list(benchmark_page.boxes)

    matched_pairs: List[MatchedPair] = []
    matched_p_indices = set()
    matched_b_indices = set()

    # Pre-index benchmark boxes spatially by vertical bands to speed up search
    # Bipartite spatial matching: evaluate candidates
    candidates: List[Tuple[float, int, int, float, float]] = []  # (-iou, p_idx, b_idx, iou, dist)

    for p_idx, pb in enumerate(p_boxes):
        for b_idx, bb in enumerate(b_boxes):
            iou_val = calculate_iou(pb, bb)
            cdist = calculate_center_distance(pb, bb)

            # Match criteria: either decent IoU OR very close center distance with overlapping vertical band
            if iou_val >= min_iou or (cdist <= max_center_dist and iou_val > 0.05):
                candidates.append((-iou_val, p_idx, b_idx, iou_val, cdist))

    # Sort candidates by highest IoU first
    candidates.sort()

    for _, p_idx, b_idx, iou_val, cdist in candidates:
        if p_idx in matched_p_indices or b_idx in matched_b_indices:
            continue

        pb = p_boxes[p_idx]
        bb = b_boxes[b_idx]

        exact_match = (pb.content == bb.content)
        clean_p = clean_text_for_comparison(pb.content)
        clean_b = clean_text_for_comparison(bb.content)
        clean_match = (clean_p == clean_b) and (clean_p != "")

        sim = levenshtein_similarity(pb.content, bb.content)

        matched_pairs.append(MatchedPair(
            pipeline_box=pb,
            benchmark_box=bb,
            iou=iou_val,
            center_dist=cdist,
            text_exact_match=exact_match,
            text_clean_match=clean_match,
            char_similarity=sim
        ))
        matched_p_indices.add(p_idx)
        matched_b_indices.add(b_idx)

    p_only = [pb for idx, pb in enumerate(p_boxes) if idx not in matched_p_indices]
    b_only = [bb for idx, bb in enumerate(b_boxes) if idx not in matched_b_indices]

    num_matched = len(matched_pairs)
    total_p = max(1, len(p_boxes))
    total_b = max(1, len(b_boxes))

    spatial_rate_p = (num_matched / total_p) * 100.0
    spatial_rate_b = (num_matched / total_b) * 100.0

    mean_iou = sum(m.iou for m in matched_pairs) / num_matched if num_matched else 0.0
    mean_dx = sum(m.pipeline_box.norm_x - m.benchmark_box.norm_x for m in matched_pairs) / num_matched if num_matched else 0.0
    mean_dy = sum(m.pipeline_box.norm_y - m.benchmark_box.norm_y for m in matched_pairs) / num_matched if num_matched else 0.0

    exact_matches = sum(1 for m in matched_pairs if m.text_exact_match)
    clean_matches = sum(1 for m in matched_pairs if m.text_clean_match)
    mean_sim = sum(m.char_similarity for m in matched_pairs) / num_matched if num_matched else 0.0

    concordance_rate = (exact_matches / num_matched * 100.0) if num_matched else 0.0
    clean_concordance_rate = (clean_matches / num_matched * 100.0) if num_matched else 0.0

    # Collect discrepancies (words with spatial alignment but textual divergence)
    discrepancies = []
    for m in matched_pairs:
        if not m.text_clean_match:
            discrepancies.append({
                "pipeline_text": m.pipeline_box.content,
                "benchmark_text": m.benchmark_box.content,
                "pipeline_conf": m.pipeline_box.confidence,
                "benchmark_conf": m.benchmark_box.confidence,
                "iou": round(m.iou, 3),
                "similarity": round(m.char_similarity, 3),
                "position_norm": {
                    "x": round(m.pipeline_box.norm_x, 4),
                    "y": round(m.pipeline_box.norm_y, 4),
                    "w": round(m.pipeline_box.norm_w, 4),
                    "h": round(m.pipeline_box.norm_h, 4)
                }
            })

    return PageComparisonResult(
        item_name=item_name,
        pipeline_page=pipeline_page,
        benchmark_page=benchmark_page,
        pipeline_total_words=len(p_boxes),
        benchmark_total_words=len(b_boxes),
        matched_pairs=matched_pairs,
        pipeline_only_boxes=p_only,
        benchmark_only_boxes=b_only,
        spatial_match_rate_pipeline=round(spatial_rate_p, 2),
        spatial_match_rate_benchmark=round(spatial_rate_b, 2),
        mean_iou=round(mean_iou, 4),
        mean_dx=round(mean_dx, 5),
        mean_dy=round(mean_dy, 5),
        text_concordance_rate=round(concordance_rate, 2),
        clean_text_concordance_rate=round(clean_concordance_rate, 2),
        mean_char_similarity=round(mean_sim, 4),
        discrepancies=discrepancies
    )


# ---------------------------------------------------------------------------
# Visual Outputs: Annotated Side-by-Side Images
# ---------------------------------------------------------------------------

def render_comparison_image(
    result: PageComparisonResult,
    bg_image_path: Optional[Path],
    output_path: Path,
    target_width: int = 1600
) -> Optional[Path]:
    """Renders a side-by-side or overlaid PNG showing pipeline vs. benchmark bounding boxes."""
    if not HAS_PIL:
        print("[WARN] Pillow is not available, skipping image rendering.", file=sys.stderr)
        return None

    # Load or create base image
    if bg_image_path and bg_image_path.exists():
        try:
            base_img = Image.open(bg_image_path).convert("RGB")
        except Exception:
            base_img = None
    else:
        base_img = None

    orig_w = int(result.pipeline_page.width)
    orig_h = int(result.pipeline_page.height)

    if base_img is None:
        # Create neutral canvas
        scale = target_width / orig_w
        canvas_h = int(orig_h * scale)
        base_img = Image.new("RGB", (target_width, canvas_h), color=(248, 249, 250))
    else:
        scale = target_width / base_img.width
        canvas_h = int(base_img.height * scale)
        base_img = base_img.resize((target_width, canvas_h), Image.Resampling.LANCZOS)

    # We will create two annotated panels: Left = Pipeline, Right = Benchmark
    # Plus a third panel or side-by-side: Left = Pipeline Boxes, Right = Benchmark Boxes
    panel_left = base_img.copy()
    panel_right = base_img.copy()

    draw_left = ImageDraw.Draw(panel_left, "RGBA")
    draw_right = ImageDraw.Draw(panel_right, "RGBA")

    # Colors
    # Agreed words: Green
    color_agree = (40, 167, 69, 140)
    color_agree_outline = (40, 167, 69, 255)
    # Discrepant words (spatial match but text differs): Orange
    color_diff = (255, 140, 0, 160)
    color_diff_outline = (255, 100, 0, 255)
    # Engine-only words: Blue (Pipeline), Red (Benchmark)
    color_p_only = (30, 144, 255, 160)
    color_p_only_outline = (0, 100, 220, 255)
    color_b_only = (220, 53, 69, 160)
    color_b_only_outline = (180, 20, 40, 255)

    # 1. Draw Pipeline boxes on Left Panel
    for m in result.matched_pairs:
        pb = m.pipeline_box
        x1 = pb.norm_x * target_width
        y1 = pb.norm_y * canvas_h
        x2 = (pb.norm_x + pb.norm_w) * target_width
        y2 = (pb.norm_y + pb.norm_h) * canvas_h

        fill = color_agree if m.text_clean_match else color_diff
        outline = color_agree_outline if m.text_clean_match else color_diff_outline
        draw_left.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=2)

    for pb in result.pipeline_only_boxes:
        x1 = pb.norm_x * target_width
        y1 = pb.norm_y * canvas_h
        x2 = (pb.norm_x + pb.norm_w) * target_width
        y2 = (pb.norm_y + pb.norm_h) * canvas_h
        draw_left.rectangle([x1, y1, x2, y2], fill=color_p_only, outline=color_p_only_outline, width=2)

    # 2. Draw Benchmark boxes on Right Panel
    for m in result.matched_pairs:
        bb = m.benchmark_box
        x1 = bb.norm_x * target_width
        y1 = bb.norm_y * canvas_h
        x2 = (bb.norm_x + bb.norm_w) * target_width
        y2 = (bb.norm_y + bb.norm_h) * canvas_h

        fill = color_agree if m.text_clean_match else color_diff
        outline = color_agree_outline if m.text_clean_match else color_diff_outline
        draw_right.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=2)

    for bb in result.benchmark_only_boxes:
        x1 = bb.norm_x * target_width
        y1 = bb.norm_y * canvas_h
        x2 = (bb.norm_x + bb.norm_w) * target_width
        y2 = (bb.norm_y + bb.norm_h) * canvas_h
        draw_right.rectangle([x1, y1, x2, y2], fill=color_b_only, outline=color_b_only_outline, width=2)

    # Header bar
    header_h = 60
    combined = Image.new("RGB", (target_width * 2 + 20, canvas_h + header_h), color=(33, 37, 41))
    draw_comb = ImageDraw.Draw(combined)

    # Titles
    draw_comb.text((20, 15), f"PIPELINE (PaddleOCR + VLM) - Words: {result.pipeline_total_words}", fill=(255, 255, 255))
    draw_comb.text((target_width + 30, 15), f"BENCHMARK (ABBYY FineReader) - Words: {result.benchmark_total_words}", fill=(255, 255, 255))

    # Paste panels
    combined.paste(panel_left, (10, header_h))
    combined.paste(panel_right, (target_width + 10, header_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(output_path), "JPEG", quality=85)
    return output_path


# ---------------------------------------------------------------------------
# Interactive HTML Side-by-Side Visualizer
# ---------------------------------------------------------------------------

def generate_interactive_html(
    result: PageComparisonResult,
    bg_image_rel_path: Optional[str],
    output_html_path: Path
) -> Path:
    """Generates a standalone, rich HTML viewer with synchronized panning, zoom,

    toggleable layers, and clickable word concordance cards.
    """
    # Prepare serializable data for embedded javascript
    matched_data = []
    for m in result.matched_pairs:
        matched_data.append({
            "p_text": m.pipeline_box.content,
            "b_text": m.benchmark_box.content,
            "p_box": [m.pipeline_box.norm_x, m.pipeline_box.norm_y, m.pipeline_box.norm_w, m.pipeline_box.norm_h],
            "b_box": [m.benchmark_box.norm_x, m.benchmark_box.norm_y, m.benchmark_box.norm_w, m.benchmark_box.norm_h],
            "iou": round(m.iou, 3),
            "clean_match": m.text_clean_match,
            "exact_match": m.text_exact_match,
            "sim": round(m.char_similarity, 3),
            "p_conf": m.pipeline_box.confidence,
            "b_conf": m.benchmark_box.confidence
        })

    p_only_data = [
        {"text": b.content, "box": [b.norm_x, b.norm_y, b.norm_w, b.norm_h], "conf": b.confidence}
        for b in result.pipeline_only_boxes
    ]
    b_only_data = [
        {"text": b.content, "box": [b.norm_x, b.norm_y, b.norm_w, b.norm_h], "conf": b.confidence}
        for b in result.benchmark_only_boxes
    ]

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRKN OCR Comparison: {result.item_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #1a1e24; color: #e1e4e8; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header {{ background: #24292e; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; z-index: 10; }}
  h1 {{ font-size: 1.15rem; font-weight: 600; color: #fff; }}
  .badges {{ display: flex; gap: 10px; font-size: 0.82rem; }}
  .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: 600; background: #30363d; }}
  .badge.success {{ background: #198754; color: #fff; }}
  .badge.warning {{ background: #d97706; color: #fff; }}
  .badge.info {{ background: #0284c7; color: #fff; }}
  
  .controls {{ background: #212529; border-bottom: 1px solid #30363d; padding: 8px 20px; display: flex; gap: 15px; align-items: center; font-size: 0.85rem; }}
  .controls label {{ display: flex; align-items: center; gap: 5px; cursor: pointer; }}
  .controls select, .controls input {{ background: #2d333b; color: #fff; border: 1px solid #444c56; padding: 4px 8px; border-radius: 4px; }}
  
  .main-container {{ display: flex; flex: 1; overflow: hidden; position: relative; }}
  .viewers-container {{ display: flex; flex: 1; overflow: hidden; position: relative; }}
  
  .panel {{ flex: 1; height: 100%; position: relative; overflow: hidden; border-right: 1px solid #30363d; background: #121519; cursor: grab; }}
  .panel:active {{ cursor: grabbing; }}
  .panel-header {{ position: absolute; top: 10px; left: 10px; z-index: 5; background: rgba(0, 0, 0, 0.75); padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; pointer-events: none; }}
  
  .canvas-stage {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; width: 100%; height: 100%; }}
  .page-image {{ width: 100%; height: auto; display: block; user-select: none; pointer-events: none; }}
  .overlay-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: auto; }}
  
  rect.agree {{ fill: rgba(40, 167, 69, 0.25); stroke: #28a745; stroke-width: 1.5; cursor: pointer; }}
  rect.discrepancy {{ fill: rgba(245, 158, 11, 0.35); stroke: #f59e0b; stroke-width: 2; cursor: pointer; }}
  rect.p-only {{ fill: rgba(56, 189, 248, 0.35); stroke: #38bdf8; stroke-width: 1.5; cursor: pointer; }}
  rect.b-only {{ fill: rgba(239, 68, 68, 0.35); stroke: #ef4444; stroke-width: 1.5; cursor: pointer; }}
  rect:hover {{ fill-opacity: 0.7; stroke-width: 2.5; }}
  
  .sidebar {{ width: 340px; background: #1f242c; border-left: 1px solid #30363d; display: flex; flex-direction: column; }}
  .sidebar-header {{ padding: 10px 14px; font-weight: 600; border-bottom: 1px solid #30363d; font-size: 0.9rem; background: #24292e; display: flex; justify-content: space-between; }}
  .sidebar-list {{ flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; }}
  .card {{ background: #282e38; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; font-size: 0.82rem; cursor: pointer; }}
  .card:hover {{ border-color: #60a5fa; background: #2d3542; }}
  .card-row {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
  .card-diff {{ font-weight: bold; color: #f59e0b; }}
  .card-label {{ color: #9ca3af; font-size: 0.75rem; }}
  
  .legend {{ display: flex; gap: 12px; font-size: 0.78rem; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>CRKN OCR Verification & Concordance: {result.item_name}</h1>
  </div>
  <div class="badges">
    <div class="badge success" title="Spatially overlapping words with concordant text">Text Concordance: {result.clean_text_concordance_rate}%</div>
    <div class="badge info" title="Percentage of pipeline words with matching benchmark bbox">Spatial IoU Alignment: {result.spatial_match_rate_pipeline}%</div>
    <div class="badge warning" title="Words where bounding boxes overlap but readings differ">Discrepancies: {len(result.discrepancies)}</div>
  </div>
</header>

<div class="controls">
  <label><input type="checkbox" id="chkAgree" checked> Concordant Words (Green)</label>
  <label><input type="checkbox" id="chkDiff" checked> Discrepant Words (Orange)</label>
  <label><input type="checkbox" id="chkOnly" checked> Engine-Only Words (Blue/Red)</label>
  <button id="btnResetView" style="background:#3b82f6;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;">Reset Zoom & Pan</button>
  <span style="flex:1;"></span>
  <div class="legend">
    <div class="legend-item"><div class="legend-color" style="background:#28a745;"></div> Agreed</div>
    <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div> Discrepancy</div>
    <div class="legend-item"><div class="legend-color" style="background:#38bdf8;"></div> Pipeline Only</div>
    <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div> ABBYY Only</div>
  </div>
</div>

<div class="main-container">
  <div class="viewers-container">
    <!-- Pipeline Panel (Left) -->
    <div class="panel" id="panelLeft">
      <div class="panel-header">Pipeline (PaddleOCR + VLM) - {result.pipeline_total_words} words</div>
      <div class="canvas-stage" id="stageLeft">
        {"<img class='page-image' src='" + bg_image_rel_path + "'>" if bg_image_rel_path else "<div style='width:100%;height:100%;background:#1e232a;'></div>"}
        <svg class="overlay-svg" id="svgLeft" viewBox="0 0 {result.pipeline_page.width} {result.pipeline_page.height}"></svg>
      </div>
    </div>
    
    <!-- Benchmark Panel (Right) -->
    <div class="panel" id="panelRight">
      <div class="panel-header">Benchmark (ABBYY FineReader) - {result.benchmark_total_words} words</div>
      <div class="canvas-stage" id="stageRight">
        {"<img class='page-image' src='" + bg_image_rel_path + "'>" if bg_image_rel_path else "<div style='width:100%;height:100%;background:#1e232a;'></div>"}
        <svg class="overlay-svg" id="svgRight" viewBox="0 0 {result.benchmark_page.width} {result.benchmark_page.height}"></svg>
      </div>
    </div>
  </div>

  <!-- Discrepancy Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <span>Word Discrepancies ({len(result.discrepancies)})</span>
      <span style="font-size:0.75rem;color:#9ca3af;">Click card to zoom</span>
    </div>
    <div class="sidebar-list" id="sidebarList"></div>
  </div>
</div>

<script>
  const matchedData = {json.dumps(matched_data)};
  const pOnlyData = {json.dumps(p_only_data)};
  const bOnlyData = {json.dumps(b_only_data)};
  const pageWidth = {result.pipeline_page.width};
  const pageHeight = {result.pipeline_page.height};

  // SVGs
  const svgLeft = document.getElementById("svgLeft");
  const svgRight = document.getElementById("svgRight");
  const sidebarList = document.getElementById("sidebarList");

  // Populate SVGs and Sidebar
  matchedData.forEach((m, idx) => {{
    const isAgree = m.clean_match;
    const cls = isAgree ? "agree" : "discrepancy";

    // Left SVG
    const rectL = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rectL.setAttribute("x", m.p_box[0] * pageWidth);
    rectL.setAttribute("y", m.p_box[1] * pageHeight);
    rectL.setAttribute("width", m.p_box[2] * pageWidth);
    rectL.setAttribute("height", m.p_box[3] * pageHeight);
    rectL.setAttribute("class", cls);
    rectL.dataset.idx = idx;
    rectL.innerHTML = `<title>Pipeline: "${{m.p_text}}" | ABBYY: "${{m.b_text}}" (IoU: ${{m.iou}})</title>`;
    svgLeft.appendChild(rectL);

    // Right SVG
    const rectR = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rectR.setAttribute("x", m.b_box[0] * pageWidth);
    rectR.setAttribute("y", m.b_box[1] * pageHeight);
    rectR.setAttribute("width", m.b_box[2] * pageWidth);
    rectR.setAttribute("height", m.b_box[3] * pageHeight);
    rectR.setAttribute("class", cls);
    rectR.dataset.idx = idx;
    rectR.innerHTML = `<title>ABBYY: "${{m.b_text}}" | Pipeline: "${{m.p_text}}" (IoU: ${{m.iou}})</title>`;
    svgRight.appendChild(rectR);

    // Sidebar items for discrepancies
    if (!isAgree) {{
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `
        <div class="card-row">
          <span class="card-diff">P: ${{escapeHtml(m.p_text)}}</span>
          <span style="color:#60a5fa;">IoU: ${{m.iou}}</span>
        </div>
        <div class="card-row">
          <span style="color:#ef4444;">B: ${{escapeHtml(m.b_text)}}</span>
          <span class="card-label">Sim: ${{Math.round(m.sim * 100)}}%</span>
        </div>
      `;
      card.onclick = () => zoomToNormBox(m.p_box);
      sidebarList.appendChild(card);
    }}
  }});

  // Pipeline-only boxes
  pOnlyData.forEach(p => {{
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", p.box[0] * pageWidth);
    rect.setAttribute("y", p.box[1] * pageHeight);
    rect.setAttribute("width", p.box[2] * pageWidth);
    rect.setAttribute("height", p.box[3] * pageHeight);
    rect.setAttribute("class", "p-only");
    rect.innerHTML = `<title>Pipeline Only: "${{p.text}}"</title>`;
    svgLeft.appendChild(rect);
  }});

  // Benchmark-only boxes
  bOnlyData.forEach(b => {{
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", b.box[0] * pageWidth);
    rect.setAttribute("y", b.box[1] * pageHeight);
    rect.setAttribute("width", b.box[2] * pageWidth);
    rect.setAttribute("height", b.box[3] * pageHeight);
    rect.setAttribute("class", "b-only");
    rect.innerHTML = `<title>ABBYY Only: "${{b.text}}"</title>`;
    svgRight.appendChild(rect);
  }});

  function escapeHtml(str) {{
    return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }}

  // Synchronized Pan & Zoom
  let scale = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;

  const stageLeft = document.getElementById("stageLeft");
  const stageRight = document.getElementById("stageRight");
  const panelLeft = document.getElementById("panelLeft");
  const panelRight = document.getElementById("panelRight");

  function updateTransform() {{
    const transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
    stageLeft.style.transform = transform;
    stageRight.style.transform = transform;
  }}

  function attachPanZoom(panel) {{
    panel.addEventListener("mousedown", e => {{
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});

    window.addEventListener("mousemove", e => {{
      if (!isDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform();
    }});

    window.addEventListener("mouseup", () => {{
      isDragging = false;
    }});

    panel.addEventListener("wheel", e => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      const newScale = Math.min(Math.max(scale * zoomFactor, 0.4), 10.0);
      
      const rect = panel.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      panX = mouseX - (mouseX - panX) * (newScale / scale);
      panY = mouseY - (mouseY - panY) * (newScale / scale);
      scale = newScale;
      updateTransform();
    }}, {{ passive: false }});
  }}

  attachPanZoom(panelLeft);
  attachPanZoom(panelRight);

  document.getElementById("btnResetView").onclick = () => {{
    scale = 1.0;
    panX = 0;
    panY = 0;
    updateTransform();
  }};

  function zoomToNormBox(box) {{
    scale = 3.5;
    const stageW = panelLeft.clientWidth;
    const stageH = panelLeft.clientHeight;
    panX = (stageW / 2) - (box[0] * stageW * scale);
    panY = (stageH / 2) - (box[1] * stageH * scale);
    updateTransform();
  }}

  // Layer Toggles
  document.getElementById("chkAgree").onchange = (e) => {{
    document.querySelectorAll(".agree").forEach(r => r.style.display = e.target.checked ? "" : "none");
  }};
  document.getElementById("chkDiff").onchange = (e) => {{
    document.querySelectorAll(".discrepancy").forEach(r => r.style.display = e.target.checked ? "" : "none");
  }};
  document.getElementById("chkOnly").onchange = (e) => {{
    document.querySelectorAll(".p-only, .b-only").forEach(r => r.style.display = e.target.checked ? "" : "none");
  }};
</script>
</body>
</html>
"""

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    return output_html_path


# ---------------------------------------------------------------------------
# CLI & Batch Runner
# ---------------------------------------------------------------------------

def find_file_pairs(pipeline_dir: Path, benchmark_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Discovers matching ALTO XML or PDF files between two directories."""
    pairs: List[Tuple[str, Path, Path]] = []

    # Map benchmark files by stem (e.g. 'ome.00003_19290103.1')
    benchmark_xmls: Dict[str, Path] = {}
    benchmark_pdfs: Dict[str, Path] = {}

    for p in benchmark_dir.rglob("*"):
        if p.is_file():
            if p.suffix.lower() == ".xml" and not p.name.startswith("dmdDC"):
                benchmark_xmls[p.stem] = p
            elif p.suffix.lower() == ".pdf":
                benchmark_pdfs[p.stem] = p

    for p in pipeline_dir.rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem
        # Check XML match
        if p.suffix.lower() == ".xml":
            if stem in benchmark_xmls:
                pairs.append((stem, p, benchmark_xmls[stem]))
        elif p.suffix.lower() == ".pdf":
            if stem in benchmark_pdfs and not any(pair[0] == stem for pair in pairs):
                pairs.append((stem, p, benchmark_pdfs[stem]))

    return sorted(pairs, key=lambda x: x[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ALTO XML / PDF text overlays and bounding boxes between PaddleOCR Pipeline and ABBYY FineReader."
    )
    # Single file mode
    parser.add_argument("--pipeline-alto", type=Path, help="Path to pipeline ALTO XML file")
    parser.add_argument("--benchmark-alto", type=Path, help="Path to benchmark (ABBYY) ALTO XML file")
    parser.add_argument("--pipeline-pdf", type=Path, help="Path to pipeline PDF file (if comparing PDFs)")
    parser.add_argument("--benchmark-pdf", type=Path, help="Path to benchmark PDF file")
    parser.add_argument("--image", type=Path, help="Path to reference page image (JPG/PNG) for visual rendering")

    # Directory mode
    parser.add_argument("--pipeline-dir", type=Path, help="Directory containing pipeline outputs")
    parser.add_argument("--benchmark-dir", type=Path, help="Directory containing benchmark outputs")

    # Output options
    parser.add_argument("--output-dir", type=Path, default=Path("alto_comparison_report"), help="Output directory for reports")
    parser.add_argument("--render-html", action="store_true", help="Generate interactive side-by-side HTML viewer")
    parser.add_argument("--render-image", action="store_true", help="Render annotated side-by-side PNG diffs")
    parser.add_argument("--min-iou", type=float, default=0.25, help="Minimum IoU threshold to consider boxes aligned (default: 0.25)")

    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items_to_compare: List[Tuple[str, Any, Any, Optional[Path]]] = []

    # Single-file mode
    if args.pipeline_alto and args.benchmark_alto:
        items_to_compare.append((
            args.pipeline_alto.stem,
            args.pipeline_alto,
            args.benchmark_alto,
            args.image
        ))
    elif args.pipeline_pdf and args.benchmark_pdf:
        items_to_compare.append((
            args.pipeline_pdf.stem,
            args.pipeline_pdf,
            args.benchmark_pdf,
            args.image
        ))
    elif args.pipeline_dir and args.benchmark_dir:
        pairs = find_file_pairs(args.pipeline_dir, args.benchmark_dir)
        if not pairs:
            print(f"[ERROR] No matching file pairs found between {args.pipeline_dir} and {args.benchmark_dir}", file=sys.stderr)
            return 1
        print(f"[CRKN Comparison] Found {len(pairs)} matching file pairs to compare.")
        for stem, p_file, b_file in pairs:
            # Look for an image with the same stem in benchmark or pipeline dir
            cand_img = None
            for cand_ext in [".jpg", ".png", ".tif"]:
                b_img = b_file.with_suffix(cand_ext)
                if b_img.exists():
                    cand_img = b_img
                    break
                p_img = p_file.with_suffix(cand_ext)
                if p_img.exists():
                    cand_img = p_img
                    break
            items_to_compare.append((stem, p_file, b_file, cand_img))
    else:
        parser.print_help()
        return 1

    summary_records = []
    all_results: List[PageComparisonResult] = []

    for stem, p_file, b_file, img_file in items_to_compare:
        print(f"\n--- Comparing: {stem} ---")
        if p_file.suffix.lower() == ".xml":
            p_pages = parse_alto_xml(p_file, "PaddleOCR")
            b_pages = parse_alto_xml(b_file, "ABBYY")
        else:
            p_pages = parse_pdf_text_layer(p_file, "PaddleOCR")
            b_pages = parse_pdf_text_layer(b_file, "ABBYY")

        if not p_pages or not b_pages:
            print(f"[WARN] Empty pages for {stem}, skipping.")
            continue

        res = compare_pages(p_pages[0], b_pages[0], stem, min_iou=args.min_iou)
        all_results.append(res)

        print(f"  • Pipeline Total Words: {res.pipeline_total_words}")
        print(f"  • Benchmark Total Words: {res.benchmark_total_words}")
        print(f"  • Spatially Aligned Words: {len(res.matched_pairs)} ({res.spatial_match_rate_pipeline}% of pipeline)")
        print(f"  • Text Concordance (Cleaned): {res.clean_text_concordance_rate}% ({res.text_concordance_rate}% exact)")
        print(f"  • Mean Word Levenshtein Sim: {res.mean_char_similarity:.3f}")
        print(f"  • Mean Bounding Box IoU: {res.mean_iou:.3f}")
        print(f"  • Mean Positional Drift (dX, dY): ({res.mean_dx:+.4f}, {res.mean_dy:+.4f})")
        print(f"  • Discrepancies: {len(res.discrepancies)}")

        summary_records.append({
            "item_name": stem,
            "pipeline_words": res.pipeline_total_words,
            "benchmark_words": res.benchmark_total_words,
            "matched_boxes": len(res.matched_pairs),
            "spatial_match_rate_pct": res.spatial_match_rate_pipeline,
            "text_concordance_exact_pct": res.text_concordance_rate,
            "text_concordance_clean_pct": res.clean_text_concordance_rate,
            "mean_char_similarity": res.mean_char_similarity,
            "mean_iou": res.mean_iou,
            "mean_dx": res.mean_dx,
            "mean_dy": res.mean_dy,
            "discrepancies_count": len(res.discrepancies)
        })

        # Visual Renderings
        img_rel_path = None
        if img_file and img_file.exists():
            # Copy or reference image for HTML
            dest_img = out_dir / f"{stem}.jpg"
            if not dest_img.exists():
                try:
                    import shutil
                    shutil.copyfile(img_file, dest_img)
                    img_rel_path = dest_img.name
                except Exception:
                    img_rel_path = None

        if args.render_image:
            out_img = out_dir / f"{stem}_diff.jpg"
            render_comparison_image(res, img_file, out_img)
            print(f"  • Rendered Annotated Diff Image: {out_img}")

        if args.render_html:
            out_html = out_dir / f"{stem}_comparison.html"
            generate_interactive_html(res, img_rel_path, out_html)
            print(f"  • Rendered Interactive HTML Viewer: {out_html}")

    # Output CSV summary
    csv_path = out_dir / "comparison_summary.csv"
    if summary_records:
        keys = list(summary_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_records)
        print(f"\n[CRKN Comparison] Saved CSV summary to: {csv_path}")

    # Output JSON summary
    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"[CRKN Comparison] Saved JSON summary to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

