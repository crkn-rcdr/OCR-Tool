"""Interactive HTML Results Dashboard Generator for Regular OCR Runs.

Generates a standalone, rich visual dashboard for single-engine OCR runs (PaddleOCR,
Indra OCR, etc.) without comparing against a benchmark.

Key Features:
- Master Home Page (index.html):
    - Top summary metrics: total runtime, pages, words, average latency, mean confidence.
    - Item-level breakdown cards/table (pages, words, runtime, avg confidence per folder/document).
    - Page-level interactive table with search, item filter, confidence filter, and direct viewer links.
- Interactive Single-Page Viewers (<page>_results.html):
    - Scanned image with overlayed ALTO word bounding boxes.
    - Color-coded by recognition confidence:
        - High (>= 90%): Green
        - Medium (75% - 89%): Orange
        - Low (< 75%): Red
    - Hover tooltips with text, exact confidence, and coordinates.
    - Top navigation bar: Previous / Next buttons, breadcrumbs, words badge, runtime badge.
    - Interactive controls: Pan & zoom, image opacity, box opacity, confidence toggles.
- Auto-open in browser when the job finishes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import webbrowser
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from services.constants import IMAGE_EXTENSIONS


@dataclass
class WordBox:
    """Individual recognized word with geometry and confidence."""
    text: str
    confidence: float  # 0.0 to 1.0
    norm_x: float      # 0.0 to 1.0 (relative to page width)
    norm_y: float      # 0.0 to 1.0 (relative to page height)
    norm_w: float
    norm_h: float
    raw_x: float
    raw_y: float
    raw_w: float
    raw_h: float

    @property
    def conf_tier(self) -> str:
        if self.confidence >= 0.90:
            return "high"
        elif self.confidence >= 0.75:
            return "medium"
        return "low"

    @property
    def conf_color(self) -> str:
        if self.confidence >= 0.90:
            return "#3fb950"  # Green
        elif self.confidence >= 0.75:
            return "#d29922"  # Amber / Orange
        return "#f85149"      # Red


@dataclass
class PageResult:
    """Page-level OCR result metadata and words."""
    page_name: str
    item_name: str
    image_path: Path
    alto_path: Optional[Path] = None
    json_path: Optional[Path] = None
    page_width: float = 1800.0
    page_height: float = 2400.0
    runtime_sec: float = 0.0
    words: List[WordBox] = field(default_factory=list)
    detected_lang: str = "auto"

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    @property
    def high_conf_count(self) -> int:
        return sum(1 for w in self.words if w.conf_tier == "high")

    @property
    def medium_conf_count(self) -> int:
        return sum(1 for w in self.words if w.conf_tier == "medium")

    @property
    def low_conf_count(self) -> int:
        return sum(1 for w in self.words if w.conf_tier == "low")


def parse_alto_words(alto_path: Path) -> Tuple[float, float, List[WordBox]]:
    """Parse words, coordinates, and confidence scores from an ALTO XML file."""
    try:
        tree = ET.parse(alto_path)
        root = tree.getroot()
    except Exception:
        return 1800.0, 2400.0, []

    # Extract page dimensions
    page_w = 1800.0
    page_h = 2400.0
    for page_elem in root.iter():
        if page_elem.tag.endswith("Page"):
            try:
                page_w = float(page_elem.get("WIDTH", page_w))
                page_h = float(page_elem.get("HEIGHT", page_h))
            except (ValueError, TypeError):
                pass
            break

    words: List[WordBox] = []
    for elem in root.iter():
        if not elem.tag.endswith("String"):
            continue

        content = elem.get("CONTENT", "").strip()
        if not content:
            continue

        try:
            rx = float(elem.get("HPOS", 0))
            ry = float(elem.get("VPOS", 0))
            rw = float(elem.get("WIDTH", 0))
            rh = float(elem.get("HEIGHT", 0))
        except (ValueError, TypeError):
            continue

        conf_str = elem.get("WC")
        try:
            conf = float(conf_str) if conf_str is not None else 0.90
        except (ValueError, TypeError):
            conf = 0.90
        conf = max(0.0, min(1.0, conf))

        # Split multi-word strings proportionally if present
        subwords = content.split()
        if len(subwords) > 1:
            weights = [max(1, len(sw)) for sw in subwords]
            total_w = sum(weights)
            curr_w = 0.0
            for sw, w in zip(subwords, weights):
                frac_start = curr_w / total_w
                frac_end = (curr_w + w) / total_w
                curr_w += w
                sub_rx = rx + rw * frac_start
                sub_rw = rw * (frac_end - frac_start)
                words.append(WordBox(
                    text=sw,
                    confidence=conf,
                    norm_x=sub_rx / page_w if page_w > 0 else 0.0,
                    norm_y=ry / page_h if page_h > 0 else 0.0,
                    norm_w=sub_rw / page_w if page_w > 0 else 0.0,
                    norm_h=rh / page_h if page_h > 0 else 0.0,
                    raw_x=sub_rx,
                    raw_y=ry,
                    raw_w=sub_rw,
                    raw_h=rh,
                ))
            continue

        words.append(WordBox(
            text=content,
            confidence=conf,
            norm_x=rx / page_w if page_w > 0 else 0.0,
            norm_y=ry / page_h if page_h > 0 else 0.0,
            norm_w=rw / page_w if page_w > 0 else 0.0,
            norm_h=rh / page_h if page_h > 0 else 0.0,
            raw_x=rx,
            raw_y=ry,
            raw_w=rw,
            raw_h=rh,
        ))

    return page_w, page_h, words


def parse_json_words(json_path: Path) -> Tuple[float, float, List[WordBox]]:
    """Parse words and confidence from OCR JSON payload."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 1800.0, 2400.0, []

    page_w = 1800.0
    page_h = 2400.0
    words: List[WordBox] = []

    # Handle PP-OCRv6 format
    if isinstance(data, list):
        for block in data:
            if not isinstance(block, dict):
                continue
            rec_texts = block.get("rec_texts", [])
            rec_scores = block.get("rec_scores", [])
            rec_polys = block.get("rec_polys", [])
            rec_boxes = block.get("rec_boxes", [])

            for idx, text in enumerate(rec_texts):
                score = float(rec_scores[idx]) if idx < len(rec_scores) else 0.90
                score = max(0.0, min(1.0, score))
                box = rec_boxes[idx] if idx < len(rec_boxes) else None
                poly = rec_polys[idx] if idx < len(rec_polys) else None

                if box and len(box) == 4:
                    x0, y0, x1, y1 = box
                elif poly and len(poly) >= 4:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue

                rw = x1 - x0
                rh = y1 - y0
                words.append(WordBox(
                    text=text,
                    confidence=score,
                    norm_x=x0 / page_w if page_w > 0 else 0.0,
                    norm_y=y0 / page_h if page_h > 0 else 0.0,
                    norm_w=rw / page_w if page_w > 0 else 0.0,
                    norm_h=rh / page_h if page_h > 0 else 0.0,
                    raw_x=x0,
                    raw_y=y0,
                    raw_w=rw,
                    raw_h=rh,
                ))

    # Handle Indra format or dict payload
    elif isinstance(data, dict):
        indra_meta = data.get("indra", {})
        if indra_meta:
            page_w = float(indra_meta.get("width", page_w))
            page_h = float(indra_meta.get("height", page_h))

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        rec_boxes = data.get("rec_boxes", [])
        for idx, text in enumerate(rec_texts):
            score = float(rec_scores[idx]) if idx < len(rec_scores) else 0.90
            score = max(0.0, min(1.0, score))
            box = rec_boxes[idx] if idx < len(rec_boxes) else None
            if not box or len(box) != 4:
                continue
            x0, y0, x1, y1 = box
            rw = x1 - x0
            rh = y1 - y0
            words.append(WordBox(
                text=text,
                confidence=score,
                norm_x=x0 / page_w if page_w > 0 else 0.0,
                norm_y=y0 / page_h if page_h > 0 else 0.0,
                norm_w=rw / page_w if page_w > 0 else 0.0,
                norm_h=rh / page_h if page_h > 0 else 0.0,
                raw_x=x0,
                raw_y=y0,
                raw_w=rw,
                raw_h=rh,
            ))

    return page_w, page_h, words


def generate_single_page_results_html(
    page: PageResult,
    prev_url: Optional[str],
    next_url: Optional[str],
    dashboard_url: str,
    output_html_path: Path,
    rel_image_path: str,
    engine_name: str = "Paddle PP-OCRv6",
) -> Path:
    """Generate interactive single-page viewer with confidence-coded bounding boxes."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    words_data = [
        {
            "idx": i,
            "text": w.text,
            "conf": round(w.confidence, 4),
            "tier": w.conf_tier,
            "color": w.conf_color,
            "box": [round(w.norm_x, 5), round(w.norm_y, 5), round(w.norm_w, 5), round(w.norm_h, 5)],
            "raw": [round(w.raw_x, 1), round(w.raw_y, 1), round(w.raw_w, 1), round(w.raw_h, 1)],
        }
        for i, w in enumerate(page.words)
    ]
    words_json = json.dumps(words_data, ensure_ascii=False)

    avg_conf_pct = page.avg_confidence * 100.0
    conf_badge_color = "#3fb950" if avg_conf_pct >= 90.0 else ("#d29922" if avg_conf_pct >= 75.0 else "#f85149")

    prev_disabled = 'class="nav-btn disabled"' if not prev_url else f'class="nav-btn" href="{prev_url}"'
    next_disabled = 'class="nav-btn disabled"' if not next_url else f'class="nav-btn" href="{next_url}"'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR Results: {page.page_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  
  /* Top Navigation Bar */
  header {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 8px 18px; display: flex; justify-content: space-between; align-items: center; z-index: 20; gap: 12px; }}
  .nav-left {{ display: flex; align-items: center; gap: 10px; font-size: 0.88rem; }}
  .nav-btn-back {{ background: #21262d; color: #58a6ff; border: 1px solid #30363d; padding: 5px 12px; border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 0.82rem; transition: all 0.15s ease; }}
  .nav-btn-back:hover {{ background: #30363d; color: #79c0ff; border-color: #58a6ff; }}
  .nav-sep {{ color: #484f58; }}
  .nav-item {{ color: #8b949e; font-weight: 500; font-size: 0.84rem; }}
  .nav-title {{ font-size: 1.05rem; font-weight: 600; color: #f0f6fc; }}
  
  .nav-center {{ display: flex; align-items: center; gap: 6px; }}
  .nav-btn {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 12px; border-radius: 4px; text-decoration: none; font-size: 0.82rem; font-weight: 500; transition: all 0.15s ease; }}
  .nav-btn:hover:not(.disabled) {{ background: #30363d; color: #58a6ff; border-color: #58a6ff; }}
  .nav-btn.disabled {{ opacity: 0.35; cursor: not-allowed; pointer-events: none; }}
  .page-current {{ font-size: 0.82rem; color: #8b949e; padding: 0 4px; }}

  .badges {{ display: flex; gap: 8px; font-size: 0.8rem; align-items: center; }}
  .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 600; background: #21262d; border: 1px solid #30363d; color: #f0f6fc; }}
  .badge.engine {{ background: #1f6feb; border-color: #388bfd; }}
  .badge.words {{ background: #238636; border-color: #2ea043; }}
  .badge.time {{ background: #30363d; color: #79c0ff; }}
  .badge.conf {{ border-color: {conf_badge_color}; color: {conf_badge_color}; }}

  /* Controls Bar */
  .controls {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 6px 18px; display: flex; gap: 16px; align-items: center; font-size: 0.82rem; flex-wrap: wrap; z-index: 15; }}
  .controls label {{ display: flex; align-items: center; gap: 6px; cursor: pointer; color: #c9d1d9; user-select: none; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .slider-group {{ display: flex; align-items: center; gap: 8px; margin-left: auto; }}
  .slider-group input[type="range"] {{ width: 100px; accent-color: #58a6ff; }}

  /* Main Workspace */
  .workspace {{ display: flex; flex: 1; overflow: hidden; position: relative; min-height: 0; }}
  .viewport {{ flex: 1; height: 100%; position: relative; overflow: hidden; background: #010409; cursor: grab; }}
  .viewport:active {{ cursor: grabbing; }}
  
  .canvas-stage {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; width: {page.page_width}px; height: {page.page_height}px; }}
  .page-image {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block; user-select: none; pointer-events: none; }}
  .overlay-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: auto; }}

  /* Bounding Boxes */
  rect.bbox {{ cursor: pointer; transition: stroke-width 0.1s; }}
  rect.bbox.tier-high {{ fill: rgba(40, 167, 69, 0.22); stroke: #28a745; stroke-width: 1.5; }}
  rect.bbox.tier-medium {{ fill: rgba(210, 153, 34, 0.32); stroke: #d29922; stroke-width: 1.8; }}
  rect.bbox.tier-low {{ fill: rgba(248, 81, 73, 0.40); stroke: #f85149; stroke-width: 2.0; }}
  rect.bbox:hover, rect.bbox.selected {{ stroke: #58a6ff !important; stroke-width: 3.0 !important; fill: rgba(88, 166, 255, 0.45) !important; }}

  /* Sidebar Details Drawer */
  .drawer {{ width: 320px; background: #161b22; border-left: 1px solid #30363d; display: flex; flex-direction: column; overflow: hidden; }}
  .drawer-header {{ padding: 12px 16px; border-bottom: 1px solid #30363d; font-size: 0.88rem; font-weight: 600; display: flex; justify-content: space-between; align-items: center; }}
  .drawer-search {{ padding: 8px 12px; border-bottom: 1px solid #30363d; }}
  .drawer-search input {{ width: 100%; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 5px 8px; font-size: 0.8rem; color: #f0f6fc; }}
  .drawer-list {{ flex: 1; overflow-y: auto; padding: 6px; }}
  .word-item {{ padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; font-size: 0.82rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border: 1px solid transparent; }}
  .word-item:hover {{ background: #21262d; border-color: #30363d; }}
  .word-item.selected {{ background: #1f6feb; color: #fff; }}
  .word-conf-tag {{ font-size: 0.72rem; font-weight: 700; padding: 2px 6px; border-radius: 3px; }}

  /* Tooltip */
  .tooltip {{ position: absolute; z-index: 1000; background: rgba(22, 27, 34, 0.95); border: 1px solid #58a6ff; color: #f0f6fc; padding: 6px 10px; border-radius: 6px; font-size: 0.82rem; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none; line-height: 1.4; }}
  .tooltip-conf {{ font-size: 0.76rem; font-weight: 600; }}
</style>
</head>
<body>

<header>
  <div class="nav-left">
    <a href="{dashboard_url}" class="nav-btn-back">← Master Dashboard</a>
    <span class="nav-sep">/</span>
    <span class="nav-item">{page.item_name}</span>
    <span class="nav-sep">/</span>
    <span class="nav-title">{page.page_name}</span>
  </div>

  <div class="nav-center">
    <a {prev_disabled}>&lsaquo; Previous</a>
    <span class="page-current">{page.page_name}</span>
    <a {next_disabled}>Next &rsaquo;</a>
  </div>

  <div class="badges">
    <span class="badge engine">{engine_name}</span>
    <span class="badge words">{page.word_count} words</span>
    <span class="badge time">{page.runtime_sec:.2f}s</span>
    <span class="badge conf">{avg_conf_pct:.1f}% Avg Conf</span>
  </div>
</header>

<div class="controls">
  <label><input type="checkbox" id="chkHigh" checked><span class="legend-dot" style="background:#28a745;"></span> High Conf (&ge; 90%) <span style="color:#8b949e;">({page.high_conf_count})</span></label>
  <label><input type="checkbox" id="chkMed" checked><span class="legend-dot" style="background:#d29922;"></span> Medium Conf (75-89%) <span style="color:#8b949e;">({page.medium_conf_count})</span></label>
  <label><input type="checkbox" id="chkLow" checked><span class="legend-dot" style="background:#f85149;"></span> Low Conf (&lt; 75%) <span style="color:#8b949e;">({page.low_conf_count})</span></label>
  
  <div class="slider-group">
    <button id="btnZoomIn" class="nav-btn" title="Zoom In">+</button>
    <button id="btnZoomOut" class="nav-btn" title="Zoom Out">&minus;</button>
    <button id="btnFit" class="nav-btn" title="Reset Fit">Fit</button>
    <label style="margin-left:8px;">Image Opacity: <input type="range" id="rngOpacity" min="0" max="100" value="85"></label>
  </div>
</div>

<div class="workspace">
  <div class="viewport" id="viewport">
    <div class="canvas-stage" id="stage">
      <img src="{rel_image_path}" class="page-image" id="pageImage" style="opacity: 0.85;">
      <svg class="overlay-svg" id="svgOverlay" viewBox="0 0 {page.page_width} {page.page_height}"></svg>
    </div>
  </div>

  <div class="drawer">
    <div class="drawer-header">
      <span>Recognized Words ({page.word_count})</span>
      <span style="font-size:0.75rem;color:#8b949e;">Click to locate</span>
    </div>
    <div class="drawer-search">
      <input type="text" id="txtWordSearch" placeholder="Filter words on this page...">
    </div>
    <div class="drawer-list" id="wordList"></div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
  const wordsData = {words_json};
  const PAGE_W = {page.page_width};
  const PAGE_H = {page.page_height};

  const viewport = document.getElementById("viewport");
  const stage = document.getElementById("stage");
  const svgOverlay = document.getElementById("svgOverlay");
  const pageImage = document.getElementById("pageImage");
  const tooltip = document.getElementById("tooltip");
  const wordList = document.getElementById("wordList");

  const chkHigh = document.getElementById("chkHigh");
  const chkMed = document.getElementById("chkMed");
  const chkLow = document.getElementById("chkLow");
  const rngOpacity = document.getElementById("rngOpacity");
  const txtWordSearch = document.getElementById("txtWordSearch");

  let scale = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let selectedIdx = null;

  // Render SVG Bounding Boxes
  function renderSvgBoxes() {{
    svgOverlay.innerHTML = "";
    wordsData.forEach(w => {{
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const rx = w.box[0] * PAGE_W;
      const ry = w.box[1] * PAGE_H;
      const rw = w.box[2] * PAGE_W;
      const rh = w.box[3] * PAGE_H;

      rect.setAttribute("x", rx);
      rect.setAttribute("y", ry);
      rect.setAttribute("width", rw);
      rect.setAttribute("height", rh);
      rect.setAttribute("class", `bbox tier-${{w.tier}}`);
      rect.setAttribute("data-idx", w.idx);
      rect.setAttribute("data-tier", w.tier);

      rect.addEventListener("mouseenter", (e) => showTooltip(e, w));
      rect.addEventListener("mousemove", (e) => moveTooltip(e));
      rect.addEventListener("mouseleave", hideTooltip);
      rect.addEventListener("click", () => selectWord(w.idx));

      svgOverlay.appendChild(rect);
    }});
    updateVisibility();
  }}

  // Render Sidebar Word List
  function renderWordList(filterText = "") {{
    wordList.innerHTML = "";
    const term = filterText.toLowerCase().trim();
    wordsData.forEach(w => {{
      if (term && !w.text.toLowerCase().includes(term)) return;
      const div = document.createElement("div");
      div.className = `word-item ${{selectedIdx === w.idx ? "selected" : ""}}`;
      div.id = `item-${{w.idx}}`;
      div.innerHTML = `
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:210px;">${{w.text}}</span>
        <span class="word-conf-tag" style="background:${{w.color}};color:#fff;">${{Math.round(w.conf * 100)}}%</span>
      `;
      div.addEventListener("click", () => selectWord(w.idx, true));
      wordList.appendChild(div);
    }});
  }}

  function selectWord(idx, panTo = false) {{
    selectedIdx = idx;
    document.querySelectorAll(".bbox").forEach(r => {{
      r.classList.toggle("selected", parseInt(r.getAttribute("data-idx")) === idx);
    }});
    document.querySelectorAll(".word-item").forEach(item => {{
      item.classList.toggle("selected", item.id === `item-${{idx}}`);
    }});

    const targetItem = document.getElementById(`item-${{idx}}`);
    if (targetItem) targetItem.scrollIntoView({{ block: "nearest", behavior: "smooth" }});

    if (panTo) {{
      const w = wordsData[idx];
      const cx = (w.box[0] + w.box[2] / 2) * PAGE_W;
      const cy = (w.box[1] + w.box[3] / 2) * PAGE_H;
      const vw = viewport.clientWidth;
      const vh = viewport.clientHeight;
      panX = (vw / 2) - cx * scale;
      panY = (vh / 2) - cy * scale;
      updateTransform();
    }}
  }}

  function showTooltip(e, w) {{
    tooltip.innerHTML = `<strong>${{w.text}}</strong><br><span class="tooltip-conf" style="color:${{w.color}}">Confidence: ${{Math.round(w.conf * 1000) / 10}}%</span>`;
    tooltip.style.display = "block";
    moveTooltip(e);
  }}

  function moveTooltip(e) {{
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top = (e.clientY + 14) + "px";
  }}

  function hideTooltip() {{
    tooltip.style.display = "none";
  }}

  function updateVisibility() {{
    const showHigh = chkHigh.checked;
    const showMed = chkMed.checked;
    const showLow = chkLow.checked;

    document.querySelectorAll("rect.bbox").forEach(r => {{
      const tier = r.getAttribute("data-tier");
      if (tier === "high") r.style.display = showHigh ? "" : "none";
      if (tier === "medium") r.style.display = showMed ? "" : "none";
      if (tier === "low") r.style.display = showLow ? "" : "none";
    }});
  }}

  function fitToViewport() {{
    const vw = viewport.clientWidth;
    const vh = viewport.clientHeight;
    scale = Math.min((vw - 40) / PAGE_W, (vh - 40) / PAGE_H, 1.2);
    panX = (vw - PAGE_W * scale) / 2;
    panY = (vh - PAGE_H * scale) / 2;
    updateTransform();
  }}

  function updateTransform() {{
    stage.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
  }}

  // Pan and Zoom
  viewport.addEventListener("mousedown", (e) => {{
    if (e.target.tagName.toLowerCase() === "rect") return;
    isDragging = true;
    startX = e.clientX - panX;
    startY = e.clientY - panY;
  }});

  window.addEventListener("mousemove", (e) => {{
    if (!isDragging) return;
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    updateTransform();
  }});

  window.addEventListener("mouseup", () => {{ isDragging = false; }});

  viewport.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    const rect = viewport.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    panX = mouseX - (mouseX - panX) * factor;
    panY = mouseY - (mouseY - panY) * factor;
    scale *= factor;
    updateTransform();
  }}, {{ passive: false }});

  document.getElementById("btnZoomIn").addEventListener("click", () => {{ scale *= 1.25; updateTransform(); }});
  document.getElementById("btnZoomOut").addEventListener("click", () => {{ scale *= 0.8; updateTransform(); }});
  document.getElementById("btnFit").addEventListener("click", fitToViewport);

  chkHigh.addEventListener("change", updateVisibility);
  chkMed.addEventListener("change", updateVisibility);
  chkLow.addEventListener("change", updateVisibility);

  rngOpacity.addEventListener("input", (e) => {{
    pageImage.style.opacity = e.target.value / 100;
  }});

  txtWordSearch.addEventListener("input", (e) => {{
    renderWordList(e.target.value);
  }});

  // Keyboard navigation Left / Right
  window.addEventListener("keydown", (e) => {{
    if (e.target.tagName.toLowerCase() === "input") return;
    if (e.key === "ArrowLeft" && "{prev_url}") window.location.href = "{prev_url}";
    if (e.key === "ArrowRight" && "{next_url}") window.location.href = "{next_url}";
  }});

  // Init
  renderSvgBoxes();
  renderWordList();
  window.addEventListener("resize", fitToViewport);
  setTimeout(fitToViewport, 50);
</script>
</body>
</html>"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_html_path


def generate_master_results_dashboard(
    pages: List[PageResult],
    dashboard_dir: Path,
    engine_name: str = "Paddle PP-OCRv6",
    total_time_sec: float = 0.0,
) -> Path:
    """Generate master index.html dashboard with item-level and page-level breakdown."""
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    total_pages = len(pages)
    total_words = sum(p.word_count for p in pages)
    avg_conf = (sum(p.avg_confidence for p in pages) / total_pages) if total_pages else 0.0
    avg_conf_pct = round(avg_conf * 100.0, 1)

    effective_time = total_time_sec if total_time_sec > 0 else sum(p.runtime_sec for p in pages)
    avg_time_per_page = (effective_time / total_pages) if total_pages else 0.0
    time_str = f"{effective_time:.1f}s" if effective_time < 60 else f"{int(effective_time // 60)}m {int(effective_time % 60)}s"

    # Group by item / folder
    items_map: Dict[str, List[PageResult]] = defaultdict(list)
    for p in pages:
        items_map[p.item_name].append(p)

    items_summary = []
    for item_name, item_pages in sorted(items_map.items()):
        i_pages = len(item_pages)
        i_words = sum(x.word_count for x in item_pages)
        i_conf = (sum(x.avg_confidence for x in item_pages) / i_pages * 100.0) if i_pages else 0.0
        i_time = sum(x.runtime_sec for x in item_pages)
        items_summary.append({
            "name": item_name,
            "pages": i_pages,
            "words": i_words,
            "conf": round(i_conf, 1),
            "time": round(i_time, 2),
        })

    # Prepare table records JSON
    table_records = []
    for idx, p in enumerate(pages):
        conf_pct = round(p.avg_confidence * 100.0, 1)
        conf_tier = "high" if conf_pct >= 90.0 else ("medium" if conf_pct >= 75.0 else "low")
        conf_color = "#3fb950" if conf_pct >= 90.0 else ("#d29922" if conf_pct >= 75.0 else "#f85149")
        rel_viewer = f"pages/{p.item_name}/{p.page_name}_results.html"

        table_records.append({
            "idx": idx + 1,
            "item": p.item_name,
            "page": p.page_name,
            "words": p.word_count,
            "time": round(p.runtime_sec, 2),
            "conf": conf_pct,
            "tier": conf_tier,
            "color": conf_color,
            "viewer": rel_viewer,
        })

    records_json = json.dumps(table_records, ensure_ascii=False)
    items_json = json.dumps(items_summary, ensure_ascii=False)

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRKN OCR Results Dashboard - {engine_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.5; }}
  
  .navbar {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }}
  .brand {{ display: flex; align-items: center; gap: 12px; }}
  .brand-badge {{ background: #1f6feb; color: #fff; font-weight: 700; font-size: 0.75rem; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.5px; text-transform: uppercase; }}
  .brand-title {{ font-size: 1.15rem; font-weight: 600; color: #f0f6fc; }}
  .nav-meta {{ font-size: 0.85rem; color: #8b949e; }}

  .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
  
  /* Hero / Summary */
  .hero {{ margin-bottom: 24px; }}
  .hero h1 {{ font-size: 1.75rem; color: #f0f6fc; margin-bottom: 6px; font-weight: 600; }}
  .hero p {{ font-size: 0.92rem; color: #8b949e; }}

  /* KPI Grid */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .kpi-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px; transition: transform 0.15s, border-color 0.15s; }}
  .kpi-card:hover {{ border-color: #58a6ff; transform: translateY(-2px); }}
  .kpi-label {{ font-size: 0.76rem; text-transform: uppercase; color: #8b949e; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 1.85rem; font-weight: 700; color: #f0f6fc; }}
  .kpi-sub {{ font-size: 0.78rem; color: #8b949e; margin-top: 4px; }}
  .kpi-value.green {{ color: #3fb950; }}
  .kpi-value.blue {{ color: #58a6ff; }}
  .kpi-value.purple {{ color: #bc8cff; }}

  /* Item Breakdown Cards */
  .section-title {{ font-size: 1.15rem; font-weight: 600; color: #f0f6fc; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }}
  .item-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .item-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px; cursor: pointer; transition: all 0.15s; }}
  .item-card:hover {{ border-color: #58a6ff; transform: translateY(-2px); }}
  .item-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .item-name {{ font-weight: 600; color: #f0f6fc; font-size: 1.0rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }}
  .item-badge {{ font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; background: #21262d; border: 1px solid #30363d; color: #8b949e; }}
  .item-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.82rem; }}
  .item-stat-val {{ font-weight: 700; color: #f0f6fc; font-size: 1.05rem; }}
  .item-stat-lbl {{ font-size: 0.72rem; color: #8b949e; text-transform: uppercase; }}

  /* Page Table Section */
  .table-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .table-controls {{ padding: 16px 20px; border-bottom: 1px solid #30363d; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  .search-input {{ flex: 1; min-width: 240px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; color: #f0f6fc; font-size: 0.85rem; }}
  .filter-select {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; color: #f0f6fc; font-size: 0.85rem; cursor: pointer; }}
  
  table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; text-align: left; }}
  th {{ background: #0d1117; color: #8b949e; padding: 12px 16px; font-weight: 600; border-bottom: 1px solid #30363d; user-select: none; cursor: pointer; }}
  th:hover {{ color: #58a6ff; }}
  td {{ padding: 12px 16px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1f242c; }}
  
  .badge-conf {{ font-size: 0.76rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; display: inline-block; }}
  .btn-view {{ background: #238636; color: #fff; padding: 5px 12px; border-radius: 4px; text-decoration: none; font-weight: 600; font-size: 0.78rem; display: inline-flex; align-items: center; gap: 4px; transition: background 0.15s; }}
  .btn-view:hover {{ background: #2ea043; }}

  /* Pagination */
  .pagination {{ padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; font-size: 0.84rem; color: #8b949e; border-top: 1px solid #30363d; }}
  .page-btns {{ display: flex; gap: 6px; }}
  .page-btn {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
  .page-btn:hover:not(:disabled) {{ background: #30363d; color: #58a6ff; }}
  .page-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
</style>
</head>
<body>

<div class="navbar">
  <div class="brand">
    <span class="brand-badge">{engine_name}</span>
    <span class="brand-title">CRKN OCR Results Dashboard</span>
  </div>
  <div class="nav-meta">
    Total Runtime: <strong style="color:#f0f6fc;">{time_str}</strong> | Pages: <strong style="color:#f0f6fc;">{total_pages}</strong>
  </div>
</div>

<div class="container">
  <div class="hero">
    <h1>OCR Processing Results</h1>
    <p>Interactive single-engine results dashboard showing all processed documents, recognized text lines, confidence distributions, and page-by-page bounding box overlays.</p>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Pages Processed</div>
      <div class="kpi-value">{total_pages:,}</div>
      <div class="kpi-sub">Across {len(items_map)} item(s)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Recognized Words</div>
      <div class="kpi-value green">{total_words:,}</div>
      <div class="kpi-sub">Avg {int(total_words / total_pages if total_pages else 0):,} words/page</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Execution Time</div>
      <div class="kpi-value blue">{time_str}</div>
      <div class="kpi-sub">~{avg_time_per_page:.2f}s per page</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Mean Confidence</div>
      <div class="kpi-value purple">{avg_conf_pct:.1f}%</div>
      <div class="kpi-sub">Word-level recognition certainty</div>
    </div>
  </div>

  <div class="section-title">📁 Item-Level Breakdown ({len(items_map)} Items)</div>
  <div class="item-grid" id="itemGrid"></div>

  <div class="section-title">📄 Page-Level Explorer & Bounding Box Viewers ({total_pages} Pages)</div>
  <div class="table-container">
    <div class="table-controls">
      <input type="text" id="txtSearch" class="search-input" placeholder="Search page name or item...">
      <select id="selItem" class="filter-select">
        <option value="">All Items ({total_pages} pages)</option>
      </select>
      <select id="selConf" class="filter-select">
        <option value="all">All Confidence Tiers</option>
        <option value="high">High Confidence (&ge; 90%)</option>
        <option value="medium">Medium (75% - 89%)</option>
        <option value="low">Needs Review (&lt; 75%)</option>
      </select>
    </div>

    <table>
      <thead>
        <tr>
          <th data-col="idx"># ⇕</th>
          <th data-col="item">Item / Folder ⇕</th>
          <th data-col="page">Page Name ⇕</th>
          <th data-col="words">Word Count ⇕</th>
          <th data-col="time">Runtime ⇕</th>
          <th data-col="conf">Avg Confidence ⇕</th>
          <th style="cursor:default;">Action</th>
        </tr>
      </thead>
      <tbody id="tblBody"></tbody>
    </table>

    <div class="pagination">
      <span id="pageInfo">Showing 0 of 0 entries</span>
      <div class="page-btns">
        <button id="btnPrev" class="page-btn">&lsaquo; Prev</button>
        <button id="btnNext" class="page-btn">Next &rsaquo;</button>
      </div>
    </div>
  </div>
</div>

<script>
  const records = {records_json};
  const items = {items_json};

  const itemGrid = document.getElementById("itemGrid");
  const selItem = document.getElementById("selItem");
  const selConf = document.getElementById("selConf");
  const txtSearch = document.getElementById("txtSearch");
  const tblBody = document.getElementById("tblBody");
  const pageInfo = document.getElementById("pageInfo");
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");

  let currentPage = 1;
  const pageSize = 50;
  let filtered = [...records];
  let sortCol = "idx";
  let sortAsc = true;

  // Render Item Cards
  items.forEach(i => {{
    const card = document.createElement("div");
    card.className = "item-card";
    card.innerHTML = `
      <div class="item-card-header">
        <div class="item-name" title="${{i.name}}">${{i.name}}</div>
        <div class="item-badge">${{i.pages}} pgs</div>
      </div>
      <div class="item-stats">
        <div>
          <div class="item-stat-val" style="color:#3fb950;">${{i.words.toLocaleString()}}</div>
          <div class="item-stat-lbl">Words</div>
        </div>
        <div>
          <div class="item-stat-val" style="color:#bc8cff;">${{i.conf}}%</div>
          <div class="item-stat-lbl">Confidence</div>
        </div>
        <div>
          <div class="item-stat-val" style="color:#58a6ff;">${{i.time}}s</div>
          <div class="item-stat-lbl">Runtime</div>
        </div>
      </div>
    `;
    card.addEventListener("click", () => {{
      selItem.value = i.name;
      applyFilters();
    }});
    itemGrid.appendChild(card);

    const opt = document.createElement("option");
    opt.value = i.name;
    opt.textContent = `${{i.name}} (${{i.pages}} pgs)`;
    selItem.appendChild(opt);
  }});

  function applyFilters() {{
    const q = txtSearch.value.toLowerCase().trim();
    const selI = selItem.value;
    const selC = selConf.value;

    filtered = records.filter(r => {{
      if (q && !r.page.toLowerCase().includes(q) && !r.item.toLowerCase().includes(q)) return false;
      if (selI && r.item !== selI) return false;
      if (selC !== "all" && r.tier !== selC) return false;
      return true;
    }});

    sortRecords();
    currentPage = 1;
    renderTable();
  }}

  function sortRecords() {{
    filtered.sort((a, b) => {{
      let va = a[sortCol];
      let vb = b[sortCol];
      if (typeof va === "string") {{
        va = va.toLowerCase();
        vb = vb.toLowerCase();
      }}
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    }});
  }}

  function renderTable() {{
    tblBody.innerHTML = "";
    const total = filtered.length;
    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, total);
    const slice = filtered.slice(start, end);

    if (slice.length === 0) {{
      tblBody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:24px;color:#8b949e;">No matching pages found</td></tr>`;
      pageInfo.textContent = "Showing 0 of 0 entries";
      btnPrev.disabled = true;
      btnNext.disabled = true;
      return;
    }}

    slice.forEach(r => {{
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color:#8b949e;">${{r.idx}}</td>
        <td style="font-weight:600;color:#f0f6fc;">${{r.item}}</td>
        <td>${{r.page}}</td>
        <td style="font-weight:600;">${{r.words.toLocaleString()}}</td>
        <td style="color:#79c0ff;">${{r.time}}s</td>
        <td><span class="badge-conf" style="background:${{r.color}}22;color:${{r.color}};border:1px solid ${{r.color}}44;">${{r.conf}}%</span></td>
        <td><a href="${{r.viewer}}" target="_blank" class="btn-view">View Bounding Boxes ↗</a></td>
      `;
      tblBody.appendChild(tr);
    }});

    pageInfo.textContent = `Showing ${{start + 1}} to ${{end}} of ${{total}} entries`;
    btnPrev.disabled = currentPage === 1;
    btnNext.disabled = end >= total;
  }}

  document.querySelectorAll("th[data-col]").forEach(th => {{
    th.addEventListener("click", () => {{
      const col = th.getAttribute("data-col");
      if (sortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        sortAsc = true;
      }}
      sortRecords();
      renderTable();
    }});
  }});

  txtSearch.addEventListener("input", applyFilters);
  selItem.addEventListener("change", applyFilters);
  selConf.addEventListener("change", applyFilters);

  btnPrev.addEventListener("click", () => {{ if (currentPage > 1) {{ currentPage--; renderTable(); }} }});
  btnNext.addEventListener("click", () => {{ if ((currentPage * pageSize) < filtered.length) {{ currentPage++; renderTable(); }} }});

  renderTable();
</script>
</body>
</html>"""

    index_path = dashboard_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    return index_path


def build_ocr_run_dashboard(
    output_root: Path | str,
    input_root: Optional[Path | str] = None,
    engine_name: str = "Paddle PP-OCRv6",
    total_time_sec: float = 0.0,
    per_page_timings: Optional[Dict[str, float]] = None,
    auto_open: bool = True,
    dashboard_folder_name: str = "HTML_Results_Dashboard",
) -> Optional[Path]:
    """Scan output folder for all completed OCR pages and build a self-contained results dashboard.

    Args:
        output_root: Root directory where OCR outputs were written.
        input_root: Original image input folder (optional, used to locate images).
        engine_name: Display name of OCR engine.
        total_time_sec: Total elapsed wall-clock time for the run.
        per_page_timings: Optional map of {stem: elapsed_seconds}.
        auto_open: If True, opens the dashboard in the default browser in a new tab.
        dashboard_folder_name: Name of the results dashboard directory.

    Returns:
        Path to the generated dashboard index.html, or None if no pages found.
    """
    output_root = Path(output_root)
    input_root = Path(input_root) if input_root else None
    per_page_timings = per_page_timings or {}
    dashboard_dir = output_root / dashboard_folder_name
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dashboard_dir / "images"
    pages_dir = dashboard_dir / "pages"

    # 1. Collect all completed ALTO XML / JSON files
    xml_files = sorted(output_root.rglob("*.xml"))
    json_files = sorted(output_root.rglob("*.json"))

    # Map stems to output files, ignoring files already in dashboard/
    items_by_stem: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for x in xml_files:
        if "dashboard" in x.parts or dashboard_folder_name in x.parts:
            continue
        items_by_stem[x.stem]["xml"] = x
        items_by_stem[x.stem]["dir"] = x.parent

    for j in json_files:
        if (
            "dashboard" in j.parts
            or dashboard_folder_name in j.parts
            or j.name.endswith("_adjudications.json")
        ):
            continue
        stem = j.stem
        if stem.endswith("_res"):
            stem = stem[:-4]
        items_by_stem[stem]["json"] = j
        if "dir" not in items_by_stem[stem]:
            items_by_stem[stem]["dir"] = j.parent

    if not items_by_stem:
        print("[Dashboard] No OCR XML/JSON outputs found to build results dashboard.")
        return None

    # 2. Build image lookup map
    image_lookup: Dict[str, Path] = {}
    search_dirs = [output_root]
    if input_root and Path(input_root).exists():
        search_dirs.append(Path(input_root))

    for s_dir in search_dirs:
        for f in s_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                if (
                    "dashboard" not in f.parts
                    and dashboard_folder_name not in f.parts
                    and "__preprocessed" not in f.name
                ):
                    image_lookup[f.stem] = f

    # 3. Parse each page
    pages_list: List[PageResult] = []
    natural_sort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

    for stem, paths in sorted(items_by_stem.items(), key=lambda x: natural_sort(x[0])):
        xml_p = paths.get("xml")
        json_p = paths.get("json")
        parent_dir = paths.get("dir", output_root)

        # Derive clean item / collection name
        try:
            rel = parent_dir.relative_to(output_root)
            item_name = str(rel).replace("\\", "/") if str(rel) != "." else parent_dir.name
        except Exception:
            item_name = parent_dir.name

        # Locate source image
        img_p = image_lookup.get(stem)
        if not img_p:
            # Check locally in parent_dir or output_root
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"]:
                cand = parent_dir / f"{stem}{ext}"
                if cand.exists():
                    img_p = cand
                    break

        # Parse words & dimensions
        page_w = 1800.0
        page_h = 2400.0
        words: List[WordBox] = []

        if xml_p and xml_p.exists():
            page_w, page_h, words = parse_alto_words(xml_p)
        elif json_p and json_p.exists():
            page_w, page_h, words = parse_json_words(json_p)

        # Fallback to image size if available
        if img_p and img_p.exists():
            try:
                with Image.open(img_p) as im:
                    page_w, page_h = im.size
            except Exception:
                pass

        runtime = per_page_timings.get(stem, 0.0)
        pages_list.append(PageResult(
            page_name=stem,
            item_name=item_name,
            image_path=img_p or (parent_dir / f"{stem}.jpg"),
            alto_path=xml_p,
            json_path=json_p,
            page_width=page_w,
            page_height=page_h,
            runtime_sec=runtime,
            words=words,
        ))

    if not pages_list:
        return None

    # Calculate average runtime if per-page timings weren't recorded
    if total_time_sec > 0 and not any(p.runtime_sec > 0 for p in pages_list):
        avg_rt = total_time_sec / len(pages_list)
        for p in pages_list:
            p.runtime_sec = avg_rt

    # 4. Generate individual page HTML viewers
    dashboard_index_rel = "../../index.html"
    for idx, page in enumerate(pages_list):
        prev_url = f"../{pages_list[idx-1].item_name}/{pages_list[idx-1].page_name}_results.html" if idx > 0 else None
        next_url = f"../{pages_list[idx+1].item_name}/{pages_list[idx+1].page_name}_results.html" if idx < len(pages_list) - 1 else None
        
        # Target viewer path
        item_pages_dir = pages_dir / page.item_name
        item_pages_dir.mkdir(parents=True, exist_ok=True)
        viewer_path = item_pages_dir / f"{page.page_name}_results.html"

        # Copy image to dashboard/images/<item>/<page>.jpg for 100% self-contained local viewer
        item_imgs_dir = images_dir / page.item_name
        item_imgs_dir.mkdir(parents=True, exist_ok=True)
        img_dest = item_imgs_dir / f"{page.page_name}.jpg"

        if page.image_path.exists():
            if not img_dest.exists() or img_dest.stat().st_size != page.image_path.stat().st_size:
                try:
                    shutil.copy2(page.image_path, img_dest)
                except Exception:
                    pass

        rel_image = f"../../images/{page.item_name}/{page.page_name}.jpg" if img_dest.exists() else f"../../images/{page.page_name}.jpg"

        generate_single_page_results_html(
            page=page,
            prev_url=prev_url,
            next_url=next_url,
            dashboard_url=dashboard_index_rel,
            output_html_path=viewer_path,
            rel_image_path=rel_image,
            engine_name=engine_name,
        )

    # 5. Generate Master Dashboard index.html
    master_index = generate_master_results_dashboard(
        pages=pages_list,
        dashboard_dir=dashboard_dir,
        engine_name=engine_name,
        total_time_sec=total_time_sec,
    )

    print(f"\n[Dashboard] Successfully built OCR Results Dashboard: {master_index}")
    print(f"[Dashboard] Total Pages: {len(pages_list)} | Total Words: {sum(p.word_count for p in pages_list):,} | Runtime: {total_time_sec:.2f}s")

    # 6. Auto-open in browser in a new tab if requested
    if auto_open and master_index.exists():
        try:
            abs_path = master_index.resolve()
            url = abs_path.as_uri()
            print(f"[Dashboard] Opening results dashboard in browser: {url}")
            opened = False
            if hasattr(os, "startfile"):
                try:
                    os.startfile(str(abs_path))
                    opened = True
                except Exception as ex_sf:
                    print(f"[Dashboard] os.startfile note: {ex_sf}")
            if not opened:
                webbrowser.open_new_tab(url)
        except Exception as exc:
            print(f"[Dashboard] Could not auto-open browser: {exc}")

    return master_index

