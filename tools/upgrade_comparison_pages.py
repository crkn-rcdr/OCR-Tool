#!/usr/bin/env python3
"""Upgrade legacy CRKN OCR comparison HTML files to the modern interactive adjudicator viewer.

Injects:
- Pinned Adjudication Inspector panel
- Manual winner selection (Paddle Win, ABBYY Win, Both Valid, Neither Valid)
- Interactive card chips (P, B, Both, None)
- Import JSON and Export JSON capabilities
- Browser LocalStorage persistence (compatible with static GitHub Pages hosting)
- Real-time QA score and discrepancy recalculation
- Keyboard shortcuts (P, B, T, N, R, Arrow keys)

Usage:
    python tools/upgrade_comparison_pages.py compare
    python tools/upgrade_comparison_pages.py "C:\\Users\\blapierre\\Documents\\OCR-Evaluations\\docs"
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_modern_html(
    matched_data: list,
    p_only_data: list,
    b_only_data: list,
    page_width: float,
    page_height: float,
    item_name: str,
    collection_name: str,
    img_src: str,
    pipeline_words: int,
    benchmark_words: int,
    rel_index_path: str = "../../index.html",
    prev_url: str | None = None,
    next_url: str | None = None,
) -> str:
    # Ensure idx field exists on each item in matched_data
    for idx, m in enumerate(matched_data):
        if "idx" not in m:
            m["idx"] = idx
        if "manual_verdict" not in m:
            m["manual_verdict"] = None

    # Calculate statistics
    exact_matches = sum(1 for m in matched_data if m.get("exact_match"))
    clean_matches = sum(1 for m in matched_data if m.get("clean_match"))
    discrepancies = [m for m in matched_data if not m.get("clean_match")]
    total_matched = len(matched_data)

    exact_pct = (exact_matches / total_matched * 100.0) if total_matched else 0.0
    clean_pct = (clean_matches / total_matched * 100.0) if total_matched else 0.0

    p_wins = sum(1 for m in discrepancies if m.get("lexical_verdict") == "PADDLE_WIN")
    b_wins = sum(1 for m in discrepancies if m.get("lexical_verdict") == "ABBYY_WIN")
    both_valid = sum(1 for m in discrepancies if m.get("lexical_verdict") == "BOTH_VALID")
    neither_valid = sum(1 for m in discrepancies if m.get("lexical_verdict") == "NEITHER_VALID")

    decisive_disputes = p_wins + b_wins
    qa_win_rate = (p_wins / decisive_disputes * 100.0) if decisive_disputes > 0 else 50.0

    # Badge class for clean concordance
    if clean_pct >= 90:
        conc_cls = "success"
    elif clean_pct >= 80:
        conc_cls = "warning"
    else:
        conc_cls = "info"

    # Badge class for QA win rate
    qa_badge_cls = "qa-win" if p_wins >= b_wins else "qa-loss"

    prev_btn_html = (
        f'<a href="{prev_url}" class="nav-page-btn">‹ Previous</a>'
        if prev_url
        else '<span class="nav-page-btn disabled">‹ Previous</span>'
    )
    next_btn_html = (
        f'<a href="{next_url}" class="nav-page-btn">Next ›</a>'
        if next_url
        else '<span class="nav-page-btn disabled">Next ›</span>'
    )

    img_tag_left = f"<img class='page-image' id='imgLeft' src='{img_src}' style='opacity:0.85;'>" if img_src else ""
    img_tag_right = f"<img class='page-image' id='imgRight' src='{img_src}' style='opacity:0.85;'>" if img_src else ""

    matched_json = json.dumps(matched_data)
    p_only_json = json.dumps(p_only_data)
    b_only_json = json.dumps(b_only_data)

    coll_disp = collection_name if collection_name else "Root"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRKN OCR Comparison: {item_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #1a1e24; color: #e1e4e8; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header {{ background: #24292e; border-bottom: 1px solid #30363d; padding: 10px 18px; display: flex; justify-content: space-between; align-items: center; z-index: 10; gap: 12px; }}
  
  .nav-left {{ display: flex; align-items: center; gap: 8px; font-size: 0.88rem; }}
  .nav-btn-back {{ background: #21262d; color: #58a6ff; border: 1px solid #30363d; padding: 5px 12px; border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 0.82rem; transition: all 0.15s ease; }}
  .nav-btn-back:hover {{ background: #30363d; color: #79c0ff; border-color: #58a6ff; }}
  .nav-sep {{ color: #484f58; }}
  .nav-coll {{ color: #8b949e; font-weight: 500; font-size: 0.84rem; }}
  h1 {{ font-size: 1.05rem; font-weight: 600; color: #f0f6fc; }}
  
  .nav-center {{ display: flex; align-items: center; gap: 6px; }}
  .nav-page-btn {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 10px; border-radius: 4px; text-decoration: none; font-size: 0.8rem; font-weight: 500; }}
  .nav-page-btn:hover:not(.disabled) {{ background: #30363d; color: #58a6ff; border-color: #58a6ff; }}
  .nav-page-btn.disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .page-current {{ font-size: 0.82rem; color: #8b949e; padding: 0 4px; }}
  
  .badges {{ display: flex; gap: 8px; font-size: 0.8rem; align-items: center; }}
  .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: 600; background: #30363d; }}
  .badge.success {{ background: #198754; color: #fff; }}
  .badge.warning {{ background: #d97706; color: #fff; }}
  .badge.info {{ background: #0284c7; color: #fff; }}
  .badge.qa-win {{ background: #238636; color: #fff; }}
  .badge.qa-loss {{ background: #8957e5; color: #fff; }}
  
  .controls {{ background: #212529; border-bottom: 1px solid #30363d; padding: 8px 18px; display: flex; gap: 14px; align-items: center; font-size: 0.84rem; flex-wrap: wrap; }}
  .controls label {{ display: flex; align-items: center; gap: 5px; cursor: pointer; }}
  .controls select, .controls input[type="range"] {{ vertical-align: middle; }}
  .slider-group {{ display: flex; align-items: center; gap: 6px; }}
  
  .main-container {{ display: flex; flex: 1; overflow: hidden; position: relative; }}
  .viewers-container {{ display: flex; flex: 1; overflow: hidden; position: relative; }}
  
  .panel {{ flex: 1; height: 100%; position: relative; overflow: hidden; border-right: 1px solid #30363d; background: #121519; cursor: grab; }}
  .panel:active {{ cursor: grabbing; }}
  .panel-header {{ position: absolute; top: 10px; left: 10px; z-index: 5; background: rgba(0, 0, 0, 0.75); padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; pointer-events: none; }}
  
  .canvas-stage {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; width: {page_width}px; height: {page_height}px; }}
  .page-image {{ position: absolute; top: 0; left: 0; width: {page_width}px; height: {page_height}px; display: block; user-select: none; pointer-events: none; transition: opacity 0.1s ease; }}
  .overlay-svg {{ position: absolute; top: 0; left: 0; width: {page_width}px; height: {page_height}px; pointer-events: auto; }}
  
  rect.agree {{ fill: rgba(40, 167, 69, 0.25); stroke: #28a745; stroke-width: 1.5; cursor: pointer; }}
  rect.discrepancy {{ fill: rgba(245, 158, 11, 0.35); stroke: #f59e0b; stroke-width: 2; cursor: pointer; }}
  rect.p-only {{ fill: rgba(56, 189, 248, 0.35); stroke: #38bdf8; stroke-width: 1.5; cursor: pointer; }}
  rect.b-only {{ fill: rgba(239, 68, 68, 0.35); stroke: #ef4444; stroke-width: 1.5; cursor: pointer; }}
  rect:hover {{ fill-opacity: 0.7; stroke-width: 2.5; }}
  
  /* Selected / Highlighted active word - completely transparent interior */
  rect.highlighted {{
    stroke: #00ffff !important;
    stroke-width: 4 !important;
    fill: none !important;
    filter: drop-shadow(0 0 6px #00ffff);
    vector-effect: non-scaling-stroke;
  }}
  
  .sidebar {{ width: 380px; background: #1f242c; border-left: 1px solid #30363d; display: flex; flex-direction: column; }}
  .sidebar-header {{ padding: 10px 14px; font-weight: 600; border-bottom: 1px solid #30363d; font-size: 0.88rem; background: #24292e; display: flex; justify-content: space-between; align-items: center; }}
  
  /* Pinned Inspector / Adjudicator Panel */
  .inspector-panel {{
    background: #161b22;
    border-bottom: 2px solid #30363d;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: all 0.2s ease;
  }}
  .inspector-panel.empty {{
    padding: 14px 12px;
    color: #8b949e;
    font-size: 0.78rem;
    text-align: center;
    background: #181c24;
  }}
  .inspector-hint {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #8b949e;
  }}
  .inspector-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: #c9d1d9;
  }}
  .inspector-title {{
    font-weight: 600;
    color: #58a6ff;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .inspector-metrics {{
    color: #8b949e;
    font-size: 0.74rem;
  }}
  .inspector-nav {{
    display: flex;
    gap: 4px;
  }}
  .inspector-nav button {{
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 1px 7px;
    font-size: 0.72rem;
    cursor: pointer;
  }}
  .inspector-nav button:hover {{
    background: #30363d;
    color: #58a6ff;
  }}
  
  .inspector-compare {{
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 0.8rem;
  }}
  .insp-row {{
    display: flex;
    align-items: baseline;
    gap: 6px;
  }}
  .insp-tag {{
    font-size: 0.68rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    min-width: 48px;
    text-align: center;
  }}
  .insp-tag.p {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }}
  .insp-tag.b {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }}
  .insp-text {{
    font-weight: 600;
    word-break: break-all;
    color: #f0f6fc;
  }}
  .insp-valid {{
    font-size: 0.75rem;
    margin-left: 2px;
  }}
  .insp-valid.yes {{ color: #3fb950; }}
  .insp-valid.no {{ color: #f85149; }}
  
  .inspector-prompt {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.76rem;
    color: #8b949e;
  }}
  .inspector-actions {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }}
  .btn-adjudicate {{
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 5px 8px;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    transition: all 0.15s ease;
  }}
  .btn-adjudicate:hover {{
    transform: translateY(-1px);
    border-color: #58a6ff;
  }}
  .btn-adjudicate.p-win {{
    color: #3fb950;
    border-color: #238636;
    background: rgba(35, 134, 54, 0.1);
  }}
  .btn-adjudicate.p-win:hover, .btn-adjudicate.p-win.active {{
    background: #238636;
    color: #fff;
  }}
  .btn-adjudicate.b-win {{
    color: #f85149;
    border-color: #da3633;
    background: rgba(218, 54, 51, 0.1);
  }}
  .btn-adjudicate.b-win:hover, .btn-adjudicate.b-win.active {{
    background: #da3633;
    color: #fff;
  }}
  .btn-adjudicate.both-valid {{
    color: #58a6ff;
    border-color: #1f6feb;
    background: rgba(56, 139, 253, 0.1);
  }}
  .btn-adjudicate.both-valid:hover, .btn-adjudicate.both-valid.active {{
    background: #1f6feb;
    color: #fff;
  }}
  .btn-adjudicate.neither-valid {{
    color: #8b949e;
    border-color: #484f58;
    background: rgba(110, 118, 129, 0.1);
  }}
  .btn-adjudicate.neither-valid:hover, .btn-adjudicate.neither-valid.active {{
    background: #484f58;
    color: #fff;
  }}
  
  .inspector-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.7rem;
    color: #8b949e;
    border-top: 1px solid #21262d;
    padding-top: 4px;
  }}
  .insp-status {{
    font-weight: 600;
  }}
  .insp-status.manual {{
    color: #d29922;
  }}
  .btn-reset-word {{
    background: none;
    border: none;
    color: #8b949e;
    cursor: pointer;
    font-size: 0.7rem;
    text-decoration: underline;
    padding: 0;
  }}
  .btn-reset-word:hover {{
    color: #f85149;
  }}
  
  .sidebar-filter-bar {{ padding: 8px 12px; background: #181b20; border-bottom: 1px solid #30363d; display: flex; gap: 8px; align-items: center; }}
  .filter-dropdown {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px 8px; font-size: 0.8rem; width: 100%; }}
  .sidebar-list {{ flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; }}
  
  .card {{ background: #282e38; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; font-size: 0.82rem; cursor: pointer; transition: all 0.15s ease; position: relative; }}
  .card:hover {{ border-color: #60a5fa; background: #2d3542; }}
  .card.active {{
    border-color: #00ffff !important;
    background: #1e3a4f !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.4);
  }}
  .card-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
  .card-diff {{ font-weight: bold; color: #f59e0b; word-break: break-all; }}
  .card-label {{ color: #9ca3af; font-size: 0.75rem; }}
  
  /* Interactive chips on cards */
  .card-chips {{
    display: flex;
    gap: 4px;
    margin-top: 6px;
    padding-top: 4px;
    border-top: 1px dashed #374151;
  }}
  .chip-btn {{
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.68rem;
    color: #8b949e;
    cursor: pointer;
    transition: all 0.1s ease;
  }}
  .chip-btn:hover {{
    border-color: #58a6ff;
    color: #58a6ff;
  }}
  .chip-btn.active.chip-p {{ background: #238636; color: #fff; border-color: #238636; }}
  .chip-btn.active.chip-b {{ background: #da3633; color: #fff; border-color: #da3633; }}
  .chip-btn.active.chip-both {{ background: #1f6feb; color: #fff; border-color: #1f6feb; }}
  .chip-btn.active.chip-neither {{ background: #484f58; color: #fff; border-color: #484f58; }}
  
  .verdict-tag {{ font-size: 0.7rem; font-weight: 600; padding: 2px 6px; border-radius: 3px; }}
  .verdict-tag.paddle-win {{ background: rgba(35, 134, 54, 0.25); color: #3fb950; border: 1px solid #238636; }}
  .verdict-tag.abbyy-win {{ background: rgba(218, 54, 51, 0.25); color: #f85149; border: 1px solid #da3633; }}
  .verdict-tag.both-valid {{ background: rgba(56, 139, 253, 0.2); color: #58a6ff; border: 1px solid #1f6feb; }}
  .verdict-tag.neither-valid {{ background: rgba(110, 118, 129, 0.2); color: #8b949e; border: 1px solid #484f58; }}
  .verdict-tag.manual {{
    box-shadow: 0 0 4px #d29922;
    border-color: #d29922 !important;
  }}
  
  .legend {{ display: flex; gap: 12px; font-size: 0.78rem; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
</style>
</head>
<body>
<header>
  <div class="nav-left">
    <a href="{rel_index_path}" class="nav-btn-back">← Master Dashboard</a>
    <span class="nav-sep">/</span>
    <span class="nav-coll">{coll_disp}</span>
    <span class="nav-sep">/</span>
    <h1>{item_name}</h1>
  </div>
  
  <div class="nav-center">
    {prev_btn_html}
    <span class="page-current">{item_name}</span>
    {next_btn_html}
  </div>

  <div class="badges">
    <span class="badge {conc_cls}">Concordance: {clean_pct:.1f}% ({clean_matches}/{total_matched})</span>
    <span class="badge {qa_badge_cls}" id="badgeQaWinRate" title="Paddle {p_wins} wins vs ABBYY {b_wins} wins on disputed words">Paddle QA Win Rate: {qa_win_rate:.2f}%</span>
  </div>
</header>

<div class="controls">
  <label><input type="checkbox" id="chkImage" checked> Show Scanned Page</label>
  <div class="slider-group">
    <label for="rngOpacity">Opacity:</label>
    <input type="range" id="rngOpacity" min="0" max="1" step="0.05" value="0.85" style="width: 80px;">
  </div>
  <label><input type="checkbox" id="chkPolys" checked> Highlight Word Bounding Boxes</label>
  <label><input type="checkbox" id="chkAgreed" checked> Show Agreed Words (Green)</label>
  
  <button id="btnResetView" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:3px 10px;border-radius:4px;font-size:0.8rem;cursor:pointer;">Reset View</button>

  <div class="legend" style="margin-left:auto;">
    <div class="legend-item"><div class="legend-color" style="background:#28a745;"></div> Agreed</div>
    <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div> Discrepancy</div>
    <div class="legend-item"><div class="legend-color" style="background:#38bdf8;"></div> Pipeline Only</div>
    <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div> ABBYY Only</div>
    <div class="legend-item"><div class="legend-color" style="background:#00ffff;"></div> Selected</div>
  </div>
</div>

<div class="main-container">
  <div class="viewers-container">
    <!-- Pipeline Panel (Left) -->
    <div class="panel" id="panelLeft">
      <div class="panel-header">Pipeline (PaddleOCR + VLM) - {pipeline_words} words</div>
      <div class="canvas-stage" id="stageLeft">
        {img_tag_left}
        <svg class="overlay-svg" id="svgLeft" viewBox="0 0 {page_width} {page_height}"></svg>
      </div>
    </div>
    
    <!-- Benchmark Panel (Right) -->
    <div class="panel" id="panelRight">
      <div class="panel-header">Benchmark (ABBYY FineReader) - {benchmark_words} words</div>
      <div class="canvas-stage" id="stageRight">
        {img_tag_right}
        <svg class="overlay-svg" id="svgRight" viewBox="0 0 {page_width} {page_height}"></svg>
      </div>
    </div>
  </div>

  <!-- Discrepancy Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <div>
        <span>Discrepancies</span>
        <span class="badge warning" id="badgeSidebarDiffCount" style="font-size:0.75rem;padding:2px 6px;margin-left:4px;">{len(discrepancies)}</span>
        <span id="lblManualCount" style="display:none;font-size:0.72rem;color:#58a6ff;font-weight:600;margin-left:6px;"></span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <button id="btnImportAdjudications" title="Import manual adjudications from JSON file" style="background:#21262d;color:#7ee787;border:1px solid #30363d;padding:2px 8px;border-radius:4px;font-size:0.72rem;cursor:pointer;font-weight:600;">Import JSON</button>
        <button id="btnExportAdjudications" title="Export manual adjudications to JSON file" style="background:#21262d;color:#58a6ff;border:1px solid #30363d;padding:2px 8px;border-radius:4px;font-size:0.72rem;cursor:pointer;font-weight:600;">Export JSON</button>
        <input type="file" id="fileImportAdjudications" accept=".json" style="display:none;" />
      </div>
    </div>

    <!-- Pinned Adjudication Inspector -->
    <div id="inspectorPanel" class="inspector-panel empty">
      <div class="inspector-hint">
        <span style="font-size:1.1rem;">🔍</span>
        <span>Click any word box or card to select winner</span>
      </div>
    </div>

    <div class="sidebar-filter-bar">
      <select id="selFilterDiscrepancies" class="filter-dropdown">
        <option value="all" id="optFilterAll">All Discrepancies ({len(discrepancies)})</option>
        <option value="PADDLE_WIN" id="optFilterPaddle">Paddle Wins ({p_wins})</option>
        <option value="ABBYY_WIN" id="optFilterAbbyy">ABBYY Wins ({b_wins})</option>
        <option value="BOTH_VALID" id="optFilterBoth">Both Valid ({both_valid})</option>
        <option value="NEITHER_VALID" id="optFilterNeither">Neither Valid ({neither_valid})</option>
      </select>
    </div>
    <div class="sidebar-list" id="sidebarList"></div>
  </div>
</div>

<script>
  const matchedData = {matched_json};
  const pOnlyData = {p_only_json};
  const bOnlyData = {b_only_json};
  const pageWidth = {page_width};
  const pageHeight = {page_height};
  const itemName = "{item_name}";
  const collectionName = "{coll_disp}";
  const STORAGE_KEY = `crkn_adj_${{itemName}}`;

  // SVGs & Sidebar
  const svgLeft = document.getElementById("svgLeft");
  const svgRight = document.getElementById("svgRight");
  const sidebarList = document.getElementById("sidebarList");
  let activeIdx = null;
  const cards = [];

  function escapeHtml(text) {{
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }}

  function formatVerdictLabel(v) {{
    if (v === "PADDLE_WIN") return "Paddle Win";
    if (v === "ABBYY_WIN") return "ABBYY Win";
    if (v === "BOTH_VALID") return "Both Valid";
    if (v === "NEITHER_VALID") return "Neither Valid";
    return v || "Unspecified";
  }}

  function buildVerdictBadgeHTML(v, isManual) {{
    const star = isManual ? ' <span title="Manually Adjudicated">✍️</span>' : '';
    const manCls = isManual ? ' manual' : '';
    if (v === "PADDLE_WIN") {{
      return `<span class="verdict-tag paddle-win${{manCls}}">Paddle Win${{star}}</span>`;
    }} else if (v === "ABBYY_WIN") {{
      return `<span class="verdict-tag abbyy-win${{manCls}}">ABBYY Win${{star}}</span>`;
    }} else if (v === "BOTH_VALID") {{
      return `<span class="verdict-tag both-valid${{manCls}}">Both Valid${{star}}</span>`;
    }} else if (v === "NEITHER_VALID") {{
      return `<span class="verdict-tag neither-valid${{manCls}}">Neither${{star}}</span>`;
    }}
    return '';
  }}

  function loadAdjudicationsFromStorage() {{
    try {{
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const map = JSON.parse(raw);
      Object.keys(map).forEach(idxStr => {{
        const idx = parseInt(idxStr, 10);
        if (matchedData[idx]) {{
          matchedData[idx].manual_verdict = map[idxStr];
        }}
      }});
    }} catch (e) {{
      console.warn("Failed to load adjudications from localStorage:", e);
    }}
  }}

  function saveAdjudicationsToStorage() {{
    const map = {{}};
    matchedData.forEach((m, idx) => {{
      if (m.manual_verdict) {{
        map[idx] = m.manual_verdict;
      }}
    }});
    try {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    }} catch (e) {{
      console.warn("localStorage quota exceeded or unavailable:", e);
    }}
  }}

  // Populate SVGs and Sidebar
  loadAdjudicationsFromStorage();

  // 1. Pipeline-only boxes (rendered in background)
  pOnlyData.forEach((p, pIdx) => {{
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", p.box[0] * pageWidth);
    rect.setAttribute("y", p.box[1] * pageHeight);
    rect.setAttribute("width", p.box[2] * pageWidth);
    rect.setAttribute("height", p.box[3] * pageHeight);
    rect.setAttribute("class", "p-only");
    rect.id = "p-only-rect-" + pIdx;
    rect.innerHTML = `<title>Pipeline Only: "${{escapeHtml(p.text)}}"</title>`;
    rect.onclick = (e) => {{ e.stopPropagation(); selectSingleBox("P", pIdx); }};
    svgLeft.appendChild(rect);
  }});

  // 2. Benchmark-only boxes (rendered in background)
  bOnlyData.forEach((b, bIdx) => {{
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", b.box[0] * pageWidth);
    rect.setAttribute("y", b.box[1] * pageHeight);
    rect.setAttribute("width", b.box[2] * pageWidth);
    rect.setAttribute("height", b.box[3] * pageHeight);
    rect.setAttribute("class", "b-only");
    rect.id = "b-only-rect-" + bIdx;
    rect.innerHTML = `<title>ABBYY Only: "${{escapeHtml(b.text)}}"</title>`;
    rect.onclick = (e) => {{ e.stopPropagation(); selectSingleBox("B", bIdx); }};
    svgRight.appendChild(rect);
  }});

  // 3. Matched Word Pairs
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
    rectL.id = "p-rect-" + idx;
    rectL.dataset.idx = idx;
    rectL.innerHTML = `<title>Pipeline: "${{escapeHtml(m.p_text)}}" | ABBYY: "${{escapeHtml(m.b_text)}}" (IoU: ${{m.iou}})</title>`;
    rectL.onclick = (e) => {{ e.stopPropagation(); selectWord(idx); }};
    svgLeft.appendChild(rectL);

    // Right SVG
    const rectR = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rectR.setAttribute("x", m.b_box[0] * pageWidth);
    rectR.setAttribute("y", m.b_box[1] * pageHeight);
    rectR.setAttribute("width", m.b_box[2] * pageWidth);
    rectR.setAttribute("height", m.b_box[3] * pageHeight);
    rectR.setAttribute("class", cls);
    rectR.id = "b-rect-" + idx;
    rectR.dataset.idx = idx;
    rectR.innerHTML = `<title>ABBYY: "${{escapeHtml(m.b_text)}}" | Pipeline: "${{escapeHtml(m.p_text)}}" (IoU: ${{m.iou}})</title>`;
    rectR.onclick = (e) => {{ e.stopPropagation(); selectWord(idx); }};
    svgRight.appendChild(rectR);

    // Sidebar items for discrepancies (or manually overridden items)
    if (!isAgree || m.manual_verdict) {{
      const card = document.createElement("div");
      card.className = "card";
      card.id = "card-" + idx;
      
      const effVerdict = m.manual_verdict || m.lexical_verdict || "NONE";
      card.dataset.verdict = effVerdict;

      const pCheck = m.p_valid ? ' <span style="color:#3fb950;font-size:0.75rem;">✓</span>' : '';
      const bCheck = m.b_valid ? ' <span style="color:#3fb950;font-size:0.75rem;">✓</span>' : '';

      card.innerHTML = `
        <div class="card-row">
          <span class="card-diff">P: ${{escapeHtml(m.p_text)}}${{pCheck}}</span>
          <span id="badge-container-${{idx}}">${{buildVerdictBadgeHTML(effVerdict, Boolean(m.manual_verdict))}}</span>
        </div>
        <div class="card-row">
          <span class="card-label">B: ${{escapeHtml(m.b_text)}}${{bCheck}}</span>
          <span class="card-label">IoU: ${{m.iou}} | Sim: ${{Math.round(m.sim * 100)}}%</span>
        </div>
        <div class="card-chips" id="chips-${{idx}}">
          <button class="chip-btn chip-p ${{effVerdict === 'PADDLE_WIN' ? 'active' : ''}}" title="Award Paddle Win" onclick="event.stopPropagation(); setWinner(${{idx}}, 'PADDLE_WIN');">P</button>
          <button class="chip-btn chip-b ${{effVerdict === 'ABBYY_WIN' ? 'active' : ''}}" title="Award ABBYY Win" onclick="event.stopPropagation(); setWinner(${{idx}}, 'ABBYY_WIN');">B</button>
          <button class="chip-btn chip-both ${{effVerdict === 'BOTH_VALID' ? 'active' : ''}}" title="Mark Both Valid" onclick="event.stopPropagation(); setWinner(${{idx}}, 'BOTH_VALID');">Both</button>
          <button class="chip-btn chip-neither ${{effVerdict === 'NEITHER_VALID' ? 'active' : ''}}" title="Mark Neither Valid" onclick="event.stopPropagation(); setWinner(${{idx}}, 'NEITHER_VALID');">None</button>
        </div>
      `;
      card.onclick = () => selectWord(idx);
      sidebarList.appendChild(card);
      cards.push({{ idx: idx, el: card }});
    }}
  }});

  function applyCurrentFilter() {{
    const selFilter = document.getElementById("selFilterDiscrepancies");
    if (!selFilter) return;
    const val = selFilter.value;
    cards.forEach(c => {{
      const v = c.el.dataset.verdict;
      if (val === "all" || v === val) {{
        c.el.style.display = "block";
      }} else {{
        c.el.style.display = "none";
      }}
    }});
  }}

  const selFilter = document.getElementById("selFilterDiscrepancies");
  if (selFilter) {{
    selFilter.onchange = applyCurrentFilter;
  }}

  function selectWord(idx) {{
    document.querySelectorAll(".highlighted").forEach(el => el.classList.remove("highlighted"));
    document.querySelectorAll(".card.active").forEach(el => el.classList.remove("active"));

    const pRect = document.getElementById("p-rect-" + idx);
    const bRect = document.getElementById("b-rect-" + idx);
    if (pRect) pRect.classList.add("highlighted");
    if (bRect) bRect.classList.add("highlighted");

    const card = document.getElementById("card-" + idx);
    if (card) {{
      card.classList.add("active");
      card.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
    }}

    activeIdx = idx;
    const m = matchedData[idx];
    zoomToNormBox(m.p_box);
    renderInspector(idx);
  }}

  function selectSingleBox(engine, boxIdx) {{
    document.querySelectorAll(".highlighted").forEach(el => el.classList.remove("highlighted"));
    document.querySelectorAll(".card.active").forEach(el => el.classList.remove("active"));
    activeIdx = null;

    if (engine === "P") {{
      const el = document.getElementById("p-only-rect-" + boxIdx);
      if (el) el.classList.add("highlighted");
      const p = pOnlyData[boxIdx];
      zoomToNormBox(p.box);
      renderInspectorEngineOnly("P", p.text);
    }} else {{
      const el = document.getElementById("b-only-rect-" + boxIdx);
      if (el) el.classList.add("highlighted");
      const b = bOnlyData[boxIdx];
      zoomToNormBox(b.box);
      renderInspectorEngineOnly("B", b.text);
    }}
  }}

  function renderInspector(idx) {{
    const insp = document.getElementById("inspectorPanel");
    if (!insp) return;

    if (idx === null || !matchedData[idx]) {{
      insp.className = "inspector-panel empty";
      insp.innerHTML = `
        <div class="inspector-hint">
          <span style="font-size:1.1rem;">🔍</span>
          <span>Click any word box or card to select winner</span>
        </div>
      `;
      return;
    }}

    const m = matchedData[idx];
    insp.className = "inspector-panel";
    const effVerdict = m.manual_verdict || m.lexical_verdict || (m.clean_match ? "BOTH_VALID" : "NEITHER_VALID");
    const isManual = Boolean(m.manual_verdict);

    const pValidIcon = m.p_valid ? '<span class="insp-valid yes">✓</span>' : '<span class="insp-valid no">✗</span>';
    const bValidIcon = m.b_valid ? '<span class="insp-valid yes">✓</span>' : '<span class="insp-valid no">✗</span>';

    insp.innerHTML = `
      <div class="inspector-header">
        <div class="inspector-title">
          <span>Adjudicate Winner</span>
          <span class="inspector-metrics">IoU: ${{m.iou}} | Sim: ${{Math.round(m.sim * 100)}}%</span>
        </div>
        <div class="inspector-nav">
          <button id="btnInspPrev" title="Previous discrepancy (Up arrow)">‹</button>
          <button id="btnInspNext" title="Next discrepancy (Down arrow)">›</button>
        </div>
      </div>
      <div class="inspector-compare">
        <div class="insp-row">
          <span class="insp-tag p">Pipeline</span>
          <span class="insp-text">${{escapeHtml(m.p_text)}}</span>
          ${{pValidIcon}}
        </div>
        <div class="insp-row">
          <span class="insp-tag b">ABBYY</span>
          <span class="insp-text">${{escapeHtml(m.b_text)}}</span>
          ${{bValidIcon}}
        </div>
      </div>
      <div class="inspector-prompt">
        <span>Select verdict:</span>
        <span class="insp-status ${{isManual ? 'manual' : ''}}">
          ${{isManual ? '✍️ Manual: ' + formatVerdictLabel(effVerdict) : 'Auto: ' + formatVerdictLabel(effVerdict)}}
        </span>
      </div>
      <div class="inspector-actions">
        <button class="btn-adjudicate p-win ${{effVerdict === 'PADDLE_WIN' ? 'active' : ''}}" id="btnWinPaddle" title="Shortcut: 'P' or '1'">
          <span>✓</span> Paddle Win
        </button>
        <button class="btn-adjudicate b-win ${{effVerdict === 'ABBYY_WIN' ? 'active' : ''}}" id="btnWinAbbyy" title="Shortcut: 'B' or '2'">
          <span>✓</span> ABBYY Win
        </button>
        <button class="btn-adjudicate both-valid ${{effVerdict === 'BOTH_VALID' ? 'active' : ''}}" id="btnWinBoth" title="Shortcut: 'T' or '3'">
          Both Valid
        </button>
        <button class="btn-adjudicate neither-valid ${{effVerdict === 'NEITHER_VALID' ? 'active' : ''}}" id="btnWinNeither" title="Shortcut: 'N' or '4'">
          Neither
        </button>
      </div>
      <div class="inspector-footer">
        <span>Shortcuts: P, B, T, N | R to reset</span>
        ${{isManual ? `<button class="btn-reset-word" id="btnResetWord" title="Revert to auto dictionary decision">Reset to Auto</button>` : ''}}
      </div>
    `;

    document.getElementById("btnWinPaddle").onclick = () => setWinner(idx, "PADDLE_WIN");
    document.getElementById("btnWinAbbyy").onclick = () => setWinner(idx, "ABBYY_WIN");
    document.getElementById("btnWinBoth").onclick = () => setWinner(idx, "BOTH_VALID");
    document.getElementById("btnWinNeither").onclick = () => setWinner(idx, "NEITHER_VALID");
    const btnReset = document.getElementById("btnResetWord");
    if (btnReset) btnReset.onclick = () => resetWinner(idx);

    document.getElementById("btnInspPrev").onclick = () => navigateDiscrepancy(-1);
    document.getElementById("btnInspNext").onclick = () => navigateDiscrepancy(1);
  }}

  function renderInspectorEngineOnly(engine, boxText) {{
    const insp = document.getElementById("inspectorPanel");
    if (!insp) return;
    const label = engine === "P" ? "Pipeline Only" : "ABBYY Only";
    const color = engine === "P" ? "#38bdf8" : "#f87171";
    insp.className = "inspector-panel";
    insp.innerHTML = `
      <div class="inspector-header">
        <div class="inspector-title" style="color:${{color}};">
          <span>${{label}}</span>
        </div>
      </div>
      <div class="inspector-compare">
        <div class="insp-row">
          <span class="insp-tag ${{engine.toLowerCase()}}">${{engine === "P" ? "Pipeline" : "ABBYY"}}</span>
          <span class="insp-text">${{escapeHtml(boxText)}}</span>
        </div>
      </div>
      <div class="inspector-footer" style="padding-top:6px;font-style:italic;">
        <span>Only one engine recognized this box (no rival box to adjudicate).</span>
      </div>
    `;
  }}

  function setWinner(idx, winner) {{
    const m = matchedData[idx];
    if (!m) return;
    m.manual_verdict = winner;
    saveAdjudicationsToStorage();
    updateWordCard(idx);
    renderInspector(idx);
    recalculateSummaryStats();
  }}

  function resetWinner(idx) {{
    const m = matchedData[idx];
    if (!m) return;
    delete m.manual_verdict;
    saveAdjudicationsToStorage();
    updateWordCard(idx);
    renderInspector(idx);
    recalculateSummaryStats();
  }}

  function updateWordCard(idx) {{
    const m = matchedData[idx];
    if (!m) return;

    const effVerdict = m.manual_verdict || m.lexical_verdict || (m.clean_match ? "BOTH_VALID" : "NEITHER_VALID");
    const isManual = Boolean(m.manual_verdict);

    let cardEl = document.getElementById("card-" + idx);
    if (!cardEl && m.manual_verdict && m.manual_verdict !== "BOTH_VALID") {{
      cardEl = document.createElement("div");
      cardEl.className = "card";
      cardEl.id = "card-" + idx;
      sidebarList.prepend(cardEl);
      cards.unshift({{ idx: idx, el: cardEl }});
      cardEl.onclick = () => selectWord(idx);
    }}

    if (cardEl) {{
      cardEl.dataset.verdict = effVerdict;
      const badgeContainer = document.getElementById("badge-container-" + idx);
      if (badgeContainer) {{
        badgeContainer.innerHTML = buildVerdictBadgeHTML(effVerdict, isManual);
      }}
      const chips = document.getElementById("chips-" + idx);
      if (chips) {{
        chips.innerHTML = `
          <button class="chip-btn chip-p ${{effVerdict === 'PADDLE_WIN' ? 'active' : ''}}" title="Award Paddle Win" onclick="event.stopPropagation(); setWinner(${{idx}}, 'PADDLE_WIN');">P</button>
          <button class="chip-btn chip-b ${{effVerdict === 'ABBYY_WIN' ? 'active' : ''}}" title="Award ABBYY Win" onclick="event.stopPropagation(); setWinner(${{idx}}, 'ABBYY_WIN');">B</button>
          <button class="chip-btn chip-both ${{effVerdict === 'BOTH_VALID' ? 'active' : ''}}" title="Mark Both Valid" onclick="event.stopPropagation(); setWinner(${{idx}}, 'BOTH_VALID');">Both</button>
          <button class="chip-btn chip-neither ${{effVerdict === 'NEITHER_VALID' ? 'active' : ''}}" title="Mark Neither Valid" onclick="event.stopPropagation(); setWinner(${{idx}}, 'NEITHER_VALID');">None</button>
        `;
      }}
    }}
  }}

  function recalculateSummaryStats() {{
    let pWins = 0;
    let bWins = 0;
    let bothValid = 0;
    let neitherValid = 0;
    let totalDiscrepancies = 0;
    let manualCount = 0;

    matchedData.forEach(m => {{
      const isDisc = !m.clean_match || (m.manual_verdict && m.manual_verdict !== "BOTH_VALID");
      const eff = m.manual_verdict || m.lexical_verdict || (m.clean_match ? "BOTH_VALID" : "NEITHER_VALID");
      if (m.manual_verdict) manualCount++;

      if (isDisc) {{
        totalDiscrepancies++;
        if (eff === "PADDLE_WIN") pWins++;
        else if (eff === "ABBYY_WIN") bWins++;
        else if (eff === "BOTH_VALID") bothValid++;
        else if (eff === "NEITHER_VALID") neitherValid++;
      }}
    }});

    const decisive = pWins + bWins;
    const winRate = decisive > 0 ? ((pWins / decisive) * 100).toFixed(2) : "50.00";

    const badgeQa = document.getElementById("badgeQaWinRate");
    if (badgeQa) {{
      badgeQa.textContent = `Paddle QA Win Rate: ${{winRate}}%`;
      badgeQa.className = `badge ${{pWins >= bWins ? 'qa-win' : 'qa-loss'}}`;
      badgeQa.title = `Paddle ${{pWins}} wins vs ABBYY ${{bWins}} wins on disputed words (${{manualCount}} manual reviews)`;
    }}

    const optAll = document.getElementById("optFilterAll");
    if (optAll) optAll.textContent = `All Discrepancies (${{totalDiscrepancies}})`;
    const optP = document.getElementById("optFilterPaddle");
    if (optP) optP.textContent = `Paddle Wins (${{pWins}})`;
    const optB = document.getElementById("optFilterAbbyy");
    if (optB) optB.textContent = `ABBYY Wins (${{bWins}})`;
    const optBoth = document.getElementById("optFilterBoth");
    if (optBoth) optBoth.textContent = `Both Valid (${{bothValid}})`;
    const optNeither = document.getElementById("optFilterNeither");
    if (optNeither) optNeither.textContent = `Neither Valid (${{neitherValid}})`;

    const badgeDiff = document.getElementById("badgeSidebarDiffCount");
    if (badgeDiff) badgeDiff.textContent = totalDiscrepancies;

    const lblManual = document.getElementById("lblManualCount");
    if (lblManual) {{
      if (manualCount > 0) {{
        lblManual.textContent = `(${{manualCount}} manual)`;
        lblManual.style.display = "inline";
      }} else {{
        lblManual.style.display = "none";
      }}
    }}
  }}

  function navigateDiscrepancy(direction) {{
    if (cards.length === 0) return;
    let currCardIndex = cards.findIndex(c => c.idx === activeIdx);
    if (currCardIndex === -1) {{
      currCardIndex = direction > 0 ? -1 : cards.length;
    }}
    let nextCardIndex = currCardIndex + direction;
    if (nextCardIndex >= 0 && nextCardIndex < cards.length) {{
      selectWord(cards[nextCardIndex].idx);
    }}
  }}

  function exportAdjudications() {{
    const overrides = [];
    matchedData.forEach((m, idx) => {{
      if (m.manual_verdict) {{
        overrides.push({{
          idx: idx,
          p_text: m.p_text,
          b_text: m.b_text,
          p_box: m.p_box,
          b_box: m.b_box,
          iou: m.iou,
          original_verdict: m.lexical_verdict,
          adjudicated_verdict: m.manual_verdict,
        }});
      }}
    }});

    const payload = {{
      item_name: itemName,
      collection: collectionName,
      exported_at: new Date().toISOString(),
      total_manual_adjudications: overrides.length,
      overrides: overrides,
    }};

    const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${{itemName}}_adjudications.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }}

  const btnExport = document.getElementById("btnExportAdjudications");
  if (btnExport) {{
    btnExport.onclick = exportAdjudications;
  }}

  const btnImport = document.getElementById("btnImportAdjudications");
  const fileInput = document.getElementById("fileImportAdjudications");
  if (btnImport && fileInput) {{
    btnImport.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {{
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {{
        try {{
          const parsed = JSON.parse(evt.target.result);
          let loaded = 0;
          const overrides = Array.isArray(parsed.overrides)
            ? parsed.overrides
            : (Array.isArray(parsed.adjudications) ? parsed.adjudications : null);

          if (overrides) {{
            overrides.forEach(entry => {{
              const idx = typeof entry.idx === "number" ? entry.idx : parseInt(entry.idx, 10);
              const verdict = entry.adjudicated_verdict || entry.verdict;
              if (!isNaN(idx) && verdict && matchedData[idx]) {{
                matchedData[idx].manual_verdict = verdict;
                updateWordCard(idx);
                loaded++;
              }}
            }});
          }} else if (typeof parsed === "object" && parsed !== null) {{
            Object.keys(parsed).forEach(k => {{
              const idx = parseInt(k, 10);
              if (!isNaN(idx) && matchedData[idx] && typeof parsed[k] === "string") {{
                matchedData[idx].manual_verdict = parsed[k];
                updateWordCard(idx);
                loaded++;
              }}
            }});
          }}

          if (loaded > 0) {{
            saveAdjudicationsToStorage();
            recalculateSummaryStats();
            applyCurrentFilter();
            if (activeIdx !== null) renderInspector(activeIdx);
            alert(`Successfully imported ${{loaded}} manual adjudications!`);
          }} else {{
            alert("No matching adjudications found in the imported JSON file for this page.");
          }}
        }} catch (err) {{
          alert("Error parsing adjudications JSON file: " + err.message);
        }}
        fileInput.value = "";
      }};
      reader.readAsText(file);
    }};
  }}

  // Keyboard navigation and winner selection
  window.addEventListener("keydown", (e) => {{
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

    if (activeIdx !== null && matchedData[activeIdx]) {{
      const key = e.key.toUpperCase();
      if (key === "P" || key === "1") {{
        e.preventDefault();
        setWinner(activeIdx, "PADDLE_WIN");
      }} else if (key === "B" || key === "A" || key === "2") {{
        e.preventDefault();
        setWinner(activeIdx, "ABBYY_WIN");
      }} else if (key === "T" || key === "3") {{
        e.preventDefault();
        setWinner(activeIdx, "BOTH_VALID");
      }} else if (key === "N" || key === "4") {{
        e.preventDefault();
        setWinner(activeIdx, "NEITHER_VALID");
      }} else if (key === "R") {{
        e.preventDefault();
        resetWinner(activeIdx);
      }}
    }}

    if (e.key === "ArrowDown") {{
      e.preventDefault();
      navigateDiscrepancy(1);
    }} else if (e.key === "ArrowUp") {{
      e.preventDefault();
      navigateDiscrepancy(-1);
    }}
  }});

  // Pan & Zoom
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

  function fitToView() {{
    const panelW = panelLeft.clientWidth || 800;
    const panelH = panelLeft.clientHeight || 900;
    scale = Math.min(panelW / pageWidth, panelH / pageHeight);
    panX = (panelW - pageWidth * scale) / 2;
    panY = 10;
    updateTransform();
  }}

  fitToView();
  window.addEventListener("resize", fitToView);

  document.getElementById("btnResetView").onclick = () => {{
    fitToView();
    document.querySelectorAll(".highlighted").forEach(el => el.classList.remove("highlighted"));
    document.querySelectorAll(".card.active").forEach(el => el.classList.remove("active"));
    activeIdx = null;
    const insp = document.getElementById("inspectorPanel");
    if (insp) {{
      insp.className = "inspector-panel empty";
      insp.innerHTML = `
        <div class="inspector-hint">
          <span style="font-size:1.1rem;">🔍</span>
          <span>Click any word box or card to select winner</span>
        </div>
      `;
    }}
  }};

  function zoomToNormBox(box) {{
    scale = 3.5;
    const panelW = panelLeft.clientWidth;
    const panelH = panelLeft.clientHeight;
    const boxCenterX = (box[0] + (box[2] / 2.0)) * pageWidth;
    const boxCenterY = (box[1] + (box[3] / 2.0)) * pageHeight;

    panX = (panelW / 2.0) - (boxCenterX * scale);
    panY = (panelH / 2.0) - (boxCenterY * scale);
    updateTransform();
  }}

  // Layer toggles
  document.getElementById("chkImage").onchange = (e) => {{
    const op = e.target.checked ? document.getElementById("rngOpacity").value : 0;
    const imgL = document.getElementById("imgLeft");
    const imgR = document.getElementById("imgRight");
    if (imgL) imgL.style.opacity = op;
    if (imgR) imgR.style.opacity = op;
  }};

  document.getElementById("rngOpacity").oninput = (e) => {{
    const imgL = document.getElementById("imgLeft");
    const imgR = document.getElementById("imgRight");
    if (document.getElementById("chkImage").checked) {{
      if (imgL) imgL.style.opacity = e.target.value;
      if (imgR) imgR.style.opacity = e.target.value;
    }}
  }};

  document.getElementById("chkPolys").onchange = (e) => {{
    const disp = e.target.checked ? "block" : "none";
    svgLeft.style.display = disp;
    svgRight.style.display = disp;
  }};

  document.getElementById("chkAgreed").onchange = (e) => {{
    const disp = e.target.checked ? "inline" : "none";
    document.querySelectorAll("rect.agree").forEach(el => {{
      el.style.display = disp;
    }});
  }};

  recalculateSummaryStats();
</script>
</body>
</html>
"""


def upgrade_comparison_html(file_path: Path) -> bool:
    """Check and upgrade an HTML comparison file to the modern adjudicator viewer."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check if already upgraded with all modern features
        if "btnImportAdjudications" in html and "inspectorPanel" in html:
            return False

        # Extract data
        m_match = re.search(r"const matchedData\s*=\s*(\[[\s\S]*?\]);\s*(?:const|let|var|\n)", html)
        if not m_match:
            return False

        matched_data = json.loads(m_match.group(1))

        p_match = re.search(r"const pOnlyData\s*=\s*(\[[\s\S]*?\]);\s*(?:const|let|var|\n)", html)
        p_only_data = json.loads(p_match.group(1)) if p_match else []

        b_match = re.search(r"const bOnlyData\s*=\s*(\[[\s\S]*?\]);\s*(?:const|let|var|\n)", html)
        b_only_data = json.loads(b_match.group(1)) if b_match else []

        w_match = re.search(r"const pageWidth\s*=\s*([\d.]+);", html)
        page_width = float(w_match.group(1)) if w_match else 1800.0

        h_match = re.search(r"const pageHeight\s*=\s*([\d.]+);", html)
        page_height = float(h_match.group(1)) if h_match else 2200.0

        item_match = re.search(r"const itemName\s*=\s*[\"']([^\"']+)[\"'];", html) or re.search(
            r"<title>CRKN OCR Comparison:\s*([^<]+)</title>", html
        )
        item_name = item_match.group(1).strip() if item_match else file_path.stem.replace("_comparison", "")

        coll_match = re.search(r"const collectionName\s*=\s*[\"']([^\"']*)[\"'];", html) or re.search(
            r"<span class=\"nav-coll\">([^<]+)</span>", html
        )
        coll_disp = coll_match.group(1).strip() if coll_match else ""

        img_match = re.search(r"id=['\"]imgLeft['\"][^>]*src=['\"]([^'\"]+)['\"]", html)
        img_src = img_match.group(1) if img_match else f"{item_name}.jpg"

        p_words_match = re.search(r"Pipeline \(PaddleOCR[^\)]*\)\s*-\s*(\d+)\s*words", html)
        pipeline_words = int(p_words_match.group(1)) if p_words_match else len(matched_data)

        b_words_match = re.search(r"Benchmark \(ABBYY[^\)]*\)\s*-\s*(\d+)\s*words", html)
        benchmark_words = int(b_words_match.group(1)) if b_words_match else len(matched_data)

        # Navigation links
        prev_match = re.search(r'<a href="([^"]+)" class="nav-page-btn"[^>]*>‹ Previous</a>', html)
        prev_url = prev_match.group(1) if prev_match else None

        next_match = re.search(r'<a href="([^"]+)" class="nav-page-btn"[^>]*>Next ›</a>', html)
        next_url = next_match.group(1) if next_match else None

        back_match = re.search(r'<a href="([^"]+)" class="nav-btn-back">', html)
        rel_index = back_match.group(1) if back_match else "../../index.html"

        modern_html = build_modern_html(
            matched_data=matched_data,
            p_only_data=p_only_data,
            b_only_data=b_only_data,
            page_width=page_width,
            page_height=page_height,
            item_name=item_name,
            collection_name=coll_disp,
            img_src=img_src,
            pipeline_words=pipeline_words,
            benchmark_words=benchmark_words,
            rel_index_path=rel_index,
            prev_url=prev_url,
            next_url=next_url,
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modern_html)

        return True
    except Exception as e:
        print(f"Error upgrading {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upgrade legacy comparison HTML files to modern adjudicator viewer.")
    parser.add_argument("target_dirs", nargs="+", help="Directories to scan for *_comparison.html")
    args = parser.parse_args()

    for target in args.target_dirs:
        p = Path(target)
        if not p.exists():
            print(f"Directory not found: {p}")
            continue

        print(f"\nScanning: {p}...")
        html_files = list(p.rglob("*_comparison.html"))
        print(f"Found {len(html_files)} comparison files.")

        upgraded = 0
        skipped = 0
        for f in html_files:
            if upgrade_comparison_html(f):
                upgraded += 1
            else:
                skipped += 1

        print(f"Done for {p}: {upgraded} upgraded, {skipped} already modern or skipped.")


if __name__ == "__main__":
    main()

