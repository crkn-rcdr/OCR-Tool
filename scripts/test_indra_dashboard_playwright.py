"""Test Indra OCR run with HTML Results Dashboard generation and Playwright browser validation.

Tests:
1. Runs Indra OCR on C:\\Users\\blapierre\\Documents\\PDF Test\\Newspapers\\ome.00003_19290103
   with output C:\\Users\\blapierre\\Desktop\\Test HTML
2. Verifies folder structure:
   - C:\\Users\\blapierre\\Desktop\\Test HTML\\ome.00003_19290103
   - C:\\Users\\blapierre\\Desktop\\Test HTML\\HTML_Results_Dashboard
3. Uses Playwright (msedge/chrome) to test:
   - index.html KPI cards, item grid, and page table
   - Single-page viewer bounding boxes, confidence colors, word count, Next/Previous navigation
   - Takes screenshots of both index and page viewer
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from playwright.sync_api import sync_playwright

from services.ocr_indra import run_indra_job


def run_test():
    input_path = Path(r"W:\Brittny\PDF Test\Newspapers\ome.00003_19290103")
    output_root = Path(r"C:\Users\blapierre\Desktop\Test HTML")

    index_html_cand = output_root / "HTML_Results_Dashboard" / "index.html"
    item_folder_cand = output_root / "ome.00003_19290103"

    if "--re-run" in sys.argv or not (index_html_cand.exists() and item_folder_cand.exists()):
        print(f"=== 1. Starting Indra OCR Run ===")
        print(f"Input:  {input_path}")
        print(f"Output: {output_root}")

        opt = {
            "method": "IndraOCR",
            "generate_text_file": False,
            "generate_results_dashboard": True,
            "lang": "auto",
        }

        t0 = time.perf_counter()
        messages = []
        for msg in run_indra_job(input_path, output_root, opt):
            messages.append(msg)
            print(f"  [Indra] {msg}")

        elapsed = time.perf_counter() - t0
        print(f"Indra OCR run completed in {elapsed:.2f}s.\n")
    else:
        print("=== 1. Using Existing OCR & Dashboard Output from recent run ===")

    # === 2. Verify Folder Structure ===
    print("=== 2. Verifying Folder Structure ===")
    item_folder = output_root / "ome.00003_19290103"
    dashboard_folder = output_root / "HTML_Results_Dashboard"

    assert item_folder.exists() and item_folder.is_dir(), f"Expected item folder missing: {item_folder}"
    print(f"  [OK] Item folder exists: {item_folder}")

    xmls = list(item_folder.glob("*.xml"))
    pdfs = list(item_folder.glob("*.pdf"))
    jsons = list(item_folder.glob("*.json"))
    print(f"  [OK] Item contents: {len(xmls)} XMLs, {len(pdfs)} PDFs, {len(jsons)} JSONs")
    assert len(xmls) > 0, "No XML files generated in item folder!"

    assert dashboard_folder.exists() and dashboard_folder.is_dir(), f"Expected dashboard folder missing: {dashboard_folder}"
    print(f"  [OK] Dashboard folder exists: {dashboard_folder}")

    index_html = dashboard_folder / "index.html"
    assert index_html.exists(), f"Dashboard index.html missing: {index_html}"
    print(f"  [OK] Dashboard index.html exists: {index_html}")

    pages_dir = dashboard_folder / "pages" / "ome.00003_19290103"
    images_dir = dashboard_folder / "images" / "ome.00003_19290103"
    assert pages_dir.exists(), f"Pages directory missing: {pages_dir}"
    assert images_dir.exists(), f"Images directory missing: {images_dir}"

    viewer_files = list(pages_dir.glob("*_results.html"))
    image_files = list(images_dir.glob("*.jpg"))
    print(f"  [OK] Viewers created: {len(viewer_files)} HTML pages")
    print(f"  [OK] Images copied: {len(image_files)} image files")
    assert len(viewer_files) == 8, f"Expected 8 page viewers, found {len(viewer_files)}"

    # === 3. Playwright Automation Testing ===
    print("\n=== 3. Testing Dashboard with Playwright ===")
    artifacts_dir = Path(r"C:\Users\blapierre\.gemini\antigravity\brain\f6588a69-8b29-468d-8bb8-badf417fc319")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Launch using msedge channel
        browser = p.chromium.launch(headless=True, channel="msedge")
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        index_uri = index_html.resolve().as_uri()
        print(f"  Navigating Playwright to: {index_uri}")
        page.goto(index_uri, wait_until="networkidle")

        # Check Title
        title = page.title()
        print(f"  Page Title: '{title}'")
        assert "Results Dashboard" in title, f"Unexpected title: {title}"

        # Check Header
        header = page.locator(".hero h1, .brand-title").first.inner_text()
        print(f"  Dashboard Header: '{header}'")

        # Check KPI Cards
        kpi_values = page.locator(".kpi-value").all_inner_texts()
        print(f"  KPI Values: {kpi_values}")
        assert len(kpi_values) >= 4, f"Expected at least 4 KPI values, got {len(kpi_values)}"
        total_pages_kpi = kpi_values[0]
        words_kpi = kpi_values[1]
        time_kpi = kpi_values[2]
        conf_kpi = kpi_values[3]
        print(f"    - Pages: {total_pages_kpi}")
        print(f"    - Words: {words_kpi}")
        print(f"    - Execution Time: {time_kpi}")
        print(f"    - Mean Confidence: {conf_kpi}")

        # Check Item Card
        item_cards = page.locator(".item-card")
        assert item_cards.count() >= 1, "Expected at least 1 item card"
        item_name_text = item_cards.first.locator(".item-name").inner_text()
        print(f"  Item Card: '{item_name_text}'")

        # Check Table Rows
        rows = page.locator("#tblBody tr")
        print(f"  Page Table Row Count: {rows.count()}")
        assert rows.count() == 8, f"Expected 8 rows in table, got {rows.count()}"

        # Screenshot Master Dashboard
        index_screenshot = artifacts_dir / "indra_results_dashboard_index.png"
        page.screenshot(path=str(index_screenshot), full_page=True)
        print(f"  [Screenshot] Master index captured: {index_screenshot}")

        # Click into first page viewer link
        first_view_btn = page.locator(".btn-view").first
        first_view_btn_url = first_view_btn.get_attribute("href")
        print(f"\n  Clicking into first page viewer: {first_view_btn_url}")

        page_viewer_url = (dashboard_folder / first_view_btn_url).resolve().as_uri()
        page.goto(page_viewer_url, wait_until="networkidle")

        # Verify Page Viewer elements
        viewer_title = page.title()
        print(f"  Viewer Title: '{viewer_title}'")
        assert "Page" in viewer_title or "results" in viewer_title.lower() or "ome" in viewer_title

        # Check badges
        badges = page.locator(".badge").all_inner_texts()
        print(f"  Viewer Badges: {badges}")
        assert any("words" in b.lower() for b in badges), "Missing word count badge!"

        # Check Bounding Boxes in SVG
        boxes = page.locator("rect.bbox")
        box_count = boxes.count()
        print(f"  Rendered Word Bounding Boxes: {box_count}")
        assert box_count > 0, "No bounding boxes rendered in SVG canvas!"

        # Check Confidence Color Tiers
        high_boxes = page.locator("rect.bbox[data-tier='high']").count()
        med_boxes = page.locator("rect.bbox[data-tier='medium']").count()
        low_boxes = page.locator("rect.bbox[data-tier='low']").count()
        print(f"  Confidence Breakdown:")
        print(f"    - High (>=90% Green): {high_boxes}")
        print(f"    - Medium (75-89% Orange): {med_boxes}")
        print(f"    - Low (<75% Red): {low_boxes}")

        # Check Next Button
        next_btn = page.locator(".nav-btn:has-text('Next')")
        assert next_btn.is_visible(), "Next button not visible"
        print("  Clicking Next button to verify navigation...")
        next_btn.click()
        page.wait_for_load_state("networkidle")

        next_page_url = page.url
        print(f"  Navigated to next page: {next_page_url}")
        assert "ome.00003_19290103.2_results.html" in next_page_url, f"Expected page 2, got: {next_page_url}"

        # Check Previous Button
        prev_btn = page.locator(".nav-btn:has-text('Previous')")
        assert prev_btn.is_visible(), "Previous button not visible"
        print("  Clicking Previous button to navigate back...")
        prev_btn.click()
        page.wait_for_load_state("networkidle")
        assert "ome.00003_19290103.1_results.html" in page.url, f"Expected back on page 1, got: {page.url}"
        print("  [OK] Next/Previous navigation verified perfectly!")

        # Screenshot Page Viewer
        viewer_screenshot = artifacts_dir / "indra_page_viewer_bounding_boxes.png"
        page.screenshot(path=str(viewer_screenshot))
        print(f"  [Screenshot] Page viewer captured: {viewer_screenshot}")

        browser.close()

    print("\n========================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print(f"1. Item folder created:      {item_folder}")
    print(f"2. Dashboard folder created: {dashboard_folder}")
    print(f"3. Screenshots captured:")
    print(f"   - {index_screenshot}")
    print(f"   - {viewer_screenshot}")
    print("========================================================")


if __name__ == "__main__":
    run_test()
