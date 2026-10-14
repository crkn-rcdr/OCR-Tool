"""Playwright browser test verifying that UI options below model selection dynamically show ONLY relevant options."""

import sys
import time
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from playwright.sync_api import sync_playwright
import app

def test_ui_options():
    print("=== Launching Gradio Server for Playwright UI Validation ===")
    port = 7865
    server_thread = threading.Thread(
        target=lambda: app.demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            show_error=True,
        ),
        daemon=True,
    )
    server_thread.start()
    time.sleep(3)

    artifacts_dir = Path(r"C:\Users\blapierre\.gemini\antigravity\brain\f6588a69-8b29-468d-8bb8-badf417fc319")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        url = f"http://127.0.0.1:{port}"
        print(f"Navigating to Gradio UI: {url}")
        page.goto(url, wait_until="networkidle")

        # 1. Default (PP-OCRv6)
        print("\n--- Testing Model: PP-OCRv6 (Default) ---")
        time.sleep(1)
        lang_vis = page.get_by_text("Select Language").first.is_visible()
        tile_vis = page.get_by_text("Image Tiling Strategy").first.is_visible()
        pre_vis = page.get_by_text("pre process").first.is_visible()
        post_vis = page.get_by_text("post process").first.is_visible()
        backend_vis = page.get_by_text("Post-OCR VLM Arbiter").first.is_visible()
        dash_vis = page.get_by_text("Generate HTML results dashboard").first.is_visible()

        print(f"  Select Language:              visible={lang_vis} (Expected: True)")
        print(f"  Image Tiling Strategy:        visible={tile_vis} (Expected: True)")
        print(f"  pre process:                  visible={pre_vis} (Expected: True)")
        print(f"  post process:                 visible={post_vis} (Expected: True)")
        print(f"  Post-OCR VLM Arbiter:         visible={backend_vis} (Expected: True)")
        print(f"  Generate HTML dashboard:      visible={dash_vis} (Expected: True)")

        assert lang_vis and tile_vis and pre_vis and post_vis and backend_vis and dash_vis
        ss_ppocr = artifacts_dir / "ui_options_ppocr.png"
        page.screenshot(path=str(ss_ppocr))
        print(f"  [Screenshot] Saved: {ss_ppocr}")

        # 2. Switch to Indra OCR
        print("\n--- Testing Model: Indra OCR (Fast CPU) ---")
        page.locator("label:has-text('Indra OCR (Fast CPU)')").click()
        time.sleep(1)

        lang_vis = page.get_by_text("Select Language").first.is_visible()
        tile_vis = page.get_by_text("Image Tiling Strategy").first.is_visible()
        pre_vis = page.get_by_text("pre process").first.is_visible()
        post_vis = page.get_by_text("post process").first.is_visible()
        backend_vis = page.get_by_text("Post-OCR VLM Arbiter").first.is_visible()
        dash_vis = page.get_by_text("Generate HTML results dashboard").first.is_visible()

        print(f"  Select Language:              visible={lang_vis} (Expected: True)")
        print(f"  Image Tiling Strategy:        visible={tile_vis} (Expected: False - HIDDEN)")
        print(f"  pre process:                  visible={pre_vis} (Expected: False - HIDDEN)")
        print(f"  post process:                 visible={post_vis} (Expected: False - HIDDEN)")
        print(f"  Post-OCR VLM Arbiter:         visible={backend_vis} (Expected: False - HIDDEN)")
        print(f"  Generate HTML dashboard:      visible={dash_vis} (Expected: True)")

        assert lang_vis, "Language dropdown should be visible for Indra"
        assert not tile_vis, "Tiling dropdown should be HIDDEN for Indra"
        assert not pre_vis, "pre process should be HIDDEN for Indra"
        assert not post_vis, "post process should be HIDDEN for Indra"
        assert not backend_vis, "Post-OCR arbiter should be HIDDEN for Indra"
        assert dash_vis, "Generate HTML dashboard should be visible"

        ss_indra = artifacts_dir / "ui_options_indra.png"
        page.screenshot(path=str(ss_indra))
        print(f"  [Screenshot] Saved: {ss_indra}")

        # 3. Switch to Dots OCR
        print("\n--- Testing Model: Dots OCR ---")
        page.locator("label:has-text('Dots OCR')").click()
        time.sleep(1)

        lang_vis = page.get_by_text("Select Language").first.is_visible()
        tile_vis = page.get_by_text("Image Tiling Strategy").first.is_visible()
        pre_vis = page.get_by_text("pre process").first.is_visible()
        post_vis = page.get_by_text("post process").first.is_visible()
        backend_vis = page.get_by_text("Post-OCR VLM Arbiter").first.is_visible()
        dash_vis = page.get_by_text("Generate HTML results dashboard").first.is_visible()

        print(f"  Select Language:              visible={lang_vis} (Expected: False - HIDDEN)")
        print(f"  Image Tiling Strategy:        visible={tile_vis} (Expected: False - HIDDEN)")
        print(f"  pre process:                  visible={pre_vis} (Expected: False - HIDDEN)")
        print(f"  post process:                 visible={post_vis} (Expected: False - HIDDEN)")
        print(f"  Post-OCR VLM Arbiter:         visible={backend_vis} (Expected: False - HIDDEN)")
        print(f"  Generate HTML dashboard:      visible={dash_vis} (Expected: True)")

        assert not lang_vis and not tile_vis and not pre_vis and not post_vis and not backend_vis
        assert dash_vis

        # 4. Switch back to PP-OCRv6
        print("\n--- Switching back to PP-OCRv6 ---")
        page.locator("label:has-text('PaddleOCR (PP-OCRv6)')").click()
        time.sleep(1)
        assert page.get_by_text("Select Language").first.is_visible()
        assert page.get_by_text("Image Tiling Strategy").first.is_visible()
        assert page.get_by_text("pre process").first.is_visible()
        assert page.get_by_text("post process").first.is_visible()
        print("  All PP-OCRv6 options restored cleanly!")

        browser.close()

    print("\n========================================================")
    print("ALL MODEL OPTION VISIBILITY TESTS PASSED!")
    print("========================================================")

if __name__ == "__main__":
    test_ui_options()
