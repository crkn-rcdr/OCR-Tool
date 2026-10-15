import unittest
from unittest.mock import MagicMock
from PIL import Image
from services.post_ocr_vl import run_backend_check, request_dotsocr_word, clean_vl_text
from services.ocr_dots import clean_dotsocr_text, text_lines, estimated_line_boxes
from services.ocr_logger import option_applies_to_strategy
from services.ocr_defaults import DEFAULT_POST_OCR_VL_BACKEND
from app import build_post_ocr_vl_options, demo

print("1. Testing text cleaning...")
fenced_text = "```markdown\nHello World\n```"
assert clean_dotsocr_text(fenced_text) == "Hello World", f"Got {clean_dotsocr_text(fenced_text)!r}"
assert text_lines("Line 1\n\nLine 2") == ["Line 1", "Line 2"]
boxes = estimated_line_boxes(["Line 1", "Line 2"], 1000, 2000)
assert len(boxes) == 2

print("2. Testing ocr_logger option_applies_to_strategy...")
assert option_applies_to_strategy("dotsocr_url", {"method": "DotsOCR"}) == True
assert option_applies_to_strategy("dotsocr_url", {"method": "PP-OCRv6"}) == False
assert option_applies_to_strategy("post_ocr_vl_dotsocr_url", {"method": "PP-OCRv6", "post_ocr_vl_enabled": True, "post_ocr_vl_backend": "dots_ocr"}) == True
assert option_applies_to_strategy("post_ocr_vl_paddlevl_url", {"method": "PP-OCRv6", "post_ocr_vl_enabled": True, "post_ocr_vl_backend": "dots_ocr"}) == False
assert option_applies_to_strategy("post_ocr_vl_paddlevl_url", {"method": "PP-OCRv6", "post_ocr_vl_enabled": True, "post_ocr_vl_backend": "paddlevl"}) == True

print("3. Testing build_post_ocr_vl_options...")
opts_dots = build_post_ocr_vl_options(True, "dots_ocr")
assert opts_dots["post_ocr_vl_backend"] == "dots_ocr"
opts_paddle = build_post_ocr_vl_options(True, "paddlevl")
assert opts_paddle["post_ocr_vl_backend"] == "paddlevl"

print("4. Testing Dots OCR mock request in post_ocr_vl...")
mock_session = MagicMock()
mock_resp = MagicMock()
mock_resp.status_code = 200
mock_resp.json.return_value = {
    "choices": [{"message": {"content": "Constitution"}}]
}
mock_session.post.return_value = mock_resp

img = Image.new("RGB", (100, 30), color="white")
res = run_backend_check("dots_ocr", img, "Constitutlon", {}, session=mock_session)
assert res == "Constitution", f"Expected Constitution, got {res}"

print("5. Testing demo gradio app compilation...")
assert demo is not None

print("\n--- ALL TESTS PASSED SUCCESSFULLY! ---")
