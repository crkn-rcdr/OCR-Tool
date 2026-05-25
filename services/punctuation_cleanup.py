"""Conservative punctuation cleanup for OCR output text.

The rules in this module are intentionally based on the artifact patterns
flagged by ``tools/compare_ocr_errors.py``. The goal is not to rewrite OCR
content aggressively, but to smooth out common punctuation debris in the text
that flows into TXT, PDF searchable text, and ALTO output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DOT_LEADER_RE = re.compile(r"\.{4,}\s*-?\s*\d+\s*$")
REPEATED_PUNCTUATION_PATTERNS = (
    (re.compile(r",{2,}"), ","),
    (re.compile(r";{2,}"), ";"),
    (re.compile(r":{2,}"), ":"),
    (re.compile(r"!{2,}"), "!"),
    (re.compile(r"\?{2,}"), "?"),
)
LINE_START_PATTERNS = (
    (re.compile(r"^\.(?=[A-Za-z])"), ""),
    (re.compile(r'^\.(["“”\']{1,3})(?=[A-Za-z])'), r"\1"),
    (re.compile(r'^["“”][.,;:!?]+(?=[A-Za-z])'), ""),
    (re.compile(r"^[-_=]{2,}(?=[A-Za-z0-9])"), ""),
    (re.compile(r"^[-_=]*[•*]+\s*(?=[A-Za-z0-9])"), ""),
)
QUOTE_DEBRIS_PATTERNS = (
    (re.compile(r'([“"\'`])[.,;:!?]{2,}'), r"\1"),
    (re.compile(r'[.,;:!?]{2,}([”"\'`])'), r"\1"),
    (re.compile(r"\)\.\)|\.\)\."), ")."),
    (re.compile(r"\)\.\)|\.\)\)"), ".)"),
)
SPACING_PATTERNS = (
    (re.compile(r"\b([A-Za-z]{2,})\.([A-Z][a-z]+)\b"), r"\1. \2"),
    (re.compile(r"\b([A-Za-z]{2,}),([A-Za-z]{2,})\b"), r"\1, \2"),
    (re.compile(r"\b([A-Za-z]{2,});([A-Za-z])"), r"\1; \2"),
    (re.compile(r"\b([A-Za-z]{2,}):([A-Za-z])"), r"\1: \2"),
    (re.compile(r"\bI\.([a-z]{2,})\b"), r"I. \1"),
)


def clean_ocr_line_text(text: Any) -> str:
    return _clean_text_value(text, allow_space_insertion=True)


def clean_ocr_token_text(text: Any) -> str:
    return _clean_text_value(text, allow_space_insertion=False)


def clean_ocr_text_content(text: str) -> str:
    value = str(text or "")
    if not value:
        return value
    return "\n".join(clean_ocr_line_text(line) for line in value.splitlines())


def clean_ocr_json_text_in_place(json_path: Path) -> bool:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    changed = clean_ocr_payload_text_in_place(payload)
    if changed:
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return changed


def clean_ocr_payload_text_in_place(payload: dict[str, Any]) -> bool:
    changed = False

    overall = payload.get("overall_ocr_res")
    if isinstance(overall, dict):
        changed = clean_ocr_payload_text_in_place(overall) or changed

    rec_texts = payload.get("rec_texts")
    if isinstance(rec_texts, list):
        for index, text in enumerate(rec_texts):
            cleaned = clean_ocr_line_text(text)
            if cleaned != text:
                rec_texts[index] = cleaned
                changed = True

    text_word = payload.get("text_word")
    if isinstance(text_word, list):
        for line_index, words in enumerate(text_word):
            if not isinstance(words, list):
                continue
            for word_index, word in enumerate(words):
                cleaned = clean_ocr_token_text(word)
                if cleaned != word:
                    words[word_index] = cleaned
                    changed = True

    ocr_blocks = payload.get("ocr_blocks")
    if isinstance(ocr_blocks, list):
        for block in ocr_blocks:
            if not isinstance(block, dict):
                continue
            for line in block.get("lines") or []:
                if not isinstance(line, dict):
                    continue
                if "text" in line:
                    cleaned = clean_ocr_line_text(line.get("text"))
                    if cleaned != line.get("text"):
                        line["text"] = cleaned
                        changed = True
                for string_item in line.get("strings") or []:
                    if not isinstance(string_item, dict) or "text" not in string_item:
                        continue
                    cleaned = clean_ocr_token_text(string_item.get("text"))
                    if cleaned != string_item.get("text"):
                        string_item["text"] = cleaned
                        changed = True

    parsing_res_list = _extract_parsing_res_list(payload)
    for block in parsing_res_list:
        if not isinstance(block, dict):
            continue
        for key in ("block_content", "text", "markdown"):
            if key not in block:
                continue
            cleaned = clean_ocr_line_text(block.get(key))
            if cleaned != block.get(key):
                block[key] = cleaned
                changed = True

    return changed


def _extract_parsing_res_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return payload["result"]["layoutParsingResults"][0]["prunedResult"][
            "parsing_res_list"
        ]
    except (KeyError, IndexError, TypeError):
        return []


def _clean_text_value(text: Any, *, allow_space_insertion: bool) -> str:
    value = str(text or "")
    if not value:
        return ""

    value = value.replace("\ufeff", "")
    value = value.rstrip("\r\n")
    if not value.strip():
        return value.strip()

    value = _clean_line_start_artifacts(value)
    value = _clean_quote_debris(value)
    value = _clean_repeated_punctuation(value)
    if allow_space_insertion:
        value = _clean_spacing_artifacts(value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _clean_line_start_artifacts(value: str) -> str:
    updated = value
    for pattern, replacement in LINE_START_PATTERNS:
        updated = pattern.sub(replacement, updated, count=1)
    return updated


def _clean_quote_debris(value: str) -> str:
    updated = value
    for pattern, replacement in QUOTE_DEBRIS_PATTERNS:
        updated = pattern.sub(replacement, updated)
    return updated


def _clean_repeated_punctuation(value: str) -> str:
    updated = value
    if not DOT_LEADER_RE.search(updated):
        updated = re.sub(r"\.{4,}", "...", updated)
        updated = re.sub(r"(?<!\.)\.\.(?!\.)", ".", updated)
    # Newspaper OCR often emits double hyphens at line ends for broken words.
    updated = re.sub(r"\b([A-Za-z]{2,})--$", r"\1-", updated)
    # A smaller set of cases are punctuation separators like "Co.-- Captain".
    updated = re.sub(r"(?<=\.)--(?=\s+[A-Z])", "", updated)
    updated = re.sub(r"-{3,}", "--", updated)
    for pattern, replacement in REPEATED_PUNCTUATION_PATTERNS:
        updated = pattern.sub(replacement, updated)
    return updated


def _clean_spacing_artifacts(value: str) -> str:
    updated = value
    previous = None
    while updated != previous:
        previous = updated
        for pattern, replacement in SPACING_PATTERNS:
            updated = pattern.sub(replacement, updated)
    return updated
