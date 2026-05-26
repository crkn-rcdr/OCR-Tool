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
    """Clean punctuation artifacts in a full OCR line of text."""
    return _clean_text_value(text, allow_space_insertion=True)


def clean_ocr_token_text(text: Any) -> str:
    """Clean punctuation artifacts in a single OCR token."""
    return _clean_text_value(text, allow_space_insertion=False)


def clean_ocr_text_content(text: str) -> str:
    """Clean a multi-line OCR text blob and preserve line breaks."""
    value = str(text or "")
    if not value:
        return value
    return "\n".join(clean_ocr_line_list(value.splitlines()))


def clean_ocr_line_list(lines: list[Any]) -> list[str]:
    """Clean OCR lines and then fix cross-line carryover artifacts."""
    cleaned_lines = [clean_ocr_line_text(line) for line in lines]
    return _clean_line_sequence(cleaned_lines)


def clean_ocr_json_text_in_place(json_path: Path) -> bool:
    """Load an OCR JSON file, clean its text fields, and save if changed."""
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
    """Clean supported text fields inside an OCR payload in place."""
    changed = False

    overall = payload.get("overall_ocr_res")
    if isinstance(overall, dict):
        changed = clean_ocr_payload_text_in_place(overall) or changed

    rec_texts = payload.get("rec_texts")
    if isinstance(rec_texts, list):
        cleaned_lines = clean_ocr_line_list(rec_texts)
        for index, text in enumerate(rec_texts):
            cleaned = cleaned_lines[index]
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
            lines = block.get("lines") or []
            cleaned_line_texts = clean_ocr_line_list(
                [
                    line.get("text")
                    if isinstance(line, dict) and "text" in line
                    else ""
                    for line in lines
                ]
            )
            for line_index, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue
                if "text" in line:
                    cleaned = cleaned_line_texts[line_index]
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
    text_keys: list[str | None] = []
    cleaned_block_texts = clean_ocr_line_list(
        [
            block.get(key)
            if isinstance(block, dict)
            and (key := next((name for name in ("block_content", "text", "markdown") if name in block), None))
            else ""
            for block in parsing_res_list
        ]
    )
    for index, block in enumerate(parsing_res_list):
        if not isinstance(block, dict):
            text_keys.append(None)
            continue
        key = next((name for name in ("block_content", "text", "markdown") if name in block), None)
        text_keys.append(key)
        if not key:
            continue
        cleaned = cleaned_block_texts[index]
        if cleaned != block.get(key):
            block[key] = cleaned
            changed = True

    return changed


def _extract_parsing_res_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the nested PaddleVL parsing result list when present."""
    try:
        return payload["result"]["layoutParsingResults"][0]["prunedResult"][
            "parsing_res_list"
        ]
    except (KeyError, IndexError, TypeError):
        return []


def _clean_line_sequence(lines: list[str]) -> list[str]:
    """Apply cleanup rules that need neighboring line context."""
    cleaned = list(lines)
    for index in range(len(cleaned) - 1):
        current = str(cleaned[index] or "")
        following = str(cleaned[index + 1] or "")
        if not current or not following:
            continue

        updated_following = _remove_duplicate_hyphen_carryover(current, following)
        updated_following = _remove_fragment_hyphen_carryover(current, updated_following)
        if updated_following != following:
            cleaned[index + 1] = updated_following
    return cleaned


def _remove_duplicate_hyphen_carryover(current: str, following: str) -> str:
    """Drop a repeated carryover word when a hyphenated line wraps badly."""
    current_match = re.search(r"([A-Za-z]{2,})-$", current)
    next_match = re.match(r"\s*([A-Za-z]{2,})([.,;:!?-]?\s*)(.*)$", following)
    if not current_match or not next_match:
        return following

    trailing_word = current_match.group(1).casefold()
    next_word = next_match.group(1)
    if next_word.casefold() != trailing_word:
        return following

    remainder = next_match.group(3).lstrip()
    if not remainder:
        return following
    return remainder


def _remove_fragment_hyphen_carryover(current: str, following: str) -> str:
    """Trim fragment duplication caused by broken hyphenated line joins."""
    fragment_match = re.search(r"([A-Za-z]{1,4})-$", current)
    next_match = re.match(r"\s*([A-Za-z]{3,})(.*)$", following)
    if not fragment_match or not next_match:
        return following

    fragment = fragment_match.group(1)
    next_word = next_match.group(1)
    if len(next_word) <= len(fragment) + 1:
        return following
    if not next_word.casefold().startswith(fragment.casefold()):
        return following

    remainder_word = next_word[len(fragment) :]
    if len(remainder_word) < 2:
        return following
    return f"{remainder_word}{next_match.group(2)}"


def _clean_text_value(text: Any, *, allow_space_insertion: bool) -> str:
    """Apply the conservative punctuation cleanup pipeline to one text value."""
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
    """Remove recurring junk punctuation found at the start of OCR lines."""
    updated = value
    for pattern, replacement in LINE_START_PATTERNS:
        updated = pattern.sub(replacement, updated, count=1)
    return updated


def _clean_quote_debris(value: str) -> str:
    """Collapse punctuation debris that accumulates around quotes or brackets."""
    updated = value
    for pattern, replacement in QUOTE_DEBRIS_PATTERNS:
        updated = pattern.sub(replacement, updated)
    return updated


def _clean_repeated_punctuation(value: str) -> str:
    """Normalize repeated punctuation runs to a single intended mark."""
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
    """Restore obvious missing spaces introduced by OCR punctuation joins."""
    updated = value
    previous = None
    while updated != previous:
        previous = updated
        for pattern, replacement in SPACING_PATTERNS:
            updated = pattern.sub(replacement, updated)
    return updated
