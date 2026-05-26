"""Normalize OCR word-box metadata so downstream text consumers stay aligned."""

from __future__ import annotations

import re
from copy import deepcopy
from numbers import Number
from typing import Any


APOSTROPHE_CHARS = {"'", "`", "’", "‘", "ʼ", "´"}


def normalize_ocr_word_boxes_in_payload(payload: dict[str, Any]) -> bool:
    """Merge apostrophe-split word-box metadata in supported OCR payloads.

    PaddleX treats apostrophes as symbols when building `text_word`, so words
    such as `queen's` can arrive as `["queen", "'", "s"]`. The recognizer line
    text normally preserves the word, so use that line text as the guardrail.
    """
    changed = False

    if isinstance(payload.get("overall_ocr_res"), dict):
        changed = normalize_ocr_word_boxes_in_payload(payload["overall_ocr_res"]) or changed

    if (
        isinstance(payload.get("rec_texts"), list)
        and isinstance(payload.get("text_word"), list)
        and isinstance(payload.get("text_word_boxes"), list)
    ):
        rec_texts = payload.get("rec_texts") or []
        text_word = payload.get("text_word") or []
        text_word_boxes = payload.get("text_word_boxes") or []

        for index in range(min(len(text_word), len(text_word_boxes))):
            line_text = rec_texts[index] if index < len(rec_texts) else None
            merged_words, merged_boxes = merge_apostrophe_word_boxes(
                text_word[index],
                text_word_boxes[index],
                line_text=line_text,
            )
            if merged_words != text_word[index] or merged_boxes != text_word_boxes[index]:
                text_word[index] = merged_words
                text_word_boxes[index] = merged_boxes
                changed = True

    return changed


def merge_apostrophe_word_boxes(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Merge apostrophe-split OCR tokens back into single word-box entries."""
    if not isinstance(words, list) or not isinstance(boxes, list):
        return words, boxes

    if len(words) != len(boxes):
        return words, boxes

    items = [
        {"text": str(word), "box": deepcopy(box)}
        for word, box in zip(words, boxes)
    ]
    if not items:
        return words, boxes

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(items):
        item = items[i]

        if is_apostrophe_token(item["text"]):
            previous = merged[-1] if merged else None
            next_item = items[i + 1] if i + 1 < len(items) else None

            if (
                previous is not None
                and next_item is not None
                and is_wordish(previous["text"])
                and is_wordish(next_item["text"])
                and should_join_apostrophe_parts(
                    [previous, item, next_item],
                    line_text=line_text,
                )
            ):
                merged[-1] = combine_word_items([previous, item, next_item])
                i += 2
                continue

            if (
                previous is not None
                and is_wordish(previous["text"])
                and should_join_apostrophe_parts(
                    [previous, item],
                    line_text=line_text,
                )
            ):
                merged[-1] = combine_word_items([previous, item])
                i += 1
                continue

            if (
                next_item is not None
                and is_wordish(next_item["text"])
                and should_join_apostrophe_parts(
                    [item, next_item],
                    line_text=line_text,
                )
            ):
                merged.append(combine_word_items([item, next_item]))
                i += 2
                continue

        merged.append(deepcopy(item))
        i += 1

    return [item["text"] for item in merged], [item["box"] for item in merged]


def should_join_apostrophe_parts(parts: list[dict[str, Any]], line_text: Any = None) -> bool:
    """Decide whether adjacent OCR tokens should be merged around apostrophes."""
    if phrase_appears_in_line(parts, line_text):
        return True

    return boxes_are_close(parts)


def phrase_appears_in_line(parts: list[dict[str, Any]], line_text: Any = None) -> bool:
    """Check whether the merged token text already appears in the line string."""
    if not line_text:
        return False

    phrase = normalize_apostrophes("".join(part["text"] for part in parts)).lower()
    line = normalize_apostrophes(str(line_text)).lower()
    return phrase in line


def boxes_are_close(parts: list[dict[str, Any]]) -> bool:
    """Return whether neighboring token boxes are close enough to merge."""
    for left, right in zip(parts, parts[1:]):
        try:
            left_rect = box_to_rect(left["box"])
            right_rect = box_to_rect(right["box"])
        except Exception:
            return False

        left_height = max(1.0, left_rect[3] - left_rect[1])
        right_height = max(1.0, right_rect[3] - right_rect[1])
        min_height = min(left_height, right_height)
        gap = right_rect[0] - left_rect[2]
        vertical_overlap = min(left_rect[3], right_rect[3]) - max(left_rect[1], right_rect[1])

        if vertical_overlap < min_height * 0.35:
            return False
        if gap > max(2.0, min_height * 0.5):
            return False

    return True


def combine_word_items(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge several token records into a single text-and-box item."""
    return {
        "text": "".join(part["text"] for part in parts),
        "box": union_boxes([part["box"] for part in parts]),
    }


def union_boxes(boxes: list[Any]) -> list[float]:
    """Return the bounding rectangle that covers all supplied boxes."""
    rects = [box_to_rect(box) for box in boxes]
    return [
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    ]


def box_to_rect(box: Any) -> tuple[float, float, float, float]:
    """Normalize a box or polygon into a plain `(x1, y1, x2, y2)` tuple."""
    if hasattr(box, "tolist"):
        box = box.tolist()

    if (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(value, Number) for value in box)
    ):
        x1, y1, x2, y2 = box
        return float(x1), float(y1), float(x2), float(y2)

    points = list(box)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def normalize_apostrophes(text: str) -> str:
    """Map several apostrophe-like glyphs onto a plain ASCII apostrophe."""
    return "".join("'" if char in APOSTROPHE_CHARS else char for char in text)


def is_apostrophe_token(text: str) -> bool:
    """Return whether a token consists only of apostrophe characters."""
    stripped = text.strip()
    return bool(stripped) and all(char in APOSTROPHE_CHARS for char in stripped)


def is_wordish(text: str) -> bool:
    """Return whether a token contains at least one letter or digit."""
    return bool(re.search(r"[A-Za-z0-9]", text or ""))
