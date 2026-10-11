"""Normalize OCR word-box metadata so downstream text consumers stay aligned."""

from __future__ import annotations

import re
from copy import deepcopy
from numbers import Number
from typing import Any


APOSTROPHE_CHARS = {"'", "`", "’", "‘", "ʼ", "´"}

COMMON_ABBREVIATIONS = {
    "no", "co", "mr", "mrs", "ms", "dr", "st", "hon", "rev", "esq", "messrs",
    "viz", "inst", "ult", "do", "vol", "vols", "cap", "sec", "art", "vic",
    "geo", "edw", "wm", "chas", "thos", "jas", "reg", "dept", "inc", "ltd",
    "gen", "col", "maj", "capt", "lieut", "sgt", "corp", "pvt", "prof",
}

ORPHAN_PUNCTUATION_RE = re.compile(r"^[.,;:!?—–\-]+$")


def normalize_ocr_word_boxes_in_payload(payload: dict[str, Any]) -> bool:
    """Merge split word-box metadata (apostrophes, abbreviations, orphan punctuation).

    PaddleX treats punctuation as standalone symbols when building `text_word`.
    Downstream consumers (ALTO XML, dictionary QA, and Searchable PDFs) require
    proper word tokens such as `queen's`, `&c.`, and `King,`.
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

        rec_scores = payload.get("rec_scores") or []

        for index in range(min(len(text_word), len(text_word_boxes))):
            line_text = rec_texts[index] if index < len(rec_texts) else None
            line_score = rec_scores[index] if index < len(rec_scores) else None
            merged_words, merged_boxes = merge_word_boxes_advanced(
                text_word[index],
                text_word_boxes[index],
                line_text=line_text,
                line_score=line_score,
            )
            if merged_words != text_word[index] or merged_boxes != text_word_boxes[index]:
                text_word[index] = merged_words
                text_word_boxes[index] = merged_boxes
                changed = True

        # Reconcile cross-line soft hyphens (e.g. 'occupa-' on line i + 'tion' on line i+1)
        if len(text_word) > 1 and len(text_word_boxes) > 1:
            hyphen_changed = reconcile_cross_line_hyphens(text_word, text_word_boxes)
            changed = changed or hyphen_changed

    return changed


def reconcile_cross_line_hyphens(text_word: list[Any], text_word_boxes: list[Any]) -> bool:
    """Reconcile split cross-line hyphenated words so the root word is evaluated as a full lexical term.
    
    When a line ends with a soft hyphen (e.g. 'occupa-') and the next line begins with a continuation
    (e.g. 'tion'), joining them produces 'occupation' which passes lexical dictionary validation.
    """
    changed = False
    for i in range(min(len(text_word) - 1, len(text_word_boxes) - 1)):
        curr_words = text_word[i]
        curr_boxes = text_word_boxes[i]
        next_words = text_word[i + 1]
        next_boxes = text_word_boxes[i + 1]

        if not curr_words or not next_words or not curr_boxes or not next_boxes:
            continue

        # Find last non-empty word in current line and first non-empty word in next line
        curr_idx = len(curr_words) - 1
        while curr_idx >= 0 and not str(curr_words[curr_idx]).strip():
            curr_idx -= 1

        next_idx = 0
        while next_idx < len(next_words) and not str(next_words[next_idx]).strip():
            next_idx += 1

        if curr_idx < 0 or next_idx >= len(next_words):
            continue

        last_w = str(curr_words[curr_idx]).strip()
        first_w = str(next_words[next_idx]).strip()

        # Check for trailing soft hyphen on last word
        if any(last_w.endswith(h) for h in ("-", "\u2010", "\u2011", "\u2013", "\u2014")) and len(last_w) >= 3:
            root_prefix = last_w.rstrip("- \u2010\u2011\u2013\u2014").strip()
            suffix = first_w.lstrip("- \u2010\u2011\u2013\u2014").strip(".,:;!?()[]{}\"'`")
            if root_prefix and suffix and root_prefix.isalpha() and suffix.isalpha():
                combined = f"{root_prefix}{suffix}"
                # Check if combined word is a valid lexical word
                try:
                    from services.historical_lexicon import is_valid_historical_word
                    valid = is_valid_historical_word(combined)
                except Exception:
                    valid = True

                if valid:
                    # Update root word on line end to contain full lexical word with hyphen notation for ALTO
                    curr_words[curr_idx] = combined
                    changed = True

    return changed


FRENCH_DIACRITICS = set("éèêëàâäîïôöùûüçÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ")


def merge_accent_split_tokens(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Recombine words that Paddle split across accented characters or isolated diacritics.
    
    Handles cases where Paddle segments words across accents:
    - Isolated diacritics: ['é', 'talage'] -> ['étalage']
    - Trailing accents: ['galvanis', 'é'] -> ['galvanisé']
    - Infix accents: ['Montr', 'é', 'al'] -> ['Montréal']
    - Split accented pieces: ['int', 'é', 'ressant'] -> ['intéressant'], ['commer', 'ç', 'ants'] -> ['commerçants']
    
    Guards strictly against merging distinct words separated by whitespace in line_text.
    """
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    try:
        from services.historical_lexicon import is_valid_historical_word
    except ImportError:
        return words, boxes

    items = [{"text": str(w), "box": deepcopy(b)} for w, b in zip(words, boxes) if str(w).strip()]
    if not items:
        return words, boxes

    raw_line = str(line_text or "").lower()

    merged: list[dict[str, Any]] = []
    i = 0
    n = len(items)

    while i < n:
        # Try 3-token merge (e.g. ['Montr', 'é', 'al'] or ['int', 'é', 'ressant'])
        if i + 2 < n:
            t1 = items[i]["text"].strip()
            t2 = items[i + 1]["text"].strip()
            t3 = items[i + 2]["text"].strip()
            joined3 = f"{t1}{t2}{t3}"
            clean3 = joined3.strip(".,:;!?()[]{}\"'`").lower()
            clean_t2 = t2.strip(".,:;!?()[]{}\"'`")

            # Must be an infix isolated diacritic
            if len(clean_t2) == 1 and clean_t2 in FRENCH_DIACRITICS:
                # Must be present in line_text without internal space OR be a verified dictionary word
                in_line = (clean3 in raw_line) if raw_line else False
                is_valid = is_valid_historical_word(clean3)
                if in_line or (is_valid and not is_valid_historical_word(t1) and not is_valid_historical_word(t3)):
                    merged.append(combine_word_items([items[i], items[i + 1], items[i + 2]]))
                    i += 3
                    continue

        # Try 2-token merge (e.g. ['é', 'talage'] or ['galvanis', 'é'])
        if i + 1 < n:
            t1 = items[i]["text"].strip()
            t2 = items[i + 1]["text"].strip()
            joined2 = f"{t1}{t2}"
            clean2 = joined2.strip(".,:;!?()[]{}\"'`").lower()
            clean_t1 = t1.strip(".,:;!?()[]{}\"'`")
            clean_t2 = t2.strip(".,:;!?()[]{}\"'`")

            is_isolated_diacritic = (
                (len(clean_t1) == 1 and clean_t1 in FRENCH_DIACRITICS)
                or (len(clean_t2) == 1 and clean_t2 in FRENCH_DIACRITICS)
            )

            if is_isolated_diacritic:
                in_line = (clean2 in raw_line) if raw_line else False
                is_valid = is_valid_historical_word(clean2)
                if in_line or is_valid:
                    merged.append(combine_word_items([items[i], items[i + 1]]))
                    i += 2
                    continue
            elif not is_valid_historical_word(clean_t1) and not is_valid_historical_word(clean_t2):
                # Neither is a valid word, but together they form a word in line_text
                in_line = (clean2 in raw_line) if raw_line else False
                if in_line and is_valid_historical_word(clean2):
                    merged.append(combine_word_items([items[i], items[i + 1]]))
                    i += 2
                    continue

        merged.append(items[i])
        i += 1

    return [m["text"] for m in merged], [m["box"] for m in merged]


def merge_word_boxes_advanced(
    words: Any,
    boxes: Any,
    line_text: Any = None,
    line_score: Any = None,
) -> tuple[Any, Any]:
    """Execute complete token consolidation pipeline."""
    # 1. Recombine accented French and line words that Paddle split across tokens
    words, boxes = merge_accent_split_tokens(words, boxes, line_text=line_text)
    words, boxes = consolidate_words_to_line_text(words, boxes, line_text=line_text)
    # 2. Merge apostrophes (queen's, o'clock, etc.)
    words, boxes = merge_apostrophe_word_boxes(words, boxes, line_text=line_text)
    # 3. Merge historical abbreviations & ligatures (&c., No., Co., etc.)
    words, boxes = merge_abbreviation_word_boxes(words, boxes, line_text=line_text)
    # 4. Attach orphan trailing punctuation to preceding word tokens
    words, boxes = merge_orphan_punctuation_word_boxes(words, boxes, line_text=line_text)
    # 5. Regularize broken microfilm words (e.g. resering -> reserving, paage -> page)
    words = regularize_word_tokens(words, line_score=line_score)
    return words, boxes


def regularize_word_tokens(words: Any, line_score: Any = None) -> Any:
    """Apply 1-edit historical dictionary regularizer to low-confidence or broken tokens."""
    if not isinstance(words, list) or not words:
        return words

    try:
        from services.historical_lexicon import regularize_broken_historical_word
    except ImportError:
        return words

    score = 0.75
    if isinstance(line_score, (int, float)):
        score = float(line_score)

    return [regularize_broken_historical_word(str(w), confidence=score) if w else w for w in words]


def consolidate_words_to_line_text(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Recombine accented French words and letterforms that Paddle split across individual tokens.
    
    PaddleX frequently segments accented French words like 'étalage' into ['é', 'talage'] or
    'intéressant' into ['int', 'é', 'ressant'] inside `text_word`. This routine checks the line text
    and greedily merges neighboring character/accent fragments that form exact words in `line_text`.
    """
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes
    if not line_text or not isinstance(line_text, str):
        return words, boxes

    expected_words = line_text.split()
    if not expected_words:
        return words, boxes

    raw_items = [{"text": str(w), "box": deepcopy(b)} for w, b in zip(words, boxes) if str(w).strip()]
    if not raw_items:
        return words, boxes

    merged: list[dict[str, Any]] = []
    item_idx = 0
    num_items = len(raw_items)

    for exp_w in expected_words:
        if item_idx >= num_items:
            break

        accum_parts = []
        accum_text = ""
        matched = False

        probe_idx = item_idx
        while probe_idx < num_items:
            curr = raw_items[probe_idx]
            accum_parts.append(curr)
            accum_text += curr["text"].strip()
            probe_idx += 1

            if accum_text == exp_w:
                merged.append(combine_word_items(accum_parts))
                item_idx = probe_idx
                matched = True
                break
            elif len(accum_text) > len(exp_w) and not exp_w.startswith(accum_text):
                break

        if not matched:
            merged.append(deepcopy(raw_items[item_idx]))
            item_idx += 1

    while item_idx < num_items:
        merged.append(deepcopy(raw_items[item_idx]))
        item_idx += 1

    return [m["text"] for m in merged], [m["box"] for m in merged]


def merge_apostrophe_word_boxes(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Merge apostrophe-split OCR tokens back into single word-box entries."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    items = [
        {"text": str(word), "box": deepcopy(box)}
        for word, box in zip(words, boxes)
    ]

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


def merge_abbreviation_word_boxes(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Merge split historical ligatures like '&c.' and standard abbreviations."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    items = [{"text": str(w), "box": deepcopy(b)} for w, b in zip(words, boxes)]
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(items):
        item = items[i]

        # 1. Historical &c. / &c pattern
        if item["text"] == "&" and i + 1 < len(items):
            next_item = items[i + 1]
            next_clean = next_item["text"].lower().rstrip(".,;:")
            if next_clean == "c":
                parts = [item, next_item]
                if i + 2 < len(items) and items[i + 2]["text"] in {".", ".,", ".;", ".:"}:
                    parts.append(items[i + 2])
                    merged.append(combine_word_items(parts))
                    i += 3
                    continue
                else:
                    merged.append(combine_word_items(parts))
                    i += 2
                    continue

        # 2. Standard historical abbreviations followed by dot
        clean_cand = item["text"].lower().rstrip(".")
        if clean_cand in COMMON_ABBREVIATIONS and i + 1 < len(items):
            next_item = items[i + 1]
            if next_item["text"] in {".", ".,", ".:", ".;"} and boxes_are_close_for_punctuation(item, next_item):
                merged.append(combine_word_items([item, next_item]))
                i += 2
                continue

        merged.append(deepcopy(item))
        i += 1

    return [item["text"] for item in merged], [item["box"] for item in merged]


def merge_orphan_punctuation_word_boxes(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Attach trailing orphan punctuation marks back into the preceding word box."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    items = [{"text": str(w), "box": deepcopy(b)} for w, b in zip(words, boxes)]
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(items):
        item = items[i]
        t = item["text"].strip()

        # Check if this token is pure punctuation following a word
        if ORPHAN_PUNCTUATION_RE.match(t) and merged and is_wordish(merged[-1]["text"]):
            prev = merged[-1]
            if boxes_are_close_for_punctuation(prev, item):
                merged[-1] = combine_word_items([prev, item])
                i += 1
                continue

        merged.append(deepcopy(item))
        i += 1

    return [item["text"] for item in merged], [item["box"] for item in merged]


def boxes_are_close_for_punctuation(left_item: dict[str, Any], right_item: dict[str, Any]) -> bool:
    """Check if punctuation token is positioned immediately after the word."""
    try:
        left_rect = box_to_rect(left_item["box"])
        right_rect = box_to_rect(right_item["box"])
    except Exception:
        return False

    left_h = max(1.0, left_rect[3] - left_rect[1])
    gap = right_rect[0] - left_rect[2]

    # Gap shouldn't exceed half the character height
    if gap > max(4.0, left_h * 0.65) or gap < -4.0:
        return False

    # Vertical baseline alignment check
    if right_rect[1] > left_rect[3] + 5 or right_rect[3] < left_rect[1] - 5:
        return False

    return True


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
