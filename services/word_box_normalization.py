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


def normalize_ocr_word_boxes_in_payload(payload: dict[str, Any], lang: Optional[str] = None) -> bool:
    """Merge split word-box metadata (apostrophes, abbreviations, orphan punctuation).

    PaddleX treats punctuation as standalone symbols when building `text_word`.
    Downstream consumers (ALTO XML, dictionary QA, and Searchable PDFs) require
    proper word tokens such as `queen's`, `&c.`, and `King,`.
    """
    changed = False

    if lang is None:
        lang = payload.get("detected_language") or payload.get("lang")
    if not lang and payload.get("rec_texts"):
        try:
            from services.language_detection import detect_document_language
            lang = detect_document_language(payload.get("rec_texts"))
        except Exception:
            pass
    if lang:
        payload["detected_language"] = lang

    if isinstance(payload.get("overall_ocr_res"), dict):
        changed = normalize_ocr_word_boxes_in_payload(payload["overall_ocr_res"], lang=lang) or changed

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
                lang=lang,
            )
            if merged_words != text_word[index] or merged_boxes != text_word_boxes[index]:
                text_word[index] = merged_words
                text_word_boxes[index] = merged_boxes
                changed = True

        # Reconcile cross-line soft hyphens (e.g. 'occupa-' on line i + 'tion' on line i+1)
        if len(text_word) > 1 and len(text_word_boxes) > 1:
            hyphen_changed = reconcile_cross_line_hyphens(text_word, text_word_boxes, lang=lang)
            changed = changed or hyphen_changed

        # Deduplicate cross-line word boxes (e.g. redundant duplicate words from tile overlap or layout overlap)
        if len(text_word) > 1 and len(text_word_boxes) > 1:
            dedupe_changed = deduplicate_cross_line_word_boxes(payload)
            changed = changed or dedupe_changed

        # Clamp vertical overlaps between adjacent lines in the same column
        if len(text_word) > 1 and len(text_word_boxes) > 1:
            clamp_changed = clamp_adjacent_line_vertical_overlap(payload)
            changed = changed or clamp_changed

    return changed


def deduplicate_cross_line_word_boxes(payload: dict[str, Any]) -> bool:
    """Eliminate redundant duplicate word boxes across lines caused by tile overlap.

    When adjacent tiles overlap, identical words can be detected in multiple tiles
    and end up in different line entries. If two word boxes across lines share
    matching normalized text and have high spatial overlap (IoU >= 0.40 or
    near-identical coordinates), keep the higher-scoring or more contextually intact
    word and drop the duplicate.
    """
    rec_texts = payload.get("rec_texts")
    text_word = payload.get("text_word")
    text_word_boxes = payload.get("text_word_boxes")

    if not isinstance(rec_texts, list) or not isinstance(text_word, list) or not isinstance(text_word_boxes, list):
        return False

    n_lines = min(len(text_word), len(text_word_boxes))
    if n_lines < 2:
        return False

    changed = False

    all_words = []
    for li in range(n_lines):
        w_list = text_word[li]
        b_list = text_word_boxes[li]
        if not isinstance(w_list, list) or not isinstance(b_list, list):
            continue
        for wi in range(min(len(w_list), len(b_list))):
            w = w_list[wi]
            b = b_list[wi]
            if w and b and len(b) >= 4:
                all_words.append((li, wi, str(w), b))

    to_remove = set()

    for i in range(len(all_words)):
        li1, wi1, w1, b1 = all_words[i]
        if (li1, wi1) in to_remove:
            continue
        norm1 = "".join(c.lower() for c in w1 if c.isalnum())
        if not norm1:
            continue

        for j in range(i + 1, len(all_words)):
            li2, wi2, w2, b2 = all_words[j]
            if li1 == li2 or (li2, wi2) in to_remove:
                continue
            norm2 = "".join(c.lower() for c in w2 if c.isalnum())
            if not norm2:
                continue

            if norm1 != norm2 and not (norm1 in norm2 or norm2 in norm1):
                continue

            x_ov = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            y_ov = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            inter = x_ov * y_ov
            a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            union = a1 + a2 - inter
            iou = inter / union if union > 0 else 0.0

            is_dup = False
            if norm1 == norm2:
                min_w = min(b1[2] - b1[0], b2[2] - b2[0])
                min_h = min(b1[3] - b1[1], b2[3] - b2[1])
                if iou >= 0.40 or (x_ov >= 0.50 * min_w and y_ov >= 0.40 * min_h):
                    is_dup = True
            elif iou >= 0.65:
                is_dup = True

            if is_dup:
                len_line1 = len(text_word[li1])
                len_line2 = len(text_word[li2])
                if len_line1 > len_line2:
                    drop_li, drop_wi = li2, wi2
                elif len_line2 > len_line1:
                    drop_li, drop_wi = li1, wi1
                else:
                    drop_li, drop_wi = li2, wi2
                to_remove.add((drop_li, drop_wi))
                if drop_li == li1 and drop_wi == wi1:
                    break

    if not to_remove:
        return False

    removals_by_line = {}
    for li, wi in to_remove:
        removals_by_line.setdefault(li, set()).add(wi)

    lines_to_delete = set()
    for li, wis in removals_by_line.items():
        w_list = text_word[li]
        b_list = text_word_boxes[li]
        new_w = [w for wi, w in enumerate(w_list) if wi not in wis]
        new_b = [b for wi, b in enumerate(b_list) if wi not in wis]
        text_word[li] = new_w
        text_word_boxes[li] = new_b
        if not new_w:
            lines_to_delete.add(li)
        else:
            if li < len(rec_texts) and isinstance(rec_texts[li], str):
                rec_texts[li] = " ".join(new_w)
        changed = True

    if lines_to_delete:
        indices_to_keep = [i for i in range(len(rec_texts)) if i not in lines_to_delete]
        for key in (
            "rec_texts",
            "rec_scores",
            "rec_polys",
            "rec_boxes",
            "dt_polys",
            "text_word",
            "text_word_boxes",
            "textline_orientation_angles",
        ):
            if isinstance(payload.get(key), list) and len(payload[key]) >= len(rec_texts):
                payload[key] = [payload[key][i] for i in indices_to_keep]

    return changed


def clamp_adjacent_line_vertical_overlap(payload: dict[str, Any]) -> bool:
    """Clamp vertical overlaps between vertically adjacent lines in the same column.

    When OCR detection polygons dilate vertically, adjacent lines within a column
    can bleed into one another (e.g. Line i bottom overlaps Line i+1 top by several pixels).
    This function finds adjacent line pairs in the same column, and if they overlap vertically,
    clamps both lines and their constituent word boxes at the inter-line gap midpoint (Y_mid).
    Non-overlapping lines and handwriting descenders with natural gaps remain untouched.
    """
    rec_texts = payload.get("rec_texts")
    rec_polys = payload.get("rec_polys")
    text_word_boxes = payload.get("text_word_boxes")

    if not isinstance(rec_texts, list) or not isinstance(text_word_boxes, list):
        return False

    n_lines = min(len(rec_texts), len(text_word_boxes))
    if rec_polys and isinstance(rec_polys, list):
        n_lines = min(n_lines, len(rec_polys))

    if n_lines < 2:
        return False

    # Extract bounding rectangle [x1, y1, x2, y2] for each line
    line_bounds: list[list[float]] = []
    for li in range(n_lines):
        rect = None
        if rec_polys and li < len(rec_polys) and rec_polys[li]:
            poly = rec_polys[li]
            if isinstance(poly[0], (int, float)) and len(poly) >= 4:
                rect = [float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3])]
            elif isinstance(poly[0], (list, tuple)) and len(poly[0]) >= 2:
                xs = [float(pt[0]) for pt in poly]
                ys = [float(pt[1]) for pt in poly]
                rect = [min(xs), min(ys), max(xs), max(ys)]
        if rect is None:
            # Fall back to word boxes
            w_boxes = text_word_boxes[li] if li < len(text_word_boxes) else []
            valid_wbs = [b for b in w_boxes if isinstance(b, (list, tuple)) and len(b) >= 4]
            if valid_wbs:
                xs = [float(b[0]) for b in valid_wbs] + [float(b[2]) for b in valid_wbs]
                ys = [float(b[1]) for b in valid_wbs] + [float(b[3]) for b in valid_wbs]
                rect = [min(xs), min(ys), max(xs), max(ys)]
        if rect is None:
            rect = [0.0, 0.0, 0.0, 0.0]
        line_bounds.append(rect)

    centers_y = [(b[1] + b[3]) / 2.0 for b in line_bounds]
    sorted_indices = sorted(range(n_lines), key=lambda idx: centers_y[idx])

    overall_changed = False

    # Perform up to 3 passes to handle multi-line cascade adjustments
    for _ in range(3):
        pass_changed = False
        for k in range(len(sorted_indices)):
            idx_b = sorted_indices[k]
            b = line_bounds[idx_b]
            cy_b = (b[1] + b[3]) / 2.0
            h_b = b[3] - b[1]

            # Find immediate predecessor above in same vertical column
            closest_a = None
            for p in range(k - 1, -1, -1):
                idx_a = sorted_indices[p]
                a = line_bounds[idx_a]
                cy_a = (a[1] + a[3]) / 2.0
                h_a = a[3] - a[1]
                min_h = min(h_a, h_b)

                if (cy_b - cy_a) < 0.35 * min_h:
                    continue  # side-by-side fragments on same row

                x_ov = min(a[2], b[2]) - max(a[0], b[0])
                min_w = min(a[2] - a[0], b[2] - b[0])

                if min_w > 0 and (x_ov / min_w >= 0.20 or x_ov >= 24.0):
                    closest_a = idx_a
                    break

            if closest_a is not None:
                a = line_bounds[closest_a]
                h_a = a[3] - a[1]
                min_h = min(h_a, h_b)

                # Check if bottom of A overlaps top of B
                if a[3] > b[1]:
                    ov = a[3] - b[1]
                    # Only clamp adjacent line dilation bleed (not full overlaps/nested ads)
                    if 0 < ov <= 0.70 * min_h:
                        y_mid = round((a[3] + b[1]) / 2.0, 1)

                        # Safeguard: ensure lines maintain minimal height (>= 4px)
                        clamp_bottom_a = max(a[1] + 4.0, min(a[3], y_mid))
                        clamp_top_b = min(b[3] - 4.0, max(b[1], y_mid))

                        if a[3] != clamp_bottom_a or b[1] != clamp_top_b:
                            # Apply to line bounds
                            a[3] = clamp_bottom_a
                            b[1] = clamp_top_b

                            # Apply to line A's word boxes (clamp bottom edge to clamp_bottom_a)
                            wbs_a = text_word_boxes[closest_a]
                            if isinstance(wbs_a, list):
                                for wb in wbs_a:
                                    if isinstance(wb, list) and len(wb) >= 4:
                                        if wb[3] > clamp_bottom_a:
                                            wb[3] = max(wb[1] + 1.0, clamp_bottom_a)

                            # Apply to line B's word boxes (clamp top edge to clamp_top_b)
                            wbs_b = text_word_boxes[idx_b]
                            if isinstance(wbs_b, list):
                                for wb in wbs_b:
                                    if isinstance(wb, list) and len(wb) >= 4:
                                        if wb[1] < clamp_top_b:
                                            wb[1] = min(wb[3] - 1.0, clamp_top_b)

                            # Apply to rec_polys if present
                            for target_idx, clamp_y, is_bottom in [
                                (closest_a, clamp_bottom_a, True),
                                (idx_b, clamp_top_b, False),
                            ]:
                                for poly_key in ("rec_polys", "dt_polys", "rec_boxes"):
                                    poly_list = payload.get(poly_key)
                                    if isinstance(poly_list, list) and target_idx < len(poly_list):
                                        poly_val = poly_list[target_idx]
                                        if isinstance(poly_val, list):
                                            if len(poly_val) == 4 and all(isinstance(v, (int, float)) for v in poly_val):
                                                if is_bottom and poly_val[3] > clamp_y:
                                                    poly_val[3] = clamp_y
                                                elif not is_bottom and poly_val[1] < clamp_y:
                                                    poly_val[1] = clamp_y
                                            elif len(poly_val) >= 3 and isinstance(poly_val[0], list):
                                                for pt in poly_val:
                                                    if isinstance(pt, list) and len(pt) >= 2:
                                                        if is_bottom and pt[1] > clamp_y:
                                                            pt[1] = clamp_y
                                                        elif not is_bottom and pt[1] < clamp_y:
                                                            pt[1] = clamp_y

                            pass_changed = True
                            overall_changed = True

        if not pass_changed:
            break

    return overall_changed


def reconcile_cross_line_hyphens(text_word: list[Any], text_word_boxes: list[Any], lang: Optional[str] = None) -> bool:
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
                    valid = is_valid_historical_word(combined, lang=lang)
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



def partition_box_horizontal(box: Any, subwords: list[str]) -> list[Any]:
    """Partition a bounding box or polygon horizontally into N sub-boxes proportional to subword lengths."""
    if not subwords or len(subwords) <= 1:
        return [box]

    if hasattr(box, "tolist"):
        box = box.tolist()

    weights = [max(1, len(str(w))) for w in subwords]
    total_w = sum(weights)
    cum_ratios = [0.0]
    curr = 0.0
    for w in weights:
        curr += w / total_w
        cum_ratios.append(curr)
    cum_ratios[-1] = 1.0

    is_polygon = (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and isinstance(box[0], (list, tuple))
        and len(box[0]) == 2
    )
    is_rect = (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(v, (int, float)) for v in box)
    )

    if is_polygon:
        p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
        out_boxes = []
        for i in range(len(subwords)):
            r_start = cum_ratios[i]
            r_end = cum_ratios[i + 1]
            tl = [p0[0] + (p1[0] - p0[0]) * r_start, p0[1] + (p1[1] - p0[1]) * r_start]
            tr = [p0[0] + (p1[0] - p0[0]) * r_end, p0[1] + (p1[1] - p0[1]) * r_end]
            br = [p3[0] + (p2[0] - p3[0]) * r_end, p3[1] + (p2[1] - p3[1]) * r_end]
            bl = [p3[0] + (p2[0] - p3[0]) * r_start, p3[1] + (p2[1] - p3[1]) * r_start]
            out_boxes.append([tl, tr, br, bl])
        return out_boxes
    elif is_rect:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        out_boxes = []
        for i in range(len(subwords)):
            r_start = cum_ratios[i]
            r_end = cum_ratios[i + 1]
            sx1 = x1 + (x2 - x1) * r_start
            sx2 = x1 + (x2 - x1) * r_end
            out_boxes.append([sx1, y1, sx2, y2])
        return out_boxes

    return [deepcopy(box) for _ in subwords]


def is_box_vertical(box: Any) -> bool:
    """Return True if bounding box or polygon has vertical text layout (height > width * 1.15)."""
    if hasattr(box, "tolist"):
        box = box.tolist()
    if isinstance(box, (list, tuple)) and len(box) == 4:
        if isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return (max(ys) - min(ys)) > (max(xs) - min(xs)) * 1.15
        elif all(isinstance(v, (int, float)) for v in box):
            return abs(box[3] - box[1]) > abs(box[2] - box[0]) * 1.15
    return False


def partition_box_vertical(box: Any, subwords: list[str]) -> list[Any]:
    """Partition a bounding box or polygon vertically (top to bottom) into N sub-boxes proportional to subword lengths."""
    if not subwords or len(subwords) <= 1:
        return [box]

    if hasattr(box, "tolist"):
        box = box.tolist()

    weights = [max(1, len(str(w))) for w in subwords]
    total_w = sum(weights)
    cum_ratios = [0.0]
    curr = 0.0
    for w in weights:
        curr += w / total_w
        cum_ratios.append(curr)
    cum_ratios[-1] = 1.0

    is_polygon = (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and isinstance(box[0], (list, tuple))
        and len(box[0]) == 2
    )
    is_rect = (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(v, (int, float)) for v in box)
    )

    if is_polygon:
        p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
        out_boxes = []
        for i in range(len(subwords)):
            r_start = cum_ratios[i]
            r_end = cum_ratios[i + 1]
            tl = [p0[0] + (p3[0] - p0[0]) * r_start, p0[1] + (p3[1] - p0[1]) * r_start]
            bl = [p0[0] + (p3[0] - p0[0]) * r_end, p0[1] + (p3[1] - p0[1]) * r_end]
            tr = [p1[0] + (p2[0] - p1[0]) * r_start, p1[1] + (p2[1] - p1[1]) * r_start]
            br = [p1[0] + (p2[0] - p1[0]) * r_end, p1[1] + (p2[1] - p1[1]) * r_end]
            out_boxes.append([tl, tr, br, bl])
        return out_boxes
    elif is_rect:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        out_boxes = []
        for i in range(len(subwords)):
            r_start = cum_ratios[i]
            r_end = cum_ratios[i + 1]
            sy1 = y1 + (y2 - y1) * r_start
            sy2 = y1 + (y2 - y1) * r_end
            out_boxes.append([x1, sy1, x2, sy2])
        return out_boxes

    return [deepcopy(box) for _ in subwords]


def partition_box_for_subwords(box: Any, subwords: list[str]) -> list[Any]:
    """Partition box vertically if vertical text, otherwise horizontally."""
    if is_box_vertical(box):
        return partition_box_vertical(box, subwords)
    return partition_box_horizontal(box, subwords)


CJK_CHAR_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u2e80-\u2fd5\u31c0-\u31ef\u3400-\u4dbf\uf900-\ufaff]"
)


def tokenize_cjk_text(text: str) -> list[str]:
    """Tokenize a CJK text string into individual characters and intact alphanumeric words."""
    if not text:
        return []
    tokens = []
    curr_latin = []
    for ch in text:
        if ch.isspace():
            if curr_latin:
                tokens.append("".join(curr_latin))
                curr_latin = []
            continue
        if CJK_CHAR_RE.search(ch):
            if curr_latin:
                tokens.append("".join(curr_latin))
                curr_latin = []
            tokens.append(ch)
        elif ch.isascii() and ch.isalnum():
            curr_latin.append(ch)
        else:
            if curr_latin:
                tokens.append("".join(curr_latin))
                curr_latin = []
            tokens.append(ch)
    if curr_latin:
        tokens.append("".join(curr_latin))
    return [t for t in tokens if t]


def chunk_cjk_tokens(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Split any CJK tokens into individual character words with appropriately partitioned bounding boxes."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    has_cjk = any(CJK_CHAR_RE.search(str(w)) for w in words if w)
    if not has_cjk:
        return words, boxes

    out_words = []
    out_boxes = []

    for w, b in zip(words, boxes):
        w_str = str(w).strip()
        if not w_str or not CJK_CHAR_RE.search(w_str):
            out_words.append(w)
            out_boxes.append(b)
            continue

        subtokens = tokenize_cjk_text(w_str)
        if len(subtokens) <= 1:
            out_words.append(w_str)
            out_boxes.append(b)
            continue

        part_boxes = partition_box_for_subwords(b, subtokens)
        for tok, pbox in zip(subtokens, part_boxes):
            out_words.append(tok)
            out_boxes.append(pbox)

    return out_words, out_boxes



def split_box_horizontally(box: Any, ratio: float) -> tuple[Any, Any]:
    """Split a bounding box or polygon horizontally by a fraction ratio (0.05 to 0.95)."""
    ratio = max(0.05, min(0.95, float(ratio)))
    if hasattr(box, "tolist"):
        box = box.tolist()

    if isinstance(box, (list, tuple)):
        # 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if len(box) == 4 and isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
            p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
            split_top = [p0[0] + (p1[0] - p0[0]) * ratio, p0[1] + (p1[1] - p0[1]) * ratio]
            split_bot = [p3[0] + (p2[0] - p3[0]) * ratio, p3[1] + (p2[1] - p3[1]) * ratio]
            box_left = [p0, split_top, split_bot, p3]
            box_right = [split_top, p1, p2, split_bot]
            return box_left, box_right
        # [x1, y1, x2, y2] rectangle
        elif len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            split_x = x1 + (x2 - x1) * ratio
            return [x1, y1, split_x, y2], [split_x, y1, x2, y2]

    return deepcopy(box), deepcopy(box)


PUNCT_SPLIT_RE = re.compile(r"^([A-Za-zÀ-ÿ0-9]+)([,;:!?&\.\-\"'\(\)]+)([A-Za-zÀ-ÿ0-9]+[.,;:!?\"'\)]*)$")
CAMEL_SPLIT_RE = re.compile(r"^([A-Za-zÀ-ÿ0-9\-]+[a-z])([A-Z][a-z0-9]+)$")
VALID_SHORT_WORDS = {'in', 'to', 'of', 'he', 'we', 'is', 'as', 'at', 'on', 'by', 'so', 'no', 'or', 'do', 'an', 'am', 'if', 'up', 'my', 'and', 'the', 'not'}


def split_punctuation_bound_tokens(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Split tokens fused together across punctuation without whitespace (e.g. 'Montréal,Avril', 'situé,la', 'of-currants', 'Fifty-TonDinosaur')."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    try:
        from services.historical_lexicon import is_valid_historical_word
    except ImportError:
        return words, boxes

    out_words = []
    out_boxes = []

    for w, b in zip(words, boxes):
        text = str(w).strip()

        # Check CamelCase split first (e.g. Fifty-TonDinosaur -> Fifty-Ton and Dinosaur)
        m_camel = CAMEL_SPLIT_RE.match(text)
        if m_camel:
            w1, w2 = m_camel.groups()
            c1 = w1.strip(".,:;!?()[]{}\"'`-").lower()
            c2 = w2.strip(".,:;!?()[]{}\"'`-").lower()
            if is_valid_historical_word(c1) and is_valid_historical_word(c2):
                ratio = len(w1) / max(1, len(text))
                b1, b2 = split_box_horizontally(b, ratio)
                out_words.extend([w1, w2])
                out_boxes.extend([b1, b2])
                continue

        m = PUNCT_SPLIT_RE.match(text)
        if m:
            w1, punct, w2 = m.groups()
            clean1 = w1.strip(".,:;!?()[]{}\"'`-").lower()
            clean2 = w2.strip(".,:;!?()[]{}\"'`-").lower()

            if is_valid_historical_word(clean1) or is_valid_historical_word(clean2):
                token1 = f"{w1}{punct}" if punct in ".,;:!?" else w1
                token2 = w2
                total_len = len(token1) + len(token2)
                ratio = len(token1) / max(1, total_len)
                b1, b2 = split_box_horizontally(b, ratio)
                out_words.extend([token1, token2])
                out_boxes.extend([b1, b2])
                continue

        out_words.append(w)
        out_boxes.append(b)

    return out_words, out_boxes


def decompound_fused_display_tokens(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Decompound display-ad words and fused conjunctions (e.g. 'ESTPERMANENT', 'Commodepour', 'andthe', 'Hadnot', 'goodthingthe')."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    try:
        from services.historical_lexicon import is_valid_historical_word
    except ImportError:
        return words, boxes

    out_words = []
    out_boxes = []

    for w, b in zip(words, boxes):
        text = str(w).strip()
        clean = text.strip(".,:;!?()[]{}\"'`")

        # Consider tokens of length >= 5 that are not already valid dictionary words
        if len(clean) >= 5 and clean.isascii() and clean.isalpha() and not is_valid_historical_word(clean.lower()):
            found_split = False
            for k in range(2, len(clean) - 1):
                p1 = clean[:k].lower()
                p2 = clean[k:].lower()
                if len(p1) == 2 and p1 not in VALID_SHORT_WORDS:
                    continue
                if len(p2) == 2 and p2 not in VALID_SHORT_WORDS:
                    continue
                if is_valid_historical_word(p1):
                    # 2-word split (e.g. andthe -> and + the, Hadnot -> Had + not, WASOIL -> WAS + OIL, Landsof -> Lands + of)
                    if is_valid_historical_word(p2):
                        token1 = text[:k]
                        token2 = text[k:]
                        ratio = len(token1) / max(1, len(text))
                        b1, b2 = split_box_horizontally(b, ratio)
                        out_words.extend([token1, token2])
                        out_boxes.extend([b1, b2])
                        found_split = True
                        break
                    # 3-word split (e.g. goodthingthe -> good + thing + the)
                    for j in range(2, len(p2) - 1):
                        p2a = p2[:j]
                        p2b = p2[j:]
                        if len(p2a) == 2 and p2a not in VALID_SHORT_WORDS:
                            continue
                        if len(p2b) == 2 and p2b not in VALID_SHORT_WORDS:
                            continue
                        if is_valid_historical_word(p2a) and is_valid_historical_word(p2b):
                            token1 = text[:k]
                            token2 = text[k:k+j]
                            token3 = text[k+j:]
                            part_boxes = partition_box_horizontal(b, [token1, token2, token3])
                            out_words.extend([token1, token2, token3])
                            out_boxes.extend(part_boxes)
                            found_split = True
                            break
                    if found_split:
                        break
            if found_split:
                continue

        out_words.append(w)
        out_boxes.append(b)

    return out_words, out_boxes


def split_whitespace_word_tokens(
    words: Any,
    boxes: Any,
    line_text: Any = None,
) -> tuple[Any, Any]:
    """Auto-split tokens with spaces into separate word items with proportionally partitioned bounding boxes."""
    if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes) or not words:
        return words, boxes

    out_words = []
    out_boxes = []

    for w, b in zip(words, boxes):
        text = str(w).strip()
        subwords = text.split()
        if len(subwords) > 1:
            part_boxes = partition_box_horizontal(b, subwords)
            out_words.extend(subwords)
            out_boxes.extend(part_boxes)
        else:
            out_words.append(w)
            out_boxes.append(b)

    return out_words, out_boxes


def merge_word_boxes_advanced(
    words: Any,
    boxes: Any,
    line_text: Any = None,
    line_score: Any = None,
    lang: Optional[str] = None,
) -> tuple[Any, Any]:
    """Execute complete token consolidation pipeline."""
    # 0. Fast-path for CJK: chunk CJK tokens into separate character words with partitioned geometry immediately
    has_cjk = any(CJK_CHAR_RE.search(str(w)) for w in words if w) or (line_text and CJK_CHAR_RE.search(str(line_text)))
    if has_cjk:
        return chunk_cjk_tokens(words, boxes, line_text=line_text)

    # 1. Recombine accented French and line words that Paddle split across tokens
    words, boxes = merge_accent_split_tokens(words, boxes, line_text=line_text)
    words, boxes = consolidate_words_to_line_text(words, boxes, line_text=line_text)
    # 2. Split punctuation-bound fused words (e.g. Montréal,Avril -> Montréal, + Avril, situé,la -> situé, + la)
    words, boxes = split_punctuation_bound_tokens(words, boxes, line_text=line_text)
    # 3. Decompound display-ad words fused across condensed typography (e.g. ESTPERMANENT -> EST + PERMANENT)
    words, boxes = decompound_fused_display_tokens(words, boxes, line_text=line_text)
    # 4. Auto-split any recognized tokens containing whitespace into separate words
    words, boxes = split_whitespace_word_tokens(words, boxes, line_text=line_text)
    # 5. Merge apostrophes (queen's, o'clock, etc.)
    words, boxes = merge_apostrophe_word_boxes(words, boxes, line_text=line_text)
    # 6. Merge historical abbreviations & ligatures (&c., No., Co., etc.)
    words, boxes = merge_abbreviation_word_boxes(words, boxes, line_text=line_text)
    # 7. Attach orphan trailing punctuation to preceding word tokens
    words, boxes = merge_orphan_punctuation_word_boxes(words, boxes, line_text=line_text)
    # 8. Regularize broken microfilm words and French dropouts (e.g. qualit -> qualité, resering -> reserving)
    words = regularize_word_tokens(words, line_score=line_score, lang=lang)
    # 9. Final safeguard: ensure no compound whitespace tokens remain
    words, boxes = split_whitespace_word_tokens(words, boxes, line_text=line_text)
    return words, boxes


def regularize_word_tokens(words: Any, line_score: Any = None, lang: Optional[str] = None) -> Any:
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

    return [regularize_broken_historical_word(str(w), confidence=score, lang=lang) if w else w for w in words]


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
    if any(CJK_CHAR_RE.search(str(w)) for w in words if w) or CJK_CHAR_RE.search(line_text):
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
