from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import csv
from difflib import SequenceMatcher
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from langdetect import DetectorFactory, LangDetectException, detect_langs
from PIL import Image
from spellchecker import SpellChecker

from services.ocr_defaults import (
    DEFAULT_OLLAMA_OPENAI_API_URL,
    DEFAULT_OLLAMA_OPENAI_MODEL,
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BACKEND,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_GROUP_LINES,
    DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_POST_OCR_VL_MAX_WORDS,
    DEFAULT_POST_OCR_VL_PADDING_PX,
    DEFAULT_POST_OCR_VL_SAVE_TILES,
)

logger = logging.getLogger(__name__)


def _get_openai_compat_url(opt: dict[str, Any]) -> str:
    return str(
        opt.get("post_ocr_vl_ollama_openai_url")
        or opt.get("post_ocr_vl_olmocr_url")
        or DEFAULT_OLLAMA_OPENAI_API_URL
    )


def _get_openai_compat_model(opt: dict[str, Any]) -> str:
    return str(
        opt.get("post_ocr_vl_ollama_openai_model")
        or opt.get("post_ocr_vl_olmocr_model")
        or DEFAULT_OLLAMA_OPENAI_MODEL
    )


def _get_openai_compat_api_key(opt: dict[str, Any]) -> str | None:
    return (
        opt.get("post_ocr_vl_ollama_openai_api_key")
        or opt.get("post_ocr_vl_olmocr_api_key")
        or os.getenv("OLLAMA_OPENAI_API_KEY")
        or os.getenv("OPENAI_COMPAT_API_KEY")
        or os.getenv("OLMOCR_API_KEY")
    )


def _native_ollama_chat_url(server_url: str) -> str | None:
    parsed = urlparse(server_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    if parsed.port != 11434:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/api/chat"


POST_OCR_VL_METHODS = {"PP-OCRv5"}
POST_OCR_VL_MAX_LINE_TARGETS = 8
WORD_RE = re.compile(r"[A-Za-z0-9]")
ODD_CHAR_RE = re.compile(r"[^A-Za-z0-9'\u2019`.,:;!?()\\[\\]{}\\-/& ]")
WORD_STRIP_CHARS = ".,:;!?()[]{}\"'`\u2018\u2019-–—"
WORDISH_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
LINE_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")
CONTROL_TAG_RE = re.compile(r"<\|[^>]*\|>")
HTML_BREAK_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[^>]+>")
OCR_WORDISH_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
LANGUAGE_KEYS = ("language", "lang", "ocr_language", "document_language")
LANGUAGE_CONTAINERS = (
    "model_settings",
    "modelSettings",
    "source_preprocessing",
    "text_det_params",
)
LANGUAGE_ALIASES = {
    "eng": "en",
    "english": "en",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "spa": "es",
    "spanish": "es",
    "por": "pt",
    "portuguese": "pt",
    "ita": "it",
    "italian": "it",
    "nld": "nl",
    "dut": "nl",
    "dutch": "nl",
    "rus": "ru",
    "russian": "ru",
}
SPELLCHECKER_LANGUAGE_MAP = {
    "ar": "ar",
    "de": "de",
    "en": "en",
    "es": "es",
    "eu": "eu",
    "fa": "fa",
    "fr": "fr",
    "it": "it",
    "lv": "lv",
    "nl": "nl",
    "pt": "pt",
    "ru": "ru",
}
_SPELLCHECKER_CACHE: dict[str, "SpellCheckerAdapter"] = {}
DetectorFactory.seed = 0


@dataclass
class WordCandidate:
    line_index: int
    word_index: int
    text: str
    box: list[float]
    line_score: float | None
    weirdness: float
    reasons: list[str] = field(default_factory=list)
    source: str = "word_box"
    line_text_start: int | None = None
    line_text_end: int | None = None


@dataclass(frozen=True)
class LineTextToken:
    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SpellcheckIssue:
    suggestion: str | None = None
    distance: int | None = None


@dataclass
class SpellCheckerAdapter:
    language: str
    checker: SpellChecker
    backend: str = "pyspellchecker"
    cache: dict[str, SpellcheckIssue | None] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return True

    def issue_for(self, token: str) -> SpellcheckIssue | None:
        if not self.available:
            return None

        value = normalize_word_token(token)
        if not value:
            return None
        if value in self.cache:
            return self.cache[value]

        try:
            if not self.checker.unknown([value]):
                self.cache[value] = None
                return None
        except Exception:
            self.cache[value] = None
            return None

        issue = SpellcheckIssue()
        self.cache[value] = issue
        return issue


@dataclass
class WordProfile:
    ngrams: Counter[str] = field(default_factory=Counter)
    token_counts: Counter[str] = field(default_factory=Counter)
    total_ngrams: int = 0
    vocab_size: int = 0
    anomaly_threshold: float | None = None
    strong_anomaly_threshold: float | None = None
    known_words: set[str] = field(default_factory=set)
    common_words_by_length: dict[int, list[str]] = field(default_factory=dict)
    training_token_count: int = 0
    language: str | None = None
    language_source: str | None = None
    spellchecker: SpellCheckerAdapter | None = None

    def score(self, text: str) -> float | None:
        token = normalize_word_token(text)
        if not is_alpha_word(token) or not self.ngrams:
            return None

        padded = f"^{token}$"
        values: list[float] = []
        denominator = self.total_ngrams + (0.5 * max(1, self.vocab_size))
        for ngram_size in (2, 3, 4):
            for index in range(len(padded) - ngram_size + 1):
                ngram = padded[index : index + ngram_size]
                values.append(math.log((self.ngrams[ngram] + 0.5) / denominator))
        return sum(values) / max(1, len(values))

    def nearest_common_word(self, text: str) -> str | None:
        match = self.nearest_common_word_match(text)
        return match[0] if match else None

    def nearest_common_word_match(self, text: str) -> tuple[str, int, int] | None:
        token = normalize_word_token(text)
        if not is_alpha_word(token) or len(token) < 4:
            return None
        if self.token_counts.get(token, 0) > 1:
            return None

        max_distance = 1 if len(token) <= 5 else 2
        best: tuple[int, int, str] | None = None
        best_word: str | None = None
        best_count = 0
        for length in range(len(token) - max_distance, len(token) + max_distance + 1):
            for candidate in self.common_words_by_length.get(length, []):
                if candidate == token:
                    continue
                count = self.token_counts[candidate]
                distance = levenshtein_limited(token, candidate, max_distance)
                if distance > max_distance:
                    continue
                ranking = (distance, -count, candidate)
                if best is None or ranking < best:
                    best = ranking
                    best_word = candidate
                    best_count = count
        if best_word is None or best is None:
            return None
        return best_word, best[0], best_count


def supports_post_ocr_vl(method: str | None, select: str | None = None) -> bool:
    effective_method = select if select in POST_OCR_VL_METHODS else method
    return effective_method in POST_OCR_VL_METHODS


def apply_post_ocr_vl_check(
    json_path: Path,
    out_dir: Path,
    opt: dict,
) -> dict[str, Any]:
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    progress_path = post_ocr_vl_progress_path(json_path)
    max_words_enabled = bool(opt.get("post_ocr_vl_max_words_enabled", False))
    max_words = (
        parse_int(
            opt.get("post_ocr_vl_max_words"),
            DEFAULT_POST_OCR_VL_MAX_WORDS,
            minimum=0,
        )
        if max_words_enabled
        else 0
    )

    metadata: dict[str, Any] = {
        "enabled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": normalize_backend(opt.get("post_ocr_vl_backend")),
        "candidate_mode": opt.get(
            "post_ocr_vl_candidate_mode",
            DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
        ),
        "max_words_enabled": max_words_enabled,
        "max_words": max_words,
        "padding_px": parse_int(
            opt.get("post_ocr_vl_padding_px"),
            DEFAULT_POST_OCR_VL_PADDING_PX,
            minimum=0,
        ),
        "low_confidence_threshold": parse_float(
            opt.get("post_ocr_vl_low_confidence_threshold"),
            DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
        ),
        "apply_corrections": bool(
            opt.get(
                "post_ocr_vl_apply_corrections",
                DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
            )
        ),
        "save_tiles": bool(
            opt.get("post_ocr_vl_save_tiles", DEFAULT_POST_OCR_VL_SAVE_TILES)
        ),
        "group_line_candidates": bool(
            opt.get("post_ocr_vl_group_lines", DEFAULT_POST_OCR_VL_GROUP_LINES)
        ),
        "progress_output": str(progress_path),
        "corrections": [],
    }

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        metadata["error"] = f"Unable to read OCR JSON: {exc}"
        write_post_ocr_vl_progress(
            progress_path,
            metadata,
            status="error",
            error=metadata["error"],
        )
        return metadata

    ocr_payload = get_ocr_payload(payload)
    word_profile = build_word_profile(
        ocr_payload,
        metadata["low_confidence_threshold"],
    )
    metadata["detected_language"] = word_profile.language
    metadata["language_detection_source"] = word_profile.language_source
    metadata["spellchecker_backend"] = (
        word_profile.spellchecker.backend if word_profile.spellchecker else None
    )
    metadata["spellchecker_language"] = (
        word_profile.spellchecker.language if word_profile.spellchecker else None
    )
    metadata["word_profile_training_tokens"] = word_profile.training_token_count
    backend_order = backends_for_request(metadata["backend"])

    include_line_text_fallback = any(
        backend in {"olmocr", "paddlevl"} for backend in backend_order
    )

    suspicious_word_count = len(
        collect_word_candidates(
            ocr_payload,
            mode="suspicious",
            max_words=0,
            low_confidence_threshold=metadata["low_confidence_threshold"],
            word_profile=word_profile,
            include_line_text_fallback=include_line_text_fallback,
        )
    )
    candidates = collect_word_candidates(
        ocr_payload,
        mode=metadata["candidate_mode"],
        max_words=0,
        low_confidence_threshold=metadata["low_confidence_threshold"],
        word_profile=word_profile,
        include_line_text_fallback=include_line_text_fallback,
    )
    metadata["qualified_count"] = len(candidates)
    if max_words > 0:
        candidates = candidates[:max_words]
    metadata["candidate_count"] = len(candidates)
    metadata["suspicious_word_count"] = suspicious_word_count
    metadata["line_text_fallback_candidate_count"] = sum(
        1 for candidate in candidates if candidate.source == "line_text"
    )
    openai_model_name = _get_openai_compat_model(opt) if "olmocr" in backend_order else ""
    metadata["line_grouping_active"] = (
        metadata["group_line_candidates"]
        and backend_order == ["olmocr"]
        and not _is_chandra_model(openai_model_name)
        and not _is_deepseek_ocr_model(openai_model_name)
    )
    candidate_batches = group_candidate_batches(
        candidates,
        group_lines=metadata["line_grouping_active"],
    )
    metadata["vl_request_count"] = len(candidate_batches)
    metadata["line_group_count"] = sum(
        1 for batch in candidate_batches if should_process_batch_as_line(batch)
    )
    metadata["line_grouped_word_count"] = sum(
        len(batch) for batch in candidate_batches if should_process_batch_as_line(batch)
    )

    write_post_ocr_vl_progress(
        progress_path,
        metadata,
        status="running" if candidates else "completed",
        source_json=str(json_path),
        suspicious_word_count=suspicious_word_count,
        qualified_word_count=metadata["qualified_count"],
        total_words_to_check=metadata["candidate_count"],
        current_position=0,
        current_word=None,
        last_completed_position=0,
    )

    if not candidates:
        payload["post_ocr_vl_check"] = metadata
        write_json(json_path, payload)
        return metadata

    image_path = resolve_image_path(payload, json_path)
    if image_path is None:
        metadata["error"] = "Could not resolve the source image path for word crops."
        write_post_ocr_vl_progress(
            progress_path,
            metadata,
            status="error",
            error=metadata["error"],
        )
        payload["post_ocr_vl_check"] = metadata
        write_json(json_path, payload)
        return metadata

    metadata["image_path"] = str(image_path)
    tile_dir = out_dir / "_post_ocr_vl_tiles" / json_path.stem
    if metadata["save_tiles"]:
        tile_dir.mkdir(parents=True, exist_ok=True)
        metadata["tile_dir"] = str(tile_dir)

    applied_count = 0
    saved_tile_count = 0
    processed_count = 0

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        for request_index, batch in enumerate(candidate_batches, start=1):
            if should_process_batch_as_line(batch):
                processed_count, applied_count, saved_tile_count = process_line_batch(
                    image=image,
                    ocr_payload=ocr_payload,
                    batch=batch,
                    opt=opt,
                    metadata=metadata,
                    progress_path=progress_path,
                    source_json=str(json_path),
                    image_path=str(image_path),
                    tile_dir=tile_dir,
                    suspicious_word_count=suspicious_word_count,
                    request_index=request_index,
                    processed_count=processed_count,
                    applied_count=applied_count,
                    saved_tile_count=saved_tile_count,
                )
                continue

            candidate = batch[0]
            position = processed_count + 1
            record = candidate_record(candidate)
            write_post_ocr_vl_progress(
                progress_path,
                metadata,
                status="running",
                source_json=str(json_path),
                image_path=str(image_path),
                suspicious_word_count=suspicious_word_count,
                qualified_word_count=metadata["qualified_count"],
                total_words_to_check=metadata["candidate_count"],
                current_position=position,
                current_word=candidate.text,
                current_line_index=candidate.line_index,
                current_word_index=candidate.word_index,
                current_line_score=candidate.line_score,
                current_weirdness=candidate.weirdness,
                last_completed_position=position - 1,
                current_vl_request=request_index,
                total_vl_requests=metadata["vl_request_count"],
                applied_count=applied_count,
                saved_tile_count=saved_tile_count,
            )

            try:
                tile, crop_box = crop_word_tile(
                    image,
                    candidate.box,
                    padding_px=metadata["padding_px"],
                )
            except Exception as exc:
                record["error"] = f"Unable to crop word tile: {exc}"
                metadata["corrections"].append(record)
                write_post_ocr_vl_progress(
                    progress_path,
                    metadata,
                    status="running",
                    source_json=str(json_path),
                    image_path=str(image_path),
                    suspicious_word_count=suspicious_word_count,
                    qualified_word_count=metadata["qualified_count"],
                    total_words_to_check=metadata["candidate_count"],
                    current_position=position,
                    current_word=candidate.text,
                    current_line_index=candidate.line_index,
                    current_word_index=candidate.word_index,
                    last_completed_position=position,
                    current_vl_request=request_index,
                    total_vl_requests=metadata["vl_request_count"],
                    applied_count=applied_count,
                    saved_tile_count=saved_tile_count,
                    last_error=record["error"],
                )
                processed_count += 1
                continue

            record["crop_box"] = crop_box
            record["crop_type"] = "word"
            tile_path = None

            if metadata["save_tiles"]:
                tile_path = tile_dir / (
                    f"{candidate.line_index:04d}_{candidate.word_index:03d}_"
                    f"{safe_filename(candidate.text)}.png"
                )
                tile.save(tile_path)
                record["tile_path"] = str(tile_path)
                saved_tile_count += 1

            results = []
            for backend in backend_order:
                try:
                    text = run_backend_check(
                        backend,
                        tile,
                        candidate.text,
                        opt,
                    )
                    results.append({"backend": backend, "text": text})
                except Exception as exc:
                    results.append({"backend": backend, "error": str(exc)})

            record["backend_results"] = results
            selected_text, reason = choose_correction(candidate.text, results)
            record["selected_text"] = selected_text
            record["reason"] = reason

            if (
                metadata["apply_corrections"]
                and selected_text
                and reason == "accepted"
            ):
                apply_candidate_correction(ocr_payload, candidate, selected_text)
                record["applied"] = True
                applied_count += 1
            else:
                record["applied"] = False

            metadata["corrections"].append(record)
            processed_count += 1
            write_post_ocr_vl_progress(
                progress_path,
                metadata,
                status="running",
                source_json=str(json_path),
                image_path=str(image_path),
                suspicious_word_count=suspicious_word_count,
                qualified_word_count=metadata["qualified_count"],
                total_words_to_check=metadata["candidate_count"],
                current_position=position,
                current_word=candidate.text,
                current_line_index=candidate.line_index,
                current_word_index=candidate.word_index,
                selected_text=record["selected_text"],
                correction_reason=record["reason"],
                last_completed_position=position,
                current_vl_request=request_index,
                total_vl_requests=metadata["vl_request_count"],
                applied_count=applied_count,
                saved_tile_count=saved_tile_count,
            )

    metadata["applied_count"] = applied_count
    metadata["saved_tile_count"] = saved_tile_count
    write_post_ocr_vl_progress(
        progress_path,
        metadata,
        status="completed",
        source_json=str(json_path),
        image_path=str(image_path),
        suspicious_word_count=suspicious_word_count,
        qualified_word_count=metadata["qualified_count"],
        total_words_to_check=metadata["candidate_count"],
        current_position=metadata["candidate_count"],
        current_word=None,
        last_completed_position=metadata["candidate_count"],
        applied_count=applied_count,
        saved_tile_count=saved_tile_count,
    )
    # Save a CSV summarizing original text, VL output and whether it was applied.
    try:
        csv_path = Path(out_dir) / f"{json_path.stem}.post_ocr_vl.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "line_index",
                "word_index",
                "original_text",
                "vl_text",
                "applied",
                "reason",
            ])
            for rec in metadata.get("corrections", []):
                writer.writerow([
                    rec.get("line_index"),
                    rec.get("word_index"),
                    rec.get("original_text") or "",
                    rec.get("selected_text") or "",
                    bool(rec.get("applied")),
                    rec.get("reason") or "",
                ])
        metadata["replacements_csv"] = str(csv_path)
    except Exception:
        # Don't let CSV errors break the main processing
        pass
    payload["post_ocr_vl_check"] = metadata
    write_json(json_path, payload)
    return metadata


def get_ocr_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("overall_ocr_res"), dict):
        return payload["overall_ocr_res"]
    return payload


def collect_word_candidates(
    ocr_payload: dict[str, Any],
    mode: str,
    max_words: int,
    low_confidence_threshold: float,
    word_profile: WordProfile | None = None,
    include_line_text_fallback: bool = True,
) -> list[WordCandidate]:
    text_word = ocr_payload.get("text_word") or []
    text_word_boxes = ocr_payload.get("text_word_boxes") or []
    rec_scores = ocr_payload.get("rec_scores") or []
    candidates: list[WordCandidate] = []
    mode = mode if mode in {"suspicious", "low_confidence", "all"} else "suspicious"
    if word_profile is None:
        word_profile = build_word_profile(ocr_payload, low_confidence_threshold)

    for line_index, words in enumerate(text_word):
        if not isinstance(words, list):
            continue

        boxes = text_word_boxes[line_index] if line_index < len(text_word_boxes) else []
        if not isinstance(boxes, list):
            continue

        line_score = parse_optional_float(
            rec_scores[line_index] if line_index < len(rec_scores) else None
        )

        for word_index, word in enumerate(words):
            text = str(word or "")
            if mode == "suspicious" and not is_word_token(text):
                continue
            if mode != "suspicious" and not is_ocr_token(text):
                continue
            if word_index >= len(boxes):
                continue

            box = geometry_to_bbox(boxes[word_index])
            if box is None or box[2] <= box[0] or box[3] <= box[1]:
                continue

            weirdness, reasons = score_word_weirdness(
                text,
                line_score,
                low_confidence_threshold,
                word_profile,
            )
            low_confidence = (
                line_score is not None and line_score < low_confidence_threshold
            )

            if mode == "suspicious" and weirdness < 0.5:
                continue
            if mode == "low_confidence" and not low_confidence:
                continue

            candidates.append(
                WordCandidate(
                    line_index=line_index,
                    word_index=word_index,
                    text=text,
                    box=box,
                    line_score=line_score,
                    weirdness=weirdness,
                    reasons=reasons,
                )
            )

    if include_line_text_fallback:
        candidates.extend(
            collect_line_text_fallback_candidates(
                ocr_payload,
                mode,
                low_confidence_threshold,
                word_profile,
            )
        )

    if mode == "all":
        return candidates[:max_words] if max_words > 0 else candidates

    candidates.sort(
        key=lambda item: (
            -item.weirdness,
            item.line_score if item.line_score is not None else 1.0,
            item.line_index,
            item.word_index,
        )
    )
    return candidates[:max_words] if max_words > 0 else candidates


def collect_line_text_fallback_candidates(
    ocr_payload: dict[str, Any],
    mode: str,
    low_confidence_threshold: float,
    word_profile: WordProfile,
) -> list[WordCandidate]:
    rec_texts = ocr_payload.get("rec_texts") or []
    rec_scores = ocr_payload.get("rec_scores") or []
    if not isinstance(rec_texts, list):
        return []

    candidates: list[WordCandidate] = []
    for line_index, line_text in enumerate(rec_texts):
        if line_has_usable_word_boxes(ocr_payload, line_index):
            continue

        text = str(line_text or "")
        if not text.strip():
            continue

        line_score = parse_optional_float(
            rec_scores[line_index] if line_index < len(rec_scores) else None
        )
        low_confidence = line_score is not None and line_score < low_confidence_threshold
        if mode == "low_confidence" and not low_confidence:
            continue

        line_box = line_bbox_for_candidates(ocr_payload, line_index, [])
        if not valid_bbox(line_box):
            continue

        for token in line_text_tokens(text):
            if mode == "suspicious" and not is_word_token(token.text):
                continue
            if mode != "suspicious" and not is_ocr_token(token.text):
                continue

            weirdness, reasons = score_word_weirdness(
                token.text,
                line_score,
                low_confidence_threshold,
                word_profile,
            )
            if mode in {"suspicious", "low_confidence"} and weirdness < 0.5:
                continue

            candidates.append(
                WordCandidate(
                    line_index=line_index,
                    word_index=token.index,
                    text=token.text,
                    box=line_box or [0.0, 0.0, 0.0, 0.0],
                    line_score=line_score,
                    weirdness=weirdness,
                    reasons=[*reasons, "line text fallback"],
                    source="line_text",
                    line_text_start=token.start,
                    line_text_end=token.end,
                )
            )

    return candidates


def line_has_usable_word_boxes(ocr_payload: dict[str, Any], line_index: int) -> bool:
    text_word = ocr_payload.get("text_word") or []
    text_word_boxes = ocr_payload.get("text_word_boxes") or []
    if not isinstance(text_word, list) or line_index >= len(text_word):
        return False
    if not isinstance(text_word_boxes, list) or line_index >= len(text_word_boxes):
        return False

    words = text_word[line_index]
    boxes = text_word_boxes[line_index]
    if not isinstance(words, list) or not isinstance(boxes, list):
        return False

    for word_index, word in enumerate(words):
        if word_index >= len(boxes):
            continue
        if not is_word_token(str(word or "")):
            continue
        if valid_bbox(geometry_to_bbox(boxes[word_index])):
            return True
    return False


def line_text_tokens(text: str) -> list[LineTextToken]:
    return [
        LineTextToken(
            index=index,
            text=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for index, match in enumerate(LINE_TEXT_TOKEN_RE.finditer(str(text or "")))
    ]


def group_candidate_batches(
    candidates: list[WordCandidate],
    group_lines: bool,
) -> list[list[WordCandidate]]:
    if not group_lines:
        return [[candidate] for candidate in candidates]

    indexed_by_line: dict[int, list[tuple[int, WordCandidate]]] = {}
    for index, candidate in enumerate(candidates):
        indexed_by_line.setdefault(candidate.line_index, []).append((index, candidate))

    batches: list[tuple[int, list[WordCandidate]]] = []
    grouped_lines: set[int] = set()
    for line_index, entries in indexed_by_line.items():
        has_line_text_candidate = any(
            candidate.source == "line_text" for _index, candidate in entries
        )
        if len(entries) < 2 and not has_line_text_candidate:
            continue

        grouped_lines.add(line_index)
        ordered = [candidate for _index, candidate in entries]
        ordered.sort(key=lambda candidate: candidate.word_index)
        for start in range(0, len(ordered), POST_OCR_VL_MAX_LINE_TARGETS):
            chunk = ordered[start : start + POST_OCR_VL_MAX_LINE_TARGETS]
            chunk_ids = {id(candidate) for candidate in chunk}
            first_index = min(
                index
                for index, candidate in entries
                if id(candidate) in chunk_ids
            )
            batches.append((first_index, chunk))

    for index, candidate in enumerate(candidates):
        if candidate.line_index in grouped_lines:
            continue
        batches.append((index, [candidate]))

    batches.sort(key=lambda item: item[0])
    return [batch for _index, batch in batches]


def should_process_batch_as_line(batch: list[WordCandidate]) -> bool:
    return len(batch) > 1 or any(candidate.source == "line_text" for candidate in batch)


def process_line_batch(
    image: Image.Image,
    ocr_payload: dict[str, Any],
    batch: list[WordCandidate],
    opt: dict,
    metadata: dict[str, Any],
    progress_path: Path,
    source_json: str,
    image_path: str,
    tile_dir: Path,
    suspicious_word_count: int,
    request_index: int,
    processed_count: int,
    applied_count: int,
    saved_tile_count: int,
) -> tuple[int, int, int]:
    line_index = batch[0].line_index
    current_word = ", ".join(candidate.text for candidate in batch)
    current_word = current_word[:160]
    start_position = processed_count + 1

    write_post_ocr_vl_progress(
        progress_path,
        metadata,
        status="running",
        source_json=source_json,
        image_path=image_path,
        suspicious_word_count=suspicious_word_count,
        qualified_word_count=metadata["qualified_count"],
        total_words_to_check=metadata["candidate_count"],
        current_position=start_position,
        current_word=current_word,
        current_line_index=line_index,
        current_word_index=[candidate.word_index for candidate in batch],
        current_line_score=batch[0].line_score,
        current_weirdness=max(candidate.weirdness for candidate in batch),
        current_vl_request=request_index,
        total_vl_requests=metadata["vl_request_count"],
        current_crop_type="line",
        line_group_size=len(batch),
        last_completed_position=processed_count,
        applied_count=applied_count,
        saved_tile_count=saved_tile_count,
    )

    records = [candidate_record(candidate) for candidate in batch]
    for record in records:
        record["crop_type"] = "line"
        record["line_group_size"] = len(batch)

    try:
        tile, crop_box = crop_line_tile(
            image,
            ocr_payload,
            line_index,
            batch,
            padding_px=metadata["padding_px"],
        )
    except Exception as exc:
        error = f"Unable to crop line tile: {exc}"
        for record in records:
            record["error"] = error
            metadata["corrections"].append(record)
        processed_count += len(batch)
        write_post_ocr_vl_progress(
            progress_path,
            metadata,
            status="running",
            source_json=source_json,
            image_path=image_path,
            suspicious_word_count=suspicious_word_count,
            qualified_word_count=metadata["qualified_count"],
            total_words_to_check=metadata["candidate_count"],
            current_position=processed_count,
            current_word=current_word,
            current_line_index=line_index,
            current_word_index=[candidate.word_index for candidate in batch],
            current_vl_request=request_index,
            total_vl_requests=metadata["vl_request_count"],
            current_crop_type="line",
            last_completed_position=processed_count,
            applied_count=applied_count,
            saved_tile_count=saved_tile_count,
            last_error=error,
        )
        return processed_count, applied_count, saved_tile_count

    tile_path = None
    if metadata["save_tiles"]:
        word_indexes = [candidate.word_index for candidate in batch]
        word_span = f"{min(word_indexes):03d}-{max(word_indexes):03d}"
        tile_path = tile_dir / f"{line_index:04d}_line_words_{word_span}.png"
        tile.save(tile_path)
        saved_tile_count += 1

    if len(batch) == 1 and batch[0].source == "line_text":
        return process_single_line_text_batch(
            tile,
            crop_box,
            batch[0],
            opt,
            metadata,
            progress_path,
            source_json,
            image_path,
            tile_path,
            suspicious_word_count,
            request_index,
            processed_count,
            applied_count,
            saved_tile_count,
            ocr_payload,
        )

    backend_result: dict[str, Any]
    line_corrections: dict[tuple[int, int], str] = {}
    try:
        response_text, line_corrections = request_olmocr_line(
            tile,
            ocr_payload,
            batch,
            opt,
        )
        backend_result = {
            "backend": "olmocr",
            "text": response_text,
            "parsed_correction_count": len(line_corrections),
        }
    except Exception as exc:
        backend_result = {"backend": "olmocr", "error": str(exc)}

    for candidate, record in zip(batch, records):
        record["crop_box"] = crop_box
        if tile_path is not None:
            record["tile_path"] = str(tile_path)

        selected_text, reason = choose_line_correction(
            candidate.text,
            line_corrections.get(candidate_key(candidate)),
            backend_result.get("error"),
        )
        record["backend_results"] = [
            {
                **backend_result,
                "target_word_index": candidate.word_index,
                "target_selected_text": selected_text,
            }
        ]
        record["selected_text"] = selected_text
        record["reason"] = reason

        if (
            metadata["apply_corrections"]
            and selected_text
            and reason == "accepted"
        ):
            apply_candidate_correction(ocr_payload, candidate, selected_text)
            record["applied"] = True
            applied_count += 1
        else:
            record["applied"] = False

        metadata["corrections"].append(record)

    processed_count += len(batch)
    write_post_ocr_vl_progress(
        progress_path,
        metadata,
        status="running",
        source_json=source_json,
        image_path=image_path,
        suspicious_word_count=suspicious_word_count,
        qualified_word_count=metadata["qualified_count"],
        total_words_to_check=metadata["candidate_count"],
        current_position=processed_count,
        current_word=current_word,
        current_line_index=line_index,
        current_word_index=[candidate.word_index for candidate in batch],
        current_vl_request=request_index,
        total_vl_requests=metadata["vl_request_count"],
        current_crop_type="line",
        selected_text=[
            record.get("selected_text")
            for record in records
            if record.get("selected_text")
        ],
        correction_reason=[
            record.get("reason")
            for record in records
            if record.get("reason")
        ],
        last_completed_position=processed_count,
        applied_count=applied_count,
        saved_tile_count=saved_tile_count,
    )
    return processed_count, applied_count, saved_tile_count


def process_single_line_text_batch(
    tile: Image.Image,
    crop_box: list[int],
    candidate: WordCandidate,
    opt: dict,
    metadata: dict[str, Any],
    progress_path: Path,
    source_json: str,
    image_path: str,
    tile_path: Path | None,
    suspicious_word_count: int,
    request_index: int,
    processed_count: int,
    applied_count: int,
    saved_tile_count: int,
    ocr_payload: dict[str, Any],
) -> tuple[int, int, int]:
    record = candidate_record(candidate)
    record["crop_type"] = "line"
    record["line_group_size"] = 1
    record["crop_box"] = crop_box
    if tile_path is not None:
        record["tile_path"] = str(tile_path)

    results = []
    for backend in backends_for_request(metadata["backend"]):
        try:
            raw_text = run_backend_check(
                backend,
                tile,
                candidate.text,
                opt,
            )
            selected_from_line = extract_line_fallback_candidate_text(
                raw_text,
                candidate.text,
                backend,
                opt,
            )
            results.append(
                {
                    "backend": backend,
                    "text": selected_from_line or raw_text,
                    "raw_text": raw_text,
                    "extracted_from_line_text": bool(
                        selected_from_line and selected_from_line != raw_text
                    ),
                }
            )
        except Exception as exc:
            results.append({"backend": backend, "error": str(exc)})

    record["backend_results"] = results
    selected_text, reason = choose_correction(candidate.text, results)
    record["selected_text"] = selected_text
    record["reason"] = reason

    if metadata["apply_corrections"] and selected_text and reason == "accepted":
        apply_candidate_correction(ocr_payload, candidate, selected_text)
        record["applied"] = True
        applied_count += 1
    else:
        record["applied"] = False

    metadata["corrections"].append(record)
    processed_count += 1
    write_post_ocr_vl_progress(
        progress_path,
        metadata,
        status="running",
        source_json=source_json,
        image_path=image_path,
        suspicious_word_count=suspicious_word_count,
        qualified_word_count=metadata["qualified_count"],
        total_words_to_check=metadata["candidate_count"],
        current_position=processed_count,
        current_word=candidate.text,
        current_line_index=candidate.line_index,
        current_word_index=[candidate.word_index],
        current_vl_request=request_index,
        total_vl_requests=metadata["vl_request_count"],
        current_crop_type="line",
        selected_text=record["selected_text"],
        correction_reason=record["reason"],
        last_completed_position=processed_count,
        applied_count=applied_count,
        saved_tile_count=saved_tile_count,
    )
    return processed_count, applied_count, saved_tile_count


def is_word_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False
    normalized = stripped.strip(".,:;!?()[]{}\"'")
    if len(normalized) < 2:
        return False
    return True


def is_ocr_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    normalized = stripped.strip(".,:;!?()[]{}\"'")
    return bool(normalized) and bool(re.search(r"[A-Za-z0-9]", normalized))


def detect_ocr_language(ocr_payload: dict[str, Any]) -> tuple[str | None, str]:
    configured = extract_configured_language(ocr_payload)
    if configured:
        return configured, "metadata"

    sample = ocr_language_sample(ocr_payload)
    if not sample:
        return None, "langdetect:insufficient_text"

    try:
        languages = detect_langs(sample)
    except LangDetectException as exc:
        return None, f"langdetect:{exc.__class__.__name__}"

    if not languages:
        return None, "langdetect:no_result"

    best = languages[0]
    language = normalize_language_code(best.lang)
    if not language:
        return None, "langdetect:unusable_result"
    return language, f"langdetect:{best.prob:.2f}"


def extract_configured_language(ocr_payload: dict[str, Any]) -> str | None:
    for key in LANGUAGE_KEYS:
        language = normalize_language_code(ocr_payload.get(key))
        if language:
            return language

    for container_key in LANGUAGE_CONTAINERS:
        container = ocr_payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in LANGUAGE_KEYS:
            language = normalize_language_code(container.get(key))
            if language:
                return language

    return None


def normalize_language_code(value: Any) -> str | None:
    if not value:
        return None
    language = str(value).strip().casefold()
    if not language or language in {"auto", "automatic", "unknown"}:
        return None
    language = language.replace("_", "-")
    language = language.split(",", 1)[0].split(".", 1)[0].split("-", 1)[0]
    language = LANGUAGE_ALIASES.get(language, language)
    if not re.fullmatch(r"[a-z]{2,3}", language):
        return None
    return language


def ocr_language_sample(ocr_payload: dict[str, Any], max_chars: int = 12000) -> str:
    texts: list[str] = []
    rec_texts = ocr_payload.get("rec_texts")
    if isinstance(rec_texts, list):
        texts.extend(str(text) for text in rec_texts if text)

    if sum(len(text) for text in texts) < max_chars:
        text_word = ocr_payload.get("text_word") or []
        for words in text_word:
            if isinstance(words, list):
                texts.append(" ".join(str(word) for word in words if word))

    sample = " ".join(texts)
    sample = re.sub(r"\s+", " ", sample).strip()
    return sample[:max_chars]


def build_spellchecker(language: str | None) -> SpellCheckerAdapter | None:
    if not language:
        return None

    spellchecker_language = SPELLCHECKER_LANGUAGE_MAP.get(language)
    if not spellchecker_language:
        return None
    if spellchecker_language not in SpellChecker.languages():
        return None

    if spellchecker_language not in _SPELLCHECKER_CACHE:
        _SPELLCHECKER_CACHE[spellchecker_language] = SpellCheckerAdapter(
            language=spellchecker_language,
            checker=SpellChecker(language=spellchecker_language, distance=2),
        )
    return _SPELLCHECKER_CACHE[spellchecker_language]


def build_word_profile(
    ocr_payload: dict[str, Any],
    low_confidence_threshold: float,
) -> WordProfile:
    language, language_source = detect_ocr_language(ocr_payload)
    spellchecker = build_spellchecker(language)
    text_word = ocr_payload.get("text_word") or []
    rec_scores = ocr_payload.get("rec_scores") or []
    minimum_training_score = max(0.92, low_confidence_threshold)

    scored_tokens: list[tuple[str, float | None]] = []
    for line_index, words in enumerate(text_word):
        if not isinstance(words, list):
            continue
        line_score = parse_optional_float(
            rec_scores[line_index] if line_index < len(rec_scores) else None
        )
        for word in words:
            token = normalize_word_token(str(word or ""))
            if is_alpha_word(token):
                scored_tokens.append((token, line_score))

    training_tokens = [
        token
        for token, line_score in scored_tokens
        if line_score is not None
        and line_score >= minimum_training_score
        and is_training_word(token)
    ]
    if len(training_tokens) < 200:
        training_tokens = [
            token for token, _line_score in scored_tokens if is_training_word(token)
        ]

    if len(training_tokens) < 50:
        return WordProfile(
            training_token_count=len(training_tokens),
            language=language,
            language_source=language_source,
            spellchecker=spellchecker,
        )

    all_token_counts = Counter(token for token, _line_score in scored_tokens)
    token_counts = Counter(training_tokens)
    ngrams: Counter[str] = Counter()
    for token in training_tokens:
        padded = f"^{token}$"
        for ngram_size in (2, 3, 4):
            for index in range(len(padded) - ngram_size + 1):
                ngrams[padded[index : index + ngram_size]] += 1

    profile = WordProfile(
        ngrams=ngrams,
        token_counts=all_token_counts,
        total_ngrams=sum(ngrams.values()),
        vocab_size=len(ngrams),
        known_words={token for token, count in all_token_counts.items() if count >= 2},
        common_words_by_length=group_common_words_by_length(all_token_counts),
        training_token_count=len(training_tokens),
        language=language,
        language_source=language_source,
        spellchecker=spellchecker,
    )

    document_scores = [
        score
        for token, _line_score in scored_tokens
        if len(token) >= 5
        for score in [profile.score(token)]
        if score is not None
    ]
    profile.anomaly_threshold = percentile(document_scores, 0.05)
    profile.strong_anomaly_threshold = percentile(document_scores, 0.02)
    return profile


def normalize_word_token(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    value = value.strip(WORD_STRIP_CHARS)
    value = re.sub(r"'{2,}", "'", value)
    return value.casefold()


def is_alpha_word(token: str) -> bool:
    return bool(WORDISH_RE.fullmatch(token))


def is_training_word(token: str) -> bool:
    if not is_alpha_word(token):
        return False
    if not 3 <= len(token) <= 18:
        return False
    if re.search(r"(.)\1\1", token):
        return False
    if len(token) >= 4 and not re.search(r"[aeiouy]", token, flags=re.IGNORECASE):
        return False
    return True


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def group_common_words_by_length(token_counts: Counter[str]) -> dict[int, list[str]]:
    words_by_length: dict[int, list[str]] = {}
    for token, count in token_counts.items():
        if count < 2 or not is_alpha_word(token) or len(token) < 3:
            continue
        words_by_length.setdefault(len(token), []).append(token)
    return words_by_length


def levenshtein_limited(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, 1):
            value = min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def score_weirdness(
    text: str,
    line_score: float | None,
    low_confidence_threshold: float,
) -> float:
    return score_word_weirdness(text, line_score, low_confidence_threshold, None)[0]


def score_word_weirdness(
    text: str,
    line_score: float | None,
    low_confidence_threshold: float,
    word_profile: WordProfile | None,
    check_subtokens: bool = True,
) -> tuple[float, list[str]]:
    stripped = text.strip()
    normalized = stripped.strip(WORD_STRIP_CHARS)
    normalized_word = normalize_word_token(stripped)
    score = 0.0
    reasons: list[str] = []

    if has_merged_word_box_text(stripped):
        score += 1.25
        reasons.append("merged word-box text")
    if ODD_CHAR_RE.search(stripped):
        score += 1.5
        reasons.append("odd glyph")
    if has_repeated_punctuation(stripped):
        score += 0.75
        reasons.append("repeated punctuation")
    if re.search(r"([A-Za-z])\1\1", normalized):
        score += 1.0
        reasons.append("repeated letter")
    if len(normalized_word) >= 4 and is_alpha_word(normalized_word):
        if not re.search(r"[AEIOUYaeiouy]", normalized):
            score += 0.75
            reasons.append("no vowel")
        if normalized_word.startswith(("rn", "vv")):
            score += 0.75
            reasons.append("OCR-confusable prefix")
        if has_odd_internal_casing(normalized):
            score += 0.75
            reasons.append("odd internal casing")
        anomaly_score = (
            word_profile.score(normalized_word)
            if word_profile is not None and len(normalized_word) >= 5
            else None
        )
        if (
            anomaly_score is not None
            and normalized_word not in word_profile.known_words
            and word_profile.strong_anomaly_threshold is not None
            and anomaly_score <= word_profile.strong_anomaly_threshold
        ):
            score += 1.0
            reasons.append("document-rare character pattern")
        elif (
            anomaly_score is not None
            and normalized_word not in word_profile.known_words
            and word_profile.anomaly_threshold is not None
            and anomaly_score <= word_profile.anomaly_threshold
        ):
            score += 0.55
            reasons.append("unusual character pattern")
        nearest_match = (
            word_profile.nearest_common_word_match(normalized_word)
            if word_profile is not None
            else None
        )
        if nearest_match and should_flag_near_common_word(
            normalized_word,
            nearest_match,
            anomaly_score,
            word_profile,
        ):
            nearest_word = nearest_match[0]
            score += 0.85
            reasons.append(f"near common document word: {nearest_word}")
        spellcheck_issue = (
            word_profile.spellchecker.issue_for(normalized_word)
            if word_profile is not None
            and word_profile.spellchecker is not None
            and should_spellcheck_word(stripped, normalized_word, word_profile)
            else None
        )
        spellcheck_weight = spellcheck_issue_weight(
            stripped,
            normalized_word,
            spellcheck_issue,
            score,
            line_score,
            low_confidence_threshold,
            word_profile,
        )
        if spellcheck_weight > 0:
            score += spellcheck_weight
            reason = "spellcheck miss"
            if word_profile is not None and word_profile.language:
                reason = f"{reason} ({word_profile.language})"
            if spellcheck_issue and spellcheck_issue.suggestion:
                reason = f"{reason}: suggested {spellcheck_issue.suggestion}"
            reasons.append(reason)
    if check_subtokens and word_profile is not None:
        subtoken_score, subtoken_reasons = score_hyphenated_subtokens(
            normalized,
            low_confidence_threshold,
            word_profile,
        )
        if subtoken_score > 0:
            score += subtoken_score
            reasons.extend(subtoken_reasons)
    if (
        re.search(r"[A-Za-z][0-9]|[0-9][A-Za-z]", normalized)
        and not is_legitimate_letter_digit_token(normalized)
    ):
        score += 1.0
        reasons.append("letter/digit mix")
    if stripped != normalized and len(normalized) >= 3:
        score += 0.15
        reasons.append("boundary punctuation")

    if line_score is not None and line_score < low_confidence_threshold:
        if line_score < max(0.5, low_confidence_threshold - 0.25):
            score += max(0.25, (low_confidence_threshold - line_score) * 2)
            reasons.append("very low line confidence")
        elif score > 0:
            score += min(0.5, max(0.1, (low_confidence_threshold - line_score) * 2))
            reasons.append("low line confidence")

    return score, reasons


def score_hyphenated_subtokens(
    text: str,
    low_confidence_threshold: float,
    word_profile: WordProfile,
) -> tuple[float, list[str]]:
    if not re.search(r"[-/]", text):
        return 0.0, []

    subtokens = [
        match.group(0)
        for match in re.finditer(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?", text)
    ]
    if len(subtokens) < 2:
        return 0.0, []

    score = 0.0
    reasons: list[str] = []
    for subtoken in subtokens:
        normalized = normalize_word_token(subtoken)
        if len(normalized) < 4 or not is_alpha_word(normalized):
            continue

        sub_score, sub_reasons = score_word_weirdness(
            subtoken,
            None,
            low_confidence_threshold,
            word_profile,
            check_subtokens=False,
        )
        if sub_score < 0.5:
            continue

        score += min(1.0, sub_score)
        reasons.append(
            f"hyphenated subtoken {subtoken}: {', '.join(sub_reasons[:2])}"
        )

    return min(score, 1.5), reasons


def has_merged_word_box_text(text: str) -> bool:
    parts = re.findall(r"[A-Za-z0-9]+", text)
    return len(parts) > 1 and bool(re.search(r"\s", text.strip()))


def has_repeated_punctuation(text: str) -> bool:
    return bool(re.search(r"\.{2,}|,{2,}|;{2,}|:{2,}|!{2,}|\?{2,}|--+", text))


def has_odd_internal_casing(text: str) -> bool:
    if len(text) < 4:
        return False
    if text.islower() or text.isupper() or text.istitle():
        return False
    if re.match(r"(?:Mc|Mac)[A-Z][a-z]", text):
        return False
    return bool(re.search(r"[a-z][A-Z][a-z]|[A-Z]{2,}[a-z]|[a-z][A-Z]{2,}", text))


def should_flag_near_common_word(
    token: str,
    match: tuple[str, int, int],
    anomaly_score: float | None,
    word_profile: WordProfile,
) -> bool:
    _word, distance, common_count = match
    if distance <= 1:
        return common_count >= 3 or len(token) >= 6
    if len(token) >= 5 and common_count >= 10:
        return True
    if len(token) < 7:
        return False
    if len(token) >= 9 and common_count >= 2:
        return True
    if common_count >= 3:
        return True
    return (
        anomaly_score is not None
        and word_profile.anomaly_threshold is not None
        and anomaly_score <= word_profile.anomaly_threshold
    )


def should_spellcheck_word(
    raw_text: str,
    token: str,
    word_profile: WordProfile,
) -> bool:
    if not is_alpha_word(token):
        return False
    if not 4 <= len(token) <= 30:
        return False
    if "'" in token:
        return False
    if raw_text.isupper() and len(token) <= 4:
        return False
    if word_profile.token_counts.get(token, 0) >= 4:
        return False
    return True


def spellcheck_issue_weight(
    raw_text: str,
    token: str,
    issue: SpellcheckIssue | None,
    existing_score: float,
    line_score: float | None,
    low_confidence_threshold: float,
    word_profile: WordProfile | None = None,
) -> float:
    if issue is None:
        return 0.0

    low_confidence = line_score is not None and line_score < low_confidence_threshold
    may_be_proper_name = raw_text[:1].isupper() and raw_text[1:].islower()
    if may_be_proper_name and existing_score <= 0 and not low_confidence:
        if should_flag_titlecase_spell_miss(token, word_profile):
            return 0.55
        if issue.distance is None or issue.distance > 1:
            return 0.0

    if issue.distance == 1:
        return 0.9
    if issue.distance == 2:
        return 0.75
    if issue.distance == 3 and len(token) >= 7:
        return 0.6
    if existing_score > 0 or low_confidence or raw_text.islower():
        return 0.55
    return 0.0


def should_flag_titlecase_spell_miss(
    token: str,
    word_profile: WordProfile | None,
) -> bool:
    if word_profile is None:
        return False
    if len(token) < 7:
        return False
    if word_profile.token_counts.get(token, 0) > 2:
        return False
    return True


def is_legitimate_letter_digit_token(text: str) -> bool:
    normalized = text.strip(".,:;!?()[]{}\"'")
    return bool(
        re.fullmatch(r"\d+(st|nd|rd|th)", normalized, flags=re.IGNORECASE)
        or re.fullmatch(r"\d+c", normalized, flags=re.IGNORECASE)
    )



def crop_word_tile(
    image: Image.Image,
    box: list[float],
    padding_px: int,
) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    contextual_padding_x = max(padding_px, int(width * 0.35))
    contextual_padding_y = max(padding_px, int(height * 0.75))

    left = max(0, int(x0 - contextual_padding_x))
    top = max(0, int(y0 - contextual_padding_y))
    right = min(image.width, int(x1 + contextual_padding_x))
    bottom = min(image.height, int(y1 + contextual_padding_y))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box: {[left, top, right, bottom]}")

    tile = image.crop((left, top, right, bottom))
    tile = upscale_small_tile(tile)
    return tile, [left, top, right, bottom]


def crop_line_tile(
    image: Image.Image,
    ocr_payload: dict[str, Any],
    line_index: int,
    candidates: list[WordCandidate],
    padding_px: int,
) -> tuple[Image.Image, list[int]]:
    box = line_bbox_for_candidates(ocr_payload, line_index, candidates)
    if box is None:
        raise ValueError(f"Could not resolve line box for line {line_index}")

    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    contextual_padding_x = max(padding_px, int(width * 0.04))
    contextual_padding_y = max(padding_px, int(height * 0.9))

    left = max(0, int(x0 - contextual_padding_x))
    top = max(0, int(y0 - contextual_padding_y))
    right = min(image.width, int(x1 + contextual_padding_x))
    bottom = min(image.height, int(y1 + contextual_padding_y))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid line crop box: {[left, top, right, bottom]}")

    tile = image.crop((left, top, right, bottom))
    tile = upscale_small_tile(tile)
    return tile, [left, top, right, bottom]


def line_bbox_for_candidates(
    ocr_payload: dict[str, Any],
    line_index: int,
    candidates: list[WordCandidate],
) -> list[float] | None:
    for key in ("rec_boxes", "rec_polys", "dt_polys"):
        geometries = ocr_payload.get(key)
        if isinstance(geometries, list) and line_index < len(geometries):
            box = geometry_to_bbox(geometries[line_index])
            if valid_bbox(box):
                return box

    text_word_boxes = ocr_payload.get("text_word_boxes") or []
    if isinstance(text_word_boxes, list) and line_index < len(text_word_boxes):
        line_word_boxes = text_word_boxes[line_index]
        if not isinstance(line_word_boxes, list):
            line_word_boxes = []
        boxes = [
            box
            for geometry in line_word_boxes
            for box in [geometry_to_bbox(geometry)]
            if valid_bbox(box)
        ]
        if boxes:
            return union_bbox(boxes)

    return union_bbox([candidate.box for candidate in candidates])


def valid_bbox(box: list[float] | None) -> bool:
    return bool(box and len(box) == 4 and box[2] > box[0] and box[3] > box[1])


def union_bbox(boxes: list[list[float]]) -> list[float] | None:
    valid_boxes = [box for box in boxes if valid_bbox(box)]
    if not valid_boxes:
        return None
    return [
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    ]


def upscale_small_tile(tile: Image.Image) -> Image.Image:
    min_height = 96
    max_width = 768
    if tile.height >= min_height:
        return tile

    scale = min(4.0, max(1.0, min_height / max(1, tile.height)))
    new_width = min(max_width, int(tile.width * scale))
    new_height = max(min_height, int(tile.height * scale))
    return tile.resize((new_width, new_height), Image.Resampling.LANCZOS)


def run_backend_check(
    backend: str,
    tile: Image.Image,
    original_text: str,
    opt: dict,
) -> str:
    if backend == "paddlevl":
        return request_paddlevl_word(tile, opt)
    if backend == "olmocr":
        return request_olmocr_word(tile, original_text, opt)
    raise ValueError(f"Unknown post-OCR VL backend: {backend}")


def request_paddlevl_word(tile: Image.Image, opt: dict) -> str:
    base_url = str(
        opt.get("post_ocr_vl_paddlevl_url") or DEFAULT_PADDLE_VL_API_URL
    ).rstrip("/")
    payload = {
        "file": image_to_base64(tile),
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "maxNewTokens": 32,
    }
    response = requests.post(
        f"{base_url}/layout-parsing",
        json=payload,
        timeout=(10, 120),
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    return clean_vl_text(extract_paddlevl_text(response.json()))


def _build_word_correction_prompt(original_text: str, model_name: str = None) -> str:
    """
    Build a prompt for word correction that's optimized for the specific model.
    
    Args:
        original_text: The OCR text that may need correction
        model_name: Name of the VLM being used (optional, for model-specific prompting)
    
    Returns:
        A prompt string optimized for the model
    """
    # For Chandra and other OCR-specialized models, use direct prompts
    if model_name and _is_chandra_model(model_name):
        return (
            "What is the correct spelling of the target word in this newspaper image? "
            "Return only the corrected word text, nothing else. "
            f"Previous OCR reading: {original_text!r}"
        )

    if model_name and _is_deepseek_ocr_model(model_name):
        return (
            "Extract the text in the image. "
            "Return only the visible target word text with no tags, no boxes, no markdown, "
            "and no explanation. "
            f"Previous OCR reading: {original_text!r}"
        )
    
    # Default prompt for general vision models
    return (
        "Read the single target word in this newspaper crop. "
        "Return only the corrected word, with visible punctuation only if it "
        f"is part of the word. The previous OCR guess was: {original_text!r}."
    )


def request_olmocr_word(tile: Image.Image, original_text: str, opt: dict) -> str:
    server_url = _get_openai_compat_url(opt).rstrip("/")
    completion_url = (
        server_url
        if server_url.endswith("/chat/completions")
        else f"{server_url}/chat/completions"
    )
    model = _get_openai_compat_model(opt)
    prompt = _build_word_correction_prompt(original_text, model_name=model)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(tile)}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    headers = {"Content-Type": "application/json"}
    api_key = _get_openai_compat_api_key(opt)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        completion_url,
        headers=headers,
        json=payload,
        timeout=(10, 120),
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if _is_chandra_model(model):
        if not str(content or "").strip():
            native_url = _native_ollama_chat_url(server_url)
            if native_url:
                native_payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_to_base64(tile)],
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 32,
                    },
                }
                native_response = requests.post(
                    native_url,
                    headers=headers,
                    json=native_payload,
                    timeout=(10, 120),
                )
                if native_response.status_code == 200:
                    native_data = native_response.json()
                    content = native_data.get("message", {}).get("content", "") or content
        return clean_vl_text(
            _extract_chandra_word_candidate(content, original_text),
            model_name=model,
        )
    if _is_deepseek_ocr_model(model):
        return clean_vl_text(
            _extract_deepseek_word_candidate(content, original_text),
            model_name=model,
        )
    return clean_vl_text(content, model_name=model)


def request_olmocr_line(
    tile: Image.Image,
    ocr_payload: dict[str, Any],
    candidates: list[WordCandidate],
    opt: dict,
) -> tuple[str, dict[tuple[int, int], str]]:
    server_url = _get_openai_compat_url(opt).rstrip("/")
    completion_url = (
        server_url
        if server_url.endswith("/chat/completions")
        else f"{server_url}/chat/completions"
    )
    model = _get_openai_compat_model(opt)
    line_index = candidates[0].line_index
    prompt = build_olmocr_line_prompt(ocr_payload, line_index, candidates, model_name=model)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(tile)}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": min(512, max(128, len(candidates) * 64)),
    }
    headers = {"Content-Type": "application/json"}
    api_key = _get_openai_compat_api_key(opt)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        completion_url,
        headers=headers,
        json=payload,
        timeout=(10, 120),
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    text = str(content or "").strip()
    return text, parse_olmocr_line_corrections(text, candidates, model_name=model)


def build_olmocr_line_prompt(
    ocr_payload: dict[str, Any],
    line_index: int,
    candidates: list[WordCandidate],
    model_name: str = None,
) -> str:
    use_line_text_tokens = any(candidate.source == "line_text" for candidate in candidates)
    line_words = line_words_for_prompt(
        ocr_payload,
        line_index,
        use_line_text_tokens=use_line_text_tokens,
    )
    indexed_line = "\n".join(
        f"{word_index}: {word!r}"
        for word_index, word in enumerate(line_words)
    )
    if len(indexed_line) > 2500:
        indexed_line = f"{indexed_line[:2500]}\n..."

    targets = "\n".join(
        f"{target_index}. word_index={candidate.word_index}, "
        f"original_text={candidate.text!r}"
        for target_index, candidate in enumerate(candidates, start=1)
    )
    
    # For Chandra and other OCR-specialized models, use simpler prompt
    if model_name and _is_chandra_model(model_name):
        return (
            "You are an OCR correction specialist. Read the newspaper line in the image.\n\n"
            "Word positions (OCR text):\n"
            f"{indexed_line}\n\n"
            "Target words to correct:\n"
            f"{targets}\n\n"
            "Return ONLY valid JSON with this exact format (no other text):\n"
            '{"corrections":[{"target":1,"word_index":0,"corrected_text":"word"}]}\n'
            "Rules:\n"
            "- Use 'target' to reference the target number (1-indexed)\n"
            "- Include all target words, even if unchanged\n"
            "- Repeat original text if correct\n"
            "- Never use placeholder strings like 'word' or 'text'\n"
            "- If you cannot read a word, repeat the original"
        )

    if model_name and _is_deepseek_ocr_model(model_name):
        return (
            "Extract the text in the image. "
            "Return strict JSON only, with no tags, no markdown, and no explanation.\n\n"
            "OCR line words by zero-based word_index:\n"
            f"{indexed_line}\n\n"
            "Target words to verify:\n"
            f"{targets}\n\n"
            "JSON schema:\n"
            '{"corrections":[{"target":1,"word_index":0,"corrected_text":"church"}]}\n'
            "Include every target once. If a target is already correct or unreadable, repeat the original text."
        )
    
    # Default prompt for general vision models
    return (
        "Read the newspaper line crop. Use the image as the source of truth; "
        "the OCR text below is only to identify word positions.\n\n"
        "OCR line words by zero-based word_index:\n"
        f"{indexed_line}\n\n"
        "Target words to verify:\n"
        f"{targets}\n\n"
        "Return strict JSON only, with this schema:\n"
        '{"corrections":[{"target":1,"word_index":0,"corrected_text":"church"}]}\n'
        "Do not return the literal string 'word' or other placeholders as a correction. "
        "For each target, repeat the original text if it is already correct. "
        "Do not include non-target words or any explanation."
    )


def line_words_for_prompt(
    ocr_payload: dict[str, Any],
    line_index: int,
    use_line_text_tokens: bool = False,
) -> list[str]:
    if use_line_text_tokens:
        rec_texts = ocr_payload.get("rec_texts") or []
        if isinstance(rec_texts, list) and line_index < len(rec_texts):
            return [token.text for token in line_text_tokens(str(rec_texts[line_index] or ""))]

    text_word = ocr_payload.get("text_word") or []
    if not isinstance(text_word, list) or line_index >= len(text_word):
        return []
    words = text_word[line_index]
    if not isinstance(words, list):
        return []
    return [str(word or "") for word in words]


def parse_olmocr_line_corrections(
    text: str,
    candidates: list[WordCandidate],
    model_name: str = None,
) -> dict[tuple[int, int], str]:
    data = parse_json_response(text)
    if data is None:
        return {}

    if isinstance(data, dict):
        raw_corrections = data.get("corrections")
        if raw_corrections is None:
            raw_corrections = data.get("targets")
    else:
        raw_corrections = data

    if not isinstance(raw_corrections, list):
        return {}

    by_target = {
        target_index: candidate
        for target_index, candidate in enumerate(candidates, start=1)
    }
    by_word_index = {candidate.word_index: candidate for candidate in candidates}
    by_original: dict[str, list[WordCandidate]] = {}
    for candidate in candidates:
        by_original.setdefault(normalize_for_compare(candidate.text), []).append(
            candidate
        )

    corrections: dict[tuple[int, int], str] = {}
    for item in raw_corrections:
        if not isinstance(item, dict):
            continue
        candidate = candidate_from_line_correction_item(
            item,
            by_target,
            by_word_index,
            by_original,
        )
        if candidate is None:
            continue
        corrected = (
            item.get("corrected_text")
            or item.get("correction")
            or item.get("text")
            or item.get("selected_text")
        )
        if corrected is None:
            continue
        corrections[candidate_key(candidate)] = clean_vl_text(corrected, model_name=model_name)

    return corrections


def candidate_from_line_correction_item(
    item: dict[str, Any],
    by_target: dict[int, WordCandidate],
    by_word_index: dict[int, WordCandidate],
    by_original: dict[str, list[WordCandidate]],
) -> WordCandidate | None:
    target = parse_optional_int(item.get("target") or item.get("target_index"))
    if target in by_target:
        return by_target[target]

    word_index = parse_optional_int(item.get("word_index"))
    if word_index in by_word_index:
        return by_word_index[word_index]

    original_text = item.get("original_text") or item.get("original")
    original_key = normalize_for_compare(str(original_text or ""))
    matches = by_original.get(original_key) or []
    if len(matches) == 1:
        return matches[0]
    return None


def parse_json_response(text: str) -> Any:
    value = str(text or "").strip()
    if not value:
        return None

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = fenced.group(1).strip()

    for opening, closing in (("[", "]"), ("{", "}")):
        start = value.find(opening)
        end = value.rfind(closing)
        if start == -1 or end == -1 or end <= start:
            continue
        fragment = value[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            continue
    return None


def choose_line_correction(
    original_text: str,
    corrected_text: str | None,
    error: Any = None,
) -> tuple[str | None, str]:
    if error:
        return None, "backend_error"
    if corrected_text is None:
        return None, "line_no_target_result"

    corrected = clean_vl_text(corrected_text)
    if not is_clean_correction(original_text, corrected):
        return None, "line_unclean_vl_text"
    if corrections_equivalent(corrected, original_text):
        return corrected, "unchanged"
    return corrected, "accepted"


def candidate_key(candidate: WordCandidate) -> tuple[int, int]:
    return candidate.line_index, candidate.word_index


def extract_paddlevl_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""

    if isinstance(data.get("text"), str):
        return data["text"]
    if isinstance(data.get("markdown"), str):
        return data["markdown"]

    rec_texts = data.get("rec_texts")
    if isinstance(rec_texts, list):
        return " ".join(str(item) for item in rec_texts if item)

    try:
        parsing_res_list = data["result"]["layoutParsingResults"][0]["prunedResult"][
            "parsing_res_list"
        ]
    except (KeyError, IndexError, TypeError):
        parsing_res_list = []

    texts = []
    for block in parsing_res_list:
        if not isinstance(block, dict):
            continue
        text = block.get("block_content") or block.get("text")
        if text:
            texts.append(str(text))

    return " ".join(texts)


def choose_correction(
    original_text: str,
    results: list[dict[str, Any]],
) -> tuple[str | None, str]:
    usable = [
        clean_vl_text(result.get("text", ""))
        for result in results
        if result.get("text")
    ]
    usable = [text for text in usable if is_clean_correction(original_text, text)]
    if not usable:
        return None, "no_clean_vl_text"

    unique = {normalize_for_acceptance(text): text for text in usable}
    if len(results) > 1 and len(unique) > 1:
        return None, "backend_disagreement"

    selected = usable[0]
    if corrections_equivalent(selected, original_text):
        return selected, "unchanged"

    return selected, "accepted"


def is_clean_correction(original_text: str, corrected_text: str) -> bool:
    corrected = corrected_text.strip()
    if not corrected or len(corrected) > 80:
        return False
    if "\n" in corrected or "\r" in corrected:
        return False
    if len(corrected.split()) > 3:
        return False
    if not WORD_RE.search(corrected):
        return False
    if len(original_text.strip()) <= 2 and len(corrected) > 6:
        return False
    return True


def apply_candidate_correction(
    ocr_payload: dict[str, Any],
    candidate: WordCandidate,
    corrected_text: str,
) -> None:
    if candidate.source == "line_text":
        apply_line_text_correction(ocr_payload, candidate, corrected_text)
        return

    apply_word_correction(
        ocr_payload,
        candidate,
        candidate.line_index,
        candidate.word_index,
        corrected_text,
    )


def apply_line_text_correction(
    ocr_payload: dict[str, Any],
    candidate: WordCandidate,
    corrected_text: str,
) -> None:
    rec_texts = ocr_payload.get("rec_texts")
    if not isinstance(rec_texts, list) or candidate.line_index >= len(rec_texts):
        return

    line_text = str(rec_texts[candidate.line_index] or "")
    replacement_span = line_text_replacement_span(line_text, candidate)
    if replacement_span is None:
        return

    start, end = replacement_span
    rec_texts[candidate.line_index] = f"{line_text[:start]}{corrected_text}{line_text[end:]}"


def line_text_replacement_span(
    line_text: str,
    candidate: WordCandidate,
) -> tuple[int, int] | None:
    if (
        candidate.line_text_start is not None
        and candidate.line_text_end is not None
        and 0 <= candidate.line_text_start <= candidate.line_text_end <= len(line_text)
    ):
        current = line_text[candidate.line_text_start : candidate.line_text_end]
        if normalize_for_compare(current) == normalize_for_compare(candidate.text):
            return candidate.line_text_start, candidate.line_text_end

    tokens = line_text_tokens(line_text)
    if 0 <= candidate.word_index < len(tokens):
        token = tokens[candidate.word_index]
        if normalize_for_compare(token.text) == normalize_for_compare(candidate.text):
            return token.start, token.end

    target = normalize_for_compare(candidate.text)
    for token in tokens:
        if normalize_for_compare(token.text) == target:
            return token.start, token.end
    return None


def apply_word_correction(
    ocr_payload: dict[str, Any],
    candidate: WordCandidate,
    line_index: int,
    word_index: int,
    corrected_text: str,
) -> None:
    text_word = ocr_payload.get("text_word") or []
    if line_index >= len(text_word):
        return
    words = text_word[line_index]
    if not isinstance(words, list) or word_index >= len(words):
        return

    words[word_index] = corrected_text
    rec_texts = ocr_payload.get("rec_texts")
    if isinstance(rec_texts, list) and line_index < len(rec_texts):
        line_text = str(rec_texts[line_index] or "")
        replacement_span = line_text_replacement_span(line_text, candidate)
        if replacement_span is not None:
            start, end = replacement_span
            rec_texts[line_index] = f"{line_text[:start]}{corrected_text}{line_text[end:]}"


def candidate_record(candidate: WordCandidate) -> dict[str, Any]:
    record = {
        "line_index": candidate.line_index,
        "word_index": candidate.word_index,
        "original_text": candidate.text,
        "box": candidate.box,
        "line_score": candidate.line_score,
        "weirdness": candidate.weirdness,
        "candidate_source": candidate.source,
    }
    if candidate.line_text_start is not None and candidate.line_text_end is not None:
        record["line_text_span"] = [candidate.line_text_start, candidate.line_text_end]
    if candidate.reasons:
        record["candidate_reasons"] = candidate.reasons
    return record


def post_ocr_vl_progress_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.post_ocr_vl_progress.json")


def write_post_ocr_vl_progress(
    progress_path: Path,
    metadata: dict[str, Any],
    **updates: Any,
) -> None:
    payload = {
        "status": updates.pop("status", "running"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "backend": metadata.get("backend"),
        "candidate_mode": metadata.get("candidate_mode"),
        "max_words_enabled": metadata.get("max_words_enabled"),
        "max_words": metadata.get("max_words"),
        "save_tiles": metadata.get("save_tiles"),
        "group_line_candidates": metadata.get("group_line_candidates"),
        "line_grouping_active": metadata.get("line_grouping_active"),
        "vl_request_count": metadata.get("vl_request_count"),
        "line_group_count": metadata.get("line_group_count"),
        "line_grouped_word_count": metadata.get("line_grouped_word_count"),
        "line_text_fallback_candidate_count": metadata.get(
            "line_text_fallback_candidate_count"
        ),
        "detected_language": metadata.get("detected_language"),
        "language_detection_source": metadata.get("language_detection_source"),
        "spellchecker_backend": metadata.get("spellchecker_backend"),
        "spellchecker_language": metadata.get("spellchecker_language"),
        **updates,
    }
    try:
        progress_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _is_chandra_model(model_name: str) -> bool:
    """Check if the model name indicates it's a Chandra model."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "chandra" in model_lower or "chandra-ocr" in model_lower


def _is_deepseek_ocr_model(model_name: str) -> bool:
    """Check if the model name indicates it's a DeepSeek OCR model."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "deepseek-ocr" in model_lower or "deepseekocr" in model_lower


def _parse_chandra_response(text: str) -> str:
    """
    Parse Chandra OCR model responses, which may return JSON or formatted text.
    
    Chandra often returns responses in formats like:
    - {"word": "corrected_text", "confidence": 0.95}
    - {"corrected_text": "word"}
    - Plain text with metadata
    """
    text = str(text or "").strip()
    if not text:
        return ""
    
    # Try to parse as JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Try common field names
            for key in ("word", "corrected_text", "correction", "text", "result"):
                if key in data and isinstance(data[key], str):
                    return str(data[key]).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    
    # If not JSON or JSON parsing failed, return text as-is for clean_vl_text to handle
    return text


def _strip_control_tag_artifacts(value: str) -> str:
    """Extract user-visible text from model outputs that include control tags."""
    if not value or not CONTROL_TAG_RE.search(value):
        return value

    segments = [
        segment.strip()
        for segment in CONTROL_TAG_RE.split(value)
        if segment and segment.strip()
    ]
    for segment in reversed(segments):
        cleaned = segment.strip().strip("{}[]()<>|+-,:;")
        if WORD_RE.search(cleaned):
            return cleaned

    collapsed = CONTROL_TAG_RE.sub(" ", value)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed.strip("{}[]()<>|+-,:;")


def _parse_deepseek_ocr_response(text: str) -> str:
    """Normalize DeepSeek OCR raw outputs that may include control tags."""
    value = str(text or "").strip()
    if not value:
        return ""

    # DeepSeek OCR may emit content through raw control-tagged text streams.
    value = _strip_control_tag_artifacts(value)
    value = HTML_BREAK_TAG_RE.sub("\n", value)
    value = HTML_TAG_RE.sub(" ", value)
    value = re.sub(r"^\+\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _extract_best_word_candidate(text: str, original_text: str, minimum_score: float = 0.55) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    normalized_original = normalize_for_compare(original_text)
    if not normalized_original:
        return value

    raw_segments = [
        segment.strip()
        for segment in re.split(r"[\r\n]+", value)
        if segment and segment.strip()
    ]
    token_candidates: list[str] = []
    for segment in raw_segments or [value]:
        token_candidates.append(segment)
        token_candidates.extend(match.group(0) for match in OCR_WORDISH_RE.finditer(segment))

    best_candidate = ""
    best_score = float("-inf")
    for candidate in token_candidates:
        cleaned = candidate.strip().strip("{}[]()<>|+-,:;")
        normalized_candidate = normalize_for_compare(cleaned)
        if not normalized_candidate:
            continue
        score = SequenceMatcher(None, normalized_candidate, normalized_original).ratio()
        if normalized_candidate == normalized_original:
            score += 1.0
        score -= max(0, len(cleaned.split()) - 1) * 0.2
        score -= abs(len(normalized_candidate) - len(normalized_original)) * 0.03
        if score > best_score:
            best_candidate = cleaned
            best_score = score

    if best_candidate and best_score >= minimum_score:
        return best_candidate
    return ""


def _extract_deepseek_word_candidate(text: str, original_text: str) -> str:
    """Pick the most plausible single-word OCR result from DeepSeek output."""
    value = _parse_deepseek_ocr_response(text)
    if not value:
        return ""
    return _extract_best_word_candidate(value, original_text, minimum_score=0.55)


def _extract_chandra_word_candidate(text: str, original_text: str) -> str:
    """Pick the most plausible word from Chandra OCR-style content."""
    value = str(text or "").strip()
    if not value:
        return ""

    value = HTML_BREAK_TAG_RE.sub("\n", value)
    value = HTML_TAG_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return _extract_best_word_candidate(value, original_text, minimum_score=0.5)


def extract_line_fallback_candidate_text(
    text: str,
    original_text: str,
    backend: str,
    opt: dict[str, Any],
) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    if backend == "olmocr":
        model = _get_openai_compat_model(opt)
        if _is_chandra_model(model):
            return clean_vl_text(
                _extract_chandra_word_candidate(value, original_text),
                model_name=model,
            )
        if _is_deepseek_ocr_model(model):
            return clean_vl_text(
                _extract_deepseek_word_candidate(value, original_text),
                model_name=model,
            )
        return clean_vl_text(
            _extract_best_word_candidate(value, original_text, minimum_score=0.45)
        )

    return clean_vl_text(
        _extract_best_word_candidate(value, original_text, minimum_score=0.45)
    )


def clean_vl_text(text: Any, model_name: str = None) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    # Handle model-specific response formats
    if model_name and _is_chandra_model(model_name):
        value = _parse_chandra_response(value)
        if not value:
            return ""
    elif model_name and _is_deepseek_ocr_model(model_name):
        value = _parse_deepseek_ocr_response(value)
        if not value:
            return ""

    value = re.sub(r"```(?:text|markdown|yaml)?", "", value, flags=re.IGNORECASE)
    value = value.replace("```", "").strip()
    value = _strip_control_tag_artifacts(value)

    if "\n" in value:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        non_metadata = [
            line
            for line in lines
            if not line.startswith("---") and ":" not in line[:24]
        ]
        if non_metadata:
            value = non_metadata[-1]
        elif lines:
            value = lines[-1]

    value = value.strip().strip('"').strip("'").strip()
    value = re.sub(r"^(word|text|answer)\s*:\s*", "", value, flags=re.IGNORECASE)

    # Reject common placeholder responses that are not real corrections
    norm = value.strip().lower()
    if norm in {"word", "text", "answer", "corrected_text", "selected_text"}:
        return ""

    return value.strip()


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resolve_image_path(payload: dict[str, Any], json_path: Path) -> Path | None:
    for key in ("input_path", "original_input_path"):
        raw = payload.get(key)
        if raw:
            path = Path(str(raw))
            if path.exists():
                return path

    file_name = payload.get("fileName") or f"{json_path.stem}.jpg"
    for parent in [json_path.parent, Path.cwd()]:
        for suffix in ["", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            candidate = parent / (str(file_name) if suffix == "" else f"{json_path.stem}{suffix}")
            if candidate.exists():
                return candidate

    return None


def normalize_backend(value: Any) -> str:
    backend = str(value or DEFAULT_POST_OCR_VL_BACKEND).strip().lower()
    if backend in {"paddle", "paddlevl", "paddle_vl", "paddle-vl"}:
        return "paddlevl"
    if backend in {"olm", "olmocr", "olm-ocr", "olm_ocr"}:
        return "olmocr"
    if backend == "both":
        return "both"
    return DEFAULT_POST_OCR_VL_BACKEND


def backends_for_request(backend: str) -> list[str]:
    if backend == "both":
        return ["paddlevl", "olmocr"]
    return [backend]


def geometry_to_bbox(geometry: Any) -> list[float] | None:
    if hasattr(geometry, "tolist"):
        geometry = geometry.tolist()

    if (
        isinstance(geometry, list)
        and len(geometry) == 4
        and all(isinstance(value, (int, float)) for value in geometry)
    ):
        return [float(value) for value in geometry]

    if (
        isinstance(geometry, list)
        and geometry
        and isinstance(geometry[0], (list, tuple))
    ):
        xs = [float(point[0]) for point in geometry]
        ys = [float(point[1]) for point in geometry]
        return [min(xs), min(ys), max(xs), max(ys)]

    return None


def normalize_for_compare(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold())


def normalize_for_acceptance(value: str) -> str:
    return " ".join(
        match.group(0).casefold()
        for match in OCR_WORDISH_RE.finditer(str(value or ""))
    )


def corrections_equivalent(left: str, right: str) -> bool:
    return normalize_for_acceptance(left) == normalize_for_acceptance(right)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return (cleaned or "word")[:40]


def parse_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
