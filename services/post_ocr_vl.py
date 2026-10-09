"""Post-process suspicious OCR words with vision-language model verification."""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import csv
from difflib import SequenceMatcher
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from langdetect import DetectorFactory, LangDetectException, detect_langs
from PIL import Image
from spellchecker import SpellChecker

from services.ocr_defaults import (
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BACKEND,
    DEFAULT_POST_OCR_VL_BATCH_SIZE,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_CONCURRENCY,
    DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_POST_OCR_VL_MAX_WORDS,
    DEFAULT_POST_OCR_VL_PADDING_PX,
    DEFAULT_POST_OCR_VL_SAVE_TILES,
)

logger = logging.getLogger(__name__)


POST_OCR_VL_METHODS = {"PP-OCRv6"}
WORD_RE = re.compile(r"[A-Za-z0-9]")
ODD_CHAR_RE = re.compile(r"[^A-Za-z0-9'\u2019`.,:;!?()\\[\\]{}\\-/& ]")
WORD_STRIP_CHARS = ".,:;!?()[]{}\"'`\u2018\u2019-–—"
WORDISH_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
LINE_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")
CONTROL_TAG_RE = re.compile(r"<\|[^>]*\|>")
OCR_WORDISH_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
JOINER_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "which",
    "with",
    "would",
    "you",
    "your",
}
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
    """Return whether the selected OCR method supports post-OCR VL checks."""
    effective_method = select if select in POST_OCR_VL_METHODS else method
    return effective_method in POST_OCR_VL_METHODS


def apply_post_ocr_vl_check(
    json_path: Path,
    out_dir: Path,
    opt: dict,
) -> dict[str, Any]:
    """Audit suspicious OCR output with VL backends and record correction metadata."""
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
        "backend": DEFAULT_POST_OCR_VL_BACKEND,
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
        "group_line_candidates": False,
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
    backend_order = [DEFAULT_POST_OCR_VL_BACKEND]

    include_line_text_fallback = "paddlevl" in backend_order

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
    batch_size = parse_int(
        opt.get("post_ocr_vl_batch_size"),
        DEFAULT_POST_OCR_VL_BATCH_SIZE,
        minimum=1,
    )
    concurrency = parse_int(
        opt.get("post_ocr_vl_concurrency"),
        DEFAULT_POST_OCR_VL_CONCURRENCY,
        minimum=1,
    )
    metadata["batch_size"] = batch_size
    metadata["concurrency"] = concurrency
    metadata["line_grouping_active"] = False
    candidate_batches = group_candidate_batches(candidates, batch_size=batch_size)
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

    with requests.Session() as session:
        adapter = HTTPAdapter(
            pool_connections=max(concurrency, 10),
            pool_maxsize=max(concurrency * 2, 20),
            max_retries=1,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
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

                    processed_count, applied_count, saved_tile_count = process_word_batch(
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
                        total_requests=len(candidate_batches),
                        processed_count=processed_count,
                        applied_count=applied_count,
                        saved_tile_count=saved_tile_count,
                        session=session,
                        executor=executor if concurrency > 1 else None,
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
    """Return the nested OCR payload object used by post-OCR helpers."""
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
    """Collect words or fallback lines that should be checked by VL backends."""
    text_word = ocr_payload.get("text_word") or []
    text_word_boxes = ocr_payload.get("text_word_boxes") or []
    rec_scores = ocr_payload.get("rec_scores") or []
    candidates: list[WordCandidate] = []
    mode = mode if mode in {"suspicious", "low_confidence", "all"} else "suspicious"
    if word_profile is None:
        word_profile = build_word_profile(ocr_payload, low_confidence_threshold)

    for line_index, words in enumerate(text_word):
        if hasattr(words, "tolist"):
            words = words.tolist()
        elif not isinstance(words, (list, tuple)):
            continue

        boxes = text_word_boxes[line_index] if line_index < len(text_word_boxes) else []
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        elif not isinstance(boxes, (list, tuple)):
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

            # Smart-Skip: Avoid sending already-clean, high-confidence words to the VLM
            if mode in {"suspicious", "low_confidence"}:
                clean_text = text.strip(".,:;!?()[]{}\"'")
                is_proper_name = (
                    (clean_text[:1].isupper() and clean_text[1:].islower())
                    or bool(re.fullmatch(r"(?:Mc|Mac|O')[A-Z][a-z]+", clean_text))
                )
                has_severe_anomaly = any(
                    r in {"odd glyph", "letter/digit mix", "merged word-box text", "repeated letter"}
                    for r in reasons
                )
                # 1. Skip high-confidence proper nouns (Titlecase or Mc/Mac/O') without severe anomalies
                if (
                    line_score is not None
                    and line_score >= 0.92
                    and is_proper_name
                    and not has_severe_anomaly
                ):
                    continue

                # 2. Skip high-confidence words that are confirmed valid dictionary words
                if (
                    line_score is not None
                    and line_score >= max(0.88, low_confidence_threshold)
                    and word_profile is not None
                    and word_profile.spellchecker is not None
                    and word_profile.spellchecker.issue_for(clean_text) is None
                    and not has_severe_anomaly
                ):
                    continue

                # 3. Skip high-confidence line-break hyphenated words without severe anomalies
                is_hyphenated_break = (
                    clean_text.endswith(("-", "–", "—"))
                    and len(clean_text) >= 3
                    and is_alpha_word(clean_text.rstrip("-–—"))
                )
                if (
                    line_score is not None
                    and line_score >= max(0.88, low_confidence_threshold)
                    and is_hyphenated_break
                    and not has_severe_anomaly
                ):
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
    """Collect fallback line candidates when suspicious text lacks word boxes."""
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
    """Return whether a line has usable word boxes for word-level VL checks."""
    text_word_boxes = ocr_payload.get("text_word_boxes") or []
    if line_index >= len(text_word_boxes):
        return False

    boxes = text_word_boxes[line_index]
    if boxes is None or len(boxes) == 0:
        return False

    for box in boxes:
        if valid_bbox(geometry_to_bbox(box)):
            return True
    return False


def line_text_tokens(text: str) -> list[LineTextToken]:
    """Split raw line text into tokens while preserving source spans."""
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
    batch_size: int = 1,
) -> list[list[WordCandidate]]:
    """Group suspicious candidates into word or line batches for VL requests."""
    if batch_size <= 1:
        return [[candidate] for candidate in candidates]

    batches: list[list[WordCandidate]] = []
    current_word_batch: list[WordCandidate] = []

    for candidate in candidates:
        if candidate.source == "line_text":
            if current_word_batch:
                batches.append(current_word_batch)
                current_word_batch = []
            batches.append([candidate])
        else:
            current_word_batch.append(candidate)
            if len(current_word_batch) >= batch_size:
                batches.append(current_word_batch)
                current_word_batch = []

    if current_word_batch:
        batches.append(current_word_batch)

    return batches


def should_process_batch_as_line(batch: list[WordCandidate]) -> bool:
    """Return whether a candidate batch should be checked as one line image."""
    return any(candidate.source == "line_text" for candidate in batch)


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
    """Run one grouped line batch through the configured VL backend."""
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

    error = "Grouped line correction is disabled for the PaddleVL-only pipeline."
    for record in records:
        record["crop_box"] = crop_box
        if tile_path is not None:
            record["tile_path"] = str(tile_path)
        record["backend_results"] = [{"backend": "paddlevl", "error": error}]
        record["selected_text"] = None
        record["reason"] = "line_group_disabled"
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
    """Run a fallback line-text batch when word geometry is unavailable."""
    record = candidate_record(candidate)
    record["crop_type"] = "line"
    record["line_group_size"] = 1
    record["crop_box"] = crop_box
    if tile_path is not None:
        record["tile_path"] = str(tile_path)

    results = []
    for backend in [DEFAULT_POST_OCR_VL_BACKEND]:
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


def process_word_batch(
    image: Image.Image,
    ocr_payload: dict[str, Any],
    batch: list[WordCandidate],
    opt: dict,
    metadata: dict[str, Any],
    progress_path: Path,
    source_json: str,
    image_path: str,
    tile_dir: Path | None,
    suspicious_word_count: int,
    request_index: int,
    total_requests: int,
    processed_count: int,
    applied_count: int,
    saved_tile_count: int,
    session: requests.Session | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[int, int, int]:
    """Process a batch of word candidates using concurrent VLM verification."""
    backend_order = [DEFAULT_POST_OCR_VL_BACKEND]
    start_position = processed_count + 1
    current_word = ", ".join(candidate.text for candidate in batch)[:160]

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
        current_line_index=batch[0].line_index,
        current_word_index=[candidate.word_index for candidate in batch],
        current_line_score=batch[0].line_score,
        current_weirdness=max(candidate.weirdness for candidate in batch),
        last_completed_position=processed_count,
        current_vl_request=request_index,
        total_vl_requests=total_requests,
        applied_count=applied_count,
        saved_tile_count=saved_tile_count,
    )

    crop_tasks: list[tuple[WordCandidate, dict[str, Any], Image.Image]] = []
    crop_failed: list[tuple[WordCandidate, dict[str, Any]]] = []

    for candidate in batch:
        record = candidate_record(candidate)
        record["crop_type"] = "word"
        try:
            tile, crop_box = crop_word_tile(
                image,
                candidate.box,
                padding_px=metadata["padding_px"],
            )
            record["crop_box"] = crop_box

            if metadata["save_tiles"] and tile_dir is not None:
                tile_path = tile_dir / (
                    f"{candidate.line_index:04d}_{candidate.word_index:03d}_"
                    f"{safe_filename(candidate.text)}.png"
                )
                tile.save(tile_path)
                record["tile_path"] = str(tile_path)
                saved_tile_count += 1

            crop_tasks.append((candidate, record, tile))
        except Exception as exc:
            record["error"] = f"Unable to crop word tile: {exc}"
            crop_failed.append((candidate, record))

    verified_results: list[tuple[WordCandidate, dict[str, Any], list[dict[str, Any]]]] = []

    if crop_tasks:
        def _check_one(item: tuple[WordCandidate, dict[str, Any], Image.Image]):
            cand, rec, tile_obj = item
            results = []
            for backend in backend_order:
                try:
                    text = run_backend_check(
                        backend,
                        tile_obj,
                        cand.text,
                        opt,
                        session=session,
                    )
                    results.append({"backend": backend, "text": text})
                except Exception as exc:
                    results.append({"backend": backend, "error": str(exc)})
            return cand, rec, results

        if executor is not None and len(crop_tasks) > 1:
            verified_results = list(executor.map(_check_one, crop_tasks))
        else:
            verified_results = [_check_one(item) for item in crop_tasks]

    all_outcomes: list[tuple[WordCandidate, dict[str, Any], list[dict[str, Any]]]] = []
    for cand, rec in crop_failed:
        all_outcomes.append((cand, rec, [{"backend": "paddlevl", "error": rec.get("error")}]))
    all_outcomes.extend(verified_results)

    cand_index_map = {id(c): idx for idx, c in enumerate(batch)}
    all_outcomes.sort(key=lambda item: cand_index_map.get(id(item[0]), 0))

    last_record = None
    for candidate, record, results in all_outcomes:
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
        last_record = record

    if last_record is not None:
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
            current_word=batch[-1].text,
            current_line_index=batch[-1].line_index,
            current_word_index=batch[-1].word_index,
            selected_text=last_record.get("selected_text"),
            correction_reason=last_record.get("reason"),
            last_completed_position=processed_count,
            current_vl_request=request_index,
            total_vl_requests=total_requests,
            applied_count=applied_count,
            saved_tile_count=saved_tile_count,
        )

    return processed_count, applied_count, saved_tile_count


def is_word_token(text: str) -> bool:
    """Return whether a token looks like a natural-language word."""
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
    """Return whether a token is suitable for OCR weirdness scoring."""
    stripped = text.strip()
    if not stripped:
        return False
    normalized = stripped.strip(".,:;!?()[]{}\"'")
    return bool(normalized) and bool(re.search(r"[A-Za-z0-9]", normalized))


def detect_ocr_language(ocr_payload: dict[str, Any]) -> tuple[str | None, str]:
    """Infer the dominant OCR language for spellchecking and candidate ranking."""
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
    """Extract an explicitly configured OCR language code from the payload."""
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
    """Normalize a language name or code into a compact ISO-style code."""
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
    """Build a representative OCR text sample for language detection."""
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
    """Build a spellchecker adapter for a detected OCR language when supported."""
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
    """Build token frequency and confidence context used for candidate scoring."""
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
    """Normalize a token for OCR comparison and spellcheck heuristics."""
    value = str(text or "").strip()
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    value = value.strip(WORD_STRIP_CHARS)
    value = re.sub(r"'{2,}", "'", value)
    return value.casefold()


def is_alpha_word(token: str) -> bool:
    """Return whether a token is composed only of letters."""
    return bool(WORDISH_RE.fullmatch(token))


def is_training_word(token: str) -> bool:
    """Return whether a token is useful for local common-word statistics."""
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
    """Return a simple percentile estimate for a numeric sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def group_common_words_by_length(token_counts: Counter[str]) -> dict[int, list[str]]:
    """Group the most common OCR words by length for near-match checks."""
    words_by_length: dict[int, list[str]] = {}
    for token, count in token_counts.items():
        if count < 2 or not is_alpha_word(token) or len(token) < 3:
            continue
        words_by_length.setdefault(len(token), []).append(token)
    return words_by_length


def levenshtein_limited(left: str, right: str, limit: int) -> int:
    """Compute Levenshtein distance with an early-exit limit."""
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
    """Score a candidate token for how likely it is to be OCR noise."""
    return score_word_weirdness(text, line_score, low_confidence_threshold, None)[0]


def score_word_weirdness(
    text: str,
    line_score: float | None,
    low_confidence_threshold: float,
    word_profile: WordProfile | None,
    check_subtokens: bool = True,
) -> tuple[float, list[str]]:
    """Combine heuristics that estimate whether a token deserves VL review."""
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
    """Score each side of a hyphenated token and return the strongest signal."""
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
    """Return whether a token looks like merged OCR text with punctuation inside."""
    parts = re.findall(r"[A-Za-z0-9]+", text)
    return len(parts) > 1 and bool(re.search(r"\s", text.strip()))


def has_repeated_punctuation(text: str) -> bool:
    """Return whether a token contains repeated punctuation artifacts."""
    return bool(re.search(r"\.{2,}|,{2,}|;{2,}|:{2,}|!{2,}|\?{2,}|--+", text))


def has_odd_internal_casing(text: str) -> bool:
    """Return whether a token has suspicious mixed casing inside the word."""
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
    """Return whether a token is suspiciously close to a common OCR word."""
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
    """Return whether a token should be considered a spellcheck miss."""
    if not is_alpha_word(token):
        return False
    if raw_text.endswith(("-", "–", "—")):
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
    """Return the weight contributed by a spellcheck issue for one token."""
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
    """Return whether a titlecase token is a conservative spellcheck miss."""
    if word_profile is None:
        return False
    if len(token) < 7:
        return False
    if word_profile.token_counts.get(token, 0) > 2:
        return False
    return True


def is_legitimate_letter_digit_token(text: str) -> bool:
    """Return whether an alphanumeric token is likely a legitimate code or label."""
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
    """Crop an image tile around one suspicious OCR word candidate."""
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
    """Crop an image tile that covers a whole suspicious line batch."""
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
    """Compute a shared bounding box that covers a candidate line batch."""
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
    """Return whether a bounding box is present and geometrically valid."""
    return bool(box and len(box) == 4 and box[2] > box[0] and box[3] > box[1])


def union_bbox(boxes: list[list[float]]) -> list[float] | None:
    """Return a bounding box that encloses all supplied boxes."""
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
    """Upscale very small cropped tiles before sending them to VL backends."""
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
    session: requests.Session | None = None,
) -> str:
    """Dispatch one cropped tile to the selected post-OCR VL backend."""
    if backend == "paddlevl":
        return request_paddlevl_word(tile, original_text, opt, session=session)
    raise ValueError(f"Unknown post-OCR VL backend: {backend}")


def request_paddlevl_word(
    tile: Image.Image,
    original_text: str,
    opt: dict,
    session: requests.Session | None = None,
) -> str:
    """Request a word-level correction from the PaddleOCR-VL backend."""
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
    client = session if session is not None else requests
    response = client.post(
        f"{base_url}/layout-parsing",
        json=payload,
        timeout=(10, 120),
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    raw_text = extract_paddlevl_text(response.json())
    # Match the OpenAI/Ollama replacement flow by extracting the best candidate
    # for the original OCR token instead of accepting raw nearby context.
    return clean_vl_text(
        _extract_best_word_candidate(raw_text, original_text, minimum_score=0.45)
    )


def extract_paddlevl_text(data: Any) -> str:
    """Extract the best text string from a PaddleOCR-VL response payload."""
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
        text = block.get("block_content") or block.get("text") or block.get("markdown")
        if text:
            texts.append(str(text))

    return " ".join(texts)


def choose_correction(
    original_text: str,
    results: list[dict[str, Any]],
) -> tuple[str | None, str]:
    """Choose an accepted correction string from one backend response."""
    usable = [
        restore_missing_spacing(original_text, clean_vl_text(result.get("text", "")))
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
    """Return whether a proposed correction is safe to apply automatically."""
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
    """Apply one accepted correction to the OCR payload in place."""
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
    """Apply a text-span replacement to a fallback line-level OCR string."""
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
    """Find the character span to replace inside a line-level OCR string."""
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
    """Apply a word-level correction to the OCR payload arrays."""
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
    """Serialize a word candidate into progress/reporting metadata."""
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
    """Return the sidecar progress file path for a post-OCR VL run."""
    return json_path.with_name(f"{json_path.stem}.post_ocr_vl_progress.json")


def write_post_ocr_vl_progress(
    progress_path: Path,
    metadata: dict[str, Any],
    **updates: Any,
) -> None:
    """Write a human-readable post-OCR VL progress report beside the JSON."""
    payload = {
        "status": updates.pop("status", "running"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "backend": metadata.get("backend"),
        "candidate_mode": metadata.get("candidate_mode"),
        "max_words_enabled": metadata.get("max_words_enabled"),
        "max_words": metadata.get("max_words"),
        "save_tiles": metadata.get("save_tiles"),
        "concurrency": metadata.get("concurrency"),
        "batch_size": metadata.get("batch_size"),
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


def _extract_best_word_candidate(text: str, original_text: str, minimum_score: float = 0.55) -> str:
    """Extract the best replacement token from a noisy free-form VL response."""
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


def extract_line_fallback_candidate_text(
    text: str,
    original_text: str,
    backend: str,
    opt: dict[str, Any],
) -> str:
    """Extract a fallback token value from a line-text correction candidate."""
    value = str(text or "").strip()
    if not value:
        return ""

    return clean_vl_text(
        _extract_best_word_candidate(value, original_text, minimum_score=0.45)
    )


def clean_vl_text(text: Any, model_name: str = None) -> str:
    """Normalize raw VL backend text before comparing or applying it."""
    value = str(text or "").strip()
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
    """Encode a PIL image as a base64 JPEG payload for API requests."""
    buffer = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resolve_image_path(payload: dict[str, Any], json_path: Path) -> Path | None:
    """Resolve the source image path associated with an OCR JSON payload."""
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


def geometry_to_bbox(geometry: Any) -> list[float] | None:
    """Normalize stored OCR geometry into a plain bounding box list."""
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
    """Normalize text for lenient equality checks between OCR variants."""
    return re.sub(r"\W+", "", value.casefold())


def normalize_for_acceptance(value: str) -> str:
    """Normalize text for stricter acceptance comparisons."""
    return " ".join(
        match.group(0).casefold()
        for match in OCR_WORDISH_RE.finditer(str(value or ""))
    )


def restore_missing_spacing(original_text: str, corrected_text: str) -> str:
    """Restore spaces that a backend may have dropped from a correction."""
    original = str(original_text or "").strip()
    corrected = str(corrected_text or "").strip()
    if not original or not corrected:
        return corrected
    if " " in original or " " in corrected:
        return corrected

    original_fold = original.casefold()
    corrected_fold = corrected.casefold()
    if original_fold == corrected_fold or len(corrected_fold) < 2:
        return corrected

    match_index = original_fold.find(corrected_fold)
    if match_index == -1:
        return corrected
    if original_fold.find(corrected_fold, match_index + 1) != -1:
        return corrected

    prefix = original[:match_index]
    suffix = original[match_index + len(corrected) :]
    if not prefix and not suffix:
        return corrected

    def _attached_word(part: str) -> str:
        candidate = part.strip()
        if not candidate:
            return ""
        if not re.fullmatch(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*", candidate):
            return ""
        return candidate

    prefix_word = _attached_word(prefix)
    suffix_word = _attached_word(suffix)
    if prefix and not prefix_word:
        return corrected
    if suffix and not suffix_word:
        return corrected

    opposite_word = prefix_word or suffix_word
    corrected_key = corrected.casefold()
    opposite_key = opposite_word.casefold() if opposite_word else ""
    split_start = match_index
    split_end = match_index + len(corrected)

    has_case_boundary = False
    if 0 < split_start < len(original):
        left = original[split_start - 1]
        right = original[split_start]
        has_case_boundary = left.isalpha() and right.isupper()
    if not has_case_boundary and 0 < split_end < len(original):
        left = original[split_end - 1]
        right = original[split_end]
        has_case_boundary = left.isalpha() and right.isupper()

    sentence_lead_split = (
        bool(prefix_word)
        and not suffix_word
        and len(prefix_word) >= 4
        and len(corrected) >= 4
        and prefix_word[:1].isupper()
        and prefix_word[1:].islower()
        and corrected.islower()
    )

    should_restore = (
        has_case_boundary
        or (
            opposite_key in JOINER_WORDS
            and (len(corrected_key) >= 4 or (opposite_key == "a" and len(corrected_key) >= 3))
        )
        or (
            corrected_key in JOINER_WORDS
            and len(opposite_key) >= 4
        )
        or sentence_lead_split
    )
    if not should_restore:
        return corrected

    parts = [part for part in (prefix_word, corrected, suffix_word) if part]
    if len(parts) <= 1:
        return corrected
    return " ".join(parts)


def corrections_equivalent(left: str, right: str) -> bool:
    """Return whether two correction strings are equivalent after normalization."""
    return normalize_for_acceptance(left) == normalize_for_acceptance(right)


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename fragment derived from free text."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return (cleaned or "word")[:40]


def parse_int(value: Any, default: int, minimum: int | None = None) -> int:
    """Parse an integer option with a fallback and optional lower bound."""
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def parse_float(value: Any, default: float) -> float:
    """Parse a float option with a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_int(value: Any) -> int | None:
    """Parse an optional integer value and return `None` when unset."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> float | None:
    """Parse an optional float value and return `None` when unset."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with the repo standard UTF-8 formatting."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
