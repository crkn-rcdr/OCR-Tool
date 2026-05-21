from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from services.ocr_defaults import (
    DEFAULT_OLMOCR_API_URL,
    DEFAULT_OLMOCR_MODEL,
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BACKEND,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_POST_OCR_VL_MAX_WORDS,
    DEFAULT_POST_OCR_VL_PADDING_PX,
    DEFAULT_POST_OCR_VL_SAVE_TILES,
)


POST_OCR_VL_METHODS = {"PP-OCRv5"}
WORD_RE = re.compile(r"[A-Za-z0-9]")
ODD_CHAR_RE = re.compile(r"[^A-Za-z0-9'\u2019`.,:;!?()\\[\\]{}\\-/& ]")


@dataclass
class WordCandidate:
    line_index: int
    word_index: int
    text: str
    box: list[float]
    line_score: float | None
    weirdness: float


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
    suspicious_word_count = len(
        collect_word_candidates(
            ocr_payload,
            mode="suspicious",
            max_words=0,
            low_confidence_threshold=metadata["low_confidence_threshold"],
        )
    )
    candidates = collect_word_candidates(
        ocr_payload,
        mode=metadata["candidate_mode"],
        max_words=0,
        low_confidence_threshold=metadata["low_confidence_threshold"],
    )
    metadata["qualified_count"] = len(candidates)
    if max_words > 0:
        candidates = candidates[:max_words]
    metadata["candidate_count"] = len(candidates)
    metadata["suspicious_word_count"] = suspicious_word_count

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

    backend_order = backends_for_request(metadata["backend"])
    applied_count = 0
    saved_tile_count = 0

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        for position, candidate in enumerate(candidates, start=1):
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
                    applied_count=applied_count,
                    saved_tile_count=saved_tile_count,
                    last_error=record["error"],
                )
                continue

            record["crop_box"] = crop_box
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
                apply_word_correction(
                    ocr_payload,
                    candidate.line_index,
                    candidate.word_index,
                    selected_text,
                )
                record["applied"] = True
                applied_count += 1
            else:
                record["applied"] = False

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
                selected_text=record["selected_text"],
                correction_reason=record["reason"],
                last_completed_position=position,
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
) -> list[WordCandidate]:
    text_word = ocr_payload.get("text_word") or []
    text_word_boxes = ocr_payload.get("text_word_boxes") or []
    rec_scores = ocr_payload.get("rec_scores") or []
    candidates: list[WordCandidate] = []
    mode = mode if mode in {"suspicious", "low_confidence", "all"} else "suspicious"

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

            weirdness = score_weirdness(text, line_score, low_confidence_threshold)
            low_confidence = (
                line_score is not None and line_score < low_confidence_threshold
            )

            if mode == "suspicious" and weirdness <= 0:
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


def score_weirdness(
    text: str,
    line_score: float | None,
    low_confidence_threshold: float,
) -> float:
    stripped = text.strip()
    normalized = stripped.strip(".,:;!?()[]{}\"'")
    score = 0.0

    if line_score is not None and line_score < low_confidence_threshold:
        score += max(0.25, (low_confidence_threshold - line_score) * 4)

    if ODD_CHAR_RE.search(stripped):
        score += 1.5
    if re.search(r"([A-Za-z])\1\1", normalized):
        score += 1.0
    if len(normalized) >= 4 and normalized.isalpha():
        if not re.search(r"[AEIOUYaeiouy]", normalized):
            score += 0.75
        if normalized.lower().startswith(("rn", "vv")):
            score += 0.25
    if re.search(r"[A-Za-z][0-9]|[0-9][A-Za-z]", normalized):
        score += 1.0
    if stripped != normalized and len(normalized) >= 3:
        score += 0.15

    return score


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


def request_olmocr_word(tile: Image.Image, original_text: str, opt: dict) -> str:
    server_url = str(
        opt.get("post_ocr_vl_olmocr_url") or DEFAULT_OLMOCR_API_URL
    ).rstrip("/")
    completion_url = (
        server_url
        if server_url.endswith("/chat/completions")
        else f"{server_url}/chat/completions"
    )
    model = str(opt.get("post_ocr_vl_olmocr_model") or DEFAULT_OLMOCR_MODEL)
    prompt = (
        "Read the single target word in this newspaper crop. "
        "Return only the corrected word, with visible punctuation only if it "
        f"is part of the word. The previous OCR guess was: {original_text!r}."
    )
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
    api_key = opt.get("post_ocr_vl_olmocr_api_key") or os.getenv("OLMOCR_API_KEY")
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
    return clean_vl_text(content)


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

    unique = {normalize_for_compare(text): text for text in usable}
    if len(results) > 1 and len(unique) > 1:
        return None, "backend_disagreement"

    selected = usable[0]
    if normalize_for_compare(selected) == normalize_for_compare(original_text):
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


def apply_word_correction(
    ocr_payload: dict[str, Any],
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
        rec_texts[line_index] = "".join(str(part) for part in words)


def candidate_record(candidate: WordCandidate) -> dict[str, Any]:
    return {
        "line_index": candidate.line_index,
        "word_index": candidate.word_index,
        "original_text": candidate.text,
        "box": candidate.box,
        "line_score": candidate.line_score,
        "weirdness": candidate.weirdness,
    }


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
        **updates,
    }
    try:
        progress_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def clean_vl_text(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    value = re.sub(r"```(?:text|markdown|yaml)?", "", value, flags=re.IGNORECASE)
    value = value.replace("```", "").strip()

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


def parse_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
