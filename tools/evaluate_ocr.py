"""Evaluate OCR accuracy (CER, WER, Bow-F1) against Ground Truth reference text.

Designed for Canadian historical collections (Canadiana, microforms, bilingual EN/FR,
historical newspapers) with support for ALTO XML, OCR JSON, PDF, and plain text.
Includes optional pre-vs-post VLM delta tracking using .post_ocr_vl.csv sidecars.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, List, Tuple
import xml.etree.ElementTree as ET

# Historical Canadian / typographic normalization mappings
LIGATURE_MAP = {
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
    "ſ": "s",  # archaic long-s ubiquitous in pre-1850s Canadian print
}

UNWANTED_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015]")
QUOTES = re.compile(r"[\u2018\u2019\u201a\u201b\u2032\u2035]")
DOUBLE_QUOTES = re.compile(r"[\u201c\u201d\u201e\u201f\u2033\u2036«»]")


@dataclass
class ErrorMetrics:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    total_ref: int = 0

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.total_ref == 0:
            return 0.0 if self.total_errors == 0 else 1.0
        return self.total_errors / self.total_ref

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.error_rate)


@dataclass
class PageEvaluation:
    name: str
    ref_length_chars: int = 0
    hyp_length_chars: int = 0
    ref_length_words: int = 0
    hyp_length_words: int = 0
    cer: ErrorMetrics = field(default_factory=ErrorMetrics)
    wer: ErrorMetrics = field(default_factory=ErrorMetrics)
    bow_f1: float = 0.0
    pre_vlm_cer: float | None = None
    pre_vlm_wer: float | None = None
    delta_cer: float | None = None
    delta_wer: float | None = None
    top_char_confusions: list[tuple[str, str, int]] = field(default_factory=list)


def normalize_historical_text(text: str) -> str:
    """Normalize historical typography, ligatures, and punctuation conventions."""
    if not text:
        return ""

    text = UNWANTED_CHARS.sub("", text)
    # Reconnect trailing hyphenations
    text = re.sub(r"(\w+)[-\u2010]\s*[\r\n]+\s*(\w+)", r"\1\2", text)

    for lig, replacement in LIGATURE_MAP.items():
        text = text.replace(lig, replacement)

    text = DASHES.sub("-", text)
    text = QUOTES.sub("'", text)
    text = DOUBLE_QUOTES.sub('"', text)
    return text


def clean_text_for_eval(text: str, mode: str = "standard") -> str:
    """Prepare text for comparison according to chosen normalization mode."""
    if not text:
        return ""

    if mode == "none":
        return text.strip()

    text = normalize_historical_text(text)

    if mode == "standard":
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    if mode == "alpha":
        text = text.casefold()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return text.strip()


def calculate_sequence_errors(
    ref_seq: list[Any] | str,
    hyp_seq: list[Any] | str,
    record_confusions: bool = False,
) -> tuple[ErrorMetrics, list[tuple[str, str, int]]]:
    """Compute insertions, deletions, substitutions via SequenceMatcher."""
    matcher = SequenceMatcher(None, ref_seq, hyp_seq, autojunk=False)
    metrics = ErrorMetrics(total_ref=len(ref_seq))
    confusions: Counter[tuple[str, str]] = Counter()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            metrics.hits += i2 - i1
        elif tag == "replace":
            ref_len = i2 - i1
            hyp_len = j2 - j1
            subs = min(ref_len, hyp_len)
            metrics.substitutions += subs
            if ref_len > hyp_len:
                metrics.deletions += ref_len - hyp_len
            elif hyp_len > ref_len:
                metrics.insertions += hyp_len - ref_len

            if record_confusions:
                # Pair characters up to substitution length
                for idx in range(subs):
                    confusions[(str(ref_seq[i1 + idx]), str(hyp_seq[j1 + idx]))] += 1
                if ref_len > hyp_len:
                    for idx in range(subs, ref_len):
                        confusions[(str(ref_seq[i1 + idx]), "<del>")] += 1
                elif hyp_len > ref_len:
                    for idx in range(subs, hyp_len):
                        confusions[("<ins>", str(hyp_seq[j1 + idx]))] += 1
        elif tag == "delete":
            metrics.deletions += i2 - i1
            if record_confusions:
                for idx in range(i1, i2):
                    confusions[(str(ref_seq[idx]), "<del>")] += 1
        elif tag == "insert":
            metrics.insertions += j2 - j1
            if record_confusions:
                for idx in range(j1, j2):
                    confusions[("<ins>", str(hyp_seq[idx]))] += 1

    top_confusions = [
        (c[0], c[1], count)
        for c, count in confusions.most_common(10)
    ]
    return metrics, top_confusions


def compute_bow_f1(ref_words: list[str], hyp_words: list[str]) -> float:
    """Calculate Bag-of-Words F1 score between reference and hypothesis tokens."""
    ref_counts = Counter(ref_words)
    hyp_counts = Counter(hyp_words)
    overlap = sum((ref_counts & hyp_counts).values())
    total_ref = len(ref_words)
    total_hyp = len(hyp_words)

    if total_ref == 0 or total_hyp == 0:
        return 1.0 if total_ref == total_hyp else 0.0

    precision = overlap / total_hyp
    recall = overlap / total_ref
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def extract_text_from_alto(xml_path: Path) -> str:
    """Extract ordered text lines from an ALTO XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = []
        for line_elem in root.iter():
            if line_elem.tag.endswith("TextLine"):
                line_words = []
                for child in line_elem:
                    if child.tag.endswith("String"):
                        content = child.attrib.get("CONTENT", "")
                        subs = child.attrib.get("SUBS_CONTENT")
                        part = child.attrib.get("SUBS_TYPE")
                        if subs and part == "HypPart1":
                            content = subs
                        elif subs and part == "HypPart2":
                            continue
                        if content:
                            line_words.append(content)
                    elif child.tag.endswith("SP"):
                        continue
                if line_words:
                    lines.append(" ".join(line_words))
        return "\n".join(lines)
    except Exception as exc:
        raise ValueError(f"Error reading ALTO XML {xml_path}: {exc}") from exc


def extract_text_from_json(json_path: Path) -> str:
    """Extract OCR transcript from normalized pipeline JSON or PaddleX output."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "rec_texts" in data and isinstance(data["rec_texts"], list):
            return "\n".join(str(t) for t in data["rec_texts"] if t)
        if "overall_ocr_res" in data and "rec_texts" in data["overall_ocr_res"]:
            return "\n".join(str(t) for t in data["overall_ocr_res"]["rec_texts"] if t)
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        if "markdown" in data and isinstance(data["markdown"], str):
            return data["markdown"]
        return json.dumps(data)
    except Exception as exc:
        raise ValueError(f"Error reading OCR JSON {json_path}: {exc}") from exc


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract visible text layer from searchable PDF using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text_parts = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(text_parts)
    except Exception as exc:
        raise ValueError(f"Error reading PDF {pdf_path}: {exc}") from exc


def load_document_text(path: Path) -> str:
    """Load text content from a text file, ALTO XML, OCR JSON, or PDF."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".gt", ".transcript"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".xml":
        return extract_text_from_alto(path)
    if suffix == ".json":
        return extract_text_from_json(path)
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def rollback_post_ocr_vl(json_path: Path, csv_path: Path) -> str | None:
    """Reconstruct pre-VLM OCR text by rolling back applied corrections in CSV."""
    if not json_path.exists() or not csv_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        rec_texts = (
            data.get("rec_texts")
            or (data.get("overall_ocr_res") or {}).get("rec_texts")
        )
        if not rec_texts or not isinstance(rec_texts, list):
            return None

        lines = list(rec_texts)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("applied")).lower() == "true":
                    line_idx = int(row["line_index"])
                    orig = row.get("original_text", "")
                    vl = row.get("vl_text", "")
                    if line_idx < len(lines) and vl and orig:
                        lines[line_idx] = lines[line_idx].replace(vl, orig, 1)

        return "\n".join(lines)
    except Exception:
        return None


def evaluate_page(
    name: str,
    ref_text_raw: str,
    hyp_text_raw: str,
    mode: str = "standard",
    pre_vlm_text_raw: str | None = None,
) -> PageEvaluation:
    """Evaluate Character and Word Error Rates for a single document."""
    ref_cleaned = clean_text_for_eval(ref_text_raw, mode=mode)
    hyp_cleaned = clean_text_for_eval(hyp_text_raw, mode=mode)

    ref_words = [w for w in ref_cleaned.split() if w]
    hyp_words = [w for w in hyp_cleaned.split() if w]

    cer_metrics, confusions = calculate_sequence_errors(
        ref_cleaned, hyp_cleaned, record_confusions=True
    )
    wer_metrics, _ = calculate_sequence_errors(
        ref_words, hyp_words, record_confusions=False
    )
    bow_f1 = compute_bow_f1(ref_words, hyp_words)

    eval_record = PageEvaluation(
        name=name,
        ref_length_chars=len(ref_cleaned),
        hyp_length_chars=len(hyp_cleaned),
        ref_length_words=len(ref_words),
        hyp_length_words=len(hyp_words),
        cer=cer_metrics,
        wer=wer_metrics,
        bow_f1=bow_f1,
        top_char_confusions=confusions,
    )

    if pre_vlm_text_raw:
        pre_cleaned = clean_text_for_eval(pre_vlm_text_raw, mode=mode)
        pre_words = [w for w in pre_cleaned.split() if w]
        pre_cer_m, _ = calculate_sequence_errors(ref_cleaned, pre_cleaned)
        pre_wer_m, _ = calculate_sequence_errors(ref_words, pre_words)
        eval_record.pre_vlm_cer = pre_cer_m.error_rate
        eval_record.pre_vlm_wer = pre_wer_m.error_rate
        eval_record.delta_cer = eval_record.cer.error_rate - eval_record.pre_vlm_cer
        eval_record.delta_wer = eval_record.wer.error_rate - eval_record.pre_vlm_wer

    return eval_record


def find_matched_pairs(
    ref_dir_or_file: Path,
    hyp_dir_or_file: Path,
) -> list[tuple[Path, Path]]:
    """Match reference ground-truth files with hypothesis outputs."""
    if ref_dir_or_file.is_file() and hyp_dir_or_file.is_file():
        return [(ref_dir_or_file, hyp_dir_or_file)]

    if not ref_dir_or_file.is_dir() or not hyp_dir_or_file.is_dir():
        raise ValueError("Both reference and hypothesis must be files or directories.")

    hyp_files = list(hyp_dir_or_file.rglob("*"))
    hyp_map = {}
    for h in hyp_files:
        if h.is_file() and h.suffix.lower() in {".txt", ".json", ".xml", ".pdf"}:
            stem = h.stem
            # Strip sub-extensions like .post_ocr_vl or .gt
            stem = stem.replace(".post_ocr_vl", "")
            hyp_map.setdefault(stem, []).append(h)

    pairs = []
    for ref in ref_dir_or_file.rglob("*"):
        if not ref.is_file() or ref.suffix.lower() not in {".txt", ".xml", ".gt"}:
            continue
        ref_stem = ref.stem.replace(".gt", "")
        if ref_stem in hyp_map:
            # Prefer .json > .xml > .txt > .pdf
            candidates = sorted(
                hyp_map[ref_stem],
                key=lambda p: [".json", ".xml", ".txt", ".pdf"].index(p.suffix.lower())
                if p.suffix.lower() in [".json", ".xml", ".txt", ".pdf"]
                else 99,
            )
            pairs.append((ref, candidates[0]))

    return pairs


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Render a GitHub-flavored Markdown / ASCII summary table."""
    all_rows = [headers] + rows
    col_widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
    
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-|-".join("-" * w for w in col_widths)
    data_lines = [
        " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths))
        for row in rows
    ]
    return f"| {header_line} |\n| {sep_line} |\n" + "\n".join(f"| {l} |" for l in data_lines)


def run_evaluation(
    ref_path: Path,
    hyp_path: Path,
    mode: str = "standard",
    check_vlm_delta: bool = True,
) -> tuple[list[PageEvaluation], dict[str, Any]]:
    """Execute the evaluation pipeline across paired files."""
    pairs = find_matched_pairs(ref_path, hyp_path)
    if not pairs:
        print(f"No matching file pairs found between {ref_path} and {hyp_path}.")
        return [], {}

    results: list[PageEvaluation] = []
    for ref_file, hyp_file in pairs:
        ref_text = load_document_text(ref_file)
        hyp_text = load_document_text(hyp_file)

        pre_vlm_text = None
        if check_vlm_delta and hyp_file.suffix.lower() == ".json":
            csv_candidate = hyp_file.with_name(f"{hyp_file.stem}.post_ocr_vl.csv")
            if csv_candidate.exists():
                pre_vlm_text = rollback_post_ocr_vl(hyp_file, csv_candidate)

        page_eval = evaluate_page(
            name=hyp_file.stem,
            ref_text_raw=ref_text,
            hyp_text_raw=hyp_text,
            mode=mode,
            pre_vlm_text_raw=pre_vlm_text,
        )
        results.append(page_eval)

    # Compute aggregate micro metrics
    total_ref_chars = sum(r.cer.total_ref for r in results)
    total_char_errs = sum(r.cer.total_errors for r in results)
    micro_cer = (total_char_errs / total_ref_chars) if total_ref_chars > 0 else 0.0

    total_ref_words = sum(r.wer.total_ref for r in results)
    total_word_errs = sum(r.wer.total_errors for r in results)
    micro_wer = (total_word_errs / total_ref_words) if total_ref_words > 0 else 0.0

    macro_cer = sum(r.cer.error_rate for r in results) / len(results)
    macro_wer = sum(r.wer.error_rate for r in results) / len(results)
    macro_f1 = sum(r.bow_f1 for r in results) / len(results)

    summary = {
        "pages_evaluated": len(results),
        "normalization_mode": mode,
        "micro_cer": micro_cer,
        "micro_wer": micro_wer,
        "macro_cer": macro_cer,
        "macro_wer": macro_wer,
        "macro_bow_f1": macro_f1,
        "char_accuracy": max(0.0, 1.0 - micro_cer),
        "word_accuracy": max(0.0, 1.0 - micro_wer),
    }

    # Aggregate VLM delta if available
    vlm_pages = [r for r in results if r.delta_cer is not None]
    if vlm_pages:
        avg_delta_cer = sum(r.delta_cer for r in vlm_pages) / len(vlm_pages)
        avg_delta_wer = sum(r.delta_wer for r in vlm_pages) / len(vlm_pages)
        summary["vlm_pages_evaluated"] = len(vlm_pages)
        summary["avg_delta_cer"] = avg_delta_cer
        summary["avg_delta_wer"] = avg_delta_wer
        summary["vlm_improved_cer"] = sum(1 for r in vlm_pages if r.delta_cer < 0)
        summary["vlm_neutral_cer"] = sum(1 for r in vlm_pages if r.delta_cer == 0)
        summary["vlm_degraded_cer"] = sum(1 for r in vlm_pages if r.delta_cer > 0)

    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR transcription accuracy against Ground Truth (CER/WER)."
    )
    parser.add_argument(
        "--ref",
        "-r",
        required=True,
        help="Reference ground-truth file or directory (.txt, .xml, .gt).",
    )
    parser.add_argument(
        "--hyp",
        "-i",
        required=True,
        help="Hypothesis OCR output file or directory (.txt, .json, .xml, .pdf).",
    )
    parser.add_argument(
        "--norm",
        choices=["standard", "raw", "alpha"],
        default="standard",
        help="Text normalization mode (default: standard with historical typography handling).",
    )
    parser.add_argument(
        "--no-vlm-delta",
        action="store_true",
        help="Skip pre-vs-post VLM delta calculation.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to write per-document evaluation metrics CSV.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write full evaluation report JSON.",
    )

    args = parser.parse_args()
    ref_path = Path(args.ref).resolve()
    hyp_path = Path(args.hyp).resolve()

    if not ref_path.exists():
        print(f"Reference path does not exist: {ref_path}", file=sys.stderr)
        return 1
    if not hyp_path.exists():
        print(f"Hypothesis path does not exist: {hyp_path}", file=sys.stderr)
        return 1

    results, summary = run_evaluation(
        ref_path=ref_path,
        hyp_path=hyp_path,
        mode=args.norm,
        check_vlm_delta=not args.no_vlm_delta,
    )
    if not results:
        return 1

    # Print summary table
    table_headers = [
        "Document",
        "Ref Chars",
        "Hyp Chars",
        "CER %",
        "WER %",
        "BoW F1 %",
    ]
    if any(r.delta_cer is not None for r in results):
        table_headers.extend(["Δ CER %", "Δ WER %"])

    table_rows = []
    for r in results:
        row = [
            r.name[:32],
            str(r.ref_length_chars),
            str(r.hyp_length_chars),
            f"{r.cer.error_rate * 100:.2f}%",
            f"{r.wer.error_rate * 100:.2f}%",
            f"{r.bow_f1 * 100:.2f}%",
        ]
        if any(res.delta_cer is not None for res in results):
            d_cer = f"{r.delta_cer * 100:+.2f}%" if r.delta_cer is not None else "N/A"
            d_wer = f"{r.delta_wer * 100:+.2f}%" if r.delta_wer is not None else "N/A"
            row.extend([d_cer, d_wer])
        table_rows.append(row)

    print("\n" + format_table(table_rows, table_headers))
    print("\n--- Summary Metrics ---")
    print(f"Pages Evaluated:   {summary['pages_evaluated']}")
    print(f"Normalization:     {summary['normalization_mode']}")
    print(f"Micro-CER:         {summary['micro_cer'] * 100:.2f}% (Char Acc: {summary['char_accuracy'] * 100:.2f}%)")
    print(f"Micro-WER:         {summary['micro_wer'] * 100:.2f}% (Word Acc: {summary['word_accuracy'] * 100:.2f}%)")
    print(f"Macro-BoW F1:      {summary['macro_bow_f1'] * 100:.2f}%")

    if "avg_delta_cer" in summary:
        print("\n--- Post-OCR VLM Verification Impact ---")
        print(f"Pages with VLM Audit:  {summary['vlm_pages_evaluated']}")
        print(f"Average Δ CER:         {summary['avg_delta_cer'] * 100:+.3f}%")
        print(f"Average Δ WER:         {summary['avg_delta_wer'] * 100:+.3f}%")
        print(
            f"Net CER Outcome:       {summary['vlm_improved_cer']} Improved, "
            f"{summary['vlm_neutral_cer']} Neutral, {summary['vlm_degraded_cer']} Regressed"
        )

    # Optional CSV export
    if args.output_csv:
        csv_path = Path(args.output_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "document",
                "ref_chars",
                "hyp_chars",
                "ref_words",
                "hyp_words",
                "cer",
                "wer",
                "char_accuracy",
                "word_accuracy",
                "bow_f1",
                "pre_vlm_cer",
                "pre_vlm_wer",
                "delta_cer",
                "delta_wer",
            ])
            for r in results:
                writer.writerow([
                    r.name,
                    r.ref_length_chars,
                    r.hyp_length_chars,
                    r.ref_length_words,
                    r.hyp_length_words,
                    round(r.cer.error_rate, 4),
                    round(r.wer.error_rate, 4),
                    round(r.cer.accuracy, 4),
                    round(r.wer.accuracy, 4),
                    round(r.bow_f1, 4),
                    round(r.pre_vlm_cer, 4) if r.pre_vlm_cer is not None else "",
                    round(r.pre_vlm_wer, 4) if r.pre_vlm_wer is not None else "",
                    round(r.delta_cer, 4) if r.delta_cer is not None else "",
                    round(r.delta_wer, 4) if r.delta_wer is not None else "",
                ])
        print(f"\nSaved CSV metrics to: {csv_path}")

    # Optional JSON export
    if args.output_json:
        json_path = Path(args.output_json).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        export_payload = {
            "summary": summary,
            "results": [
                {
                    "name": r.name,
                    "ref_length_chars": r.ref_length_chars,
                    "hyp_length_chars": r.hyp_length_chars,
                    "ref_length_words": r.ref_length_words,
                    "hyp_length_words": r.hyp_length_words,
                    "cer": r.cer.error_rate,
                    "wer": r.wer.error_rate,
                    "bow_f1": r.bow_f1,
                    "pre_vlm_cer": r.pre_vlm_cer,
                    "pre_vlm_wer": r.pre_vlm_wer,
                    "delta_cer": r.delta_cer,
                    "delta_wer": r.delta_wer,
                    "top_confusions": [
                        {"ref": ref_c, "hyp": hyp_c, "count": cnt}
                        for ref_c, hyp_c, cnt in r.top_char_confusions
                    ],
                }
                for r in results
            ],
        }
        json_path.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON report to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
