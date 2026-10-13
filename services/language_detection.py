"""Language and script detection services for CRKN OCR pipeline.

Provides automatic script and language detection across Canadian heritage,
newspaper, and serial collections, ensuring seamless multi-lingual processing.
Supports English, French, Italian, German, Spanish, and Traditional/Simplified Chinese.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# Collection IDs and path markers for Traditional Chinese Canadiana collections
# e.g., Simon Fraser University Chinese Times collection (sfu.00001), Dahan Gongbao, etc.
CHINESE_TRADITIONAL_MARKERS = {
    "sfu.00001",
    "sfu.00002",
    "chinese",
    "chinese_cht",
    "cht",
    "ch_tra",
    "trad_chinese",
    "traditional_chinese",
    "dahan",
    "gongbao",
    "wah_kew",
    "kuomintang",
    "shing_wah",
    "tai_sing",
    "zh-tw",
    "zh-hk",
    "zh_cht",
}

CHINESE_SIMPLIFIED_MARKERS = {
    "chinese_sim",
    "simp_chinese",
    "simplified_chinese",
    "zh-cn",
    "zh_cn",
}

# Italian Canadian newspapers and collections in Canadiana / BAnQ
# e.g., qmbn.00001 is Il Cittadino Canadese (Montreal's Italian-language newspaper)
ITALIAN_MARKERS = {
    "qmbn.00001",
    "cittadino",
    "canadese",
    "italiano",
    "italiana",
    "italiani",
    "corriere_italiano",
    "il_cittadino",
    "italian",
    "italien",
}

# German Canadian historical serials
GERMAN_MARKERS = {
    "berliner_journal",
    "deutsche",
    "german",
    "allemand",
}

FRENCH_MARKERS = {
    "francais",
    "francaise",
    "french",
    "quebec",
    "québec",
    "banq",
    "oocihm.8_06619",
    "oocihm.8_",
    "prix_courant",
    "canadiana_fr",
    "bas_canada",
    "canada_est",
    "devoir",
    "la_presse",
}

# Canadian federal bilingual records and official publications (RCMP, StatCan annuals, Parliamentary)
# Paddle's 'fr' model is a strict superset of 'en' (contains all Latin letters A-Z + all French accented vowels)
BILINGUAL_CANADIAN_MARKERS = {
    "sru.00007",
    "sru.00004",
    "rcmp",
    "grc",
    "hansard",
    "parliament",
    "statutes",
    "debates",
    "bilingual",
}


def detect_language_for_path(path: Path | str, fallback: str = "en") -> str:
    """
    Auto-detect the optimal OCR language code from path components,
    collection identifiers, and file naming conventions.

    Returns:
        Paddle language code, e.g. "it", "chinese_cht", "ch", "fr", or fallback ("en").
    """
    p_str = str(path).lower().replace("\\", "/")

    # 1. Italian Canadian collections (e.g. qmbn.00001 Il Cittadino Canadese)
    for marker in ITALIAN_MARKERS:
        if marker in p_str:
            return "it"

    # 2. Traditional Chinese collections (e.g. sfu.00001 The Chinese Times)
    for marker in CHINESE_TRADITIONAL_MARKERS:
        if marker in p_str:
            return "chinese_cht"

    # 3. Simplified Chinese collections
    for marker in CHINESE_SIMPLIFIED_MARKERS:
        if marker in p_str:
            return "ch"

    # 4. German Canadian historical serials
    for marker in GERMAN_MARKERS:
        if marker in p_str:
            return "de"

    # 5. French Canadian collections & serials
    for marker in FRENCH_MARKERS:
        if marker in p_str:
            return "fr"

    # 6. Canadian federal bilingual serials (StatCan annuals, RCMP forms, Parliamentary records)
    for marker in BILINGUAL_CANADIAN_MARKERS:
        if marker in p_str:
            return "fr"

    return fallback


def detect_document_language(
    text_or_lines: Union[str, List[str]],
    fallback: str = "en",
) -> str:
    """Detect natural language from recognized document text or line lists.

    Returns ISO language code: 'en', 'fr', 'it', 'de', 'es', 'chinese_cht', or 'bilingual'.
    """
    if isinstance(text_or_lines, list):
        text = " ".join(str(t) for t in text_or_lines if t)
    else:
        text = str(text_or_lines or "")

    text = text.strip()
    if not text:
        return fallback

    # 1. CJK detection for Chinese Canadiana serials
    cjk_count = sum(1 for ch in text if CJK_CHAR_RE.match(ch))
    if cjk_count >= 10 or (len(text) > 0 and cjk_count / len(text) > 0.15):
        return "chinese_cht"

    # 2. Extract alphabetical words for statistical n-gram language detection
    words = [w for w in text.split() if len(w) > 2 and w.isalpha()]
    sample = " ".join(words[:250])
    if len(sample) < 20:
        return fallback

    try:
        from langdetect import detect_langs
        langs = detect_langs(sample)
        if not langs:
            return fallback

        top = langs[0]

        # Check for Canadian bilingual English / French records
        lang_probs = {l.lang: l.prob for l in langs}
        if "en" in lang_probs and "fr" in lang_probs:
            if lang_probs["en"] >= 0.20 and lang_probs["fr"] >= 0.20:
                return "bilingual"

        # Check high-confidence detections
        if top.lang in {"it", "fr", "en", "de", "es", "pt"}:
            if top.prob >= 0.50:
                return top.lang

        # Return top language code
        return top.lang
    except Exception:
        return fallback


def map_lang_to_alto_code(lang: Optional[str]) -> str:
    """Map pipeline language identifier to valid ALTO XML ISO 639-1/639-2 language code."""
    if not lang:
        return "en"
    l = lang.lower().strip()
    if l in {"chinese_cht", "ch", "zh-cn", "zh-tw", "zh_cht", "zh"}:
        return "zh"
    if l in {"it", "italian", "italiano"}:
        return "it"
    if l in {"fr", "french", "francais"}:
        return "fr"
    if l in {"de", "german", "deutsch"}:
        return "de"
    if l in {"es", "spanish", "espanol"}:
        return "es"
    if l in {"bilingual"}:
        return "en"
    return l


def map_lang_to_paddle_code(lang: Optional[str]) -> str:
    """Map language identifier to supported PaddleOCR language model parameter."""
    if not lang:
        return "en"
    l = lang.lower().strip()
    if l in {"chinese_cht", "zh-tw", "zh_cht"}:
        return "chinese_cht"
    if l in {"ch", "zh-cn", "zh_cn", "zh"}:
        return "ch"
    if l in {"it", "italian", "italiano"}:
        return "it"
    if l in {"fr", "french", "bilingual"}:
        return "fr"
    if l in {"de", "german", "deutsch"}:
        return "german"
    if l in {"es", "spanish"}:
        return "es"
    return "en"


def resolve_effective_language(
    lang_setting: Optional[str],
    path: Path | str,
    text: Optional[Union[str, List[str]]] = None,
) -> str:
    """Resolve the effective language to use given the UI/config setting, document path, and optional text.

    If the setting is 'auto', None, or empty:
    1. If text is provided, attempts text-based detection.
    2. Otherwise, detects from the path.
    """
    if not lang_setting or lang_setting in {"auto", "None", "none"}:
        if text:
            detected = detect_document_language(text, fallback="")
            if detected:
                return detected
        return detect_language_for_path(path, fallback="en")
    return lang_setting
