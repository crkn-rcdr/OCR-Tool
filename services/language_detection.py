"""Language and script detection services for CRKN OCR pipeline.

Provides automatic script and language detection across Canadian heritage,
newspaper, and serial collections, ensuring seamless multi-lingual processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


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

FRENCH_MARKERS = {
    "francais",
    "francaise",
    "french",
    "quebec",
    "québec",
    "banq",
    "canadiana_fr",
    "bas_canada",
    "canada_est",
}


def detect_language_for_path(path: Path | str, fallback: str = "en") -> str:
    """
    Auto-detect the optimal OCR language code from path components,
    collection identifiers, and file naming conventions.

    Returns:
        Paddle language code, e.g. "chinese_cht", "ch", "fr", or fallback ("en").
    """
    p_str = str(path).lower().replace("\\", "/")

    # 1. Traditional Chinese collections (e.g. sfu.00001 The Chinese Times)
    for marker in CHINESE_TRADITIONAL_MARKERS:
        if marker in p_str:
            return "chinese_cht"

    # 2. Simplified Chinese collections
    for marker in CHINESE_SIMPLIFIED_MARKERS:
        if marker in p_str:
            return "ch"

    # 3. French Canadian collections & serials
    for marker in FRENCH_MARKERS:
        if marker in p_str:
            return "fr"

    return fallback


def resolve_effective_language(lang_setting: Optional[str], path: Path | str) -> str:
    """
    Resolve the effective language to use given the UI/config setting and document path.
    If the setting is 'auto', None, or empty, auto-detects from the path.
    Otherwise, returns the explicitly selected language setting.
    """
    if not lang_setting or lang_setting in {"auto", "None", "none"}:
        return detect_language_for_path(path, fallback="en")
    return lang_setting

