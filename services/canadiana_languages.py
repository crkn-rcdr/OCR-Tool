"""Official Canadiana Language Taxonomy and Registry.

Encompasses all 54 languages documented in the Canadiana collections,
with document counts, ISO codes, scripts, and spellchecking policies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class CanadianaLanguage:
    name: str
    doc_count: int
    iso639_1: Optional[str]  # 2-letter code if standard
    iso639_2: str            # 3-letter bibliographic code
    script: str              # latin, cjk, hebrew, cyrillic, canadian_syllabics, greek, indic, etc.
    spellcheck_supported: bool  # True if validated dictionary available
    spellcheck_lang_code: Optional[str] = None  # Code for pyspellchecker
    alto_code: str = "en"
    paddle_code: str = "en"


# Complete census of all 54 languages in Canadiana archives
CANADIANA_LANGUAGE_CATALOG: Dict[str, CanadianaLanguage] = {
    # 1. Official & Major National Languages
    "en": CanadianaLanguage(
        name="English",
        doc_count=287983,
        iso639_1="en",
        iso639_2="eng",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="en",
        alto_code="en",
        paddle_code="en",
    ),
    "fr": CanadianaLanguage(
        name="French",
        doc_count=87877,
        iso639_1="fr",
        iso639_2="fra",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="fr",
        alto_code="fr",
        paddle_code="fr",
    ),
    "und": CanadianaLanguage(
        name="Undetermined",
        doc_count=46263,
        iso639_1=None,
        iso639_2="und",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="und",
        paddle_code="en",
    ),
    "zh": CanadianaLanguage(
        name="Chinese",
        doc_count=23006,
        iso639_1="zh",
        iso639_2="zho",
        script="cjk",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="zh",
        paddle_code="chinese_cht",
    ),
    "de": CanadianaLanguage(
        name="German",
        doc_count=5540,
        iso639_1="de",
        iso639_2="deu",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="de",
        alto_code="de",
        paddle_code="german",
    ),
    "yi": CanadianaLanguage(
        name="Yiddish",
        doc_count=5516,
        iso639_1="yi",
        iso639_2="yid",
        script="hebrew",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="yi",
        paddle_code="en",
    ),
    "uk": CanadianaLanguage(
        name="Ukrainian",
        doc_count=3245,
        iso639_1="uk",
        iso639_2="ukr",
        script="cyrillic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="uk",
        paddle_code="en",
    ),
    "it": CanadianaLanguage(
        name="Italian",
        doc_count=1747,
        iso639_1="it",
        iso639_2="ita",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="it",
        alto_code="it",
        paddle_code="it",
    ),
    "fi": CanadianaLanguage(
        name="Finnish",
        doc_count=606,
        iso639_1="fi",
        iso639_2="fin",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="fi",
        paddle_code="en",
    ),
    "la": CanadianaLanguage(
        name="Latin",
        doc_count=546,
        iso639_1="la",
        iso639_2="lat",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="la",
        paddle_code="en",
    ),
    "is": CanadianaLanguage(
        name="Icelandic",
        doc_count=459,
        iso639_1="is",
        iso639_2="isl",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="is",
        paddle_code="en",
    ),
    "hu": CanadianaLanguage(
        name="Hungarian",
        doc_count=383,
        iso639_1="hu",
        iso639_2="hun",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="hu",
        paddle_code="en",
    ),
    # Indigenous Languages of Canada
    "cr": CanadianaLanguage(
        name="Cree",
        doc_count=257,
        iso639_1="cr",
        iso639_2="cre",
        script="canadian_syllabics",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="cr",
        paddle_code="en",
    ),
    "chn": CanadianaLanguage(
        name="Chinook jargon",
        doc_count=210,
        iso639_1=None,
        iso639_2="chn",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="chn",
        paddle_code="en",
    ),
    "nl": CanadianaLanguage(
        name="Dutch",
        doc_count=108,
        iso639_1="nl",
        iso639_2="nld",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="nl",
        alto_code="nl",
        paddle_code="en",
    ),
    "mic": CanadianaLanguage(
        name="Mi'kmaq",
        doc_count=86,
        iso639_1=None,
        iso639_2="mic",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="mic",
        paddle_code="en",
    ),
    "oj": CanadianaLanguage(
        name="Ojibwa",
        doc_count=80,
        iso639_1="oj",
        iso639_2="oji",
        script="canadian_syllabics",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="oj",
        paddle_code="en",
    ),
    "es": CanadianaLanguage(
        name="Spanish",
        doc_count=76,
        iso639_1="es",
        iso639_2="spa",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="es",
        alto_code="es",
        paddle_code="es",
    ),
    "moh": CanadianaLanguage(
        name="Mohawk",
        doc_count=50,
        iso639_1=None,
        iso639_2="moh",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="moh",
        paddle_code="en",
    ),
    "sv": CanadianaLanguage(
        name="Swedish",
        doc_count=32,
        iso639_1="sv",
        iso639_2="swe",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="sv",
        paddle_code="en",
    ),
    "el": CanadianaLanguage(
        name="Greek, Modern (1453-)",
        doc_count=31,
        iso639_1="el",
        iso639_2="ell",
        script="greek",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="el",
        paddle_code="en",
    ),
    "alg": CanadianaLanguage(
        name="Algonquian languages",
        doc_count=30,
        iso639_1=None,
        iso639_2="alg",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="alg",
        paddle_code="en",
    ),
    "ru": CanadianaLanguage(
        name="Russian",
        doc_count=28,
        iso639_1="ru",
        iso639_2="rus",
        script="cyrillic",
        spellcheck_supported=True,
        spellcheck_lang_code="ru",
        alto_code="ru",
        paddle_code="en",
    ),
    "gd": CanadianaLanguage(
        name="Gaelic",
        doc_count=23,
        iso639_1="gd",
        iso639_2="gla",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="gd",
        paddle_code="en",
    ),
    "iu": CanadianaLanguage(
        name="Inuktitut",
        doc_count=20,
        iso639_1="iu",
        iso639_2="iku",
        script="canadian_syllabics",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="iu",
        paddle_code="en",
    ),
    "da": CanadianaLanguage(
        name="Danish",
        doc_count=17,
        iso639_1="da",
        iso639_2="dan",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="da",
        paddle_code="en",
    ),
    "iro": CanadianaLanguage(
        name="Iroquoian languages",
        doc_count=13,
        iso639_1=None,
        iso639_2="iro",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="iro",
        paddle_code="en",
    ),
    "no": CanadianaLanguage(
        name="Norwegian",
        doc_count=13,
        iso639_1="no",
        iso639_2="nor",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="no",
        paddle_code="en",
    ),
    "he": CanadianaLanguage(
        name="Hebrew",
        doc_count=12,
        iso639_1="he",
        iso639_2="heb",
        script="hebrew",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="he",
        paddle_code="en",
    ),
    "tsi": CanadianaLanguage(
        name="Tsimshian",
        doc_count=11,
        iso639_1=None,
        iso639_2="tsi",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="tsi",
        paddle_code="en",
    ),
    "ang": CanadianaLanguage(
        name="English, Old (ca.450-1100)",
        doc_count=10,
        iso639_1=None,
        iso639_2="ang",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="ang",
        paddle_code="en",
    ),
    "pl": CanadianaLanguage(
        name="Polish",
        doc_count=8,
        iso639_1="pl",
        iso639_2="pol",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="pl",
        paddle_code="en",
    ),
    "grc": CanadianaLanguage(
        name="Greek, Ancient (to 1453)",
        doc_count=5,
        iso639_1=None,
        iso639_2="grc",
        script="greek",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="grc",
        paddle_code="en",
    ),
    "ga": CanadianaLanguage(
        name="Irish",
        doc_count=4,
        iso639_1="ga",
        iso639_2="gle",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="ga",
        paddle_code="en",
    ),
    "hi": CanadianaLanguage(
        name="Hindi",
        doc_count=4,
        iso639_1="hi",
        iso639_2="hin",
        script="indic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="hi",
        paddle_code="en",
    ),
    "bla": CanadianaLanguage(
        name="Siksika",
        doc_count=3,
        iso639_1=None,
        iso639_2="bla",
        script="canadian_syllabics",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="bla",
        paddle_code="en",
    ),
    "chp": CanadianaLanguage(
        name="Chipewyan",
        doc_count=3,
        iso639_1=None,
        iso639_2="chp",
        script="canadian_syllabics",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="chp",
        paddle_code="en",
    ),
    "cy": CanadianaLanguage(
        name="Welsh",
        doc_count=3,
        iso639_1="cy",
        iso639_2="cym",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="cy",
        paddle_code="en",
    ),
    "del": CanadianaLanguage(
        name="Delaware",
        doc_count=3,
        iso639_1=None,
        iso639_2="del",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="del",
        paddle_code="en",
    ),
    "ng": CanadianaLanguage(
        name="Ndonga",
        doc_count=3,
        iso639_1="ng",
        iso639_2="ndo",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="ng",
        paddle_code="en",
    ),
    "pt": CanadianaLanguage(
        name="Portuguese",
        doc_count=3,
        iso639_1="pt",
        iso639_2="por",
        script="latin",
        spellcheck_supported=True,
        spellcheck_lang_code="pt",
        alto_code="pt",
        paddle_code="en",
    ),
    "bn": CanadianaLanguage(
        name="Bengali",
        doc_count=2,
        iso639_1="bn",
        iso639_2="ben",
        script="indic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="bn",
        paddle_code="en",
    ),
    "hai": CanadianaLanguage(
        name="Haida",
        doc_count=2,
        iso639_1=None,
        iso639_2="hai",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="hai",
        paddle_code="en",
    ),
    "cho": CanadianaLanguage(
        name="Choctaw",
        doc_count=1,
        iso639_1=None,
        iso639_2="cho",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="cho",
        paddle_code="en",
    ),
    "chu": CanadianaLanguage(
        name="Church Slavic",
        doc_count=1,
        iso639_1="cu",
        iso639_2="chu",
        script="cyrillic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="cu",
        paddle_code="en",
    ),
    "cop": CanadianaLanguage(
        name="Coptic",
        doc_count=1,
        iso639_1=None,
        iso639_2="cop",
        script="coptic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="cop",
        paddle_code="en",
    ),
    "dak": CanadianaLanguage(
        name="Dakota",
        doc_count=1,
        iso639_1=None,
        iso639_2="dak",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="dak",
        paddle_code="en",
    ),
    "den": CanadianaLanguage(
        name="Slave (Athapascan)",
        doc_count=1,
        iso639_1=None,
        iso639_2="den",
        script="canadian_syllabics",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="den",
        paddle_code="en",
    ),
    "egy": CanadianaLanguage(
        name="Egyptian (Ancient)",
        doc_count=1,
        iso639_1=None,
        iso639_2="egy",
        script="hieroglyphic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="egy",
        paddle_code="en",
    ),
    "fa": CanadianaLanguage(
        name="Persian",
        doc_count=1,
        iso639_1="fa",
        iso639_2="fas",
        script="arabic",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="fa",
        paddle_code="en",
    ),
    "lg": CanadianaLanguage(
        name="Ganda",
        doc_count=1,
        iso639_1="lg",
        iso639_2="lug",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="lg",
        paddle_code="en",
    ),
    "man": CanadianaLanguage(
        name="Mandingo",
        doc_count=1,
        iso639_1=None,
        iso639_2="man",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="man",
        paddle_code="en",
    ),
    "ro": CanadianaLanguage(
        name="Romanian",
        doc_count=1,
        iso639_1="ro",
        iso639_2="ron",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="ro",
        paddle_code="en",
    ),
    "tem": CanadianaLanguage(
        name="Timne",
        doc_count=1,
        iso639_1=None,
        iso639_2="tem",
        script="latin",
        spellcheck_supported=False,
        spellcheck_lang_code=None,
        alto_code="tem",
        paddle_code="en",
    ),
}

# Lookup indices
NAME_TO_LANG: Dict[str, CanadianaLanguage] = {
    l.name.lower(): l for l in CANADIANA_LANGUAGE_CATALOG.values()
}
ISO3_TO_LANG: Dict[str, CanadianaLanguage] = {
    l.iso639_2.lower(): l for l in CANADIANA_LANGUAGE_CATALOG.values()
}
ISO2_TO_LANG: Dict[str, CanadianaLanguage] = {
    l.iso639_1.lower(): l for l in CANADIANA_LANGUAGE_CATALOG.values() if l.iso639_1
}

# Canonical set of languages where full dictionary spellchecking is authorized
AUTHORIZED_SPELLCHECK_LANGUAGES: Set[str] = {
    code
    for code, l in CANADIANA_LANGUAGE_CATALOG.items()
    if l.spellcheck_supported
} | {"bilingual"}

# Script detection regexes
CANADIAN_SYLLABICS_RE = re.compile(r"[\u1400-\u167f]")
HEBREW_SCRIPT_RE = re.compile(r"[\u0590-\u05ff\ufb1d-\ufb4f]")
CYRILLIC_SCRIPT_RE = re.compile(r"[\u0400-\u04ff\u0500-\u052f]")
GREEK_SCRIPT_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")
DEVANAGARI_SCRIPT_RE = re.compile(r"[\u0900-\u097f]")
BENGALI_SCRIPT_RE = re.compile(r"[\u0980-\u09ff]")


COMMON_ALIASES: Dict[str, str] = {
    "chinese_cht": "zh",
    "chinese_sim": "zh",
    "ch": "zh",
    "cht": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh_cht": "zh",
    "traditional_chinese": "zh",
    "simplified_chinese": "zh",
    "german": "de",
    "deutsch": "de",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "english": "en",
    "italian": "it",
    "italiano": "it",
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "dutch": "nl",
    "nederlands": "nl",
    "portuguese": "pt",
    "português": "pt",
    "russian": "ru",
    "русский": "ru",
    "ukrainian": "uk",
    "українська": "uk",
    "yiddish": "yi",
    "ייִדיש": "yi",
    "latin": "la",
    "latina": "la",
    "cree": "cr",
    "inuktitut": "iu",
    "ojibwa": "oj",
    "mohawk": "moh",
}


def get_canadiana_language(identifier: Optional[str]) -> Optional[CanadianaLanguage]:
    """Retrieve CanadianaLanguage entry by code, 3-letter ISO code, or full English name."""
    if not identifier:
        return None
    raw = identifier.strip().lower()
    if raw in CANADIANA_LANGUAGE_CATALOG:
        return CANADIANA_LANGUAGE_CATALOG[raw]
    if raw in COMMON_ALIASES:
        target_code = COMMON_ALIASES[raw]
        if target_code in CANADIANA_LANGUAGE_CATALOG:
            return CANADIANA_LANGUAGE_CATALOG[target_code]
    if raw in ISO2_TO_LANG:
        return ISO2_TO_LANG[raw]
    if raw in ISO3_TO_LANG:
        return ISO3_TO_LANG[raw]
    if raw in NAME_TO_LANG:
        return NAME_TO_LANG[raw]
    return None


def is_spellcheck_authorized(lang: Optional[str]) -> bool:
    """Return True only if language has an authorized, verified dictionary spellchecker."""
    if not lang:
        return True  # Fallback to English/French
    norm = lang.strip().lower()
    if norm in {"bilingual", "en", "fr", "it", "de", "es", "nl", "pt", "ru"}:
        return True
    cl = get_canadiana_language(norm)
    return cl.spellcheck_supported if cl else False


def detect_canadiana_script(text: str) -> Optional[str]:
    """Detect non-Latin indigenous and international writing scripts common to Canadiana."""
    if not text:
        return None
    sample = text[:1000]
    if len(CANADIAN_SYLLABICS_RE.findall(sample)) >= 5:
        return "canadian_syllabics"
    if len(HEBREW_SCRIPT_RE.findall(sample)) >= 5:
        return "hebrew"
    if len(CYRILLIC_SCRIPT_RE.findall(sample)) >= 5:
        return "cyrillic"
    if len(GREEK_SCRIPT_RE.findall(sample)) >= 5:
        return "greek"
    if len(ARABIC_SCRIPT_RE.findall(sample)) >= 5:
        return "arabic"
    if len(DEVANAGARI_SCRIPT_RE.findall(sample)) >= 5:
        return "indic"
    if len(BENGALI_SCRIPT_RE.findall(sample)) >= 5:
        return "bengali"
    return None
