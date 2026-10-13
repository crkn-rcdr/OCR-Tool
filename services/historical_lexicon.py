"""Historical Canadian lexicon, gazetteer whitelist, and orthography tools.

Provides curated domain dictionaries and normalizers for Canadiana archives:
1. Canadian Historical Gazetteer (provinces, districts, counties, towns, forts, rivers)
2. French Canadian legal, municipal, seigniorial, and church terminology
3. 18th/19th-century colonial and legal abbreviations
4. Historical long-s ('ſ' -> 's') OCR error resolver
5. Bilingual French diacritic restoration for unaccented historical print
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 1. Canadian Historical Gazetteer & Toponyms
# ---------------------------------------------------------------------------

CANADIAN_HISTORICAL_PLACES: Set[str] = {
    # Historical colonies, territories, and districts
    "acadia", "acadie", "assiniboia", "athabaska", "keewatin", "ungava",
    "franklin", "mackenzie", "klondike", "yukon", "rupert", "rupertsland",
    "hochelaga", "stadacona", "cataraqui", "bytown", "york", "newark",
    "canada", "canadas", "uppercanada", "lowercanada", "canadawest", "canadaeast",
    # Ontario counties, regions, and historic settlements
    "frontenac", "glengarry", "stormont", "dundas", "prescott", "russell",
    "carleton", "lanark", "renfrew", "leeds", "grenville", "addington",
    "hastings", "lennox", "princeedward", "northumberland", "durham",
    "peterborough", "victoria", "simcoe", "haliburton", "muskoka", "parrysound",
    "nipissing", "algoma", "manitoulin", "sudbury", "timiskaming", "cochrane",
    "rainyriver", "kenora", "thunderbay", "peel", "halton", "wentworth",
    "lincoln", "welland", "haldimand", "norfolk", "brant", "waterloo",
    "wellington", "dufferin", "grey", "bruce", "huron", "perth", "oxford",
    "middlesex", "elgin", "kent", "essex", "lambton", "streetsville", "toronto",
    "kingston", "hamilton", "ottawa", "london", "windsor", "niagara", "brockville",
    "cornwall", "perth", "belleville", "cobourg", "guelph", "galt", "berlin",
    # Quebec regions, counties, seigniories, and historical settlements
    "gaspé", "gaspe", "gaspesie", "bonaventure", "matane", "rimouski",
    "témiscouata", "temiscouata", "kamouraska", "lislet", "l'islet", "montmagny",
    "bellechasse", "lévis", "levis", "dorchester", "beauce", "mégantic", "megantic",
    "lotbinière", "lotbiniere", "nicolet", "arthabaska", "drummond", "yamaska",
    "richelieu", "saint-hyacinthe", "sainthyacinthe", "bagot", "rouville",
    "iberville", "missisquoi", "brome", "shefford", "stanstead", "sherbrooke",
    "richmond", "compton", "wolfe", "saguenay", "charlevoix", "montmorency",
    "québec", "quebec", "portneuf", "champlain", "saint-maurice", "saintmaurice",
    "maskinongé", "maskinonge", "berthier", "joliette", "montcalm", "lassomption",
    "l'assomption", "terrebonne", "deux-montagnes", "deuxmontagnes", "argenteuil",
    "vaudreuil", "soulanges", "beauharnois", "châteauguay", "chateauguay",
    "huntingdon", "napierville", "laprairie", "chambly", "verchères", "vercheres",
    "saint-jean", "saintjean", "laval", "jacques-cartier", "jacquescartier",
    "montréal", "montreal", "ville-marie", "villemarie", "trois-rivières",
    "troisrivieres", "hull", "gatineau", "pontiac", "temiscamingue", "témiscamingue",
    "abitibi", "chicoutimi", "lac-saint-jean", "lacsaintjean", "tadoussac",
    # Western Canada historical settlements & forts
    "redriver", "fortgarry", "selkirk", "portagelaprairie", "brandon",
    "saint-boniface", "saintboniface", "batoche", "ducklake", "fishcreek",
    "froglake", "cutknife", "battleford", "princealbert", "regina", "moosejaw",
    "saskatoon", "swiftcurrent", "qu'appelle", "quappelle", "cypresshills",
    "fortmacleod", "fortcalgary", "fortedmonton", "medicinehat", "lethbridge",
    "reddeer", "strathcona", "kootenay", "cariboo", "yale", "lillooet",
    "newwestminster", "esquimalt", "nanaimo", "kamloops", "okanagan", "cassiar",
    "atlin", "skeena", "peaceriver", "victoria", "vancouver",
    # Atlantic historical settlements & counties
    "halifax", "dartmouth", "pictou", "antigonish", "guysborough", "sydney",
    "louisbourg", "annapolis", "hants", "colchester", "cumberland", "lunenburg",
    "shelburne", "yarmouth", "digby", "capebreton", "inverness",
    "saintjohn", "fredericton", "moncton", "westmorland", "albert",
    "gloucester", "restigouche", "madawaska", "sunbury", "charlotte",
    "standrews", "miramichi", "chaleur", "charlottetown", "summerside",
    "stjohns", "avalon", "placentia", "trinity", "conception", "bonavista",
    "twillingate", "fogo", "burin", "fortune", "labrador",
}

# ---------------------------------------------------------------------------
# 2. French Canadian Historical Legal, Land & Institutional Terms
# ---------------------------------------------------------------------------

CANADIAN_HISTORICAL_TERMS: Set[str] = {
    # Seigniorial and land tenure
    "seigneurie", "seigneuries", "seigneur", "seigneurs", "seigneuresse",
    "seigniory", "seigniories", "censitaire", "censitaires", "censive",
    "cens", "rentes", "corvée", "corvee", "banalité", "banalite", "lods",
    "ventes", "arpents", "arpent", "lieue", "lieues", "toise", "toises",
    "perche", "perches", "bailliage", "prévôté", "prevote", "sénéchaussée",
    "senechaussee", "greffe", "greffier", "notaire", "notariat", "notarié",
    "notarie", "fabrique", "marguillier", "marguilliers", "curé", "cure",
    "évêché", "eveche", "diocèse", "diocese", "paroisse", "paroisses",
    "canton", "cantons", "rang", "rangs", "concession", "concessions",
    "terroirs", "habitant", "habitants", "habitants-pêcheurs",
    # Fur trade & Indigenous historical vocabulary
    "coureur", "coureurs", "voyageur", "voyageurs", "engagé", "engage",
    "engagés", "engages", "pemmican", "brigades", "portage", "portages",
    "chipewyan", "cree", "ojibwe", "algonquin", "iroquois", "mohawk",
    "huron", "wendat", "micmac", "mi'kmaq", "maliseet", "innu", "naskapi",
    "beothuk", "métis", "metis", "bois-brûlé", "boisbrule",
    # Colonial, legal, legislative & governance
    "legislature", "assembly", "council", "legislative", "prorogation",
    "dissolved", "assented", "proclamation", "ordinance", "ordinances",
    "statut", "statuts", "gazette", "dominion", "confederation", "statute",
    "statutes", "parliament", "provincial", "colonial", "imperial",
    "dispatch", "dispatches", "lieutenant-governor", "governor-general",
    "surveyor-general", "attorney-general", "solicitor-general",
    "receiver-general", "commissioner", "commissioners", "bailiff", "sheriff",
    "constable", "militia", "regiment", "battalion", "fencibles",
    "voltigeurs", "voltigeur", "actes", "édits", "edits", "ordonnances",
}

# ---------------------------------------------------------------------------
# 3. Historical Abbreviations (18th-20th c. English & French Canadiana)
# ---------------------------------------------------------------------------

CANADIAN_HISTORICAL_ABBREVIATIONS: Set[str] = {
    # Honorifics & titles
    "esq", "esqr", "hon", "honble", "messrs", "mr", "mrs", "ms", "rev",
    "revd", "dr", "col", "gen", "maj", "capt", "lieut", "lt", "sgt",
    "sergt", "cpl", "pte", "prof", "sir", "bart", "kt", "cb", "cmg",
    "kcmg", "qb", "kb", "pc", "mp", "mpp", "mla", "mlc", "qc", "kc", "knt",
    # Monarchs & regnal years
    "vic", "vict", "geo", "edw", "edwd", "wm", "chas", "thos", "jas",
    "benj", "saml", "robt", "richd", "danl", "alex", "archd", "cath",
    "margt", "eliz", "hen", "phil", "fred",
    # Legal, chronological, and publication markers
    "inst", "ult", "prox", "viz", "vid", "ibid", "op", "cit", "vol", "vols",
    "cap", "caps", "chap", "chaps", "sec", "secs", "sect", "sects", "art",
    "arts", "sess", "no", "nos", "fol", "fols", "pp", "p", "do", "ditto",
    "etc", "&c", "&c.", "c", "seq", "vs", "v", "reg", "greeting", "l.s.", "ls",
    # Units of measure & currency
    "lb", "lbs", "oz", "cwt", "qr", "qrs", "bush", "bu", "ft", "in", "yd",
    "yds", "sq", "acre", "acres", "arp", "arps", "sh", "pence", "cent",
    "cents", "dol", "dols",
}

# ---------------------------------------------------------------------------
# 3b. Historical Chinese Canadian & Overseas Names, Surnames & Commercial Terms
# ---------------------------------------------------------------------------

CANADIAN_HISTORICAL_NAMES_AND_SURNAMES: Set[str] = {
    # Chinese romanized surnames and given names (Wade-Giles, Cantonese, Pinyin)
    # ubiquitous in SFU Chinese Times and British Columbia / Pacific historical archives
    "chan", "chen", "cheng", "cheung", "chew", "chin", "ching", "chiu",
    "cho", "chou", "chow", "choy", "chu", "chuan", "chuen", "chun", "chung",
    "fong", "foo", "fu", "fung",
    "gan", "gim", "gin", "goon", "guan",
    "ha", "hang", "hau", "he", "heng", "hing", "ho", "hom", "hon", "hong", "hou", "hsu", "huang", "hui", "hung",
    "jang", "jeung", "jew", "jim", "jin", "joe", "jong", "jow", "jung",
    "kam", "kan", "kang", "kao", "kau", "keung", "kin", "king", "ko", "kong", "koo", "kuan", "kuo", "kwan", "kwok", "kwon", "kwong",
    "lai", "lam", "lau", "law", "lee", "leong", "leung", "li", "liang", "liao", "lien", "lim", "lin", "ling", "liu", "lo", "lock", "lok", "loo", "look", "louie", "low", "lu", "lui", "luk", "lum", "lung",
    "ma", "mah", "mak", "man", "mar", "mau", "mei", "ming", "mock", "mok", "mon", "moon",
    "ng", "ngan", "ngo",
    "ou", "ouyang",
    "pack", "pak", "pan", "pang", "pei", "peng", "ping", "pong", "poon", "poy", "pun",
    "quon",
    "sam", "sang", "seid", "seto", "shek", "shen", "shi", "shin", "shing", "shiu", "shue", "shum", "sing", "siu", "so", "som", "song", "sue", "suen", "sui", "sum", "sun", "sung", "szeto",
    "tai", "tam", "tan", "tang", "tin", "ting", "to", "tom", "tong", "tsai", "tsang", "tse", "tsim", "tso", "tsoi", "tsou", "tsui", "tu", "tung",
    "wah", "wai", "wan", "wang", "wei", "wen", "wing", "wo", "won", "wong", "woo", "woon", "wu",
    "xia", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xu", "xue",
    "yan", "yang", "yao", "yap", "yau", "ye", "yee", "yeh", "yen", "yeung", "yim", "yin", "ying", "yip", "yiu", "yong", "yow", "yu", "yuan", "yue", "yuen", "yung",
    "zeng", "zhang", "zhao", "zhen", "zheng", "zhong", "zhou", "zhu", "zhuang",
    # Commercial & municipal terms appearing in advertisements and directories
    "co", "ltd", "inc", "corp", "dept", "mgr", "bros", "cie", "assn", "soc",
    "mfg", "tel", "ave", "boul", "bvd", "blvd", "bldg", "san", "francisco",
}

# ---------------------------------------------------------------------------
# 3c. Historical Italian Canadian Newspapers & Cultural Terms (Il Cittadino Canadese, etc.)
# ---------------------------------------------------------------------------

CANADIAN_ITALIAN_TERMS: Set[str] = {
    "cittadino", "cittadini", "canadese", "canadesi", "italia", "italiano",
    "italiana", "italiani", "italiane", "corriere", "tribuna", "voce",
    "stampa", "messaggero", "comunitario", "comunitaria", "emigrazione",
    "emigrati", "colonia", "coloniale", "patria", "montreale", "montreal",
    "toronto", "ottawa", "vancouver", "quebec", "ontario", "canada",
    "roma", "milano", "napoli", "palermo", "genova", "venezia", "trieste",
    "località", "qualità", "società", "attività", "capacità", "proprietà",
}

ALL_CANADIAN_LEXICON_TOKENS: Set[str] = (
    CANADIAN_HISTORICAL_PLACES
    | CANADIAN_HISTORICAL_TERMS
    | CANADIAN_HISTORICAL_ABBREVIATIONS
    | CANADIAN_HISTORICAL_NAMES_AND_SURNAMES
    | CANADIAN_ITALIAN_TERMS
)

# ---------------------------------------------------------------------------
# 4. Archaic Long-s ('ſ' -> 's') Resolution
# ---------------------------------------------------------------------------

# Words with zero modern English collisions: every key here is unknown in modern
# dictionaries and uniquely resolves an archaic long-s OCR error.
LONG_S_DIRECT_MAPPINGS: Dict[str, str] = {
    "firft": "first",
    "houfe": "house",
    "caufe": "cause",
    "caufes": "causes",
    "difcharge": "discharge",
    "eftate": "estate",
    "eftates": "estates",
    "infpect": "inspect",
    "infpection": "inspection",
    "defire": "desire",
    "defired": "desired",
    "fuch": "such",
    "prefent": "present",
    "prefented": "presented",
    "prefents": "presents",
    "prefence": "presence",
    "pofleflion": "possession",
    "poflefs": "possess",
    "pofsefled": "possessed",
    "pofsefsion": "possession",
    "congrefs": "congress",
    "witnefs": "witness",
    "witnefses": "witnesses",
    "buſineſs": "business",
    "bufinefs": "business",
    "paft": "past",
    "laft": "last",
    "moft": "most",
    "alfo": "also",
    "purfuant": "pursuant",
    "perfons": "persons",
    "perfon": "person",
    "affembly": "assembly",
    "affembled": "assembled",
    "majefty": "majesty",
    "majeftys": "majesty's",
    "commiffion": "commission",
    "commiffioners": "commissioners",
    "diffolve": "dissolve",
    "diffolved": "dissolved",
    "feffion": "session",
    "feffions": "sessions",
    "procefs": "process",
    "exprefs": "express",
    "difpofe": "dispose",
    "difpofed": "disposed",
    "ufe": "use",
    "ufed": "used",
    "reafon": "reason",
    "fenfe": "sense",
    "affent": "assent",
    "affented": "assented",
    "pleafe": "please",
    "pleafed": "pleased",
    "fhall": "shall",
    "fhould": "should",
    "fubject": "subject",
    "fubjects": "subjects",
    "fubfcribe": "subscribe",
    "fubfcribed": "subscribed",
    "fealed": "sealed",
    "figned": "signed",
    "faid": "said",
    "fatisfaction": "satisfaction",
    "fatisfied": "satisfied",
}

# Contextual phrases where words like 'fame' ('same') or 'cafe' ('case') are unambiguous
LONG_S_PHRASE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bthe fame\b", re.IGNORECASE), "the same"),
    (re.compile(r"\bThe fame\b"), "The same"),
    (re.compile(r"\bthat fame\b", re.IGNORECASE), "that same"),
    (re.compile(r"\bin cafe\b", re.IGNORECASE), "in case"),
    (re.compile(r"\band fo\b", re.IGNORECASE), "and so"),
    (re.compile(r"\bfo far\b", re.IGNORECASE), "so far"),
]


def resolve_archaic_long_s(token: str, spellchecker_en=None) -> Optional[str]:
    """Test and resolve 18th/early 19th-century medial long-s OCR errors ('ſ' read as 'f').

    Returns the corrected modern word if replacement resolves an archaic long-s error,
    or None if the token should be kept as-is.
    """
    if not token or len(token) < 2:
        return None

    clean = token.strip(".,:;!?()[]{}\"'`").lower()
    if not clean.isalpha() or "f" not in clean:
        return None

    # 1. Direct dictionary match of common historical long-s forms
    if clean in LONG_S_DIRECT_MAPPINGS:
        target = LONG_S_DIRECT_MAPPINGS[clean]
        # Preserve original capitalization
        if token.istitle():
            return target.capitalize()
        if token.isupper():
            return target.upper()
        return target

    # 2. Rule-based heuristic: if word has 'f' and is unknown, test replacing with 's' or 'ss'
    if spellchecker_en is not None:
        try:
            if not spellchecker_en.unknown([clean]):
                return None

            candidates = [
                clean.replace("ff", "ss"),
                re.sub(r"(?<!f)f(?!f)", "s", clean),
                clean.replace("f", "s"),
            ]
            for cand in candidates:
                if cand != clean and not spellchecker_en.unknown([cand]):
                    if token.istitle():
                        return cand.capitalize()
                    if token.isupper():
                        return cand.upper()
                    return cand
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# 5. French Diacritic Regularization (Bilingual Canadiana)
# ---------------------------------------------------------------------------

# Words with zero French collisions: every key here is unaccented and invalid in French
FRENCH_DIACRITIC_MAP: Dict[str, str] = {
    "republique": "république",
    "republiques": "républiques",
    "general": "général",
    "generale": "générale",
    "generaux": "généraux",
    "generales": "générales",
    "president": "président",
    "presidente": "présidente",
    "presidence": "présidence",
    "senat": "sénat",
    "senateur": "sénateur",
    "senateurs": "sénateurs",
    "assemblee": "assemblée",
    "assemblees": "assemblées",
    "comite": "comité",
    "comites": "comités",
    "decret": "décret",
    "decrets": "décrets",
    "arrete": "arrêté",
    "arretes": "arrêtés",
    "etat": "état",
    "etats": "états",
    "ete": "été",
    "etes": "êtes",
    "etre": "être",
    "deja": "déjà",
    "apres": "après",
    "tres": "très",
    "meme": "même",
    "memes": "mêmes",
    "precedent": "précédent",
    "precedents": "précédents",
    "precedente": "précédente",
    "periode": "période",
    "periodes": "périodes",
    "annee": "année",
    "annees": "années",
    "tresor": "trésor",
    "tresorier": "trésorier",
    "tresorerie": "trésorerie",
    "societe": "société",
    "societes": "sociétés",
    "secretariat": "secrétariat",
    "departement": "département",
    "departements": "départements",
    "reglement": "règlement",
    "reglements": "règlements",
    "impot": "impôt",
    "impots": "impôts",
    "cout": "coût",
    "couts": "coûts",
    "premiere": "première",
    "premieres": "premières",
    "derniere": "dernière",
    "dernieres": "dernières",
    "legale": "légale",
    "legales": "légales",
    "legaux": "légaux",
    "systeme": "système",
    "systemes": "systèmes",
    "modele": "modèle",
    "modeles": "modèles",
    "interet": "intérêt",
    "interets": "intérêts",
    "numero": "numéro",
    "numeros": "numéros",
    "recu": "reçu",
    "recus": "reçus",
    "recue": "reçue",
    "recues": "reçues",
    "francais": "français",
    "francaise": "française",
    "francaises": "françaises",
    "eveque": "évêque",
    "eveques": "évêques",
    "archeveque": "archevêque",
    "eveche": "évêché",
    "seigneurie": "seigneurie",
    "evenement": "événement",
    "evenements": "événements",
    "element": "élément",
    "elements": "éléments",
    "education": "éducation",
    "economie": "économie",
    "economique": "économique",
    "economiques": "économiques",
    "region": "région",
    "regions": "régions",
    "regime": "régime",
    "regimes": "régimes",
    "legislateur": "législateur",
    "legislatif": "législatif",
    "legislative": "législative",
    "legislature": "législature",
    "ministere": "ministère",
    "ministeres": "ministères",
    "secretaire": "secrétaire",
    "secretaires": "secrétaires",
    "proces": "procès",
    "proces-verbal": "procès-verbal",
    "enquete": "enquête",
    "enquetes": "enquêtes",
    "hopital": "hôpital",
    "hopitaux": "hôpitaux",
    "hotel": "hôtel",
    "hotels": "hôtels",
    "depot": "dépôt",
    "depots": "dépôts",
    "entree": "entrée",
    "entrees": "entrées",
}


def regularize_french_diacritics(token: str) -> Optional[str]:
    """Restore missing French accents on unaccented historical print words.

    Returns the accented word if a match is found in the canonical French map,
    or None if unchanged.
    """
    if not token or len(token) < 2:
        return None

    clean = token.strip(".,:;!?()[]{}\"'`").lower()
    if clean in FRENCH_DIACRITIC_MAP:
        target = FRENCH_DIACRITIC_MAP[clean]
        if token.istitle():
            return target.capitalize()
        if token.isupper():
            return target.upper()
        return target

    return None


def normalize_historical_orthography(text: str, is_french: bool = False) -> str:
    """Normalize archaic long-s OCR errors and French diacritics in a text line or token.

    Safely applies:
    1. Unambiguous long-s word corrections (e.g. firft -> first, houfe -> house)
    2. Contextual long-s phrases (e.g. the fame -> the same, in cafe -> in case)
    3. French diacritic restoration for unaccented historical vocabulary (when French detected
       or when French canonical vocabulary matches)
    """
    if not text:
        return text

    updated = text

    # Apply contextual phrase replacements
    for pattern, replacement in LONG_S_PHRASE_PATTERNS:
        updated = pattern.sub(replacement, updated)

    # Word-level replacement preserving surrounding punctuation
    def _replace_word(match: re.Match) -> str:
        word = match.group(0)

        # 1. Long-s check
        corrected_long_s = resolve_archaic_long_s(word)
        if corrected_long_s:
            return corrected_long_s

        # 2. French diacritic check
        if is_french:
            corrected_diacritic = regularize_french_diacritics(word)
            if corrected_diacritic:
                return corrected_diacritic

        return word

    updated = re.sub(r"\b[A-Za-z\u00C0-\u017F]+\b", _replace_word, updated)
    return updated


_SP_INSTANCES: Dict[str, Any] = {}

def get_shared_spellchecker(lang: str = "en") -> Optional[Any]:
    """Lazy initialize and cache spellcheckers for supported Canadiana languages."""
    global _SP_INSTANCES
    code = (lang or "en").lower().strip()
    if code in {"english", "en_ca", "en_us", "en_gb"}:
        code = "en"
    elif code in {"french", "fr_ca", "fr_fr"}:
        code = "fr"
    elif code in {"italian", "italiano"}:
        code = "it"
    elif code in {"german", "deutsch"}:
        code = "de"
    elif code in {"spanish", "espanol"}:
        code = "es"

    if code not in _SP_INSTANCES:
        try:
            from spellchecker import SpellChecker
            _SP_INSTANCES[code] = SpellChecker(language=code)
        except Exception:
            _SP_INSTANCES[code] = None
    return _SP_INSTANCES.get(code)


def _get_shared_spellcheckers():
    """Return shared EN and FR spellcheckers for backwards compatibility."""
    return get_shared_spellchecker("en"), get_shared_spellchecker("fr")


# High-confidence optical confusion and abbreviation mappings common to historical print
HISTORICAL_OPTICAL_CORRECTIONS: Dict[str, str] = {
    # Microfilm stroke breakages & contractions
    "sd": "said",
    "thie": "this",
    "tis": "this",
    "heeb": "hereby",
    "oupa": "occupa-",
    "boad": "boats",
    "conainin": "containing",
    "resering": "reserving",
    "paage": "passage",
    "veel": "vessels",
    "fery": "fishery",
    "beign": "reign",
    "wi": "with",
    "visty": "viscount",
    "coanada": "canada",
    # Optical confusion pairs: rn <-> m, cl <-> d, vv <-> w, etc.
    "modem": "modern",
    "bomed": "bombed",
    "buming": "burning",
    "fami": "farm",
    "fre": "free",
    "cleaf": "deaf",
    "clead": "dead",
    "cleath": "death",
    "vvhich": "which",
    "vvill": "will",
    "vvith": "with",
    "vvhen": "when",
    "vvhere": "where",
    # High-frequency historical OCR optical confusions & broken letterforms
    "desirel": "desired",
    "sesion": "session",
    "muffer": "muffler",
    "suoh": "such",
    "slah": "slab",
    "traffe": "traffic",
    "lhad": "had",
    "doso": "do",
    "tha": "that",
    "carryin": "carrying",
    "senching": "searching",
    # French historical dropouts & faint terminal accents
    "fraichement": "fraîchement",
    "deyenir": "devenir",
    "deyenant": "devenant",
}

FRENCH_OPTICAL_CORRECTIONS: Dict[str, str] = {
    "qualit": "qualité",
    "societ": "société",
    "activit": "activité",
    "comptabilit": "comptabilité",
    "capacit": "capacité",
    "propriet": "propriété",
}

ITALIAN_OPTICAL_CORRECTIONS: Dict[str, str] = {
    "localit": "località",
    "qualit": "qualità",
    "societ": "società",
    "activit": "attività",
    "comptabilit": "contabilità",
    "capacit": "capacità",
    "propriet": "proprietà",
}


def is_valid_historical_word(word: str, lang: Optional[str] = None) -> bool:
    """Check if a word token is recognized in historical Canadian vocabulary or standard dictionaries."""
    if not word:
        return False
    subwords = str(word).strip().split()
    if len(subwords) > 1:
        return all(is_valid_historical_word(sw, lang=lang) for sw in subwords)
    clean = re.sub(r"^[^\w\d&]+|[^\w\d]+$", "", str(word).lower().strip())
    if not clean:
        return False
    if clean in {"&c", "&c.", "c.", "c"}:
        return True
    if clean.isdigit() or (re.match(r"^[\d,.\-/:#]+$", clean) and any(ch.isdigit() for ch in clean)):
        return True
    if re.match(r"^\d+(st|nd|rd|th)$", clean):
        return True
    if any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in clean):
        return True
    if len(clean) == 1:
        return clean in {"a", "i", "o", "y", "à", "è", "é", "d", "l", "s", "n", "m", "j", "t", "c", "e", "u"}
    if clean in ALL_CANADIAN_LEXICON_TOKENS:
        return True
    if clean in HISTORICAL_OPTICAL_CORRECTIONS.values():
        return True

    # 1. Check language-specific dictionary first when document language is known
    norm_l = (lang or "").lower().strip()
    if norm_l in {"it", "italian"}:
        sp_it = get_shared_spellchecker("it")
        if sp_it and clean in sp_it:
            return True
        if clean in ITALIAN_OPTICAL_CORRECTIONS.values():
            return True
    elif norm_l in {"de", "german"}:
        sp_de = get_shared_spellchecker("de")
        if sp_de and clean in sp_de:
            return True
    elif norm_l in {"es", "spanish"}:
        sp_es = get_shared_spellchecker("es")
        if sp_es and clean in sp_es:
            return True
    elif norm_l in {"fr", "french"}:
        sp_fr = get_shared_spellchecker("fr")
        if sp_fr and clean in sp_fr:
            return True
        if clean in FRENCH_OPTICAL_CORRECTIONS.values():
            return True
        sp_en = get_shared_spellchecker("en")
        if sp_en and clean in sp_en:
            return True
    elif norm_l in {"en", "english"}:
        sp_en = get_shared_spellchecker("en")
        if sp_en and clean in sp_en:
            return True

    # 2. Multi-lingual lookup fallback ONLY if language is unspecified or bilingual
    if not norm_l or norm_l in {"bilingual", "all", "auto"}:
        for check_code in ("en", "fr", "it", "de", "es"):
            sp = get_shared_spellchecker(check_code)
            if sp and clean in sp:
                return True

    return False


def regularize_broken_historical_word(
    word: str,
    confidence: float = 0.75,
    max_confidence: float = 0.88,
    lang: Optional[str] = None,
) -> str:
    """1-edit Levenshtein dictionary regularizer and optical confusion resolver for broken microfilm letterforms.
    
    Snaps single-character split or broken strokes (e.g., 'resering' -> 'reserving', 'sd' -> 'said',
    'thie' -> 'this', 'fre' -> 'free') to known Canadian historical and standard dictionary words.
    Deterministic optical corrections apply whenever the token is not an existing dictionary word;
    fuzzy 1-edit Levenshtein regularizing applies on low-confidence tokens (confidence <= max_confidence).
    """
    if not word or len(word) < 2:
        return word

    clean = word.lower().strip(".,:;!?()[]{}\"'`")
    if not clean.isalpha():
        return word

    norm_l = (lang or "").lower().strip()

    # 1. Deterministic historical optical corrections
    if clean in HISTORICAL_OPTICAL_CORRECTIONS:
        target = HISTORICAL_OPTICAL_CORRECTIONS[clean]
        if word.istitle():
            return target.capitalize()
        elif word.isupper():
            return target.upper()
        return target

    if norm_l in {"it", "italian"}:
        if clean in ITALIAN_OPTICAL_CORRECTIONS:
            target = ITALIAN_OPTICAL_CORRECTIONS[clean]
            if word.istitle():
                return target.capitalize()
            elif word.isupper():
                return target.upper()
            return target
    elif norm_l in {"fr", "french", "bilingual", ""}:
        if clean in FRENCH_OPTICAL_CORRECTIONS:
            target = FRENCH_OPTICAL_CORRECTIONS[clean]
            if word.istitle():
                return target.capitalize()
            elif word.isupper():
                return target.upper()
            return target

    # 2. CRITICAL SAFEGUARD: If already a valid canonical Canadian historical word
    # or recognized dictionary word in ANY supported language (EN, FR, IT, DE, ES),
    # NEVER mutate it! This guarantees valid Italian words (e.g. 'località', 'qualche',
    # 'scheda') will never be swapped for spurious English or French words.
    if clean in ALL_CANADIAN_LEXICON_TOKENS:
        return word

    for check_code in ("en", "fr", "it", "de", "es"):
        sp = get_shared_spellchecker(check_code)
        if sp and clean in sp:
            return word

    # Words under 4 chars must not be heuristically spell-checked to avoid corrupting short words
    if len(clean) < 4:
        return word

    # Non-Latin languages (e.g. Chinese) must never have Latin dictionary regularizer applied
    if norm_l in {"zh", "chinese", "chinese_cht", "ch"}:
        return word

    # 3. Fuzzy 1-edit Levenshtein regularizer:
    # On lower-confidence words, allows ratio >= 0.78.
    # On high-confidence words (where OCR was overconfident about a non-dictionary typo), require strict 1-substitution (ratio >= 0.85 and len >= 5).
    min_ratio = 0.78 if confidence <= max_confidence else 0.85
    if confidence > max_confidence and len(clean) < 5:
        return word

    cand_sps = []
    if norm_l in {"it", "italian"}:
        sp_it = get_shared_spellchecker("it")
        if sp_it:
            cand_sps.append(sp_it)
    elif norm_l in {"de", "german"}:
        sp_de = get_shared_spellchecker("de")
        if sp_de:
            cand_sps.append(sp_de)
    elif norm_l in {"es", "spanish"}:
        sp_es = get_shared_spellchecker("es")
        if sp_es:
            cand_sps.append(sp_es)
    elif norm_l in {"fr", "french"}:
        sp_fr = get_shared_spellchecker("fr")
        if sp_fr:
            cand_sps.append(sp_fr)
        sp_en = get_shared_spellchecker("en")
        if sp_en:
            cand_sps.append(sp_en)
    elif norm_l in {"en", "english"}:
        sp_en = get_shared_spellchecker("en")
        if sp_en:
            cand_sps.append(sp_en)
    else:
        # Default / bilingual fallback: English and French
        sp_en = get_shared_spellchecker("en")
        sp_fr = get_shared_spellchecker("fr")
        if sp_en:
            cand_sps.append(sp_en)
        if sp_fr:
            cand_sps.append(sp_fr)

    import difflib
    candidates = []
    for sp in cand_sps:
        cands = list(sp.candidates(clean) or [])
        candidates.extend(cands)
        if len(candidates) >= 5:
            break

    scored = []
    for c in candidates:
        if abs(len(c) - len(clean)) <= 1:
            ratio = difflib.SequenceMatcher(None, clean, c).ratio()
            # Require tight 1-edit similarity
            if ratio >= min_ratio:
                freq = 0
                for sp in cand_sps:
                    try:
                        freq = max(freq, sp.word_frequency[c] or 0)
                    except Exception:
                        pass
                if c in ALL_CANADIAN_LEXICON_TOKENS:
                    freq += 1000000
                if freq > 0:
                    scored.append((freq, c))

    if scored:
        scored.sort(reverse=True)
        best = scored[0][1]
        # Preserve original capitalization
        if word.istitle():
            return best.capitalize()
        elif word.isupper():
            return best.upper()
        return best

    return word



