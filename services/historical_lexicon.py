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

ALL_CANADIAN_LEXICON_TOKENS: Set[str] = (
    CANADIAN_HISTORICAL_PLACES
    | CANADIAN_HISTORICAL_TERMS
    | CANADIAN_HISTORICAL_ABBREVIATIONS
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

