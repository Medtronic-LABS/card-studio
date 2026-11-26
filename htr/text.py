"""Bengali text: normalization, grapheme clusters, words, conjuncts. Pure Python, no third-party module.

The reader's CTC units are NFC codepoints (a vowel sign, a virama and a nukta are each one class), because grapheme clusters in handwritten
Bangla run to more than a thousand and would starve a small model. Accuracy is still *measured* on clusters (`graphemes`), which is what a
person sees as one letter: বাংলাদেশ is 8 codepoints and 4 clusters (বাং লা দে শ); ক্ষ is 3 codepoints and 1 cluster.

normalize(s, kind):
    NFC (this composes ে+া into ো and ে+ৗ into ৌ, and decomposes ড় ঢ় য় into base letter + nukta, which are the classes we keep),
    joiners (ZWJ, ZWNJ) dropped unless keep_joiners, whitespace collapsed to one space, and for the number/date/bp kinds every Bangla
    digit mapped to 0-9 (those fields are validated and stored as ASCII; the text kind keeps ০-৯ as written).
problems(s): what makes a string unwritable (a virama with nothing before or after it, a vowel sign on nothing), for a label check.
"""
import unicodedata

ZWNJ, ZWJ = "‌", "‍"
VIRAMA, NUKTA, AU_LENGTH = "্", "়", "ৗ"
CANDRABINDU, ANUSVARA, VISARGA = "ঁ", "ং", "ঃ"
KHANDA_TA = "ৎ"
DANDA = "।"
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
BN2EN = str.maketrans(BN_DIGITS, "0123456789")
INDEPENDENT_VOWELS = "অআইঈউঊঋঌএঐওঔৠৡ"
CONSONANTS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ" + "ড়ঢ়য়ৰৱ"      # the three nukta letters appear here only so `is_consonant` answers for un-normalized text
VOWEL_SIGNS = "ািীুূৃৄেৈোৌৗৢৣ"
MARKS = set(VOWEL_SIGNS) | {CANDRABINDU, ANUSVARA, VISARGA, NUKTA, VIRAMA}     # never start a cluster
NUMERIC_KINDS = ("number", "date", "bp")


def is_bengali(ch): return "ঀ" <= ch <= "৿"
def is_consonant(ch): return ch in CONSONANTS or ch == KHANDA_TA
def is_vowel_sign(ch): return ch in VOWEL_SIGNS
def is_bn_digit(ch): return ch in BN_DIGITS
def is_digit(ch): return ch in BN_DIGITS or "0" <= ch <= "9"


def normalize(s, kind="text", keep_joiners=False):
    """-> the canonical string the reader is trained on and scored against. kind: text | number | date | bp | mark."""
    s = unicodedata.normalize("NFC", str(s))
    if not keep_joiners: s = s.replace(ZWJ, "").replace(ZWNJ, "")
    s = " ".join(s.split())
    if kind in NUMERIC_KINDS: s = s.translate(BN2EN)
    return s


def problems(s):
    """-> [reason, ...] for sequences no hand could write: a virama or a vowel sign with no consonant before it, two viramas in a row,
    a virama at the end. Empty when the string is fine. Works on normalized text."""
    out = []; prev = ""
    for i, ch in enumerate(s):
        if ch == VIRAMA:
            if not prev or not (is_consonant(prev) or prev == NUKTA): out.append(f"virama at {i} follows {prev!r}, not a consonant")
            elif i == len(s) - 1: out.append("virama at the end")
            elif s[i + 1] in (VIRAMA,): out.append(f"two viramas at {i}")
        elif ch in MARKS and ch != NUKTA:
            if not prev or not (is_consonant(prev) or prev in (NUKTA, VIRAMA) or is_vowel_sign(prev) and ch in (CANDRABINDU, ANUSVARA, VISARGA)) and not (is_vowel_sign(prev) and ch == AU_LENGTH):
                if not (prev and is_bengali(prev) and prev not in MARKS and ch in (CANDRABINDU, ANUSVARA, VISARGA)): out.append(f"mark {ch!r} at {i} follows {prev!r}")
        elif ch == NUKTA and not (prev and is_consonant(prev)): out.append(f"nukta at {i} follows {prev!r}")
        prev = ch
    return out


def graphemes(s):
    """Grapheme clusters as a Bengali reader sees them: a base letter with its marks, and a conjunct (consonant + virama + consonant...) as one.
    Any other script: one cluster per codepoint plus following combining marks. Deterministic, no `regex` dependency."""
    out = []; cur = ""; join_next = False
    for ch in s:
        if cur and (ch in MARKS or ch in (ZWJ, ZWNJ) or join_next or (unicodedata.combining(ch) and not is_bengali(ch))):
            cur += ch
        else:
            if cur: out.append(cur)
            cur = ch
        join_next = ch in (VIRAMA, ZWJ, ZWNJ)           # a consonant after a virama (or a joiner) continues the conjunct
    if cur: out.append(cur)
    return out


def words(s):
    return s.split()


def conjuncts(s):
    """The clusters of s that are conjuncts: two or more consonants joined by a virama (ক্ষ, ত্র, জ্ঞ, র্ক ...)."""
    return [g for g in graphemes(s) if VIRAMA in g and sum(is_consonant(c) for c in g) >= 2]


def has_numerals(s): return any(is_digit(c) for c in s)
