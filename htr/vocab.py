"""The reader's classes: a versioned vocabulary (vocabulary.json) with kind masks. Pure Python.

    {"version": "bn-v1", "sha256": "...", "blank": 0, "unk": 1, "classes": ["<blank>", "<unk>", " ", "0", ... "ৎ"],
     "groups": {"consonant": [...], "vowel_sign": [...], "bn_digit": [...], "ascii_digit": [...], "punct": [...], ...},
     "kinds": {"number": "-.0123456789০১২৩৪৫৬৭৮৯", "date": ..., "bp": ..., "mark": "-xy", "text": "*"}}

Index 0 is the CTC blank, 1 is <unk> (a label character outside the vocabulary; it is never emitted by decoding). The default `bn-v1`
holds ASCII (digits, letters, the punctuation cards use), the Bengali block and ০-৙ as their own classes: a number field *reads* what is
written and `text.normalize(kind="number")` maps it to 0-9 afterwards, so its masks allow both digit sets.

A vocabulary made from an old model_meta.json (`Vocabulary.from_classes`) keeps that pack's 129 classes and index order, so old packs and
registered weights keep working; its sha differs, and weights are only ever loaded for the sha they were trained with.
"""
import hashlib, json
from pathlib import Path

from . import text as T

BLANK, UNK = "<blank>", "<unk>"
ASCII_PUNCT = " ().,/-:|"
BENGALI_PUNCT = T.DANDA
KIND_EXTRA = {"number": "-.", "date": "-./", "bp": "/", "mark": "-xy"}                       # plus both digit sets for the first three
DEFAULT_VERSION = "bn-v1"


def _bengali_block():
    """Every assigned letter, sign and digit of the Bengali block that a handwritten line may hold, in codepoint order; the composed ড় ঢ় য় are left out
    (NFC writes them as base + nukta) and so are the archaic ৰ ৱ (they share their shapes with র and ব)."""
    skip = {0x09DC, 0x09DD, 0x09DF, 0x09F0, 0x09F1}
    out = []
    for cp in range(0x0980, 0x09FF):
        if cp in skip: continue
        ch = chr(cp)
        try: name = __import__("unicodedata").name(ch)
        except ValueError: continue
        if "CURRENCY" in name or "FRACTION" in name or "ISSHAR" in name or "GANDA" in name or "ABBREVIATION" in name: continue
        out.append(ch)
    return out


def default_classes():
    cls = [BLANK, UNK] + list(ASCII_PUNCT) + [chr(c) for c in range(ord("0"), ord("9") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    cls += [BENGALI_PUNCT] + _bengali_block()
    assert len(cls) == len(set(cls))
    return cls


def groups_of(classes):
    g = {"consonant": [], "independent_vowel": [], "vowel_sign": [], "sign": [], "bn_digit": [], "ascii_digit": [], "latin": [], "punct": [], "control": []}
    for i, c in enumerate(classes):
        if c in (BLANK, UNK): g["control"].append(i)
        elif T.is_consonant(c): g["consonant"].append(i)
        elif c in T.INDEPENDENT_VOWELS: g["independent_vowel"].append(i)
        elif T.is_vowel_sign(c): g["vowel_sign"].append(i)
        elif c in (T.VIRAMA, T.NUKTA, T.CANDRABINDU, T.ANUSVARA, T.VISARGA, T.AU_LENGTH) or (T.is_bengali(c) and c not in T.BN_DIGITS): g["sign"].append(i)
        elif T.is_bn_digit(c): g["bn_digit"].append(i)
        elif "0" <= c <= "9": g["ascii_digit"].append(i)
        elif c.isascii() and c.isalpha(): g["latin"].append(i)
        else: g["punct"].append(i)
    return g


def default_kinds():
    digits = "0123456789" + T.BN_DIGITS
    return {"number": KIND_EXTRA["number"] + digits, "date": KIND_EXTRA["date"] + digits, "bp": KIND_EXTRA["bp"] + digits, "mark": KIND_EXTRA["mark"], "text": "*"}


class Vocabulary:
    def __init__(self, classes, kinds=None, version=DEFAULT_VERSION, blank=0, unk=1):
        if classes[blank] != BLANK: raise ValueError(f"class {blank} must be {BLANK}")
        if unk is not None and (unk >= len(classes) or classes[unk] != UNK): raise ValueError(f"class {unk} must be {UNK}")
        if len(set(classes)) != len(classes): raise ValueError("duplicate classes")
        self.classes, self.blank, self.unk, self.version = list(classes), blank, unk, version
        self.kinds = dict(kinds or default_kinds())
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.groups = groups_of(self.classes)
        self._masks = {}

    # ---- construction
    @classmethod
    def default(cls): return cls(default_classes())

    @classmethod
    def from_classes(cls, classes, version="legacy-129"):
        """An old pack's class list (model_meta.json: blank first, no <unk>), kept in its order so the weights still fit."""
        return cls(list(classes), kinds={"number": "-.0123456789", "date": "-./0123456789", "bp": "/0123456789", "mark": "-xy", "text": "*"}, version=version, unk=None)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        v = cls(d["classes"], d.get("kinds"), d.get("version", "?"), d.get("blank", 0), d.get("unk"))
        if d.get("sha256") and d["sha256"] != v.sha256: raise ValueError(f"{path}: sha256 does not match its classes (file edited?)")
        return v

    def to_dict(self):
        return {"version": self.version, "sha256": self.sha256, "blank": self.blank, "unk": self.unk, "n": len(self.classes), "classes": self.classes, "groups": self.groups, "kinds": self.kinds}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps({"classes": self.classes, "kinds": self.kinds, "blank": self.blank, "unk": self.unk}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def __len__(self): return len(self.classes)

    # ---- text <-> ids
    def encode(self, s, kind="text", strict=False):
        """normalized string -> class ids. Unknown characters become <unk> (or raise when strict, or when the vocabulary has no <unk>)."""
        out = []
        for ch in s:
            i = self.index.get(ch)
            if i is None:
                if strict or self.unk is None: raise ValueError(f"{ch!r} (U+{ord(ch):04X}) is not in vocabulary {self.version}")
                i = self.unk
            out.append(i)
        return out

    def decode(self, ids):
        return "".join(self.classes[i] for i in ids if i != self.blank and i != self.unk)

    def unknown_chars(self, s):
        return sorted({ch for ch in s if ch not in self.index})

    # ---- kind masks
    def allowed(self, kind):
        """Class indices a field of this kind may emit (always includes the blank). Pure, cached."""
        if kind not in self.kinds: raise ValueError(f"unknown field kind {kind!r} (use one of {', '.join(self.kinds)})")
        if kind not in self._masks:
            spec = self.kinds[kind]
            if spec == "*": idx = [i for i in range(len(self.classes)) if i != self.unk]
            else: idx = sorted({self.blank} | {self.index[c] for c in spec if c in self.index})
            self._masks[kind] = idx
        return list(self._masks[kind])

    def mask(self, kind):
        """-> list of bool, one per class."""
        m = [False] * len(self.classes)
        for i in self.allowed(kind): m[i] = True
        return m

    def chars(self, kind):
        """The characters (no blank) a kind may hold: what a label of that kind is checked against."""
        return "".join(self.classes[i] for i in self.allowed(kind) if i not in (self.blank, self.unk))
