"""The worked examples from the HTR plan, as tests: NFC composition/decomposition, grapheme clustering, conjunct detection, numeral
mapping on number/date/bp fields only, and the shape rules that flag an unwritable sequence."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.text as T


class Normalize(unittest.TestCase):
    def test_composes_and_decomposes(self):
        self.assertEqual(T.normalize("ড়"), "ড়")              # already NFC
        self.assertEqual([hex(ord(c)) for c in T.normalize("ড়")], ["0x9a1", "0x9bc"])   # precomposed -> base + nukta
        self.assertEqual(T.normalize("ে" + "া"), "ো")                   # e-matra + aa-matra composes to o-matra

    def test_numeral_mapping_only_for_numeric_kinds(self):
        self.assertEqual(T.normalize("১২৫", kind="number"), "125")
        self.assertEqual(T.normalize("১২৫", kind="date"), "125")
        self.assertEqual(T.normalize("১২৫", kind="text"), "১২৫")

    def test_joiners_and_whitespace(self):
        self.assertEqual(T.normalize("a‌b‍ c"), "ab c")
        self.assertEqual(T.normalize("  a   b  "), "a b")


class Graphemes(unittest.TestCase):
    def test_bangladesh(self):
        s = T.normalize("বাংলাদেশ")
        self.assertEqual(len(s), 8)
        self.assertEqual(T.graphemes(s), ["বাং", "লা", "দে", "শ"])

    def test_conjunct(self):
        s = "ক্ষ"
        self.assertEqual(len(s), 3)
        self.assertEqual(T.graphemes(s), ["ক্ষ"])
        self.assertEqual(T.conjuncts(s), ["ক্ষ"])

    def test_plain_consonant_is_not_a_conjunct(self):
        self.assertEqual(T.conjuncts("কলম"), [])


class Problems(unittest.TestCase):
    def test_well_formed_text_has_none(self):
        self.assertEqual(T.problems(T.normalize("বাংলাদেশ")), [])
        self.assertEqual(T.problems(T.normalize("ক্ষমতা")), [])

    def test_stray_virama_and_vowel_sign_are_flagged(self):
        self.assertTrue(T.problems("্ক"))                 # virama with nothing before it
        self.assertTrue(T.problems("া"))                   # vowel sign with nothing before it
        self.assertTrue(T.problems("ক্্"))             # two viramas


if __name__ == "__main__":
    unittest.main()
