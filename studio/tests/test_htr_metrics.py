"""htr.eval.metrics: CER at codepoint/grapheme level, WER, exact, numeral/conjunct accuracy, breakdown with Wilson CIs."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.eval.metrics as M


class Cer(unittest.TestCase):
    def test_codepoint_vs_grapheme_differ_on_a_conjunct_substitution(self):
        ref, hyp = "ক্ষমতা", "কষমতা"   # conjunct ক্ষ (3 codepoints, 1 cluster) replaced by two plain letters
        self.assertGreater(M.cer_codepoint([(ref, hyp)]), 0)
        self.assertGreater(M.cer_grapheme([(ref, hyp)]), 0)

    def test_exact_match_zero_cer(self):
        self.assertEqual(M.cer_grapheme([("বাংলাদেশ", "বাংলাদেশ")]), 0.0)
        self.assertEqual(M.exact([("বাংলাদেশ", "বাংলাদেশ")]), 1.0)

    def test_wer_counts_words(self):
        self.assertEqual(M.wer([("a b c", "a x c")]), 1 / 3)


class Restricted(unittest.TestCase):
    def test_numeral_accuracy(self):
        self.assertEqual(M.numeral_accuracy([("125", "125")]), 1.0)
        self.assertLess(M.numeral_accuracy([("125", "128")]), 1.0)
        self.assertTrue(__import__("math").isnan(M.numeral_accuracy([("abc", "abc")])))

    def test_conjunct_accuracy(self):
        self.assertEqual(M.conjunct_accuracy([("ক্ষমতা", "ক্ষমতা")]), 1.0)
        self.assertLess(M.conjunct_accuracy([("ক্ষমতা", "কষমতা")]), 1.0)


class Breakdown(unittest.TestCase):
    def test_groups_and_wilson_ci(self):
        rows = [{"ref": "ab", "hyp": "ab", "doc": "x"}, {"ref": "cd", "hyp": "ce", "doc": "x"}, {"ref": "ef", "hyp": "ef", "doc": "y"}]
        b = M.breakdown(rows, "doc")
        self.assertEqual(set(b), {"x", "y"})
        self.assertEqual(b["x"]["n"], 2); self.assertEqual(b["y"]["exact"], 1.0)
        lo, hi = b["x"]["exact_ci"]
        self.assertTrue(0 <= lo <= b["x"]["exact"] <= hi <= 1)


if __name__ == "__main__":
    unittest.main()
