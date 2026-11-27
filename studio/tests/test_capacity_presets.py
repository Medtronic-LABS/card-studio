"""big-crnn and attn (schema-2 presets) are the two readers studio/capacity.py compares by hand. Run: python -m unittest discover -s tests"""
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack"))
import htr


class CapacityPresets(unittest.TestCase):
    def test_both_are_selectable_presets_that_normalize(self):
        for k in ("big-crnn", "attn"):
            self.assertIn(k, htr.PRESETS); self.assertEqual(htr.describe(htr.normalize(k))["time_steps"], 48)

    def test_parameter_counts_match_capacity_py(self):
        try: import capacity, torch
        except ImportError: self.skipTest("torch / capacity.py dependencies not installed")
        n = 129
        mine = lambda k: htr.describe(htr.normalize(k), n)["params"]; theirs = lambda v: sum(p.numel() for p in capacity.make(v, n).parameters())
        self.assertEqual(mine("big-crnn"), theirs("big_crnn"))
        self.assertLess(abs(mine("attn") - theirs("attn")) / theirs("attn"), 0.005)           # capacity.py's attn also learns a 48 x 288 position table the catalog's transformer lacks


if __name__ == "__main__":
    unittest.main()
