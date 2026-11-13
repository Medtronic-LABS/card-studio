"""Grammar decoding: candidates, scoring, and how the pack and the lab use it. Run: python -m unittest discover -s tests"""
import json, sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, constrained as X

try:
    import torch; HAVE_TORCH = True
except Exception: HAVE_TORCH = False

M = cards.meta(); CL, BL = M["classes"], M["blank_idx"]
ALLOWED = cards.allowed_idx("date", CL, BL)


def logits_for(text, steps=24, sure=8.0):
    """A reader output that says `text` clearly: one frame per character with a blank frame between, every other class low."""
    lg = np.full((steps, len(CL)), -6.0, np.float32); lg[:, BL] = 0.0; t = 1
    for ch in text:
        lg[t, :] = -6.0; lg[t, CL.index(ch)] = sure; t += 2
    return lg


class Candidates(unittest.TestCase):
    def test_shapes_and_sizes(self):
        d = X.candidates("date"); self.assertIn("20/9/26", d); self.assertIn("08.9.24", d); self.assertIn("2/2/21", d); self.assertNotIn("32/1/26", d); self.assertNotIn("20/13/26", d); self.assertNotIn("20/9/31", d)
        self.assertEqual(len(d), 40 * 21 * 8 * 2); self.assertEqual(len(X.candidates("date", None, [24, 26])), 40 * 21 * 3 * 2)
        b = X.candidates("bp"); self.assertIn("120/80", b); self.assertNotIn("60/80", b); self.assertEqual(len(b), 151 * 91)
        n = X.candidates("number", [30, 150]); self.assertIn("57", n); self.assertIn("57.5", n); self.assertNotIn("29", n); self.assertNotIn("150.5", n); self.assertEqual(len(n), 121 + 120 * 10)
        with self.assertRaises(ValueError): X.candidates("mark")

    def test_importing_it_needs_no_torch(self):
        import subprocess
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = [%r, %r]; import constrained; assert 'torch' not in sys.modules" % (str(HERE), str(HERE.parent / "cardpack"))], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Decoding(unittest.TestCase):
    def test_a_clear_read_comes_back_unchanged_with_a_margin_near_zero(self):
        text, margin = X.decode(logits_for("20/9/26"), ALLOWED, "date"); self.assertEqual(text, "20/9/26"); self.assertGreater(margin, -1.0)

    def test_a_broken_read_is_pulled_to_the_nearest_valid_date(self):
        # the reader dropped the year's second digit and the second separator: "20/9/2"
        text, margin = X.decode(logits_for("20/9/2"), ALLOWED, "date"); self.assertRegex(text, r"^\d{1,2}[/.]\d{1,2}[/.]\d{2}$"); self.assertTrue(text.startswith("20/9/2"))
        self.assertLess(margin, 0.0)

    def test_the_year_range_is_a_rule(self):
        text, _ = X.decode(logits_for("20/9/31"), ALLOWED, "date", None, [24, 27]); self.assertNotEqual(text, "20/9/31"); self.assertIn(text[-2:], ("24", "25", "26", "27"))

    def test_numbers_and_pressures(self):
        al = cards.allowed_idx("number", CL, BL); self.assertEqual(X.decode(logits_for("57.5"), al, "number", [30, 150])[0], "57.5")
        self.assertEqual(X.decode(logits_for("160"), al, "number", [30, 150])[0] in X.candidates("number", [30, 150]), True)       # out of range: forced inside it
        ab = cards.allowed_idx("bp", CL, BL); self.assertEqual(X.decode(logits_for("120/80"), ab, "bp")[0], "120/80")

    def test_score_orders_candidates_by_ctc_likelihood(self):
        s = X.score(logits_for("20/9/26"), ALLOWED, ["20/9/26", "20/9/25", "2/9/26"]); self.assertEqual(int(s.argmax()), 0)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class PackAndLab(unittest.TestCase):
    def test_validator_rules_for_decode(self):
        import cardpack as cp, synthetic, tempfile
        def check(change):
            m = synthetic.build_members(); fl = json.loads(m["fields.json"]); change(fl["fields"]); m["fields.json"] = json.dumps(fl).encode()
            man = json.loads(m["manifest.json"]); man["files"] = {n: cp.sha256(m[n]) for n in cp.hashed(m)}; m["manifest.json"] = json.dumps(man).encode()
            with tempfile.TemporaryDirectory() as d: p = Path(d) / "s_v1.cardpack"; cp.build(m, p); return cp.validate(p, check_reader=False)
        kinds = {f["key"]: f for f in json.loads(synthetic.build_members()["fields.json"])["fields"]}
        self.assertEqual(check(lambda fs: [f.update(decode="grammar") for f in fs if f["key"] == "bp"]), [])
        self.assertTrue(any("decode grammar" in e for e in check(lambda fs: [f.update(decode="grammar") for f in fs if f["kind"] == "mark"])))
        self.assertTrue(any("decode grammar" in e for e in check(lambda fs: [f.update(decode="grammar", range=None) for f in fs if f["key"] == "pulse"])))
        self.assertTrue(any("date.years" in e for e in check(lambda fs: [f.update(date={"years": [26, 20]}) for f in fs if f["key"] == "bp"])))

    def test_the_lab_reads_a_grammar_field_through_the_grammar(self):
        import cleanup
        class P:
            classes, blank = CL, BL; in_name, out_name = "image", "logits"; external = None
            spec = {"d": {"allowed_idx": ALLOWED, "kind": "date", "range": None, "decode": "grammar", "date": {"years": [24, 27]}}, "g": {"allowed_idx": ALLOWED, "kind": "date", "range": None}}
            class session:
                @staticmethod
                def run(_, feed): return [logits_for("20/9/31")[:, None, :]]
        px = np.zeros((48, 192), np.uint8)
        text, conf = cleanup.read_cell(P, px, "d"); self.assertIn(text[-2:], ("24", "25", "26", "27")); self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(cleanup.read_cell(P, px, "g")[0], "20/9/31")                      # no decode key: greedy, as before


if __name__ == "__main__":
    unittest.main()
