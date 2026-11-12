"""Grammar decoding reference (grammar.py) and its stored vectors. numpy only. Run: python cardpack/test_grammar.py"""
import json, sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grammar as G

VECTORS = json.loads((HERE / "reference" / "grammar_vectors.json").read_text())


class Candidates(unittest.TestCase):
    def test_what_each_kind_can_hold(self):
        d = G.candidates("date"); self.assertEqual(len(d), 40 * 21 * 8 * 2); self.assertEqual(d, sorted(d))
        for ok in ("20/9/26", "08.9.24", "2/2/21"): self.assertIn(ok, d)
        for bad in ("32/1/26", "20/13/26", "20/9/28", "20/9/19", "20-9-26"): self.assertNotIn(bad, d)
        self.assertEqual(len(G.candidates("date", None, [24, 26])), 40 * 21 * 3 * 2)
        b = G.candidates("bp"); self.assertEqual(len(b), 151 * 91); self.assertIn("120/80", b); self.assertNotIn("60/80", b)
        n = G.candidates("number", [30, 150]); self.assertEqual(len(n), 121 + 120 * 10); self.assertIn("57.5", n); self.assertNotIn("29", n)
        with self.assertRaises(ValueError): G.candidates("mark")


class Decoding(unittest.TestCase):
    def test_every_stored_vector_reproduces(self):
        for v in VECTORS["vectors"]:
            text, margin = G.decode(np.array(v["logits"]), v["chars"], v["blank"], v["kind"], v["range"], v["years"])
            self.assertEqual(text, v["text"], v["note"]); self.assertAlmostEqual(margin, v["margin"], places=9, msg=v["note"])

    def test_clear_reads_come_back_as_read_and_broken_ones_are_pulled_to_valid_values(self):
        by = {v["note"]: v for v in VECTORS["vectors"] if v["kind"] == "date"}
        self.assertEqual(by["clear read"]["text"], "20/9/26"); self.assertGreater(by["clear read"]["margin"], -1)
        self.assertEqual(by["year outside the range: pulled inside it"]["text"][-2:], "27"); self.assertLess(by["year outside the range: pulled inside it"]["margin"], -5)
        for v in VECTORS["vectors"]: self.assertIn(v["text"], G.candidates(v["kind"], v["range"], v["years"]))

    def test_a_candidate_with_a_character_the_field_cannot_emit_is_never_chosen(self):
        v = next(x for x in VECTORS["vectors"] if x["note"] == "clear read with dots"); keep = [i for i, c in enumerate(v["chars"]) if c != "."]
        chars = [v["chars"][i] for i in keep]; lg = np.array(v["logits"])[:, keep]
        text, _ = G.decode(lg, chars, chars.index("<blank>"), "date"); self.assertNotIn(".", text)

    def test_likelihood_is_a_sum_over_alignments_so_the_margin_can_be_positive(self):
        v = next(x for x in VECTORS["vectors"] if x["note"] == "broken read: year cut short"); self.assertGreater(v["margin"], 0)

    def test_nothing_writable_gives_nothing(self):
        self.assertEqual(G.decode(np.zeros((6, 2)), ["<blank>", "x"], 0, "bp"), ("", float("-inf")))


class TwoLineDate(unittest.TestCase):
    """The top line of a date written on two lines (day.month) is its own grammar; the year line is checked against it."""
    def test_daymonth_holds_only_days_that_month_has(self):
        c = G.candidates("daymonth"); self.assertEqual(c, sorted(c))
        for ok in ("22.2", "1/12", "01.02", "29/2", "30.4", "31.12"): self.assertIn(ok, c)
        for bad in ("30.2", "31.4", "00.5", "5.13", "32.1", "22-2", "22.2.26"): self.assertNotIn(bad, c)

    def test_date_ok_checks_the_two_lines_together(self):
        for top, yr in (("22.2", "26"), ("29/2", "2024"), ("01.02", "26")): self.assertTrue(G.date_ok(top, yr), (top, yr))
        for top, yr in (("29.2", "25"), ("22.2", ""), ("22.2", "2"), ("22.2", "202"), ("", "26"), ("22", "26")): self.assertFalse(G.date_ok(top, yr), (top, yr))

    def test_an_impossible_greedy_read_is_pulled_to_a_real_day(self):
        chars = ["<b>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/", "."]; blank = 0
        def frame(**p):
            z = np.full(len(chars), -8.0); z[blank] = -8.0
            for ch, v in p.items(): z[chars.index({"dot": ".", "b": "<b>"}.get(ch, ch.lstrip("c")))] = v
            return z
        # frames spell "31.4": greedy would say 31.4, but April has 30 days and the "1" frame hedges towards a blank
        lg = np.stack([frame(c3=6), frame(b=6), frame(c1=2, b=1.5, c0=1.5), frame(b=6), frame(dot=6), frame(b=6), frame(c4=6)])
        text, margin = G.decode(lg, chars, blank, "daymonth")
        self.assertNotEqual(text, "31.4"); self.assertIn(text, G.candidates("daymonth")); self.assertTrue(text.endswith(".4")); self.assertTrue(np.isfinite(margin))


if __name__ == "__main__":
    unittest.main()
