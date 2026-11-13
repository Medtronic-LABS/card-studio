"""Card checks are read from a card's fields.json, and a two-line date that is not a real day is flagged by the clean-up lab's reader."""
import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cards, cleanup, subcells


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name) / "epi"; d.mkdir()
        self.fp = d / "fields.json"; self.cards = mock.patch.object(cards, "CARDS", Path(self.tmp.name))
        self.cards.start(); self.addCleanup(self.cards.stop)
        self.fields = mock.patch.object(cards, "fields", lambda c: [{"key": k} for k in ("bcg", "penta", "opv", "visit_date")]); self.fields.start(); self.addCleanup(self.fields.stop)

    def test_none_asked(self):
        self.assertEqual(cards.checks("epi"), {})
        self.fp.write_text(json.dumps({"fields": {}})); self.assertEqual(cards.checks("epi"), {})

    def test_rules_are_read_and_validated(self):
        self.fp.write_text(json.dumps({"checks": {"same_day": "bcg, penta, opv", "increasing": ["visit_date"]}}))
        self.assertEqual(cards.checks("epi"), {"same_day": ["bcg", "penta", "opv"], "increasing": ["visit_date"]})
        for bad, msg in (({"same_day": ["bcg", "nope"]}, "not in the grid"), ({"same_day": ["bcg"]}, "at least two"), ({"weekly": ["bcg"]}, "unknown rule")):
            self.fp.write_text(json.dumps({"checks": bad}))
            with self.assertRaisesRegex(ValueError, msg): cards.checks("epi")


class StackedDate(unittest.TestCase):
    def _pack(self, decode):
        classes = ["<b>"] + list("0123456789/."); p = mock.Mock(); p.classes = classes; p.blank = 0; p.external = None
        p.spec = {"bcg": {"kind": "date", "layout": "stacked", "decode": decode, "allowed_idx": list(range(len(classes)))}}
        return p

    def _frames(self, classes, text):
        out = []
        for ch in text:
            for hot in (classes.index(ch), 0):
                z = np.full(len(classes), -8.0); z[hot] = 6.0; out.append(z)
        return np.stack(out)

    def _read(self, decode, top, year):
        pack = self._pack(decode); logits = iter([self._frames(pack.classes, top), self._frames(pack.classes, year)])
        pack.session.run = lambda *a, **k: [next(logits)[None]]
        with mock.patch.object(subcells, "parts", lambda crop, layout: (np.zeros((48, 192), np.uint8),) * 2), mock.patch.object(cleanup, "prepare", lambda piece, pack: piece):
            return cleanup.read_two_line(pack, np.zeros((40, 60), np.uint8), "bcg", "stacked")

    def test_a_real_day_keeps_its_confidence_and_a_leap_day_without_a_leap_year_is_flagged(self):
        text, conf = self._read("greedy", "22.2", "26"); self.assertEqual(text, "22/2/26"); self.assertGreater(conf, 0.5)
        text, conf = self._read("greedy", "29.2", "25"); self.assertEqual(text, "29/2/25"); self.assertLessEqual(conf, cleanup.UNREAL_DATE_CONFIDENCE)

    def test_grammar_decode_pulls_an_impossible_top_line_to_a_real_one(self):
        text, _ = self._read("greedy", "31.4", "26"); self.assertEqual(text, "31/4/26")      # April has 30 days: greedy keeps what it saw
        text, conf = self._read("grammar", "31.4", "26"); top = text.rsplit("/", 1)[0]
        self.assertNotEqual(top, "31/4"); self.assertTrue(cleanup.grammar.date_ok(top, "26"), text)
        text, conf = self._read("grammar", "22.2", "26"); self.assertEqual(text, "22/2/26")


if __name__ == "__main__":
    unittest.main()
