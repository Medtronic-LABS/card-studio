"""AI suggestions for labeling, and what labeling records about them. A stand-in reader is used (no trained model). Run: python -m unittest discover -s tests"""
import json, sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, cleanup, labeling, scans, suggest
from test_scans import Base as ScanBase, ROWS

F = [f"f1_{i}" for i in range(ROWS)]


class FakeReader:
    """Answers "57" for every cell, less sure on each later cell, so order and thresholds can be checked."""
    def __init__(self, name):
        m = cards.meta(); self.classes, self.blank = m["classes"], m["blank_idx"]; self.name = name
        self.man = {"reader": {"input": {"height": 48, "width": 192}}}; self.in_name, self.out_name = "image", "logits"; self.session = self; self.n = 0; self.spec = {}          # a pack's per-field rules; none means greedy reads

    def run(self, _, feed):
        x = feed["image"]; T, V = 8, len(self.classes); out = np.full((T, x.shape[0], V), -20.0, np.float32); out[:, :, self.blank] = 0.0
        for b in range(x.shape[0]):
            margin = 6.0 - 1.0 * self.n; self.n += 1                              # how far the chosen character is ahead of the rest
            for t, ch in ((1, "5"), (3, "7")): out[t, b, self.blank] = -20.0; out[t, b, self.classes.index(ch)] = margin; out[t, b, self.classes.index("3")] = 0.0
        return [out]


class Base(ScanBase):
    def setUp(self):
        super().setUp(); cards.PACKS.mkdir(exist_ok=True)
        for i in range(2): scans.add("alpha", self.scan("alpha", cells=((0, 0), (1, 1), (4, 2)), seed=i), f"p-{i:03d}.png")
        labeling.set_field("alpha", F[0], "number", [30, 150]); labeling.set_field("alpha", F[1], "date")           # F[4] stays text
        self.old_load = cleanup.load_pack; cleanup.load_pack = lambda p: FakeReader(Path(p).stem.replace("_v", " v"))

    def tearDown(self):
        cleanup.load_pack = self.old_load; super().tearDown()

    def pack(self, name): (cards.PACKS / f"{name}.cardpack").write_bytes(b"stand-in")


class Decode(unittest.TestCase):
    def test_masked_greedy_ctc_and_lowest_character_confidence(self):
        m = cards.meta(); c, blank = m["classes"], m["blank_idx"]; lg = np.full((5, len(c)), -9.0); lg[:, blank] = 0.0
        for t, ch, v in ((0, "1", 3.0), (1, "1", 3.0), (3, "2", 1.0)): lg[t, c.index(ch)] = v
        lg[3, c.index("x")] = 9.0                                                   # the best class overall, but not allowed in a number field
        text, conf = suggest._decode(lg, cards.allowed_idx("number", c, blank), c, blank)
        self.assertEqual(text, "12"); self.assertTrue(0.5 < conf < 0.8)             # repeated frames merge; confidence is that of the weakest character
        self.assertEqual(suggest._decode(np.where(np.arange(len(c)) == blank, 0.0, -9.0)[None].repeat(3, 0), [blank, c.index("1")], c, blank), ("", 0.0))


class Source(Base):
    def test_no_reader_then_a_borrowed_one_then_the_cards_own(self):
        self.assertEqual(suggest.status("alpha")["available"], False)
        with self.assertRaises(ValueError): suggest.make("alpha")
        self.pack("other_v3"); s = suggest.status("alpha"); self.assertEqual((s["reader"], s["own_reader"], s["stale"]), ("other v3", False, True))
        self.pack("alpha_v1"); s = suggest.status("alpha"); self.assertEqual((s["reader"], s["own_reader"]), ("alpha v1", True))

    def test_a_borrowed_reader_is_not_asked_about_free_text_and_its_suggestions_are_never_filled_in(self):
        self.pack("other_v3"); r = suggest.make("alpha")
        self.assertEqual((r["reader"], r["own_reader"], r["cells"], r["skipped_text_fields"]), ("other v3", False, 4, F[2:5]))
        self.assertEqual({k[1] for k in suggest.read("alpha")}, {F[0], F[1]})
        q = labeling.queue("alpha")["cells"]; self.assertTrue(all(c["suggestion"] is None or not c["suggestion"]["sure"] for c in q))
        self.assertEqual([c["suggestion"]["text"] for c in q if c["suggestion"]], ["57"] * 4)
        self.assertEqual(suggest.status("alpha")["stale"], False)

    def test_the_cards_own_reader_fills_in_what_it_is_sure_of_and_new_readers_make_suggestions_stale(self):
        self.pack("alpha_v1"); r = suggest.make("alpha"); self.assertEqual((r["own_reader"], r["cells"], r["skipped_text_fields"]), (True, 6, []))
        sure = [c["suggestion"]["sure"] for c in labeling.queue("alpha")["cells"]]; self.assertTrue(any(sure) and not all(sure))
        self.pack("alpha_v2"); s = suggest.status("alpha"); self.assertEqual((s["stored_reader"], s["reader"], s["stale"]), ("alpha v1", "alpha v2", True))


class WhatLabelingRecords(Base):
    def setUp(self):
        super().setUp(); self.pack("alpha_v1"); suggest.make("alpha")

    def test_typed_accepted_and_corrected(self):
        self.assertEqual(labeling.save("alpha", "p-000", F[0], "c1", "57")["how"], "accepted")
        self.assertEqual(labeling.save("alpha", "p-001", F[0], "c1", "58")["how"], "corrected")
        self.assertEqual(labeling.save("alpha", "p-000", F[1], "c2", note="unreadable")["how"], "corrected")
        rows = labeling._read("alpha"); self.assertEqual((rows[("p-001", F[0], "c1")]["suggested"], rows[("p-001", F[0], "c1")]["how"]), ("57", "corrected"))
        o = labeling.overview("alpha"); f = o["fields"][0]
        self.assertEqual((f["ai_right"], f["ai_wrong"], o["ai"]["right"], o["ai"]["wrong"], o["ai"]["count"], o["ai"]["own_reader"]), (1, 1, 1, 2, 6, True))
        self.assertEqual([r["label"] for r in cards.labels("alpha")[0]], ["57", "58"])                    # both are a person's labels and are trained on

    def test_a_cell_without_a_suggestion_is_typed(self):
        (cards.CARDS / "alpha" / "suggestions.csv").unlink()
        self.assertEqual(labeling.save("alpha", "p-000", F[0], "c1", "57")["how"], "typed")

    def test_order_by_confidence(self):
        conf = lambda order: [c["suggestion"]["confidence"] for c in labeling.queue("alpha", order_by=order)["cells"]]
        self.assertEqual(conf("unsure"), sorted(conf("reading"))); self.assertEqual(conf("sure"), sorted(conf("reading"), reverse=True))
        with self.assertRaises(ValueError): labeling.queue("alpha", order_by="random")

    def test_disagreements_are_saved_labels_the_ai_reads_differently_now(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57"); labeling.save("alpha", "p-001", F[0], "c1", "58"); labeling.save("alpha", "p-000", F[1], "c2", note="blank")
        self.assertEqual([(c["page"], c["label"], c["suggestion"]["text"]) for c in labeling.queue("alpha", show="disagree")["cells"]], [("p-001", "58", "57")])

    def test_labels_saved_before_suggestions_existed_still_read(self):
        p = cards.CARDS / "alpha" / "labels.csv"; p.write_text("page,row,col,label,note\n" + f"p-000,{F[0]},c1,60,\n")
        self.assertEqual(labeling._read("alpha")[("p-000", F[0], "c1")], {"label": "60", "note": "", "how": "", "suggested": ""})
        self.assertEqual(labeling.overview("alpha")["fields"][0]["labeled"], 1)


if __name__ == "__main__":
    unittest.main()
