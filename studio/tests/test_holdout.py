"""The held-out test set: reserved once (whole writers when codes exist), invisible to everything that trains or tunes, scored with a look counter."""
import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE.parent / "work_phaseA"))
import cards, testset


def rows(n_pages, per_page=3):
    return [{"page": f"p-{i:03d}", "row": "weight", "col": f"c{j}", "label": "50", "crop": ""} for i in range(n_pages) for j in range(per_page)]


class Reserve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(cards, "CARDS", Path(self.tmp.name)); p.start(); self.addCleanup(p.stop)
        (Path(self.tmp.name) / "c1").mkdir()
        self.rows = rows(40); q = mock.patch.object(cards, "_labels", lambda card: (list(self.rows), {})); q.start(); self.addCleanup(q.stop)

    def test_pages_are_reserved_once_and_left_out_of_every_label_read(self):
        before = cards.labels("c1")[0]; self.assertEqual(len(before), 120)
        h = cards.make_holdout("c1"); self.assertEqual((h["unit"], len(h["pages"])), ("page", 8))
        train, test = cards.labels("c1")[0], cards.labels("c1", holdout=True)[0]
        self.assertEqual((len(train), len(test)), (96, 24)); self.assertFalse({r["page"] for r in train} & {r["page"] for r in test}); self.assertEqual(h["labeled_cells"], 24)
        self.assertEqual(cards.make_holdout.__name__, "make_holdout")
        with self.assertRaisesRegex(ValueError, "already has a test set"): cards.make_holdout("c1")
        st = cards.holdout_status("c1"); self.assertEqual((st["exists"], st["test_pages"], st["train_pages"], st["test_cells"]), (True, 8, 32, 24))

    def test_the_draw_is_deterministic_and_whole_writers_when_codes_exist(self):
        w = Path(self.tmp.name) / "c1" / "writers.csv"
        w.write_text("page,writer\n" + "".join(f"p-{i:03d},W{i % 8:02d}\n" for i in range(40)))
        h = cards.make_holdout("c1"); self.assertEqual((h["unit"], h["units"]), ("writer", 2))
        writers = {r["page"]: f"W{int(r['page'][2:]) % 8:02d}" for r in self.rows}
        held = {writers[p] for p in h["pages"]}; kept = {writers[r["page"]] for r in cards.labels("c1")[0]}
        self.assertEqual(len(held), 2); self.assertFalse(held & kept)                       # a writer is on one side only
        again = tempfile.TemporaryDirectory(); self.addCleanup(again.cleanup)
        cards.holdout_path("c1").unlink(); self.assertEqual(cards.make_holdout("c1")["pages"], h["pages"])

    def test_too_few_pages_or_a_bad_fraction_is_refused_and_writes_nothing(self):
        self.rows = rows(11)
        with self.assertRaisesRegex(ValueError, "need 10"): cards.make_holdout("c1")
        self.assertIsNone(cards.holdout("c1"))
        self.rows = rows(40)
        for bad in (0.01, 0.9):
            with self.assertRaisesRegex(ValueError, "5% to 50%"): cards.make_holdout("c1", bad)
        self.assertIsNone(cards.holdout("c1"))


class Looks(unittest.TestCase):
    def test_a_run_trained_on_a_test_page_is_refused_and_no_set_is_refused(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(cards, "CARDS", Path(t)), mock.patch.object(testset, "RUNS", Path(t) / "runs"):
            (Path(t) / "c1").mkdir()
            with self.assertRaisesRegex(ValueError, "no test set"): testset.score("c1")
            (Path(t) / "c1" / "holdout.json").write_text(json.dumps({"pages": ["p-001"], "unit": "page", "units": 1, "created": "x"}))
            with self.assertRaisesRegex(ValueError, "no finished run"): testset.score("c1")
            r = Path(t) / "runs" / "run-1"; r.mkdir(parents=True)
            (r / "results.json").write_text(json.dumps({"card": "c1"})); (r / "reader.onnx").write_bytes(b"x"); (r / "folds.json").write_text(json.dumps({"folds": [["p-001", "p-002"]]}))
            with self.assertRaisesRegex(ValueError, "trained on 1 page"): testset.score("c1", "run-1")
            self.assertEqual(testset.looks("c1"), [])                                       # a refused look is not counted
            self.assertEqual(testset.newest_run("c1"), "run-1")


if __name__ == "__main__":
    unittest.main()
