"""Seed experiments: paired comparison over cells with a page bootstrap, the seed spread beside it, and the count of settings tried."""
import json, random, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE.parent / "work_phaseA"))
import seedexp as X


def experiment(root, card, name, p_right, seeds=3, pages=20, per_page=5, seed0=1, settings=None):
    """Fabricate finished seeds: every cell is read right with probability p_right (drawn per seed)."""
    d = root / card / name
    for s in range(seeds):
        rng = random.Random(seed0 * 100 + s); preds = []
        for pg in range(pages):
            for c in range(per_page):
                ok = rng.random() < p_right
                preds.append({"page": f"p-{pg:03d}", "row": "weight", "col": f"c{c}", "label": "50", "pred": "50" if ok else "5", "conf": 0.5})
        (d / f"seed-{s}").mkdir(parents=True); (d / f"seed-{s}" / "cv_predictions.json").write_text(json.dumps(preds))
    (d / "status.json").write_text(json.dumps({"state": "done", "seeds": seeds, "seeds_done": seeds, "arch_label": "the shipped CRNN", "settings_hash": X.settings_hash(settings), "started": "x", "detail": ""}))


class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name) / "seedexp"
        p = mock.patch.object(X, "ROOT", self.root); p.start(); self.addCleanup(p.stop)

    def test_a_clearly_better_setting_is_called_better_with_an_interval_above_zero(self):
        experiment(self.root, "c", "a", 0.30, seed0=1); experiment(self.root, "c", "b", 0.60, seed0=2, settings={"hp": {"lr": 0.001}})
        r = X.compare("c", "a", "b")
        self.assertEqual(r["verdict"], "b is better"); self.assertGreater(r["ci95"][0], 0); self.assertGreater(r["diff"], 0.15); self.assertGreater(r["only_b_better"], r["only_a_better"]); self.assertLess(r["sign_p"], 0.05)
        self.assertEqual((r["cells"], r["pages"]), (100, 20)); self.assertIsNone(r["warning"] if r["diff"] > r["noise"] else "x")

    def test_the_same_setting_twice_cannot_be_told_apart_and_says_so(self):
        experiment(self.root, "c", "a", 0.5, seed0=1); experiment(self.root, "c", "b", 0.5, seed0=3)
        r = X.compare("c", "a", "b")
        self.assertEqual(r["verdict"], "no clear difference"); self.assertLessEqual(r["ci95"][0], 0); self.assertGreaterEqual(r["ci95"][1], 0); self.assertIn("cannot tell", r["warning"])

    def test_a_difference_smaller_than_the_seed_spread_is_flagged_even_if_the_interval_misses_zero(self):
        experiment(self.root, "c", "a", 0.50, seeds=4, pages=60, seed0=1); experiment(self.root, "c", "b", 0.50, seeds=4, pages=60, seed0=4)
        with mock.patch.object(X, "summary", lambda card, name: {"name": name, "seeds": 4, "cells": 300, "exact": [0.4, 0.5, 0.6, 0.45], "mean": 0.49, "spread": [0.4, 0.6], "by_field": {}}):
            r = X.compare("c", "a", "b")
        self.assertAlmostEqual(r["noise"], 0.2, places=3)

    def test_summary_gives_each_seeds_exact_and_the_spread(self):
        experiment(self.root, "c", "a", 0.5, seeds=3)
        s = X.summary("c", "a"); self.assertEqual((s["seeds"], s["cells"], len(s["exact"])), (3, 100, 3)); self.assertEqual(s["spread"], [min(s["exact"]), max(s["exact"])]); self.assertIn("weight", s["by_field"])

    def test_missing_seeds_are_refused(self):
        with self.assertRaisesRegex(ValueError, "finished seeds"): X.summary("c", "nope")
        with self.assertRaisesRegex(ValueError, "both experiments"): X.compare("c", "a", "b")


class Guardrails(unittest.TestCase):
    def test_run_checks_its_arguments_before_doing_anything(self):
        with mock.patch.object(X.cards, "exists", lambda c: c == "c"):
            for args, msg in ((("c", "bad name!"), "letters"), (("nope", "ok"), "no such card"), (("c", "ok", None, 1), "2 to"), (("c", "ok", None, 99), "2 to")):
                with self.assertRaisesRegex(ValueError, msg): X.run(*args)

    def test_the_count_of_settings_tried_warns_after_a_few(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(X, "ROOT", Path(t) / "seedexp"), mock.patch.object(X, "HERE", Path(t)):
            root = Path(t) / "seedexp"
            for i in range(3): experiment(root, "c", f"e{i}", 0.5, seeds=2, seed0=i, settings={"hp": {"lr": 0.0001 * (i + 1)}})
            t3 = X.tried("c"); self.assertEqual((t3["settings_tried"], t3["message"]), (3, None))
            experiment(root, "c", "e3", 0.5, seeds=2, seed0=9, settings={"hp": {"lr": 0.5}}); t4 = X.tried("c")
            self.assertEqual(t4["settings_tried"], 4); self.assertIn("optimistic", t4["message"])
            experiment(root, "c", "e4", 0.5, seeds=2, seed0=10, settings={"hp": {"lr": 0.5}}); self.assertEqual(X.tried("c")["settings_tried"], 4)           # the same settings again are not a new try


if __name__ == "__main__":
    unittest.main()
