"""How much labeled data is enough: the evidence rule, the curve fit, and the per-card summary. Run: python -m unittest discover -s tests"""
import json, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import datasize as D, history


class Evidence(unittest.TestCase):
    def test_four_verdicts(self):
        self.assertEqual(D.evidence(33, 34, 0.85)["verdict"], "ship")
        self.assertEqual(D.evidence(18, 25, 0.85), {"verdict": "not-accurate", "need": None})            # 72% right: no number of cells proves 85%
        self.assertEqual(D.evidence(0, 0, 0.85), {"verdict": "no-data", "need": None})
        e = D.evidence(32, 34, 0.85); self.assertEqual(e["verdict"], "more-cells"); self.assertGreater(e["need"], 34)

    def test_need_is_the_smallest_size_that_clears_the_target_at_the_same_share_right(self):
        for k, n in ((32, 34), (8, 8), (22, 23), (1, 1)):
            need = D.evidence(k, n, 0.85)["need"]; p = k / n
            self.assertGreaterEqual(D.wilson_lo(int(p * need), need), 0.85); self.assertLess(D.wilson_lo(int(p * (need - 1)), need - 1), 0.85)
        self.assertEqual(D.evidence(8, 8, 0.85)["need"], 22)             # even a perfect record needs 22 cells before the low end reaches 85%

    def test_an_accuracy_barely_above_the_target_needs_more_cells_than_can_be_estimated(self):
        self.assertEqual(D.evidence(8501, 10000, 0.85), {"verdict": "more-cells", "need": None})


class Curve(unittest.TestCase):
    def test_a_power_law_is_recovered_and_extended(self):
        sizes = [50, 100, 200, 400]; err = [2.0 * s ** -0.5 for s in sizes]
        a, b = D.fit_power(sizes, err); self.assertAlmostEqual(a, 2.0, places=6); self.assertAlmostEqual(b, 0.5, places=6)
        self.assertEqual(D.extrapolate(sizes, err, 0.05)["need"], 1600)                                    # 2 * n^-0.5 = 0.05 at n = 1600

    def test_no_number_when_error_is_not_falling_or_points_are_missing(self):
        flat = D.extrapolate([50, 100, 200, 400], [0.30, 0.31, 0.29, 0.30], 0.15); self.assertIsNone(flat["need"]); self.assertIn("not falling", flat["note"])
        self.assertIn("too few", D.extrapolate([50, 100], [0.3, 0.2], 0.05)["note"])
        slow = D.extrapolate([50, 100, 200, 400], [0.5 * s ** -0.12 for s in (50, 100, 200, 400)], 0.001); self.assertIsNone(slow["need"]); self.assertIn("too slowly", slow["note"])

    def test_far_beyond_what_was_measured_is_rounded_and_called_out_of_reach(self):
        sizes = [74, 147, 221, 296]; r = D.extrapolate(sizes, [2.6 * n ** -0.263 for n in sizes], 0.15)
        self.assertEqual(r["need"] % 1000, 0); self.assertGreater(r["need"], D.FAR * 296); self.assertIn("will not get there", r["note"])
        self.assertIsInstance(D.extrapolate([74.3, 147.3], [0.3, 0.1], 0.15)["need"], int)

    def test_a_target_already_met_at_a_measured_size(self):
        r = D.extrapolate([50, 100, 200], [0.3, 0.12, 0.08], 0.15); self.assertEqual((r["need"], r["note"]), (100, "already reached at a size that was measured"))


class Summary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name); self.old = (history.RUNS, D.CURVES); history.RUNS, D.CURVES = t / "runs", t / "curves"
        for name, cells, fields in (("run-1", 100, [("pulse", 20, 12)]), ("run-2", 200, [("pulse", 50, 49), ("edema", 34, 32), ("weight_kg", 20, 9)])):
            d = history.RUNS / name; d.mkdir(parents=True); (d / "status.json").write_text(json.dumps({"stage": "done"}))
            fl = [{"field": f, "n": n, "k": k, "exact": k / n, "lo": D.wilson_lo(k, n), "hi": 1.0, "cer": 0.1, "status": "needs-review", "reason": ""} for f, n, k in fields]
            (d / "results.json").write_text(json.dumps({"target": 85.0, "min_n": 20, "n_cells": cells, "n_pages": cells // 10, "fields": fl, "unlabeled_fields": ["temp_f"],
                                                        "cv_all": {"exact": 0.5 + cells / 1000, "lo": 0.4, "hi": 0.8, "cer": 0.2}}))
        import os; os.utime(history.RUNS / "run-1" / "status.json", (1, 1))

    def tearDown(self):
        history.RUNS, D.CURVES = self.old; self.tmp.cleanup()

    def test_latest_run_is_judged_field_by_field_and_runs_are_listed_oldest_first(self):
        s = D.summary("anc")
        self.assertEqual((s["latest_run"], [h["cells"] for h in s["history"]], s["counts"]), ("run-2", [100, 200], {"ship": 1, "more-cells": 1, "not-accurate": 1, "no-data": 1}))
        f = {x["field"]: x for x in s["fields"]}
        self.assertEqual((f["pulse"]["verdict"], f["weight_kg"]["verdict"], f["temp_f"]["verdict"], f["edema"]["more"]), ("ship", "not-accurate", "no-data", f["edema"]["need"] - 34))
        self.assertEqual((s["more_cells_total"], f["pulse"]["kind"], s["curve"], s["status"]), (f["edema"]["more"], "number", None, None))

    def test_a_stricter_target_changes_the_verdicts_and_unknown_cards_are_refused(self):
        self.assertEqual(D.summary("anc", 97)["counts"]["ship"], 0)
        with self.assertRaises(KeyError): D.summary("nosuch")

    def test_a_stored_curve_is_returned(self):
        d = D.CURVES / "anc"; d.mkdir(parents=True); (d / "curve.json").write_text(json.dumps({"groups": {}})); (d / "status.json").write_text(json.dumps({"stage": "done"}))
        s = D.summary("anc"); self.assertEqual((s["curve"], s["status"]["stage"]), ({"groups": {}}, "done"))


if __name__ == "__main__":
    unittest.main()
