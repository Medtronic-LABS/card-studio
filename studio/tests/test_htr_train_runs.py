"""htr.train.runs: writing one line-dataset run's results.json and scanning a runs root for the /api/runs/table comparison table.
Pure Python (no torch): synthetic dicts stand in for a real htr.train.loop.train() result."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.train.runs as R


class WriteAndScan(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_then_scan_round_trips(self):
        rec = R.write_result("run1", arch_name="crnn-base", arch_hash="abc123", n_params=330000, best_epoch=12,
                              metrics={"exact": 0.81, "cer_grapheme": 0.06, "cer_codepoint": 0.05, "wer": 0.2, "n": 400},
                              model_size_bytes=1_300_000, dataset_version="bn-v1", root=self.tmp)
        self.assertEqual(rec["name"], "run1")
        rows = R.scan(self.tmp)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ok"])
        self.assertEqual(rows[0]["metrics"]["exact"], 0.81)
        self.assertEqual(rows[0]["arch_hash"], "abc123")

    def test_scan_sorts_by_name_and_skips_nothing(self):
        for name in ("zeta", "alpha", "mid"):
            (self.tmp / name).mkdir()
            (self.tmp / name / "results.json").write_text(json.dumps({
                "name": name, "arch_name": "crnn", "arch_hash": "h", "n_params": 1, "best_epoch": 1,
                "metrics": {"exact": 0.5, "cer_grapheme": 0.1, "cer_codepoint": 0.1, "wer": 0.3, "n": 10}}))
        rows = R.scan(self.tmp)
        self.assertEqual([r["name"] for r in rows], ["alpha", "mid", "zeta"])

    def test_scan_reports_a_broken_run_instead_of_dropping_it(self):
        (self.tmp / "bad1").mkdir(); (self.tmp / "bad1" / "results.json").write_text("not json")
        (self.tmp / "bad2").mkdir(); (self.tmp / "bad2" / "results.json").write_text(json.dumps({"name": "bad2"}))
        rows = R.scan(self.tmp)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["ok"] is False for r in rows))
        self.assertIn("missing", rows[1]["problem"])

    def test_scan_of_a_missing_root_is_an_empty_list(self):
        self.assertEqual(R.scan(self.tmp / "no-such-dir"), [])

    def test_a_directory_with_no_results_json_is_skipped(self):
        (self.tmp / "half-written").mkdir()
        self.assertEqual(R.scan(self.tmp), [])

    def test_bad_names_are_refused(self):
        with self.assertRaises(ValueError): R.write_result("a/b", arch_name="x", arch_hash="h", n_params=1, best_epoch=1, metrics={}, root=self.tmp)
        with self.assertRaises(ValueError): R.write_result("", arch_name="x", arch_hash="h", n_params=1, best_epoch=1, metrics={}, root=self.tmp)
        with self.assertRaises(ValueError): R.write_result("..", arch_name="x", arch_hash="h", n_params=1, best_epoch=1, metrics={}, root=self.tmp)


if __name__ == "__main__":
    unittest.main()
