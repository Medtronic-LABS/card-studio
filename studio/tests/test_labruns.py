"""The experiments panel: what a status file and result files become. Stand-in files only. Run: python -m unittest discover -s tests"""
import json, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import labruns as L


def score(k, n=100):
    return {"all": {"n": n, "correct": k, "exact": k / n, "lo": k / n - .08, "hi": k / n + .08, "cer": .2}, "marks": {"n": 40, "correct": 38, "exact": .95, "lo": .85, "hi": .99, "cer": .05},
            "dates": {"n": 10, "correct": 1, "exact": .1, "lo": 0, "hi": .4, "cer": .5}}


def preds(right):
    return [{"page": f"p-{i}", "row": "pulse", "col": "c1", "label": "57", "pred": "57" if i in right else "58"} for i in range(40)]


class Panel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.old = (L.CAP, L._processes); L.CAP = Path(self.tmp.name); self.procs = {}; L._processes = lambda: self.procs

    def tearDown(self):
        L.CAP, L._processes = self.old; self.tmp.cleanup()

    def write(self, rel, doc): p = L.CAP / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(doc))

    def test_nothing_run_means_nothing_listed(self):
        self.assertEqual(L.jobs(), [])

    def test_state_comes_from_the_process_not_from_how_old_the_status_is(self):
        self.write("full/status.json", {"stage": "finetune", "detail": "big_crnn_small: cross-validation", "elapsed": 7000})
        self.assertEqual(L.jobs()[0]["state"], "stopped")                                # unfinished and no process
        self.procs = {"capacity.py": "01:02:03"}; j = L.jobs()[0]; self.assertEqual((j["state"], j["process_time"], j["stage"]), ("running", "01:02:03", "finetune"))
        self.write("full/status.json", {"stage": "done", "detail": "summary written", "elapsed": 9000}); self.assertEqual(L.jobs()[0]["state"], "done")
        self.write("full/status.json", {"stage": "error", "detail": "Traceback"}); self.procs = {}; self.assertEqual(L.jobs()[0]["state"], "error")

    def test_running_experiments_come_first(self):
        self.write("full/status.json", {"stage": "done", "detail": ""}); self.write("v2/full/status.json", {"stage": "pretrain", "detail": "x"}); self.procs = {"synthexp.py": "00:10"}
        self.assertEqual([j["id"] for j in L.jobs()], ["generator", "capacity"])

    def test_capacity_table_progress_and_paired_test_without_leaking_cell_values(self):
        self.write("full/status.json", {"stage": "pretrain", "detail": "attn_small: pretrain"})
        self.write("full/cv_base_small.json", {"variant": "base", "cond": "small", "synthetic": {"sequences": 4000, "epochs": 15}, "score": score(60), "preds": preds(set(range(20)))})
        self.write("full/cv_big_crnn_large.json", {"variant": "big_crnn", "cond": "large", "synthetic": {"sequences": 30000, "epochs": 4}, "score": score(70), "preds": preds(set(range(35)))})
        j = L.jobs()[0]; self.assertEqual(j["progress"], {"done": 2, "total": 6}); self.assertEqual(len(j["rows"]), 2)
        self.assertEqual(j["rows"][0][-1], "the reference"); self.assertIn("15 cells right that it got wrong, 0 the other way", j["rows"][1][-1]); self.assertIn("p = 0.000", j["rows"][1][-1])
        self.assertEqual((j["rows"][1][0], j["rows"][1][1]), ("bigger CRNN", "4.29M, 17.2 MB"))
        self.assertNotIn('"57"', json.dumps(j)); self.assertNotIn("preds", json.dumps(j))

    def test_other_experiments_have_tables(self):
        self.write("constrained/full/status.json", {"stage": "done", "detail": "x"})
        self.write("constrained/full/summary.json", {"weights": "w.pt", "results": {"dates": {"greedy": {"n": 32, "correct": 3, "exact": .09}, "constrained": {"n": 32, "correct": 8, "exact": .25}}, "label_outside_grammar": {"dates": 0}}})
        self.write("xscript/full/status.json", {"stage": "stopped", "detail": "stopped by the user"})
        self.write("xscript/full/bangla.json", {"scripts": ["bangla"], "tests": {t: {"exact": .95 if t == "bangla" else .06} for t in ("bangla", "kannada", "kannada_dig", "devanagari", "telugu")}})
        by = {j["id"]: j for j in L.jobs()}
        self.assertEqual(by["grammar"]["rows"], [["dates", "3/32 (9%)", "8/32 (25%)"]])
        self.assertEqual((by["crossscript"]["state"], by["crossscript"]["progress"], by["crossscript"]["rows"][0][:3]), ("stopped", {"done": 1, "total": 5}, ["bangla", "95%", "6% *"]))


if __name__ == "__main__":
    unittest.main()
