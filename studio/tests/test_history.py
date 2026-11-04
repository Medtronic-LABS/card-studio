"""Run history, compare, logs, downloads. Run: python -m unittest discover -s tests"""
import csv, io, json, os, sys, tempfile, unittest, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import history as H


def field(name, k, n, lo, hi, status):
    return {"field": name, "n": n, "k": k, "exact": k / n, "lo": lo, "hi": hi, "cer": 0.1, "status": status, "reason": "x"}


def make_run(root, name, fields, cells=369, pages=49, target=85.0, preds=None, log=None, legacy_log=None, stage="done"):
    d = root / name; d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"stage": stage, "detail": "pack written", "target": target, "min_n": 20}))
    if fields is not None:
        (d / "results.json").write_text(json.dumps({"config": "light_lr3e-4", "target": target, "min_n": 20, "n_cells": cells, "n_pages": pages, "minutes": 12.2,
            "cv_all": {"n": cells, "k": 260, "exact": 0.7, "lo": 0.66, "hi": 0.75, "cer": 0.27}, "fields": fields, "pack": {"id": "anc", "version": 4}}))
    if preds is not None: (d / "cv_predictions.json").write_text(json.dumps(preds))
    if log is not None: (d / "log.txt").write_text(log)
    if legacy_log is not None: (root / f"runs_{name}.log").write_text(legacy_log)
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.old = H.RUNS; H.RUNS = Path(self.tmp.name)

    def tearDown(self):
        H.RUNS = self.old; self.tmp.cleanup()


class Listing(Base):
    def test_list_skips_folders_without_status_and_counts_shipped(self):
        make_run(H.RUNS, "run-a", [field("pulse", 18, 25, .5, .86, "needs-review"), field("edema", 31, 32, .84, .99, "ship")], log="x")
        (H.RUNS / "idtest").mkdir(); (H.RUNS / "idtest" / "report.html").write_text("x")
        rs = H.list_runs()
        self.assertEqual([r["name"] for r in rs], ["run-a"])
        self.assertEqual((rs[0]["shipped"], rs[0]["fields"], rs[0]["has_log"], rs[0]["pack"]), (1, 2, True, {"id": "anc", "version": 4}))

    def test_unfinished_run_listed_without_metrics(self):
        make_run(H.RUNS, "run-b", None, stage="cv")
        r = H.list_runs()[0]
        self.assertEqual((r["state"], r["exact"], r["shipped"]), ("cv", None, 0))


class Compare(Base):
    def setUp(self):
        super().setUp()
        make_run(H.RUNS, "a", [field("pulse", 18, 25, .52, .86, "needs-review"), field("edema", 31, 32, .84, .99, "needs-review"), field("bp", 0, 7, 0, .35, "needs-review")])
        make_run(H.RUNS, "b", [field("pulse", 24, 25, .80, .99, "needs-review"), field("edema", 32, 32, .89, 1.0, "ship"), field("glucose", 5, 20, .1, .5, "needs-review")], cells=400, pages=52, target=80.0)

    def test_verdicts_and_status_change(self):
        c = H.compare("a", "b"); rows = {r["field"]: r for r in c["rows"]}
        self.assertEqual(rows["pulse"]["verdict"], "indistinguishable")      # .86 hi vs .80 lo: intervals overlap
        self.assertAlmostEqual(rows["pulse"]["delta"], 24 / 25 - 18 / 25)
        self.assertEqual(rows["edema"]["status_change"], "needs-review -> ship")
        self.assertEqual(rows["bp"]["verdict"], "only in a"); self.assertEqual(rows["glucose"]["verdict"], "only in b")

    def test_clearly_separated_is_better_or_worse(self):
        make_run(H.RUNS, "c", [field("pulse", 5, 25, .08, .39, "needs-review")])
        make_run(H.RUNS, "d", [field("pulse", 24, 25, .80, .99, "needs-review")])
        self.assertEqual(H.compare("c", "d")["rows"][0]["verdict"], "better")
        self.assertEqual(H.compare("d", "c")["rows"][0]["verdict"], "worse")

    def test_caveats_flag_different_data_and_gate(self):
        text = " ".join(H.compare("a", "b")["caveats"])
        self.assertIn("different label sets", text); self.assertIn("different gate settings", text); self.assertIn("correlated", text)

    def test_same_data_has_no_dataset_caveat(self):
        make_run(H.RUNS, "e", [field("pulse", 18, 25, .52, .86, "needs-review")])
        make_run(H.RUNS, "f", [field("pulse", 19, 25, .55, .88, "needs-review")])
        text = " ".join(H.compare("e", "f")["caveats"])
        self.assertNotIn("different label sets", text); self.assertNotIn("different gate", text)

    def test_unfinished_run_refused(self):
        make_run(H.RUNS, "g", None, stage="cv")
        with self.assertRaises(ValueError): H.compare("a", "g")

    def test_bad_names(self):
        for bad in ("../x", "nope", "a b", ""):
            with self.assertRaises(KeyError): H.compare("a", bad)


class Logs(Base):
    def test_tail_is_capped_and_legacy_location_found(self):
        make_run(H.RUNS, "r1", [], log="x" * 500_000)
        t = H.run_log("r1", max_bytes=1000); self.assertEqual((len(t["text"]), t["cut"]), (1000, True))
        make_run(H.RUNS, "r2", [], legacy_log="old log")
        self.assertEqual(H.run_log("r2")["text"], "old log")
        make_run(H.RUNS, "r3", [])
        self.assertFalse(H.run_log("r3")["found"])


class Downloads(Base):
    def test_predictions_csv_content_and_formula_guard(self):
        preds = [{"page": "p-1", "row": "pulse", "col": "c3", "label": "98", "pred": "98", "conf": .91234},
                 {"page": "p-1", "row": "bp", "col": "c1", "label": "120/80", "pred": '12,0"/8', "conf": .5},
                 {"page": "p-2", "row": "edema", "col": "c2", "label": "-", "pred": "-", "conf": .99},
                 {"page": "p-2", "row": "x", "col": "c2", "label": "=SUM(A1)", "pred": "-5", "conf": .1},
                 {"page": "p-3", "row": "x", "col": "c2", "label": "-cmd", "pred": "@x", "conf": .1}]
        make_run(H.RUNS, "p", [], preds=preds)
        rows = list(csv.reader(io.StringIO(H.predictions_csv("p").decode())))
        self.assertEqual(rows[0], ["page", "field", "col", "label", "read", "confidence", "correct"])
        self.assertEqual(rows[1], ["p-1", "pulse", "c3", "98", "98", "0.9123", "1"])
        self.assertEqual(rows[2][4], '12,0"/8')                       # commas and quotes survive a round trip
        self.assertEqual(rows[3][3:5], ["-", "-"])                    # a bare mark is data, left alone
        self.assertEqual(rows[4][3:5], ["'=SUM(A1)", "-5"])           # formula guarded; a negative number is not
        self.assertEqual(rows[5][3:5], ["'-cmd", "'@x"])

    def test_run_file_whitelist(self):
        d = make_run(H.RUNS, "q", [], preds=[]); (d / "reader.onnx").write_bytes(b"onnx")
        self.assertEqual(H.run_file("q", "reader.onnx")[::2], (b"onnx", "q_reader.onnx"))
        for bad in ("../status.json", "merged_labels.csv", "status.json"):
            with self.assertRaises(KeyError): H.run_file("q", bad)
        with self.assertRaises(ValueError): H.run_file("q", "log.txt")
        with self.assertRaises(KeyError): H.run_file("../q", "reader.onnx")

    def test_template_zip(self):
        t = Path(self.tmp.name) / "templates" / "mycard" / "v2"; t.mkdir(parents=True)
        (t / "template.png").write_bytes(b"png"); (t / "layout.json").write_text("{}"); (t / "meta.json").write_text("{}")
        data, name = H.template_zip(Path(self.tmp.name) / "templates", "mycard", 2)
        self.assertEqual(name, "mycard_v2_template.zip")
        self.assertEqual(sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()), ["layout.json", "meta.json", "template.png"])
        self.assertEqual(H.template_zip(Path(self.tmp.name) / "templates", "mycard", 2)[0], data)   # deterministic
        for bad in (("../x", 1), ("mycard", 9)):
            with self.assertRaises(KeyError): H.template_zip(Path(self.tmp.name) / "templates", *bad)


if __name__ == "__main__":
    unittest.main()
