"""Failure browser, paired comparison, recalibration, experiments, provenance. Run: python -m unittest discover -s tests"""
import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis as A, mlstats as M, experiments as E, provenance as P


def mkrun(root, name, cells, folds=None, pack_missing=True):
    d = Path(root) / name; d.mkdir(parents=True)
    (d / "cv_predictions.json").write_text(json.dumps([{"page": p, "row": f, "col": "c1", "label": l, "pred": r, "conf": c} for p, f, l, r, c in cells]))
    pages = sorted({c[0] for c in cells}); folds = folds or [pages[i::5] for i in range(5)]
    (d / "folds.json").write_text(json.dumps({"folds": folds}))
    (d / "results.json").write_text(json.dumps({"pack": {"path": "/nonexistent.cardpack", "id": "anc", "version": 1}, "config": "x", "fields": []}))
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.old = (A.RUNS, M.RUNS); A.RUNS = M.RUNS = Path(self.tmp.name)

    def tearDown(self):
        A.RUNS, M.RUNS = self.old; self.tmp.cleanup()


class Failures(Base):
    def test_kinds(self):
        mkrun(self.tmp.name, "r", [("p1", "pulse", "98", "98", .9), ("p1", "pulse", "98", "96", .95), ("p2", "pulse", "98", "96", .3), ("p2", "bp", "120/80", "", 0.0),
                                    ("p3", "pulse", "70", "70", .2), ("p3", "bp", "1", "7", .85)])
        k = lambda kind, **kw: [(i["page"], i["field"]) for i in A.errors("r", kind=kind, **kw)["items"]]
        self.assertEqual(sorted(k("wrong")), [("p1", "pulse"), ("p2", "bp"), ("p2", "pulse"), ("p3", "bp")])
        self.assertEqual(k("confident_wrong"), [("p1", "pulse"), ("p3", "bp")])                       # >= 0.8, most confident first
        self.assertEqual(k("blank_read"), [("p2", "bp")])
        self.assertEqual(k("flagged_right", field="pulse"), [])                                      # the pack file is missing, so every minimum is 0 and nothing is flagged
        self.assertEqual(A.errors("r", field="bp", kind="wrong")["total"], 2)
        with self.assertRaises(ValueError): A.errors("r", kind="nope")
        with self.assertRaises(KeyError): A.errors("../x")

    def test_ops_are_returned_for_the_crop_caption(self):
        mkrun(self.tmp.name, "r", [("p1", "pulse", "98", "96", .5)] * 1 + [(f"p{i}", "pulse", "1", "1", .9) for i in range(2, 7)])
        it = A.errors("r")["items"][0]
        self.assertEqual([o["op"] for o in it["ops"]], ["ok", "sub"]); self.assertEqual(it["crop"], "/api/crop/p1/pulse/c1.png")


class Paired(Base):
    def test_mcnemar_exact_hand_values(self):
        self.assertEqual(A.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(A.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(A.mcnemar_exact(0, 10), 2 * 1 / 1024)                                    # 2 * P(X <= 0), n = 10
        self.assertAlmostEqual(A.mcnemar_exact(2, 8), 2 * (1 + 10 + 45) / 1024)

    def test_counts_and_interval(self):
        cells_a = [(f"p{i % 10}", f"f{i}", "1", "1" if i % 2 else "2", .9) for i in range(60)]          # A right on odd i
        cells_b = [(f"p{i % 10}", f"f{i}", "1", "1" if (i % 2 or i < 40) else "2", .9) for i in range(60)]   # B also right on even i < 40
        mkrun(self.tmp.name, "a", cells_a); mkrun(self.tmp.name, "b", cells_b)
        r = A.paired("a", "b")
        self.assertEqual((r["common"], r["only_a"], r["only_b"], r["both"], r["neither"]), (60, 0, 20, 30, 10))
        self.assertAlmostEqual(r["diff"], 20 / 60); lo, hi = r["ci_page_bootstrap"]; self.assertLess(lo, r["diff"]); self.assertGreater(hi, r["diff"]); self.assertGreater(lo, 0)
        self.assertAlmostEqual(r["p_mcnemar"], 2 * 1 / 2 ** 20, places=12)
        self.assertEqual(sum(f["n"] for f in r["fields"]), 60)

    def test_only_common_cells_and_refusal(self):
        mkrun(self.tmp.name, "a", [(f"p{i}", "x", "1", "1", .9) for i in range(30)]); mkrun(self.tmp.name, "b", [(f"p{i}", "x", "1", "1", .9) for i in range(20, 50)])
        self.assertEqual(A.paired("a", "b")["common"], 10)
        mkrun(self.tmp.name, "c", [(f"q{i}", "x", "1", "1", .9) for i in range(30)])
        with self.assertRaises(ValueError): A.paired("a", "c")


class Calibration(Base):
    def test_pav(self):
        e, m = A.pav([1, 2, 3, 4, 5], [0, 1, 0, 1, 1])
        self.assertEqual(m.tolist(), [0.0, 0.5, 1.0, 1.0]); self.assertEqual(e.tolist(), [1.0, 3.0, 4.0, 5.0])      # 0 | (1,0 pooled to .5) | 1 | 1
        self.assertAlmostEqual(float(A.apply_iso(e, m, np.array([5]))[0]), 1.0); self.assertAlmostEqual(float(A.apply_iso(e, m, np.array([2.5]))[0]), 0.5)

    def test_overconfident_model_is_pulled_down_and_ece_drops(self):
        rs = np.random.RandomState(0); cells = []
        for i in range(600):
            conf = rs.uniform(.5, 1.0); right = rs.rand() < conf - 0.25                                 # true accuracy is 25 points below the stated confidence
            cells.append((f"p{i % 25}", "f", "1", "1" if right else "2", float(conf)))
        mkrun(self.tmp.name, "r", cells); r = A.recalibrate("r")
        self.assertGreater(r["ece_before"], 0.2); self.assertLess(r["ece_after"], 0.08)
        pc = [c["p_correct"] for c in r["curve"]]; self.assertTrue(all(b >= a - 1e-12 for a, b in zip(pc, pc[1:])))

    def test_policy_cells_shape(self):
        mkrun(self.tmp.name, "r", [(f"p{i}", "f", "1", "1" if i % 3 else "2", .5 + i / 100) for i in range(60)])
        d = A.policy_cells("r"); self.assertEqual(len(d["cells"]), 60); self.assertEqual(set(d["cells"][0]), {"f", "c", "cc", "ok", "r"})


class Experiments(unittest.TestCase):
    def test_seed_stats(self):
        s = E.seed_stats([{"k": 35, "n": 62, "cer": .3}, {"k": 34, "n": 62, "cer": .4}, {"k": 37, "n": 62, "cer": .2}])
        self.assertAlmostEqual(s["mean"], (35 + 34 + 37) / 3 / 62); self.assertEqual(s["seeds"], 3); self.assertAlmostEqual(s["mean_cer"], .3)
        self.assertAlmostEqual(s["sd"], float(np.std([35 / 62, 34 / 62, 37 / 62], ddof=1))); self.assertIsNone(E.seed_stats([{"k": 1, "n": 2}])["sd"])

    def test_save_load_roundtrip_and_id_guard(self):
        with tempfile.TemporaryDirectory() as d:
            old = E.DIR; E.DIR = Path(d)
            try:
                E.save({"id": "t1", "title": "T", "baselines": [{"name": "b", "k": 5, "n": 10}], "treatments": [{"name": "x", "seeds": [{"k": 6, "n": 10}, {"k": 8, "n": 10}]}]})
                e = E.load("t1"); self.assertAlmostEqual(e["baselines"][0]["exact"], .5); self.assertAlmostEqual(e["treatments"][0]["stats"]["mean"], .7)
                self.assertEqual([x["id"] for x in E.list_all()], ["t1"])
                for bad in ("../x", "a b", ""):
                    with self.assertRaises((KeyError, ValueError)): E.load(bad)
                with self.assertRaises(ValueError): E.save({"id": "../x", "title": "x"})
            finally:
                E.DIR = old


class Provenance(Base):
    def test_run_without_records_says_so(self):
        d = mkrun(self.tmp.name, "r", [("p1", "f", "1", "1", .9)]); (d / "merged_labels.csv").write_text("page,row,col,label\np1,f,c1,1\n")
        out = P.describe(d); self.assertFalse(out["recorded_by_trainer"])
        self.assertTrue(any("not recorded" in i["value"] for i in out["items"])); self.assertEqual(out["items"][0]["name"], "labels file hash")

    def test_recorded_run_flags_changed_code(self):
        d = mkrun(self.tmp.name, "r", [("p1", "f", "1", "1", .9)])
        rec = {"recorded_at": "x", "labels_sha256": "a" * 64, "crops": {"n": 1, "missing": 0, "sha256": "b" * 64}, "split_sha256": "c" * 64, "seed": 0, "config": "cfg",
               "preprocessing": "none", "code_sha256": {"studio/trainer.py": "deadbeef"}, "versions": {"python": "3", "numpy": "1"}}
        res = json.loads((d / "results.json").read_text()); res["provenance"] = rec; (d / "results.json").write_text(json.dumps(res))
        items = {i["name"]: i for i in P.describe(d)["items"]}; self.assertIn("changed since the run", items["code"]["value"]); self.assertTrue(P.describe(d)["recorded_by_trainer"])


if __name__ == "__main__":
    unittest.main()
