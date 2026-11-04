"""ML-engineer metrics. Hand-computable cases, then a check against the numbers the trainer itself wrote. Run: python -m unittest discover -s tests"""
import json, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mlstats as M


def row(label, pred, conf, f="pulse", page="p-1"):
    return {"page": page, "row": f, "col": "c1", "label": label, "pred": pred, "conf": conf}


class Alignment(unittest.TestCase):
    def ops(self, a, b): return [o for o in M.align_ops(a, b) if o[0] != "ok"]

    def test_basic_edits(self):
        self.assertEqual(self.ops("98", "98"), [])
        self.assertEqual(self.ops("98", "96"), [("sub", "8", "6")])
        self.assertEqual(self.ops("120/80", "12/80"), [("del", "0", "")])
        self.assertEqual(self.ops("98", "918"), [("ins", "", "1")])
        self.assertEqual(self.ops("x", ""), [("del", "x", "")])
        self.assertEqual(self.ops("", "y"), [("ins", "", "y")])

    def test_edit_count_is_levenshtein(self):
        import itertools
        for a, b in itertools.product(["", "1", "12", "120/80", "x", "7.5"], repeat=2):
            self.assertEqual(len(self.ops(a, b)), self._lev(a, b), (a, b))

    @staticmethod
    def _lev(a, b):
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev, dp[0] = dp[0], i
            for j, cb in enumerate(b, 1):
                cur = dp[j]; dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1]); prev = cur
        return dp[len(b)]


class Metrics(unittest.TestCase):
    def test_field_stats(self):
        rows = [row("98", "98", .9), row("98", "96", .6), row("120/80", "12/80", .8, f="bp"), row("x", "", 0.0, f="edema")]
        fs = {f["field"]: f for f in M.field_stats(rows, {"pulse": 0.7})}
        self.assertEqual((fs["pulse"]["n"], fs["pulse"]["exact"], fs["pulse"]["sub"], fs["pulse"]["flagged"]), (2, 0.5, 1, 1))
        self.assertAlmostEqual(fs["pulse"]["cer"], 1 / 4)
        self.assertEqual((fs["bp"]["del"], fs["bp"]["sub"]), (1, 0)); self.assertAlmostEqual(fs["bp"]["cer"], 1 / 6)
        self.assertEqual((fs["edema"]["no_read"], fs["edema"]["del"], fs["edema"]["mean_conf"]), (1, 1, None))

    def test_calibration_and_ece(self):
        rows = [row("1", "1", .95), row("1", "1", .95), row("1", "2", .15), row("1", "1", .15), row("1", "", 0)]
        c = M.calibration(rows)
        self.assertEqual((c["read"], c["no_read"]), (4, 1))
        self.assertEqual(c["bins"][9]["n"], 2); self.assertEqual(c["bins"][9]["acc"], 1.0)
        self.assertEqual(c["bins"][1]["n"], 2); self.assertEqual(c["bins"][1]["acc"], 0.5)
        self.assertAlmostEqual(c["ece"], 0.5 * abs(1.0 - 0.95) + 0.5 * abs(0.5 - 0.15))

    def test_risk_coverage_and_policy(self):
        rows = [row("1", "1", .9), row("1", "2", .8), row("1", "1", .4), row("1", "", 0.0), row("1", "1", .75, f="bp")]
        rc = M.risk_coverage(rows, {"pulse": 0.7, "bp": 0.7})
        at = {round(c["t"], 2): c for c in rc["curve"]}
        self.assertEqual((at[0.0]["coverage"], at[0.0]["accuracy"]), (4 / 5, 3 / 4))             # the blank read is never accepted
        self.assertEqual((at[0.5]["coverage"], at[0.5]["accuracy"]), (3 / 5, 2 / 3))
        self.assertIsNone(at[1.0]["accuracy"])
        p = rc["policy"]
        self.assertEqual((p["accepted"], p["silent_errors"], p["flagged_for_review"], p["blank_reads_of_filled_cells"]), (3, 1, 1, 1))
        self.assertAlmostEqual(p["accuracy"], 2 / 3); self.assertAlmostEqual(p["silent_error_rate"], 1 / 5)

    def test_confusions(self):
        rows = [row("98", "96", .9), row("58", "56", .9), row("1", "7", .9), row("12", "1", .9), row("1", "12", .9)]
        c = M.confusions(rows)
        self.assertEqual(c["top_substitutions"][0], {"true": "8", "read": "6", "n": 2})
        self.assertEqual(c["totals"], {"sub": 3, "ins": 1, "del": 1})
        i, j = c["matrix"]["chars"].index("8"), c["matrix"]["chars"].index("6"); self.assertEqual(c["matrix"]["sub"][i][j], 2)


class Positions(unittest.TestCase):
    def test_first_middle_last(self):
        rows = [row("98", "28", .9), row("98", "96", .9), row("120/80", "12/80", .9), row("120/80", "120/8", .9), row("5", "6", .9), row("98", "98", .9), row("98", "8", .9)]
        p = M.positions(rows)
        self.assertEqual(p, {"multi_cells": 6, "wrong": 5, "first": 2, "middle": 1, "last": 2, "first_deleted": 1})   # 98->28 first sub; 98->96 last; 120/80->12/80 middle; ->120/8 last; 98->8 first del


@unittest.skipUnless((HERE / "runs" / "run1" / "results.json").exists(), "needs runs/run1")
class AgainstTrainerNumbers(unittest.TestCase):
    def test_matches_what_the_trainer_wrote(self):
        rep = M.run_report("run1"); res = json.loads((HERE / "runs" / "run1" / "results.json").read_text())
        self.assertEqual(rep["overall"]["n"], res["cv_all"]["n"])
        self.assertAlmostEqual(rep["overall"]["exact"], res["cv_all"]["exact"], places=9)
        self.assertAlmostEqual(rep["overall"]["cer"], res["cv_all"]["cer"], places=6)
        theirs = {f["field"]: f for f in res["fields"]}
        for f in rep["fields"]:
            self.assertEqual((f["n"], round(f["exact"], 9)), (theirs[f["field"]]["n"], round(theirs[f["field"]]["exact"], 9)), f["field"])
        o = rep["overall"]; self.assertEqual(o["sub"] + o["ins"] + o["del"], round(o["cer"] * rep["data"]["cells"] * rep["data"]["mean_len"]))
        self.assertGreater(rep["model"]["latency_ms_median"], 0); self.assertGreater(rep["model"]["onnx_bytes"], 100000)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless((HERE / "runs" / "run1" / "results.json").exists(), "needs runs/run1")
class Overview(unittest.TestCase):
    def test_run_summary_matches_the_full_report(self):
        full = M.run_report("run1"); one = M.run_summary(HERE / "runs" / "run1")
        self.assertEqual(one["pack_id"], "anc")
        self.assertAlmostEqual(one["exact"], full["overall"]["exact"], places=9); self.assertAlmostEqual(one["cer"], full["overall"]["cer"], places=9)
        self.assertAlmostEqual(one["ece"], full["calibration"]["ece"], places=9)
        self.assertAlmostEqual(one["silent_error_rate"], full["risk_coverage"]["policy"]["silent_error_rate"], places=9)

    def test_overview_groups_by_pack_id_oldest_first(self):
        g = M.overview(); self.assertIn("anc", g)
        self.assertEqual([r["name"] for r in g["anc"]], sorted((r["name"] for r in g["anc"]), key=lambda n: next(x["ts"] for x in g["anc"] if x["name"] == n)))
