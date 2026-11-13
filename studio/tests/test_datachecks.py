"""The before-you-train checklist: each rule, both outcomes. Labels, writers, fields, test set and tuning history are fabricated."""
import sys, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE.parent / "work_phaseA"))
import cards, datachecks as D


def rows(pages, field="weight", per_page=1):
    return [{"page": f"p-{i:03d}", "row": field, "col": f"c{c}", "label": "5", "crop": ""} for i in range(pages) for c in range(per_page)]


def check(card_rows, how=None, writers=None, fields=None, holdout=True, tried=0, msg=None):
    fields = fields if fields is not None else [{"key": "weight", "kind": "number"}]
    ps = [mock.patch.object(cards, "labels", lambda c, holdout=False: (list(card_rows), {})),
          mock.patch.object(D, "_how", lambda c: how or {}),
          mock.patch.object(cards, "writers", lambda c: writers or {}),
          mock.patch.object(cards, "fields", lambda c: fields),
          mock.patch.object(cards, "holdout_status", lambda c: {"exists": holdout}),
          mock.patch.object(D.seedexp, "tried", lambda c: {"settings_tried": tried, "message": msg})]
    for p in ps: p.start()
    try: r = D.checklist("x")
    finally:
        for p in ps: p.stop()
    return r, {c["id"]: c for c in r["checks"]}


class Rules(unittest.TestCase):
    def test_pages_bad_warn_ok(self):
        r, c = check(rows(9)); self.assertEqual(c["pages"]["level"], "bad"); self.assertIn("at least 10 labeled pages", c["pages"]["message"]); self.assertFalse(r["ready"])
        r, c = check(rows(10)); self.assertEqual(c["pages"]["level"], "warn"); self.assertTrue(r["ready"])
        r, c = check(rows(29)); self.assertEqual(c["pages"]["level"], "warn")
        r, c = check(rows(30)); self.assertEqual(c["pages"]["level"], "ok")

    def test_pages_do_not_count_pages_made_only_of_ai_reads(self):
        rs = rows(30); how = {(r["page"], r["row"], r["col"]): "ai-read" for r in rs[:25]}
        r, c = check(rs, how=how); self.assertEqual(c["pages"]["level"], "bad"); self.assertIn("5 person-checked", c["pages"]["message"])

    def test_ai_reads_warn_and_ok(self):
        rs = rows(30); how = {(r["page"], r["row"], r["col"]): "ai-read" for r in rs[:3]}
        r, c = check(rs, how=how); self.assertEqual(c["ai_reads"]["level"], "warn"); self.assertIn("3 of 30 labels are AI reads accepted unchecked", c["ai_reads"]["message"])
        self.assertEqual(check(rows(30), how={"x": "typed"})[1]["ai_reads"]["level"], "ok")

    def test_writers_ok_and_warn(self):
        rs = rows(30); w = {r["page"]: f"W{i % 6}" for i, r in enumerate(rs)}
        self.assertEqual(check(rs, writers=w)[1]["writers"]["level"], "ok")
        self.assertEqual(check(rs, writers={r["page"]: f"W{i % 4}" for i, r in enumerate(rs)})[1]["writers"]["level"], "warn")      # four writers
        r, c = check(rs, writers=dict(list(w.items())[:20])); self.assertEqual(c["writers"]["level"], "warn")                     # not every page coded
        r, c = check(rs); self.assertIn("no writer codes", c["writers"]["message"])

    def test_fields_ok_warn_info(self):
        f = [{"key": "a", "kind": "number"}, {"key": "b", "kind": "mark"}, {"key": "t", "kind": "text"}]
        rs = rows(20, "a", 1) + rows(20, "b", 1)
        self.assertEqual([c["level"] for c in check(rs, fields=f)[0]["checks"] if c["id"] == "fields"], ["ok"])
        r, c = check(rows(5, "a"), fields=f)
        lv = {x["level"]: x["message"] for x in r["checks"] if x["id"] == "fields"}
        self.assertIn("a", lv["warn"]); self.assertIn("b", lv["info"]); self.assertNotIn("t", lv["info"].split(": ")[1].split(", "))

    def test_fields_names_at_most_eight(self):
        f = [{"key": f"f{i}", "kind": "number"} for i in range(12)]; rs = sum((rows(3, k["key"]) for k in f), [])
        m = [x for x in check(rs, fields=f)[0]["checks"] if x["id"] == "fields" and x["level"] == "warn"][0]["message"]
        self.assertIn("12 non-text", m); self.assertIn("f7", m); self.assertNotIn("f8", m); self.assertIn("4 more", m)

    def test_test_set(self):
        self.assertEqual(check(rows(30), holdout=True)[1]["test_set"]["level"], "ok")
        r, c = check(rows(30), holdout=False); self.assertEqual(c["test_set"]["level"], "warn"); self.assertIn("no test set reserved", c["test_set"]["message"])

    def test_tried(self):
        self.assertEqual(check(rows(30), tried=3)[1]["tried"]["level"], "ok")
        r, c = check(rows(30), tried=4, msg="4 different settings have been tried"); self.assertEqual(c["tried"]["level"], "warn"); self.assertIn("4 different", c["tried"]["message"])

    def test_ready_ignores_warnings_and_every_check_has_the_four_keys(self):
        r, c = check(rows(12), holdout=False, tried=5, msg="m")
        self.assertTrue(r["ready"])
        for x in r["checks"]: self.assertEqual(set(x), {"id", "level", "message", "hint"}); self.assertIn(x["level"], ("ok", "info", "warn", "bad"))

    def test_legacy_card_has_no_how_column(self):
        self.assertEqual(D._how(cards.LEGACY), {})


if __name__ == "__main__":
    unittest.main()
