"""The cross-card dataset table: one row per card from the functions the per-card views use. Run: python -m unittest discover -s tests"""
import sys, unittest
from pathlib import Path
from unittest import mock
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import datasets, cards, datachecks


def card(cid, cells, pages, packs=()):
    return {"id": cid, "labels": {"cells": cells, "pages": pages}, "pack_versions": list(packs), "blocker": "" if pages >= 10 else "few pages"}


CK = {"a": [{"id": "pages", "level": "ok", "message": "m"}, {"id": "writers", "level": "info", "message": "m"}],
      "b": [{"id": "pages", "level": "bad", "message": "4 pages"}, {"id": "writers", "level": "warn", "message": "no writers"}]}


class Overview(unittest.TestCase):
    def run_overview(self, listing, boom=None):
        def ck(c):
            if c == boom: raise OSError("disk gone")
            return {"checks": CK.get(c, CK["a"]), "ready": True}
        with mock.patch.object(cards, "list_cards", return_value=listing), mock.patch.object(cards, "writer_status", side_effect=lambda c: {"writers": 3 if c == "a" else 0, "split_unit": "page"}), \
             mock.patch.object(cards, "holdout_status", side_effect=lambda c: {"test_pages": 8 if c == "a" else 0}), mock.patch.object(datachecks, "checklist", side_effect=ck):
            return datasets.overview()

    def test_one_row_per_card_with_worst_level_and_the_failing_checks(self):
        r = self.run_overview([card("a", 100, 12, [1, 2, 7]), card("b", 8, 4)])
        a, b = r["cards"]
        self.assertEqual((a["id"], a["labels"], a["pages"], a["writers"], a["test_pages"], a["pack_version"], a["level"], a["failing"]), ("a", 100, 12, 3, 8, 7, "info", []))
        self.assertEqual((b["level"], [f["id"] for f in b["failing"]], b["pack_version"], b["blocker"]), ("bad", ["pages", "writers"], None, "few pages"))
        self.assertEqual(r["totals"], {"cards": 2, "labels": 108, "pages": 16, "writers": 3, "levels": {"ok": 0, "info": 1, "warn": 0, "bad": 1}})

    def test_a_card_that_errors_gets_an_error_row_not_a_failed_table(self):
        r = self.run_overview([card("a", 1, 1), card("b", 2, 2)], boom="b")
        self.assertIsNone(r["cards"][0]["error"]); self.assertIn("disk gone", r["cards"][1]["error"]); self.assertEqual(r["cards"][1]["level"], "bad")

    def test_it_writes_nothing(self):
        before = sorted(p.name for p in (HERE).iterdir()); self.run_overview([card("a", 1, 1)]); self.assertEqual(before, sorted(p.name for p in HERE.iterdir()))


if __name__ == "__main__":
    unittest.main()
