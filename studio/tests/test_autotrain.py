"""studio/autotrain.py: trigger thresholds (new approved cells, new pages, a new writer's pages), snapshot/version bookkeeping.
Drawn cards only (no patient data)."""
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import autotrain as AT
import cards, labeling, review, scans
from test_labeling import Base, F, ROWS, COLS


def _approve_all(card):
    review.sync(card)
    for key, row in review.read(card).items():
        if row["status"] == "pending": review.set_status(card, *key, "approved")


def _add_pages(test, card, page_nums):
    """Scan extra, fully-inked pages so labeling.save has a cell to record there (Base's setUp only scans p-000..p-002)."""
    cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
    for i in page_nums:
        scans.add(card, test.scan(card, cells=cells, turn=None, seed=i), f"p-{i:03d}.png")


class Trigger(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._old = AT.DATASETS
        AT.DATASETS = Path(self._tmp.name)

    def tearDown(self):
        AT.DATASETS = self._old; self._tmp.cleanup()


class NoSnapshotYet(Base, Trigger):
    def setUp(self):
        Base.setUp(self); Trigger.setUp(self)

    def tearDown(self):
        Trigger.tearDown(self); Base.tearDown(self)

    def test_no_trigger_below_threshold(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57"); _approve_all("alpha")
        should, reason = AT.should_trigger("alpha")
        self.assertFalse(should)

    def test_triggers_once_enough_pages_with_no_prior_snapshot(self):
        _add_pages(self, "alpha", range(3, 5))
        for i in range(5): labeling.save("alpha", f"p-{i:03d}", F[0], "c1", "57")
        _approve_all("alpha")
        should, reason = AT.should_trigger("alpha")
        self.assertTrue(should); self.assertIn("no snapshot yet", reason)

    def test_manual_always_snapshots(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57"); _approve_all("alpha")
        rec = AT.check_and_snapshot("alpha", manual=True)
        self.assertIsNotNone(rec); self.assertEqual(rec["reason"], "manual"); self.assertEqual(rec["version"], 1)


class AfterASnapshot(Base, Trigger):
    def setUp(self):
        Base.setUp(self); Trigger.setUp(self)
        labeling.save("alpha", "p-000", F[0], "c1", "57"); _approve_all("alpha")
        self.base = AT.snapshot("alpha", reason="manual")

    def tearDown(self):
        Trigger.tearDown(self); Base.tearDown(self)

    def test_new_approved_cells_trigger(self):
        # 5 pages (3 existing + 2 new, kept under the 5-new-page threshold) x every field x every visit: past the 50-cell
        # threshold on cells, not pages, so this exercises that branch specifically.
        _add_pages(self, "alpha", range(3, 5))
        label_for = {F[0]: "60", F[1]: "1/1/20", F[2]: "120/80", F[3]: "y", F[4]: "x"}
        for p in ("p-000", "p-001", "p-002", "p-003", "p-004"):
            for f in F:
                for c in range(1, COLS + 1): labeling.save("alpha", p, f, f"c{c}", label_for[f], confirm=True)
        _approve_all("alpha")
        should, reason = AT.should_trigger("alpha")
        self.assertTrue(should); self.assertIn("new approved cells", reason)

    def test_new_pages_trigger(self):
        _add_pages(self, "alpha", range(1, 6))
        for i in range(1, 6): labeling.save("alpha", f"p-{i:03d}", F[0], "c1", "60")
        _approve_all("alpha")
        should, reason = AT.should_trigger("alpha")
        self.assertTrue(should); self.assertIn("new approved pages", reason)

    def test_new_writer_with_enough_pages_triggers(self):
        _add_pages(self, "alpha", range(1, 4))
        for i in range(1, 4): labeling.save("alpha", f"p-{i:03d}", F[0], "c1", "60")
        cards.set_writers("alpha", "W01", [f"p-{i:03d}" for i in range(1, 4)])
        _approve_all("alpha")
        should, reason = AT.should_trigger("alpha")
        self.assertTrue(should); self.assertIn("new writer", reason)

    def test_check_and_snapshot_bumps_version(self):
        _add_pages(self, "alpha", range(1, 6))
        for i in range(1, 6): labeling.save("alpha", f"p-{i:03d}", F[0], "c1", "60")
        _approve_all("alpha")
        rec = AT.check_and_snapshot("alpha")
        self.assertIsNotNone(rec); self.assertEqual(rec["version"], 2)
        self.assertIsNone(AT.check_and_snapshot("alpha"))   # nothing new since v2


if __name__ == "__main__":
    unittest.main()
