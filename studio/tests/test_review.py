"""Review status (studio/review.py): sync creates/resets rows, set_status records a decision, migrate seeds a sensible starting
state from `how`, and cards.labels(min_status=...) filters on it. Drawn cards only (no patient data)."""
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, labeling, review
from test_labeling import Base, F


class Sync(Base):
    def test_adds_pending_rows_for_new_labels(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        out = review.sync("alpha")
        self.assertEqual(out["added"], 1)
        self.assertEqual(review.read("alpha")[("p-000", F[0], "c1")]["status"], "pending")

    def test_editing_a_reviewed_label_resets_it_to_pending(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.sync("alpha")
        review.set_status("alpha", "p-000", F[0], "c1", "approved", reviewer="r1")
        labeling.save("alpha", "p-000", F[0], "c1", "58", confirm=True)
        out = review.sync("alpha")
        self.assertEqual(out["reset"], 1)
        self.assertEqual(review.read("alpha")[("p-000", F[0], "c1")]["status"], "pending")

    def test_set_status_requires_a_labeled_cell(self):
        with self.assertRaises(KeyError): review.set_status("alpha", "p-000", F[0], "c1", "approved")

    def test_set_status_rejects_unknown_status(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        with self.assertRaises(ValueError): review.set_status("alpha", "p-000", F[0], "c1", "maybe")


class Migrate(Base):
    def test_typed_and_corrected_become_approved_accepted_depends_on_confidence(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")   # no suggestion existed -> how=typed
        out = review.migrate("alpha")
        self.assertEqual(out["approved"], 1)
        self.assertEqual(review.read("alpha")[("p-000", F[0], "c1")]["status"], "approved")

    def test_migrate_is_a_noop_once_review_csv_exists(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.migrate("alpha")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        self.assertIsNone(review.migrate("alpha"))
        self.assertNotIn(("p-001", F[0], "c1"), review.read("alpha"))


class LabelsFilter(Base):
    def test_min_status_none_is_unchanged_behavior(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        rows, _ = cards.labels("alpha")
        self.assertEqual(len(rows), 1)

    def test_min_status_approved_excludes_pending(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.sync("alpha")
        rows, _ = cards.labels("alpha", min_status="approved")
        self.assertEqual(rows, [])
        review.set_status("alpha", "p-000", F[0], "c1", "approved")
        rows, _ = cards.labels("alpha", min_status="approved")
        self.assertEqual(len(rows), 1)

    def test_min_status_pending_excludes_only_rejected(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        review.sync("alpha")
        review.set_status("alpha", "p-001", F[0], "c1", "rejected")
        rows, _ = cards.labels("alpha", min_status="pending")
        self.assertEqual({(r["page"]) for r in rows}, {"p-000"})


if __name__ == "__main__":
    unittest.main()
