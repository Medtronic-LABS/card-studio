"""studio/active.py: the weighted score, round-robin queue ordering, and labeling.queue(order_by="active") wiring. Drawn cards
only (no patient data)."""
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import active, cards, labeling
from test_labeling import Base, F


class Score(unittest.TestCase):
    def test_weights_match_the_plan(self):
        self.assertEqual(active.score({"low_conf": 1.0}), 1.0)
        self.assertEqual(active.score({"hard_graphemes": 1.0}), 0.8)
        self.assertEqual(active.score({"unseen_writer": 1.0}), 0.7)
        self.assertEqual(active.score({"new_doc_type": 1.0}), 0.5)
        self.assertEqual(active.score({"version_disagree": 1.0}), 0.9)

    def test_missing_keys_count_as_zero(self):
        self.assertEqual(active.score({}), 0.0)

    def test_all_signals_combine_additively(self):
        s = {"low_conf": 0.5, "hard_graphemes": 0.5, "unseen_writer": 1.0, "new_doc_type": 0.0, "version_disagree": 0.0}
        self.assertAlmostEqual(active.score(s), 0.5 * 1.0 + 0.5 * 0.8 + 1.0 * 0.7)


class ComputeSignals(unittest.TestCase):
    def test_no_suggestion_is_maximally_unsure(self):
        s = active.compute_signals(None, "w1", None, set(), set())
        self.assertEqual(s["low_conf"], 1.0); self.assertEqual(s["hard_graphemes"], 0.0)

    def test_confident_suggestion_lowers_low_conf(self):
        s = active.compute_signals({"text": "ab", "confidence": 0.9}, "w1", None, set(), set())
        self.assertAlmostEqual(s["low_conf"], 0.1)

    def test_conjunct_text_scores_harder_than_plain(self):
        plain = active.compute_signals({"text": "কলম", "confidence": 0.9}, "w1", None, set(), set())
        conjunct = active.compute_signals({"text": "ক্ষমতা", "confidence": 0.9}, "w1", None, set(), set())
        self.assertGreater(conjunct["hard_graphemes"], plain["hard_graphemes"])

    def test_unseen_writer_flag(self):
        self.assertEqual(active.compute_signals(None, "new", None, {"old"}, set())["unseen_writer"], 1.0)
        self.assertEqual(active.compute_signals(None, "old", None, {"old"}, set())["unseen_writer"], 0.0)

    def test_version_disagree_needs_both_suggestion_and_other_version(self):
        self.assertEqual(active.compute_signals({"text": "a", "confidence": 0.9}, "w", None, set(), set(), other_version_text="b")["version_disagree"], 1.0)
        self.assertEqual(active.compute_signals({"text": "a", "confidence": 0.9}, "w", None, set(), set(), other_version_text="a")["version_disagree"], 0.0)
        self.assertEqual(active.compute_signals(None, "w", None, set(), set(), other_version_text="b")["version_disagree"], 0.0)


class QueueOrder(unittest.TestCase):
    def test_round_robins_across_fields(self):
        cells = [{"field": "a", "score": 0.9}, {"field": "a", "score": 0.8}, {"field": "b", "score": 0.1}]
        out = active.queue_order(cells, "field")
        self.assertEqual([c["field"] for c in out], ["a", "b", "a"])   # field b's one cell interleaved, not pushed to the end

    def test_within_a_field_highest_score_first(self):
        cells = [{"field": "a", "score": 0.1}, {"field": "a", "score": 0.9}]
        out = active.queue_order(cells, "field")
        self.assertEqual([c["score"] for c in out], [0.9, 0.1])


class LabelingQueueActive(Base):
    def test_order_by_active_accepted_and_returns_every_cell(self):
        q = labeling.queue("alpha", order_by="active")
        self.assertEqual(q["total"], 9)   # same 9 inked cells test_labeling.Queue expects for "todo"

    def test_a_label_with_no_suggestion_is_still_scored_and_returned(self):
        q = labeling.queue("alpha", order_by="active", limit=1)
        self.assertEqual(len(q["cells"]), 1)

    def test_unknown_order_by_still_refused(self):
        with self.assertRaises(ValueError): labeling.queue("alpha", order_by="nonsense")


if __name__ == "__main__":
    unittest.main()
