"""The helpers behind confidence.py: grammar posteriors, equivalent spellings of a date, AUROC, accuracy at coverage, and a cutoff chosen on one half of the pages and applied to the other."""
import math, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack"))
import confidence as K


class Helpers(unittest.TestCase):
    def test_shares_sum_to_one_and_follow_the_likelihoods(self):
        sh = K.softmax_shares({"a": math.log(3.0), "b": math.log(1.0)}); self.assertAlmostEqual(sum(sh.values()), 1.0); self.assertAlmostEqual(sh["a"], 0.75)

    def test_every_spelling_of_a_day_and_month_is_one_class(self):
        self.assertEqual({K.date_class(t) for t in ("5.3", "05/03", "5-3", "05.03.26", " 5/3 ")}, {(5, 3)}); self.assertIsNone(K.date_class("x")); self.assertNotEqual(K.date_class("3.5"), K.date_class("5.3"))

    def test_auroc_is_one_for_a_perfect_ranking_half_for_no_information_and_none_without_both(self):
        self.assertEqual(K.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0); self.assertEqual(K.auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]), 0.0)
        self.assertAlmostEqual(K.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]), 0.5); self.assertIsNone(K.auroc([0.5, 0.6], [True, True]))

    def test_accuracy_of_the_most_confident_cells(self):
        conf = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]; ok = [True] * 5 + [False] * 5
        self.assertEqual(K.at_coverage(conf, ok, 0.4), 1.0); self.assertEqual(K.at_coverage(conf, ok, 1.0), 0.5)

    def test_a_cutoff_chosen_on_one_half_is_measured_on_the_other(self):
        pages = [f"p{i % 8}" for i in range(80)]; conf = [1 - i / 80 for i in range(80)]; ok = [i < 40 for i in range(80)]            # the top half is right, the bottom half wrong
        share, wrong = K.accepted_at_cap(conf, ok, pages, 0.02)
        self.assertGreater(share, 0.3); self.assertLessEqual(wrong, 0.05)
        share, wrong = K.accepted_at_cap([0.5] * 80, [(i // 8) % 2 == 0 for i in range(80)], pages, 0.02); self.assertEqual((share, wrong), (0.0, None))           # nothing separates right from wrong: nothing is accepted


if __name__ == "__main__":
    unittest.main()
