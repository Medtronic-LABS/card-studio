"""Card-level checks: same-day dose columns (EPI) and increasing visit dates (ANC)."""
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cardlogic as L

EPI = ["bcg", "penta", "opv", "pcv"]


class DateKey(unittest.TestCase):
    def test_real_dates_only(self):
        self.assertEqual(L.date_key("22/2/26"), (22, 2, 2026)); self.assertEqual(L.date_key("02.03.2026"), (2, 3, 2026)); self.assertEqual(L.date_key(" 29/2/24 "), (29, 2, 2024))
        for bad in ("", None, "31/4/26", "29/2/25", "22/13/26", "0/1/26", "22/2", "22/2/2", "12/80"): self.assertIsNone(L.date_key(bad), bad)


class SameDay(unittest.TestCase):
    def test_a_dissenting_cell_gets_the_majority_date_as_a_suggestion_and_keeps_its_read(self):
        reads = {("bcg", 2): "22/2/26", ("penta", 2): "22.2.26", ("opv", 2): "22/2/26", ("pcv", 2): "23/2/26"}
        f = L.same_day(reads, EPI)
        self.assertEqual([(x["field"], x["visit"], x["suggestion"]) for x in f], [("pcv", 2, "22/2/26")])
        self.assertIn("23/2/26", f[0]["note"])

    def test_not_a_date_is_corrected_too_but_a_blank_cell_is_left_alone(self):
        reads = {("bcg", 1): "5/3/26", ("penta", 1): "5/3/26", ("opv", 1): "0/3/26", ("pcv", 1): ""}
        self.assertEqual([x["field"] for x in L.same_day(reads, EPI)], ["opv"])

    def test_a_tie_or_a_single_date_decides_nothing(self):
        self.assertEqual(L.same_day({("bcg", 1): "5/3/26", ("penta", 1): "6/3/26"}, EPI), [])
        self.assertEqual(L.same_day({("bcg", 1): "5/3/26", ("penta", 1): "5/3/26", ("opv", 1): "6/3/26", ("pcv", 1): "6/3/26"}, EPI), [])
        self.assertEqual(L.same_day({("bcg", 1): "5/3/26", ("penta", 1): "garbage"}, EPI), [])

    def test_visits_are_judged_apart_and_weak_cells_do_not_vote(self):
        reads = {("bcg", 1): "5/3/26", ("penta", 1): "5/3/26", ("opv", 1): "9/9/26", ("bcg", 2): "7/4/26", ("penta", 2): "7/4/26", ("opv", 2): "7/4/26"}
        self.assertEqual([(x["field"], x["visit"]) for x in L.same_day(reads, EPI)], [("opv", 1)])
        self.assertEqual(L.same_day(reads, EPI, weak={("penta", 1)}), [])           # only one trusted vote left in visit 1
        self.assertEqual(L.same_day(reads, ["bcg"]), [])                              # a rule needs at least two fields

    def test_fields_the_rule_does_not_name_are_ignored(self):
        self.assertEqual(L.same_day({("bcg", 1): "5/3/26", ("penta", 1): "5/3/26", ("weight", 1): "9/9/26"}, "bcg, penta"), [])


class Increasing(unittest.TestCase):
    def test_an_earlier_visit_date_is_noted_and_nothing_is_suggested(self):
        reads = {("visit_date", 1): "10/1/26", ("visit_date", 2): "2/2/26", ("visit_date", 3): "1/2/26", ("visit_date", 4): "9/3/26"}
        f = L.increasing(reads, ["visit_date"])
        self.assertEqual([(x["visit"], x["suggestion"]) for x in f], [(3, None)]); self.assertIn("visit 2", f[0]["note"])

    def test_year_rolls_over_blank_and_unreadable_visits_are_skipped_equal_dates_pass(self):
        reads = {("visit_date", 1): "20/12/25", ("visit_date", 2): "", ("visit_date", 3): "junk", ("visit_date", 4): "5/1/26", ("visit_date", 5): "5/1/26"}
        self.assertEqual(L.increasing(reads, "visit_date"), [])

    def test_check_runs_both(self):
        reads = {("bcg", 1): "5/3/26", ("penta", 1): "5/3/26", ("opv", 1): "6/3/26", ("visit_date", 1): "9/1/26", ("visit_date", 2): "8/1/26"}
        self.assertEqual([x["rule"] for x in L.check(reads, {"same_day": EPI, "increasing": "visit_date"})], ["same_day", "increasing"])
        self.assertEqual(L.check(reads, None), [])


if __name__ == "__main__":
    unittest.main()
