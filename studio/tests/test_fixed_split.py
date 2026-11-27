"""The optional fixed train/validation/test split (trainer.make_split, --val-frac/--test-frac). 5-fold cross-validation stays the default.
Run: python -m unittest discover -s tests"""
import subprocess, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import trainer


def data(writers=10, per=4):
    pages = [f"p-{i:03d}" for i in range(writers * per)]; units = {p: f"W{i % writers}" for i, p in enumerate(pages)}
    rows = [{"page": p, "row": f, "col": "c1", "label": "5"} for p in pages for f in ("pulse", "weight")]
    return rows, units, pages


class MakeSplit(unittest.TestCase):
    def test_parts_cover_every_page_once_and_never_share_a_writer(self):
        rows, units, pages = data()
        tr, va, te = trainer.make_split(rows, 0.2, 0.2, units=units)
        self.assertEqual(sorted(tr + va + te), pages)
        self.assertEqual((len({units[p] for p in va}), len({units[p] for p in te}), len({units[p] for p in tr})), (2, 2, 6))
        for a, b in ((tr, va), (tr, te), (va, te)): self.assertFalse({units[p] for p in a} & {units[p] for p in b})

    def test_same_seed_same_split_and_without_units_it_is_by_page(self):
        rows, units, pages = data()
        self.assertEqual(trainer.make_split(rows, 0.15, 0.15, units=units), trainer.make_split(rows, 0.15, 0.15, units=units))
        tr, va, te = trainer.make_split(rows, 0.25, 0.25); self.assertEqual(sorted(tr + va + te), pages); self.assertEqual((len(va), len(te)), (10, 10))

    def test_refuses_a_split_with_no_training_unit_left(self):
        rows, units, _ = data(writers=3, per=2)
        with self.assertRaises(ValueError): trainer.make_split(rows, 0.5, 0.5, units=units)

    def test_default_cross_validation_is_unchanged(self):
        rows, units, pages = data()
        folds = trainer.make_folds(rows, units=units); self.assertEqual(len(folds), 5); self.assertEqual(sorted(sum(folds, [])), pages)


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(HERE / "trainer.py"), "RUN", *args], capture_output=True, text=True, cwd=str(HERE))

    def test_the_two_fractions_go_together_and_stay_in_range(self):
        r = self.run_cli("--val-frac", "0.2"); self.assertEqual(r.returncode, 2); self.assertIn("go together", r.stderr)
        r = self.run_cli("--val-frac", "0.5", "--test-frac", "0.5"); self.assertEqual(r.returncode, 2); self.assertIn("at most 0.6", r.stderr)
        r = self.run_cli("--val-frac", "0.01", "--test-frac", "0.2"); self.assertEqual(r.returncode, 2); self.assertIn("0.05 to 0.5", r.stderr)


if __name__ == "__main__":
    unittest.main()
