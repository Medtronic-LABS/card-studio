"""Early stopping decisions, checkpoint retention, atomic curves, resume argument refusal. Run: python -m unittest discover -s tests"""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "work_phaseA"))
try:
    import torch
    import train as T
except Exception:            # no torch: the torch-dependent parts skip
    torch = T = None
needs_torch = unittest.skipIf(T is None, "torch not available")


def run_es(es, seq):
    """seq: [(exact, cer, loss)] -> (list of improvement flags, epoch the stop fired at or None)"""
    flags, stop = [], None
    for ep, (ex, c, vl) in enumerate(seq, 1):
        flags.append(es.update(ep, ex, c, vl))
        if es.should_stop(ep): stop = ep; break
    return flags, stop


@needs_torch
class EarlyStopTest(unittest.TestCase):
    def test_default_is_the_original_rule(self):
        es = T.EarlyStop("exact", 0.0, 3)
        flags, stop = run_es(es, [(.5, .2, 1), (.5, .1, 1), (.5, .1, 1), (.4, .0, 1), (.6, .3, 1), (.6, .3, 1), (.5, 0, 1), (.5, 0, 1), (.5, 0, 1)])
        self.assertEqual(flags, [True, True, False, False, True, False, False, False])      # equal exact: lower CER wins; equal both: not better
        self.assertEqual((es.best_ep, stop), (5, 8))

    def test_cer_and_loss_lower_is_better(self):
        es = T.EarlyStop("cer", 0.0, 10); flags, _ = run_es(es, [(.9, .30, 5), (.1, .20, 6), (.9, .20, 4), (.95, .25, 1)])
        self.assertEqual(flags, [True, True, True, False]); self.assertEqual(es.best_ep, 3)           # a tie on CER goes to the higher exact
        es = T.EarlyStop("loss", 0.0, 10); flags, _ = run_es(es, [(.9, .3, 5), (.1, .9, 4), (.9, .1, 4.5)])
        self.assertEqual(flags, [True, True, False]); self.assertEqual(es.best_ep, 2)

    def test_min_delta_is_absolute_in_the_metrics_units(self):
        es = T.EarlyStop("exact", 0.01, 5); flags, _ = run_es(es, [(.50, .2, 1), (.505, .1, 1), (.52, .1, 1), (.525, .1, 1)])
        self.assertEqual(flags, [True, False, True, False]); self.assertEqual(es.best_ep, 3)
        es = T.EarlyStop("cer", 0.05, 5); flags, _ = run_es(es, [(0, .50, 1), (0, .46, 1), (0, .44, 1)])
        self.assertEqual(flags, [True, False, True])
        es = T.EarlyStop("loss", 0.5, 5); flags, _ = run_es(es, [(0, 0, 3.0), (0, 0, 2.6), (0, 0, 2.4)])
        self.assertEqual(flags, [True, False, True])           # compared with the best so far (3.0 then 2.4 is 0.6 better)

    def test_patience_counts_epochs_since_improvement(self):
        es = T.EarlyStop("exact", 0.0, 3); _, stop = run_es(es, [(.5, 0, 0), (.4, 0, 0), (.4, 0, 0), (.4, 0, 0), (.9, 0, 0)])
        self.assertEqual(stop, 4)
        es = T.EarlyStop("exact", 0.0, 3); _, stop = run_es(es, [(.5, 0, 0), (.4, 0, 0), (.6, 0, 0), (.4, 0, 0), (.4, 0, 0), (.4, 0, 0)])
        self.assertEqual((stop, es.best_ep), (6, 3))

    def test_state_round_trip(self):
        es = T.EarlyStop("cer", 0.0, 4); run_es(es, [(0, .3, 0), (0, .2, 0)]); st = json.loads(json.dumps(es.state()))
        es2 = T.EarlyStop("cer", 0.0, 4); es2.load(st); self.assertEqual((es2.best_ep, es2.best), (es.best_ep, es.best))
        self.assertFalse(es2.update(3, 0, .25, 0))


@needs_torch
class RetentionTest(unittest.TestCase):
    def test_keep_last_never_touches_best_or_last(self):
        with tempfile.TemporaryDirectory() as d:
            cp = T.Checkpointer({"dir": d, "name": "fold-0", "every": 1, "keep_last": 2})
            cp.save_best({"w": torch.zeros(1)}); cp.save_last({"epoch": 9})
            for ep in range(1, 8): cp.save_numbered(ep, {"w": torch.ones(1) * ep})
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["fold-0-best.pt", "fold-0-e006.pt", "fold-0-e007.pt", "fold-0-last.pt"])

    def test_other_names_are_left_alone_and_keep_one_at_least(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("fold-1-e001.pt", "fold-1-e002.pt", "fold-10-e001.pt", "final-e001.pt"): (Path(d) / n).write_bytes(b"x")
            self.assertEqual(T.prune_numbered(d, "fold-1", 1), ["fold-1-e001.pt"])
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["final-e001.pt", "fold-1-e002.pt", "fold-10-e001.pt"])

    def test_cadence(self):
        cp = T.Checkpointer({"dir": tempfile.mkdtemp(), "name": "f", "every": 3, "keep_last": 1}); self.assertEqual([e for e in range(1, 10) if cp.due(e)], [3, 6, 9])
        cp = T.Checkpointer({"dir": tempfile.mkdtemp(), "name": "f", "every": 0, "keep_last": 1}); self.assertFalse(any(cp.due(e) for e in range(1, 10)))


@needs_torch
class CurvesTest(unittest.TestCase):
    def test_atomic_write_leaves_no_temp_and_replaces_whole(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "curves" / "fold-0.json"
            for n in range(1, 4): T.write_json_atomic(p, [{"epoch": i} for i in range(n)]); self.assertEqual(len(json.loads(p.read_text())), n)
            self.assertEqual([f.name for f in p.parent.iterdir()], ["fold-0.json"])

    def test_stop_file(self):
        with tempfile.TemporaryDirectory() as d:
            cp = T.Checkpointer({"dir": d + "/c", "name": "f", "stop": d + "/STOP"}); self.assertFalse(cp.stop_requested())
            (Path(d) / "STOP").write_text(""); self.assertTrue(cp.stop_requested())


class ResumeArgsTest(unittest.TestCase):
    def run_trainer(self, *args):
        return subprocess.run([sys.executable, str(HERE / "trainer.py"), *args], capture_output=True, text=True, timeout=120)

    def test_resume_needs_trainer_args(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_trainer(d, "--resume"); self.assertNotEqual(r.returncode, 0); self.assertIn("trainer_args.json", r.stderr)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_resume_refuses_other_args(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "trainer_args.json").write_text(json.dumps({"args": {}}))
            r = self.run_trainer(d, "--resume", "--max-epochs", "5", "--card", "x"); self.assertNotEqual(r.returncode, 0)
            self.assertIn("--max-epochs", r.stderr); self.assertIn("--card", r.stderr)


if __name__ == "__main__":
    unittest.main()
