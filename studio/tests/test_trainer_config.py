"""The training presets the trainer offers are the ones train.py defines, and an unknown one is refused before anything runs."""
import subprocess, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))


class TrainerConfig(unittest.TestCase):
    def test_names_match_train_configs(self):
        try: import train as T
        except ImportError: self.skipTest("needs torch")
        import trainer
        self.assertEqual(set(trainer.CONFIG_NAMES), set(T.CONFIGS)); self.assertIn(trainer.CFG, trainer.CONFIG_NAMES)

    def test_unknown_preset_is_refused(self):
        r = subprocess.run([sys.executable, str(HERE / "trainer.py"), "/nonexistent-run", "--config", "huge"], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0); self.assertIn("invalid choice", r.stderr)


if __name__ == "__main__":
    unittest.main()
