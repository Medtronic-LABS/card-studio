"""The settings check behind the Train card (nothing is started) and its bounds. Run: python -m unittest discover -s tests"""
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import server, trainer


class CheckSettings(unittest.TestCase):
    def test_a_value_out_of_range_names_the_field(self):
        r = server.check_settings({"settings": {"hp": {"batch": 512}}, "pretrained": "none"})
        self.assertFalse(r["ok"]); self.assertEqual(r["errors"][0]["key"], "batch"); self.assertIn("256", r["errors"][0]["message"])

    def test_an_unknown_architecture_names_arch(self):
        r = server.check_settings({"settings": {"arch": "nope"}, "pretrained": "none"})
        self.assertFalse(r["ok"]); self.assertEqual(r["errors"][0]["key"], "arch")

    def test_good_settings_pass_and_start_nothing(self):
        before = server._proc
        self.assertEqual(server.check_settings({"settings": {"hp": {"lr": 0.0005}, "arch": "crnn-base"}, "pretrained": "none"}), {"ok": True, "errors": []})
        self.assertIs(server._proc, before)

    def test_every_hp_range_the_ui_accepts_fits_the_line_trainer_limits(self):
        import settings as S
        from htr.train import config as K
        pairs = {"lr": "optim.lr", "weight_decay": "optim.weight_decay", "step_every": "optim.step_every", "step_gamma": "optim.step_gamma", "plateau_patience": "optim.plateau_patience",
                 "batch": "loop.batch_size", "patience": "loop.patience", "min_delta": "loop.min_delta", "grad_clip": "loop.grad_clip"}
        hp = S.describe_all()["hp"]
        for k, lim in pairs.items():
            lo, hi = K.LIMITS[lim]; self.assertGreaterEqual(hp[k]["min"], lo, k); self.assertLessEqual(hp[k]["max"], hi, k)


class Bounds(unittest.TestCase):
    def test_split_preview_refuses_fractions_out_of_range(self):
        for v, t in ((0.01, 0.2), (0.2, 0.7), (0.4, 0.4)):
            with self.assertRaises(ValueError): trainer.split_preview("anc", v, t)


if __name__ == "__main__":
    unittest.main()
