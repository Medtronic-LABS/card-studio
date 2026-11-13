"""The settings of a training run: presets, every value with its limits, and what a settings panel needs. The checks that build a network need torch."""
import sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack"))
import settings as S

try: import torch  # noqa: F401
except ImportError: torch = None


class Panel(unittest.TestCase):
    def test_describe_all_gives_a_panel_everything_without_torch(self):
        d = S.describe_all()
        self.assertEqual(set(d["hp"]), {"lr", "optimizer", "weight_decay", "batch", "schedule", "step_every", "step_gamma", "grad_clip", "balance", "patience"})
        self.assertIn("noise", d["aug"]); self.assertEqual(d["default_preset"], "light_lr3e-4"); self.assertEqual(d["arch_default"], "crnn")
        self.assertEqual({k: v["time_steps"] for k, v in d["arch_presets"].items()}, {k: 48 for k in d["arch_presets"]})
        self.assertEqual(d["arch_presets"]["crnn"]["params"], 538305)
        self.assertEqual(d["hp"]["optimizer"]["options"], ["adam", "adamw", "sgd"]); self.assertEqual((d["hp"]["batch"]["min"], d["hp"]["batch"]["max"]), (4, 256))


@unittest.skipUnless(torch, "needs torch")
class Load(unittest.TestCase):
    def test_nothing_set_is_the_shipped_recipe(self):
        import train as T
        r = S.load(None); self.assertIsNone(r["arch"]); self.assertEqual(r["config_label"], "light_lr3e-4"); self.assertEqual(r["cfg"], T.resolve_cfg("light_lr3e-4"))
        self.assertEqual(r["cfg"]["lr"], T.CONFIGS["light_lr3e-4"]["lr"]); self.assertEqual((r["cfg"]["optimizer"], r["cfg"]["batch"], r["cfg"]["balance"]), ("adam", 64, True))
        self.assertIsNone(S.load({"arch": "crnn"})["arch"])           # naming the shipped network is the shipped network

    def test_presets_in_the_panel_match_the_trainers(self):
        import train as T
        for k, v in S.PRESET_VALUES.items():
            self.assertEqual(T.CONFIGS[k]["lr"], v["lr"], k); self.assertEqual({a: T.CONFIGS[k]["aug"][a] for a in v["aug"]}, v["aug"], k)

    def test_values_and_architecture_are_checked_and_recorded(self):
        r = S.load({"preset": "nbaug_lr1e-4", "hp": {"lr": 5e-4, "schedule": "cosine", "aug": {"erase": 0.3}}, "arch": "gru-crnn"})
        self.assertEqual((r["cfg"]["lr"], r["cfg"]["schedule"], r["cfg"]["aug"]["erase"], r["cfg"]["aug"]["rot"]), (5e-4, "cosine", 0.3, 3.0)); self.assertEqual(r["config_label"], "nbaug_lr1e-4+custom")
        self.assertEqual(r["record"]["arch_name"], "gru-crnn"); self.assertEqual(r["record"]["arch_params"], 472257); self.assertIn("gru-crnn", r["arch_label"])
        for bad, msg in (({"hp": {"lr": 5}}, "between"), ({"hp": {"optimizer": "rmsprop"}}, "optimizer"), ({"hp": {"aug": {"zoom": 1}}}, "augmentation"), ({"hp": {"nope": 1}}, "unknown training"),
                         ({"arch": {"conv": []}}, "blocks"), ({"preset": "x"}, "unknown preset"), ({"extra": 1}, "unknown settings"), ([], "object")):
            with self.assertRaisesRegex(ValueError, msg): S.load(bad)


if __name__ == "__main__":
    unittest.main()
