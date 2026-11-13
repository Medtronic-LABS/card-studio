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
        self.assertEqual(set(d["hp"]), {"lr", "optimizer", "weight_decay", "batch", "schedule", "step_every", "step_gamma", "grad_clip", "balance", "patience", "monitor", "min_delta", "restore", "plateau_patience"})
        self.assertIn("noise", d["aug"]); self.assertEqual(d["default_preset"], "light_lr3e-4"); self.assertEqual(d["arch_default"], "crnn")
        self.assertEqual({k: v["time_steps"] for k, v in d["arch_presets"].items()}, {k: 48 for k in d["arch_presets"]})
        self.assertEqual(d["arch_presets"]["crnn"]["params"], 538305)
        self.assertEqual(d["hp"]["optimizer"]["options"], ["adam", "adamw", "sgd"]); self.assertEqual(d["hp"]["monitor"]["options"], ["exact", "cer", "loss"]); self.assertIn("plateau", d["hp"]["schedule"]["options"])
        self.assertEqual({k: (v["default"], v["min"], v["max"]) for k, v in d["run"].items()}, {"checkpoint_every": (10, 0, 100), "keep_last": (2, 1, 10)}); self.assertEqual((d["hp"]["batch"]["min"], d["hp"]["batch"]["max"]), (4, 256))


class Load(unittest.TestCase):
    def test_nothing_set_is_the_shipped_recipe(self):
        import hpspec
        r = S.load(None); self.assertIsNone(r["arch"]); self.assertEqual(r["config_label"], "light_lr3e-4"); self.assertEqual(r["cfg"], hpspec.resolve("light_lr3e-4"))
        self.assertEqual(r["cfg"]["lr"], hpspec.PRESETS["light_lr3e-4"]["lr"]); self.assertEqual((r["cfg"]["optimizer"], r["cfg"]["batch"], r["cfg"]["balance"]), ("adam", 64, True))
        self.assertIsNone(S.load({"arch": "crnn"})["arch"])           # naming the shipped network is the shipped network

    @unittest.skipUnless(torch, "needs torch")
    def test_the_trainers_presets_are_the_panels_presets(self):
        import hpspec, train as T
        self.assertEqual(T.CONFIGS, hpspec.PRESETS); self.assertEqual(S.describe_all()["presets"], hpspec.PRESETS)

    def test_early_stopping_and_run_settings_are_validated_and_recorded_but_do_not_make_a_run_custom(self):
        r = S.load({"hp": {"monitor": "cer", "min_delta": 0.01, "restore": "last", "schedule": "plateau", "plateau_patience": 5}, "run": {"checkpoint_every": 0, "keep_last": 3}})
        self.assertEqual((r["cfg"]["monitor"], r["cfg"]["min_delta"], r["cfg"]["restore"], r["cfg"]["plateau_patience"]), ("cer", 0.01, "last", 5)); self.assertEqual(r["run"], {"checkpoint_every": 0, "keep_last": 3}); self.assertEqual(r["record"]["run"], r["run"])
        only_run = S.load({"run": {"checkpoint_every": 5}}); self.assertEqual((only_run["config_label"], only_run["run"]["checkpoint_every"]), ("light_lr3e-4", 5))
        d = S.load(None); self.assertEqual((d["cfg"]["monitor"], d["cfg"]["min_delta"], d["cfg"]["restore"], d["run"]), ("exact", 0.0, "best", {"checkpoint_every": 10, "keep_last": 2}))
        for bad, msg in (({"hp": {"monitor": "x"}}, "monitor"), ({"hp": {"restore": "x"}}, "restore"), ({"hp": {"min_delta": 1}}, "between"), ({"run": {"keep_last": 0}}, "between"), ({"run": {"zzz": 1}}, "unknown run"), ({"run": [1]}, "object")):
            with self.assertRaisesRegex(ValueError, msg): S.load(bad)

    def test_values_and_architecture_are_checked_and_recorded(self):
        r = S.load({"preset": "nbaug_lr1e-4", "hp": {"lr": 5e-4, "schedule": "cosine", "aug": {"erase": 0.3}}, "arch": "gru-crnn"})
        self.assertEqual((r["cfg"]["lr"], r["cfg"]["schedule"], r["cfg"]["aug"]["erase"], r["cfg"]["aug"]["rot"]), (5e-4, "cosine", 0.3, 3.0)); self.assertEqual(r["config_label"], "nbaug_lr1e-4+custom")
        self.assertEqual(r["record"]["arch_name"], "gru-crnn"); self.assertEqual(r["record"]["arch_params"], 472257); self.assertIn("gru-crnn", r["arch_label"])
        for bad, msg in (({"hp": {"lr": 5}}, "between"), ({"hp": {"optimizer": "rmsprop"}}, "optimizer"), ({"hp": {"aug": {"zoom": 1}}}, "augmentation"), ({"hp": {"nope": 1}}, "unknown training"),
                         ({"arch": {"conv": []}}, "blocks"), ({"preset": "x"}, "unknown preset"), ({"extra": 1}, "unknown settings"), ([], "object")):
            with self.assertRaisesRegex(ValueError, msg): S.load(bad)


if __name__ == "__main__":
    unittest.main()
