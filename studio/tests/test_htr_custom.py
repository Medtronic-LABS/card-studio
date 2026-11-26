"""htr.nn.custom: loading a custom architecture module by name (never by path or uploaded source), the dry-run contract check
(forward shape, monotonic output_lengths, the input_lengths=None case), and the worked example fixture
(htr/custom/example_tiny.py) actually passing it."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    import htr.nn.custom as C
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class Load(unittest.TestCase):
    def test_example_fixture_is_available_and_loads(self):
        self.assertIn("example_tiny", C.available())
        net = C.load("example_tiny", n_classes=40)
        self.assertEqual(net.n_classes, 40); self.assertEqual(net.custom_name, "example_tiny")

    def test_unknown_name_refused(self):
        with self.assertRaises(ValueError): C.load("no_such_architecture_xyz")

    def test_bad_name_shape_refused_before_even_looking_on_disk(self):
        for bad in ("Example", "../etc/passwd", "has space", "123start", ""):
            with self.assertRaises(ValueError): C.load(bad)


class _NoBuild:
    """A module stand-in with no build() at all."""


class _BadReturn(nn.Module if torch else object):
    def forward(self, images, input_lengths=None): return images   # not a tuple


class _WrongBatch(nn.Module if torch else object):
    def forward(self, images, input_lengths=None):
        logits = torch.zeros(10, 3, 5)   # batch 3, but the dry run always uses batch 2
        return (logits, torch.tensor([10, 10, 10])) if input_lengths is not None else (logits, None)


class _NonMonotonic(nn.Module if torch else object):
    def forward(self, images, input_lengths=None):
        logits = torch.zeros(10, 2, 5)
        if input_lengths is None: return logits, None
        return logits, torch.tensor([2, 9])   # the shorter input (index 1) gets the LONGER output: backwards


class _IgnoresNone(nn.Module if torch else object):
    def forward(self, images, input_lengths=None):
        logits = torch.zeros(10, 2, 5)
        return logits, torch.tensor([5, 5])   # output_lengths is never None, even when input_lengths is


@unittest.skipIf(torch is None, "torch not installed")
class SpecEngine(unittest.TestCase):
    """normalize/describe/spec_hash/is_default/flow/summary: the same interface htr.spec/htr.presets offer, so
    work_phaseA/archnet.py's shim can dispatch to this module the same way it dispatches to the other two."""
    def test_normalize_round_trips_a_valid_config(self):
        self.assertEqual(C.normalize({"custom": "example_tiny"}), {"custom": "example_tiny"})

    def test_normalize_rejects_unknown_or_malformed(self):
        with self.assertRaises(ValueError): C.normalize({"custom": "no_such_thing"})
        with self.assertRaises(ValueError): C.normalize({"custom": "example_tiny", "extra": 1})
        with self.assertRaises(ValueError): C.normalize("example_tiny")

    def test_describe_builds_and_counts_real_params(self):
        net = C.load("example_tiny", 40)
        d = C.describe({"custom": "example_tiny"}, 40)
        self.assertEqual(d["params"], sum(p.numel() for p in net.parameters()))
        self.assertEqual(d["time_steps"], 48)

    def test_spec_hash_is_the_source_files_own_hash(self):
        import hashlib
        path = ROOT / "htr" / "custom" / "example_tiny.py"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self.assertEqual(C.spec_hash({"custom": "example_tiny"}), expected)

    def test_is_default_is_always_false(self):
        self.assertFalse(C.is_default({"custom": "example_tiny"}))

    def test_flow_traces_child_modules_in_order(self):
        f = C.flow({"custom": "example_tiny"}, 40)
        self.assertEqual(f[0]["label"], "cell (gray)")
        self.assertEqual([x["label"] for x in f[1:]], ["cnn", "fc"])
        self.assertEqual(f[-1]["shape"][-1], 40)   # the head's own output width is n_classes

    def test_summary_bundles_everything_and_reports_problems_cleanly(self):
        ok = C.summary({"custom": "example_tiny"}, 40)
        self.assertTrue(ok["ok"]); self.assertIn("flow", ok)
        bad = C.summary({"custom": "no_such_thing"}, 40)
        self.assertFalse(bad["ok"]); self.assertTrue(bad["problems"])


@unittest.skipIf(torch is None, "torch not installed")
class HtrTopLevelDispatch(unittest.TestCase):
    """htr/__init__.py's _engine() (the direct successor to work_phaseA/archnet.py, now deleted) must route a {"custom": name}
    config here, not to htr.spec/htr.presets."""
    def setUp(self):
        import htr
        self.htr = htr

    def test_normalize_describe_hash_is_default_all_dispatch_to_custom(self):
        H, cfg = self.htr, {"custom": "example_tiny"}
        self.assertEqual(H.normalize(cfg), cfg)
        self.assertEqual(H.describe(cfg, 40)["params"], C.describe(cfg, 40)["params"])
        self.assertEqual(H.spec_hash(cfg), C.spec_hash(cfg))
        self.assertFalse(H.is_default(cfg))

    def test_tflite_support_refused_for_custom(self):
        self.assertFalse(self.htr.tflite_support({"custom": "example_tiny"})["supported"])

    def test_build_wraps_to_bare_logits_same_as_other_engines(self):
        net = self.htr.build({"custom": "example_tiny"}, 40).eval()
        with torch.no_grad(): y = net(torch.zeros(2, 1, 48, 192))
        self.assertEqual(tuple(y.shape), (48, 2, 40))   # bare tensor, not a (logits, lengths) tuple

    def test_check_succeeds_and_reports_tflite_unsupported(self):
        r = self.htr.check({"custom": "example_tiny"}, 40, runs=2)
        self.assertTrue(r["ok"], r.get("problems"))
        self.assertFalse(r["tflite"]["supported"])
        self.assertFalse(r["shipped_weights_fit"])


@unittest.skipIf(torch is None, "torch not installed")
class CompatForCustom(unittest.TestCase):
    def test_onnx_always_supported_rebuild_never(self):
        import htr.export.compat as compat
        c = compat.check({"custom": "example_tiny"})
        self.assertTrue(c["onnx"]["supported"])
        self.assertFalse(c["rebuild"]["supported"])


@unittest.skipIf(torch is None, "torch not installed")
class LoadRejectsABrokenModule(unittest.TestCase):
    """A real, temporary file under htr/custom/, exercising load()'s own error paths (not just _dry_run's) - removed in tearDown."""
    def setUp(self):
        self.path = ROOT / "htr" / "custom" / "test_no_build_fixture.py"
        self.path.write_text("# intentionally has no build() function, for test_htr_custom.py\nTHING = 1\n")

    def tearDown(self):
        self.path.unlink(missing_ok=True)
        import sys as _sys
        _sys.modules.pop("htr.custom.test_no_build_fixture", None)

    def test_missing_build_function_refused(self):
        with self.assertRaisesRegex(ValueError, "no build"): C.load("test_no_build_fixture", n_classes=5)


@unittest.skipIf(torch is None, "torch not installed")
class DryRun(unittest.TestCase):
    def test_non_tuple_return_refused(self):
        with self.assertRaisesRegex(ValueError, "must return"): C._dry_run(_BadReturn(), n_classes=5)

    def test_wrong_batch_size_refused(self):
        with self.assertRaisesRegex(ValueError, "batch"): C._dry_run(_WrongBatch(), n_classes=5)

    def test_non_monotonic_lengths_refused(self):
        with self.assertRaisesRegex(ValueError, "monotonic"): C._dry_run(_NonMonotonic(), n_classes=5)

    def test_output_lengths_must_be_none_when_input_lengths_is_none(self):
        with self.assertRaisesRegex(ValueError, "None"): C._dry_run(_IgnoresNone(), n_classes=5)

    def test_example_fixture_passes_dry_run_directly(self):
        import htr.custom.example_tiny as ex
        C._dry_run(ex.build(5), n_classes=5)   # must not raise


if __name__ == "__main__":
    unittest.main()
