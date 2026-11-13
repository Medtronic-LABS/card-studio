"""The reader's architecture as a spec: validation, presets, parameter counts, and (with torch) that the default spec IS the shipped CRNN."""
import sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "work_phaseA"))
import archnet as A

try: import torch  # noqa: F401
except ImportError: torch = None


class Spec(unittest.TestCase):
    def test_every_preset_is_valid_with_48_time_steps_and_its_hash_is_stable(self):
        hashes = set()
        for name in A.PRESETS:
            s = A.normalize(name); d = A.describe(s); self.assertEqual(d["time_steps"], 48, name); self.assertEqual(A.normalize(s), s, name)
            hashes.add(A.spec_hash(name))
        self.assertEqual(len(hashes), len(A.PRESETS)); self.assertEqual(A.describe(A.normalize("crnn"))["params"], 538305)          # the shipped reader's 0.54M parameters

    def test_the_last_block_always_collapses_the_height_and_defaults_are_filled(self):
        s = A.normalize({"conv": [{"out": 16, "pool": [2, 2]}, {"out": 32, "pool": [2, 1]}], "rnn": {"type": "gru", "hidden": 32}})
        self.assertEqual(s["conv"][0], {"out": 16, "k": 3, "pool": [2, 2]}); self.assertEqual(s["conv"][1]["pool"], "collapse")
        self.assertEqual(s["rnn"], {"type": "gru", "hidden": 32, "layers": 1, "bidirectional": True, "dropout": 0.0}); self.assertEqual(s["dropout"], 0.3)

    def test_nonsense_is_refused_with_a_reason(self):
        ok = {"conv": [{"out": 16}], "rnn": {"type": "lstm", "hidden": 32}}
        for bad, msg in (({"conv": []}, "blocks"), ({**ok, "conv": [{"out": 16, "k": 4}]}, "k must"), ({**ok, "conv": [{"out": 16, "pool": [3, 1]}]}, "pool"), ({**ok, "conv": [{"out": 1000}]}, "between"),
                         ({**ok, "conv": [{"out": 16, "pool": "collapse"}, {"out": 16}]}, "only the last"), ({**ok, "conv": [{"out": 8, "pool": [1, 4]}] * 3 + [{"out": 8}]}, "time steps"),
                         ({"conv": [{"out": 16}], "rnn": None}, "context"), ({**ok, "rnn": {"type": "transformer", "hidden": 32}}, "lstm or gru"), ({**ok, "rnn": {"type": "lstm", "hidden": 2}}, "between"),
                         ({**ok, "extra": 1}, "unknown"), ({**ok, "dropout": 0.99}, "between"), ("no-such-preset", "unknown architecture preset"), (5, "preset name or an object")):
            with self.assertRaisesRegex(ValueError, msg): A.normalize(bad)

    def test_a_cnn_only_reader_needs_a_context_convolution_and_a_huge_one_is_refused(self):
        A.normalize({"conv": [{"out": 32}], "context": [{"out": 64, "k": 5}], "rnn": None})
        with self.assertRaisesRegex(ValueError, "limit"): A.normalize({"conv": [{"out": 512, "k": 7}] * 8, "rnn": {"type": "lstm", "hidden": 512, "layers": 3}})

    def test_the_default_is_recognised(self):
        self.assertTrue(A.is_default("crnn")); self.assertFalse(A.is_default("crnn-wide")); self.assertTrue(A.is_default(A.PRESETS["crnn"]["spec"]))


@unittest.skipUnless(torch, "needs torch")
class Network(unittest.TestCase):
    def test_the_default_spec_has_exactly_the_shipped_networks_parameters_and_output(self):
        import common as C
        ref = C.CRNN(len(C.CLASSES)).eval(); net = A.build("crnn", len(C.CLASSES)).eval()
        self.assertEqual(list(ref.state_dict()), list(net.state_dict())); net.load_state_dict(ref.state_dict())
        x = torch.randn(2, 1, 48, 192)
        with torch.no_grad(): self.assertEqual(float((ref(x) - net(x)).abs().max()), 0.0)

    def test_every_preset_builds_to_the_counted_size_and_the_reader_contract(self):
        for name in A.PRESETS:
            net = A.build(name, 129).eval(); d = A.describe(A.normalize(name))
            self.assertEqual(sum(p.numel() for p in net.parameters()), d["params"], name)
            with torch.no_grad(): self.assertEqual(tuple(net(torch.zeros(3, 1, 48, 192)).shape), (d["time_steps"], 3, 129), name)

    def test_check_is_a_dry_run_that_reports_instead_of_raising(self):
        r = A.check("cnn-ctc", 129, runs=2); self.assertTrue(r["ok"], r["problems"]); self.assertTrue(r["onnx_ok"]); self.assertGreater(r["params"], 0)
        self.assertFalse(r["shipped_weights_fit"]); self.assertTrue(A.check("crnn", 129, runs=2)["shipped_weights_fit"])
        bad = A.check({"conv": []}, 129); self.assertFalse(bad["ok"]); self.assertIn("blocks", bad["problems"][0])


if __name__ == "__main__":
    unittest.main()
