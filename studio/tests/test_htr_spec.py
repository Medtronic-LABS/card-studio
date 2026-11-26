"""htr.spec must keep behaving exactly as the old work_phaseA/archnet.py (now deleted - htr/__init__.py is its direct successor) did:
same presets, same hashes, same parameter counts, because old settings.json records, registered pretrained weights and old
results.json entries carry its hash. The expected values below were captured from archnet.py before it was deleted, by running
archnet.spec_hash/describe/flow against every legacy preset one last time; htr.nn.compile must also produce a network whose state
dict matches work_phaseA/common.CRNN's, so the shipped weights still load."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "work_phaseA"))
sys.path.insert(0, str(ROOT))
import htr.spec as S            # noqa: E402

try: import torch  # noqa: F401
except ImportError: torch = None

# Frozen from archnet.py (the module htr.spec replaces) just before it was deleted: {preset: (hash, params, len(flow))}.
FROZEN = {"crnn": ("11e4af9f47f7", 538305, 7), "crnn-small": ("1d805b3cb3fe", 309681, 7), "crnn-wide": ("cf047e452f71", 2589825, 7),
          "gru-crnn": ("cd96c3d4662a", 472257, 7), "cnn-ctc": ("b009c18ce34b", 767169, 8)}


class SpecStaysStable(unittest.TestCase):
    """Every legacy preset's hash, parameter count and flow length must never move: a pretrained-weight record's arch_sha256 or an
    old results.json's hash would silently stop matching if they did."""
    def test_every_legacy_preset_matches_its_frozen_hash_params_and_flow_length(self):
        self.assertEqual(set(FROZEN), set(S.PRESETS))
        for name, (h, params, flow_len) in FROZEN.items():
            self.assertEqual(S.spec_hash(name), h, name)
            self.assertEqual(S.describe(S.normalize(name))["params"], params, name)
            self.assertEqual(len(S.flow(name)), flow_len, name)

    def test_shipped_crnn_is_538305_parameters(self):
        self.assertEqual(S.describe(S.normalize("crnn"))["params"], 538305)

    def test_summary_hash_matches_spec_hash(self):
        self.assertEqual(S.summary("crnn")["hash"], S.spec_hash("crnn"))

    def test_htr_top_level_presets_include_the_labs_schema2_presets_too(self):
        import htr
        import htr.presets as P
        self.assertEqual(set(htr.PRESETS), set(S.PRESETS) | set(P.PRESETS))

    def test_bad_specs_are_still_refused(self):
        with self.assertRaises(ValueError): S.normalize({"conv": []})
        with self.assertRaises(ValueError): S.normalize({"conv": [{"out": 16, "k": 4}]})


@unittest.skipIf(torch is None, "torch not installed")
class CompileMatchesLegacyCRNN(unittest.TestCase):
    def test_state_dict_and_forward_match_common_CRNN(self):
        import torch
        import common as C
        import htr.legacy as L
        from htr.nn.compile import compile as build_net

        old = C.CRNN(len(C.CLASSES)).eval()
        self.assertTrue(L.state_dict_matches(old.state_dict(), len(C.CLASSES)))
        net = build_net("crnn", len(C.CLASSES)).eval()
        net.load_state_dict(old.state_dict())
        x = torch.randn(2, 1, C.IMG_H, C.IMG_W)
        with torch.no_grad():
            y_new, lengths = net(x, torch.full((2,), C.IMG_W, dtype=torch.long))
            y_old = old(x)
        self.assertTrue(torch.allclose(y_new, y_old, atol=1e-5))
        self.assertEqual(lengths.tolist(), [y_new.shape[0]] * 2)

    def test_check_reports_ok_for_every_preset(self):
        import htr
        for name in S.PRESETS:
            r = htr.check(name, n_classes=129, runs=2)
            self.assertTrue(r["ok"], r.get("problems"))
            self.assertEqual(r["hash"], S.spec_hash(name))


if __name__ == "__main__":
    unittest.main()
