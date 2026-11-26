"""htr.spec must behave exactly as work_phaseA/archnet.py did (same presets, same hashes, same parameter counts), because old
settings.json records, registered pretrained weights and old results.json entries carry its hash. htr.nn.compile must produce a
network whose state dict matches work_phaseA/common.CRNN's, so the shipped weights still load."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "work_phaseA"))
sys.path.insert(0, str(ROOT))
import archnet as OLD           # noqa: E402  the module this package replaces
import htr.spec as S            # noqa: E402

try: import torch  # noqa: F401
except ImportError: torch = None


class MatchesOldArchnet(unittest.TestCase):
    """OLD.PRESETS is now archnet's merged list (legacy + the lab's schema-2 presets, since archnet.py is itself a shim over this
    package - see work_phaseA/archnet.py). What must hold is that every LEGACY preset (S.PRESETS) still normalizes/hashes/describes
    through OLD exactly as htr.spec does; a schema-2 name is covered separately (test_htr_catalog.py)."""
    def test_every_legacy_preset_normalizes_describes_and_hashes_the_same_as_before(self):
        for name in S.PRESETS:
            self.assertEqual(OLD.normalize(name), S.normalize(name), name)
            self.assertEqual(OLD.spec_hash(name), S.spec_hash(name), name)
            self.assertEqual(OLD.describe(OLD.normalize(name)), S.describe(S.normalize(name)), name)

    def test_shipped_crnn_is_538305_parameters(self):
        self.assertEqual(S.describe(S.normalize("crnn"))["params"], 538305)

    def test_flow_and_summary_unchanged(self):
        for name in S.PRESETS:
            self.assertEqual(OLD.flow(name), S.flow(name), name)
        self.assertEqual(OLD.summary("crnn")["hash"], S.summary("crnn")["hash"])

    def test_archnet_presets_also_include_the_labs_schema2_presets(self):
        import htr.presets as P
        self.assertEqual(set(OLD.PRESETS), set(S.PRESETS) | set(P.PRESETS))

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
