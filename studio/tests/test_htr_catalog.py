"""htr.catalog / htr.presets: the schema-2 layer catalog and its 5 presets (crnn-tiny/base/large, cnn-transformer, svtr-tiny),
plus htr.nn.compile's dispatch between the legacy and schema-2 paths. Every preset must validate, its describe()d parameter count
must match what actually gets built, and htr.check must pass (shape, ONNX parity) the same way it does for a legacy preset."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.catalog as K
import htr.presets as P
import htr.spec as S

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


class Normalize(unittest.TestCase):
    def test_every_preset_normalizes(self):
        for name in P.PRESETS:
            s = P.normalize(name)
            self.assertEqual(s["schema"], 2)

    def test_last_backbone_block_always_collapses_height_even_if_its_pool_does_not_say_so(self):
        # same convention as htr.spec (legacy): whatever the last block's own pool asks for, height ends at 1
        cfg = {"schema": 2, "backbone": [{"type": "conv2d", "out": 16, "k": 3, "pool": [2, 2]}], "sequence": [{"type": "rnn", "hidden": 32}]}
        s = P.normalize(cfg)
        self.assertEqual(P.describe(s)["time_steps"], K.IMG_W_REF // 2)   # height collapsed, width still halved by the same pool's width factor

    def test_too_many_backbone_blocks_leave_too_few_time_steps(self):
        cfg = {"schema": 2, "backbone": [{"type": "conv2d", "out": 16, "k": 3, "pool": [2, 4]}, {"type": "conv2d", "out": 16, "k": 3, "pool": [2, 4]},
                                          {"type": "conv2d", "out": 16, "k": 3, "pool": [2, 4]}, {"type": "conv2d", "out": 16, "k": 3, "pool": "collapse"}],
               "sequence": []}
        with self.assertRaises(ValueError): P.normalize(cfg)

    def test_sequence_block_needs_collapsed_height(self):
        with self.assertRaises(ValueError): K.validate_block("rnn", {"hidden": 32}, h=2, w=48, c=16)

    def test_unknown_layer_type_rejected(self):
        with self.assertRaises(ValueError): K.validate_block("resnet_block", {}, h=1, w=48, c=16)

    def test_rnn_cell_must_be_lstm_or_gru(self):
        with self.assertRaises(ValueError): K.validate_block("rnn", {"hidden": 32, "cell": "transformer"}, h=1, w=48, c=16)

    def test_dim_must_divide_by_heads(self):
        with self.assertRaises(ValueError): K.validate_block("transformer", {"dim": 100, "heads": 3}, h=1, w=48, c=16)


class SpecHash(unittest.TestCase):
    def test_stable_and_changes_with_content(self):
        a, b = P.normalize("crnn-tiny"), P.normalize("crnn-tiny")
        self.assertEqual(P.spec_hash(a), P.spec_hash(b))
        changed = {**a, "head": {"dropout": 0.5}}
        self.assertNotEqual(P.spec_hash(a), P.spec_hash(changed))

    def test_no_collision_with_legacy_preset_names(self):
        self.assertEqual(set(P.PRESETS) & set(S.PRESETS), set())


@unittest.skipIf(torch is None, "torch not installed")
class CompileDispatch(unittest.TestCase):
    def test_legacy_name_builds_through_legacy_path(self):
        from htr.nn.compile import _is_schema2, compile_legacy
        self.assertFalse(_is_schema2("crnn"))
        net = compile_legacy("crnn", 131)
        self.assertTrue(hasattr(net, "cnn"))

    def test_schema2_name_builds_through_schema2_path(self):
        from htr.nn.compile import _is_schema2, compile_schema2
        self.assertTrue(_is_schema2("crnn-tiny"))
        net = compile_schema2("crnn-tiny", 131)
        self.assertTrue(hasattr(net, "backbone")); self.assertTrue(hasattr(net, "sequence"))

    def test_check_reports_ok_and_exact_param_count_for_every_preset(self):
        import htr
        for name in P.PRESETS:
            r = htr.check(name, n_classes=131, runs=2)
            self.assertTrue(r["ok"], r.get("problems"))
            self.assertEqual(r["params"], r["params_built"], name)
            self.assertLess(r["onnx_max_diff"], 1e-3, name)

    def test_forward_respects_dynamic_lengths(self):
        from htr.nn.compile import compile
        net = compile("cnn-transformer", 131).eval()
        x = torch.zeros(2, 1, K.IMG_H, K.IMG_W_REF)
        lengths = torch.tensor([K.IMG_W_REF, K.IMG_W_REF // 2])
        with torch.no_grad():
            logits, out_lengths = net(x, lengths)
        self.assertEqual(out_lengths.tolist(), [l // net.width_stride for l in lengths.tolist()])
        self.assertEqual(logits.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
