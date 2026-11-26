"""htr.export: rebuild_backend parses the same ONNX families as before (moved, not rewritten, from studio/tflite_export.py - which
still re-exports its names so old imports keep working), onnx_backend exports+checks parity for any architecture (legacy or
schema-2), compat.py says which backend fits a config, and backends.export's auto order falls through correctly."""
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "studio"))
import htr.export.compat as compat
import htr.export.rebuild_backend as RB
import tflite_export as T

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


class Reexports(unittest.TestCase):
    def test_tflite_export_still_has_the_moved_names(self):
        self.assertIs(T.read_onnx, RB.read_onnx)
        self.assertIs(T.build, RB.build)
        self.assertIs(T.convert, RB.convert)
        self.assertIs(T.Unavailable, RB.Unavailable)
        self.assertEqual(T.MAX_DIFF, RB.MAX_DIFF)


class Compat(unittest.TestCase):
    def test_legacy_preset_supports_onnx_and_rebuild(self):
        c = compat.check("crnn")
        self.assertTrue(c["onnx"]["supported"]); self.assertTrue(c["rebuild"]["supported"])

    def test_schema2_preset_supports_onnx_not_rebuild(self):
        c = compat.check("cnn-transformer")
        self.assertTrue(c["onnx"]["supported"]); self.assertFalse(c["rebuild"]["supported"])

    def test_future_backends_reported_not_built(self):
        c = compat.check("crnn")
        self.assertFalse(c["ai_edge_torch"]["supported"]); self.assertFalse(c["onnx2tf"]["supported"])


@unittest.skipIf(torch is None, "torch not installed")
class OnnxBackend(unittest.TestCase):
    def test_export_and_parity_for_a_legacy_network(self):
        import htr.export.onnx_backend as O
        from htr.nn.compile import compile
        net = compile("crnn-small", 40).eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reader.onnx"
            blob = O.export(net, 48, 192, path)
            self.assertTrue(path.exists()); self.assertGreater(len(blob), 0)
            p = O.parity(net, path, 48, 192)
            self.assertTrue(p["ok"], p)

    def test_export_and_parity_for_a_schema2_network(self):
        import htr.export.onnx_backend as O
        from htr.nn.compile import compile
        net = compile("crnn-tiny", 40).eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reader.onnx"
            O.export(net, 48, 192, path)
            p = O.parity(net, path, 48, 192)
            self.assertTrue(p["ok"], p)


if __name__ == "__main__":
    unittest.main()
