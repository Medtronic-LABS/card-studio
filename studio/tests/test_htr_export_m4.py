"""M4 export surface not exercised above: ai_edge_backend/onnx2tf_backend report Unavailable cleanly when their packages are not
installed (true in this environment - see each module's own docstring on what is and is not verified), compat.py's dynamic
onnx2tf/ai_edge_torch answers, and backends.export's auto fallthrough to the legacy rebuild backend."""
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.export.ai_edge_backend as AE
import htr.export.compat as compat
import htr.export.onnx2tf_backend as O2T

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


class Availability(unittest.TestCase):
    def test_ai_edge_torch_not_installed_here(self):
        self.assertFalse(AE.available())

    def test_onnx2tf_not_installed_here(self):
        self.assertFalse(O2T.available())

    def test_convert_raises_unavailable_not_some_other_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(O2T.Unavailable): O2T.convert(Path(d) / "x.onnx", d)


class Compat(unittest.TestCase):
    def test_reports_both_future_backends_unavailable_for_a_legacy_preset(self):
        c = compat.check("crnn")
        self.assertFalse(c["ai_edge_torch"]["supported"]); self.assertIn("not installed", c["ai_edge_torch"]["reason"])
        self.assertFalse(c["onnx2tf"]["supported"]); self.assertIn("not installed", c["onnx2tf"]["reason"])

    def test_reports_both_future_backends_unavailable_for_a_schema2_preset(self):
        c = compat.check("cnn-transformer")
        self.assertFalse(c["ai_edge_torch"]["supported"]); self.assertFalse(c["onnx2tf"]["supported"])


@unittest.skipIf(torch is None, "torch not installed")
class BackendsAuto(unittest.TestCase):
    def test_auto_falls_through_to_rebuild_for_a_legacy_config_when_others_unavailable(self):
        import htr.export.backends as B
        from htr.nn.compile import compile
        net = compile("crnn-small", 37).eval()
        with tempfile.TemporaryDirectory() as d:
            onnx_path = Path(d) / "reader.onnx"; tflite_path = Path(d) / "reader.tflite"
            try:
                report = B.export(net, "crnn-small", 48, 192, onnx_path, tflite_path=tflite_path)
            except B.NoBackend as e:
                # rebuild itself needs a TensorFlow venv this environment may not have; that is a legitimate "no backend yet"
                self.assertIn("rebuild", e.attempts)
                return
            self.assertEqual(report["backend"], "rebuild")

    def test_onnx_always_written_even_with_no_tflite_path(self):
        import htr.export.backends as B
        from htr.nn.compile import compile
        net = compile("crnn-small", 37).eval()
        with tempfile.TemporaryDirectory() as d:
            onnx_path = Path(d) / "reader.onnx"
            report = B.export(net, "crnn-small", 48, 192, onnx_path)
            self.assertTrue(onnx_path.exists()); self.assertIsNone(report["tflite"])


if __name__ == "__main__":
    unittest.main()
