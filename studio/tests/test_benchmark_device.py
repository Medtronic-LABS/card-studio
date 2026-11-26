"""tools/benchmark_device.py's pure logic: adb-output parsing, benchmark_model-output parsing, the devices.json matrix, and the
patient-data path refusal. Nothing here touches a real device (none is required or used)."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import benchmark_device as BD


class ParseAdbDevices(unittest.TestCase):
    def test_only_ready_devices_are_listed(self):
        out = "List of devices attached\nABCD1234\tdevice\nEFGH5678\toffline\nIJKL9999\tunauthorized\n"
        self.assertEqual(BD.parse_adb_devices(out), ["ABCD1234"])

    def test_no_devices(self):
        self.assertEqual(BD.parse_adb_devices("List of devices attached\n"), [])


class ParseBenchmarkModel(unittest.TestCase):
    def test_parses_avg_inference_and_peak_memory(self):
        text = "Inference (avg): 12345.6\noverall_resident set size (KB): 51200\n"
        r = BD.parse_benchmark_model_output(text)
        self.assertAlmostEqual(r["avg_ms"], 12.3456, places=3)
        self.assertAlmostEqual(r["peak_rss_mb"], 50.0, places=3)

    def test_missing_fields_are_none(self):
        r = BD.parse_benchmark_model_output("nothing useful here")
        self.assertIsNone(r["avg_ms"]); self.assertIsNone(r["peak_rss_mb"])


class Devices(unittest.TestCase):
    def test_default_matrix_loads_and_has_entry_and_mid(self):
        devices = BD.load_devices()
        ids = {d["id"] for d in devices}
        self.assertIn("entry-12nm", ids); self.assertIn("mid-6nm", ids)

    def test_device_record_by_id(self):
        d = BD.device_record("entry-12nm")
        self.assertEqual(d["targets"]["p50_ms"], 300)

    def test_unknown_device_id_raises(self):
        with self.assertRaises(KeyError): BD.device_record("nonexistent-device")


class RefusePatientPaths(unittest.TestCase):
    def test_refuses_a_path_under_cards(self):
        with self.assertRaises(ValueError): BD.refuse_patient_paths(str(ROOT / "studio" / "cards" / "anc" / "x.tflite"))

    def test_refuses_a_path_under_work(self):
        with self.assertRaises(ValueError): BD.refuse_patient_paths(str(ROOT / "work" / "x.tflite"))

    def test_allows_a_synthetic_export_path(self):
        BD.refuse_patient_paths(str(ROOT / "bengali_dataset" / "exports" / "reader.tflite"))   # must not raise


class NoDeviceAttached(unittest.TestCase):
    def test_run_raises_no_device_when_none_attached(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tflite = Path(d) / "reader.tflite"; tflite.write_bytes(b"fake")
            fake_bm = Path(d) / "benchmark_model"; fake_bm.write_bytes(b"fake")
            if not BD.list_devices():
                with self.assertRaises(BD.NoDevice): BD.run(tflite, "entry-12nm", fake_bm)
            else:
                self.skipTest("a real adb device is attached in this environment")


if __name__ == "__main__":
    unittest.main()
