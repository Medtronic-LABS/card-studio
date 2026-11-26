"""/api/arch/catalog and /api/arch/compat (M3 backend for a future architecture builder UI): serve htr.catalog's layer registry
and htr.export.compat's per-backend answers over the existing Card Studio server, for both a legacy and a schema-2 architecture."""
import http.client, json, sys, threading, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class ArchRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown()

    def call(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else b""
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True); c.putheader("Host", f"127.0.0.1:{self.port}")
        if method == "POST": c.putheader("X-Card-Studio", "1"); c.putheader("Content-Length", str(len(data)))
        c.endheaders(data); r = c.getresponse(); out = json.loads(r.read()); c.close()
        return r.status, out

    def test_catalog_lists_layer_types_and_limits(self):
        st, b = self.call("POST", "/api/arch/catalog", {})
        self.assertEqual(st, 200)
        self.assertIn("conv2d", b["layer_types"]); self.assertIn("transformer", b["layer_types"])
        self.assertIn("rebuild", b["layer_types"]["conv2d"]["backends"])   # conv2d itself is rebuild-backend compatible
        self.assertNotIn("rebuild", b["layer_types"]["transformer"]["backends"])   # transformer is not
        self.assertIn("params", b["limits"])
        self.assertEqual(set(b["stages"]), {"backbone", "sequence"})

    def test_compat_for_legacy_preset(self):
        st, b = self.call("POST", "/api/arch/compat", {"arch": "crnn"})
        self.assertEqual(st, 200)
        self.assertTrue(b["onnx"]["supported"]); self.assertTrue(b["rebuild"]["supported"])

    def test_compat_for_schema2_preset(self):
        st, b = self.call("POST", "/api/arch/compat", {"arch": "cnn-transformer"})
        self.assertEqual(st, 200)
        self.assertTrue(b["onnx"]["supported"]); self.assertFalse(b["rebuild"]["supported"])

    def test_compat_default_arch_when_none_given(self):
        st, b = self.call("POST", "/api/arch/compat", {})
        self.assertEqual(st, 200); self.assertTrue(b["rebuild"]["supported"])   # the shipped crnn

    def test_custom_lists_the_example_fixture_as_ok(self):
        st, b = self.call("GET", "/api/arch/custom")
        self.assertEqual(st, 200)
        names = {a["name"]: a for a in b["architectures"]}
        self.assertIn("example_tiny", names)
        self.assertTrue(names["example_tiny"]["ok"], names["example_tiny"])
        self.assertGreater(names["example_tiny"]["params"], 0)

    def test_check_and_describe_and_compat_accept_a_custom_arch(self):
        st, b = self.call("POST", "/api/arch/check", {"arch": {"custom": "example_tiny"}})
        self.assertEqual(st, 200); self.assertTrue(b["ok"], b.get("problems"))
        st, b = self.call("POST", "/api/arch/describe", {"arch": {"custom": "example_tiny"}})
        self.assertEqual(st, 200); self.assertTrue(b["ok"])
        st, b = self.call("POST", "/api/arch/compat", {"arch": {"custom": "example_tiny"}})
        self.assertEqual(st, 200); self.assertTrue(b["onnx"]["supported"]); self.assertFalse(b["rebuild"]["supported"])

    def test_export_bench_reports_real_size_and_timing(self):
        st, b = self.call("POST", "/api/export/bench", {"arch": "crnn-small", "width": 96, "runs": 3})
        self.assertEqual(st, 200)
        self.assertGreater(b["size_bytes"], 0); self.assertEqual(b["runs"], 3); self.assertGreaterEqual(b["p95_ms"], b["p50_ms"])

    def test_export_bench_refuses_a_bad_spec(self):
        st, b = self.call("POST", "/api/export/bench", {"arch": {"conv": []}})
        self.assertEqual(st, 400)


if __name__ == "__main__":
    unittest.main()
