"""GET /api/runs/table: the line-dataset comparison table (htr.train.runs.scan) served over the existing Card Studio server."""
import http.client, json, shutil, sys, tempfile, threading, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import server
import htr.train.runs as R


class RunsTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown()

    def call(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest("GET", path, skip_host=True, skip_accept_encoding=True); c.putheader("Host", f"127.0.0.1:{self.port}")
        c.endheaders(); r = c.getresponse(); out = json.loads(r.read()); c.close()
        return r.status, out

    def test_empty_when_no_runs_exist(self):
        tmp = Path(tempfile.mkdtemp())
        orig = R.RUNS_ROOT
        try:
            R.RUNS_ROOT = tmp / "no-such-runs-dir"
            st, b = self.call("/api/runs/table")
            self.assertEqual(st, 200); self.assertEqual(b["rows"], [])
        finally:
            R.RUNS_ROOT = orig; shutil.rmtree(tmp, ignore_errors=True)

    def test_lists_a_real_written_run(self):
        tmp = Path(tempfile.mkdtemp())
        orig = R.RUNS_ROOT
        try:
            R.RUNS_ROOT = tmp
            R.write_result("demo", arch_name="crnn-base", arch_hash="h1", n_params=100, best_epoch=3,
                            metrics={"exact": 0.7, "cer_grapheme": 0.1, "cer_codepoint": 0.09, "wer": 0.3, "n": 50})
            st, b = self.call("/api/runs/table")
            self.assertEqual(st, 200); self.assertEqual(len(b["rows"]), 1)
            self.assertEqual(b["rows"][0]["name"], "demo"); self.assertTrue(b["rows"][0]["ok"])
        finally:
            R.RUNS_ROOT = orig; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
