"""Backend guards: Host / Origin / header checks, body limit, upload expiry. Run: python -m unittest discover -s tests"""
import http.client, json, os, sys, tempfile, threading, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class Guards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def call(self, method, path, headers=None, body=b""):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": f"127.0.0.1:{self.port}", **(headers or {})}
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in h.items(): c.putheader(k, v)
        if method == "POST" and "Content-Length" not in h: c.putheader("Content-Length", str(len(body)))
        c.endheaders(body)
        r = c.getresponse(); data = r.read(); c.close()
        return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)

    def test_get_with_right_host_works(self):
        self.assertEqual(self.call("GET", "/api/packs")[0], 200)

    def test_rebinding_host_refused(self):
        for host in ("evil.example", f"evil.example:{self.port}", "127.0.0.1", f"127.0.0.1.evil.example:{self.port}"):
            self.assertEqual(self.call("GET", "/api/packs", {"Host": host})[0], 421, host)

    def test_localhost_name_allowed(self):
        self.assertEqual(self.call("GET", "/api/packs", {"Host": f"localhost:{self.port}"})[0], 200)

    def test_cross_site_get_refused(self):
        self.assertEqual(self.call("GET", "/api/packs", {"Sec-Fetch-Site": "cross-site"})[0], 403)

    def test_post_needs_header(self):
        self.assertEqual(self.call("POST", "/api/train/start", body=b"{}")[0], 403)

    def test_post_foreign_origin_refused(self):
        for o in ("http://evil.example", "null", f"http://127.0.0.1:{self.port + 1}"):
            self.assertEqual(self.call("POST", "/api/save", {"X-Card-Studio": "1", "Origin": o}, b"{}")[0], 403, o)

    def test_post_same_origin_reaches_handler(self):
        st, body = self.call("POST", "/api/save", {"X-Card-Studio": "1", "Origin": f"http://127.0.0.1:{self.port}"}, b"{}")
        self.assertEqual(st, 400)  # got past the guard; rejected by validation (no name)

    def test_oversize_and_bad_length(self):
        self.assertEqual(self.call("POST", "/api/analyze", {"X-Card-Studio": "1", "Content-Length": str(server.MAX_BODY + 1)})[0], 413)
        self.assertEqual(self.call("POST", "/api/analyze", {"X-Card-Studio": "1", "Content-Length": "abc"})[0], 400)


class Uploads(unittest.TestCase):
    def test_purge_removes_only_old(self):
        with tempfile.TemporaryDirectory() as d:
            old_up = server.UPLOADS; server.UPLOADS = Path(d)
            try:
                a, b = Path(d) / "a.png", Path(d) / "b.png"
                a.write_bytes(b"x"); b.write_bytes(b"x")
                os.utime(a, (time.time() - 3 * 86400,) * 2)
                self.assertEqual(server.purge_uploads(), 1)
                self.assertFalse(a.exists()); self.assertTrue(b.exists())
            finally:
                server.UPLOADS = old_up


if __name__ == "__main__":
    unittest.main()
