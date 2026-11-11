"""Writer codes and writer-disjoint splits. Drawn cards only. Run: python -m unittest discover -s tests"""
import collections, json, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, scans, server
from test_scans import Base as ScanBase, ROWS
from test_labeling import F


class Store(ScanBase):
    def setUp(self):
        super().setUp()
        for i in range(8): scans.add("alpha", self.scan("alpha", cells=((0, 0),), seed=i), f"p-{i:03d}.png")

    def test_codes_not_names(self):
        for bad in ("Dr. Rahman", "name" + chr(64) + "site", "x" * 25, "w 07"):
            with self.assertRaises(ValueError, msg=bad): cards.set_writers("alpha", bad, ["p-000"])
            with self.assertRaises(ValueError, msg=bad): scans.add("alpha", self.scan("alpha"), "z.png", bad)
        with self.assertRaises(ValueError): cards.set_writers("alpha", "W1", [])
        with self.assertRaises(ValueError): cards.set_writers("alpha", "W1", ["../x"])

    def test_assign_range_clear_and_scan_listing(self):
        self.assertEqual(cards.set_writers("alpha", "W1", ["p-000", "p-001"]), 2); self.assertEqual(cards.set_writers("alpha", "W1", ["p-000", "p-001"]), 0)
        self.assertEqual(cards.pages_between("alpha", "p-003", "p-005"), ["p-003", "p-004", "p-005"]); self.assertEqual(cards.pages_between("alpha", "p-005", "p-003"), ["p-003", "p-004", "p-005"])
        cards.set_writers("alpha", "W2", cards.pages_between("alpha", "p-002", "p-003"))
        self.assertEqual({s["page"]: s["writer"] for s in scans.summary("alpha")["scans"]}, {"p-000": "W1", "p-001": "W1", "p-002": "W2", "p-003": "W2", **{f"p-00{i}": "" for i in range(4, 8)}})
        cards.set_writers("alpha", "", ["p-000"]); self.assertNotIn("p-000", cards.writers("alpha"))

    def test_a_batch_writer_is_recorded_and_a_removed_scan_forgets_it(self):
        scans.add("alpha", self.scan("alpha", cells=((2, 2),), seed=40), "q.png", "W9"); self.assertEqual(cards.writers("alpha"), {"q": "W9"})
        scans.remove("alpha", "q"); self.assertEqual(cards.writers("alpha"), {})

    def test_split_units_need_every_page_and_enough_writers(self):
        pages = [f"p-{i:03d}" for i in range(8)]
        self.assertEqual(cards.split_units("alpha", pages)[1:], ("page", 8))
        for i, p in enumerate(pages): cards.set_writers("alpha", f"W{i % 4}", [p])                  # four writers: too few for five folds
        self.assertEqual(cards.split_units("alpha", pages)[1:], ("page", 8))
        for i, p in enumerate(pages): cards.set_writers("alpha", f"W{i % 6}", [p])
        units, unit, n = cards.split_units("alpha", pages); self.assertEqual((unit, n, units["p-000"], units["p-006"]), ("writer", 6, "W0", "W0"))
        cards.set_writers("alpha", "", ["p-007"]); self.assertEqual(cards.split_units("alpha", pages)[1], "page")        # one page without a writer: back to page
        w = cards.writer_status("alpha"); self.assertEqual((w["pages"], w["assigned"]), (8, 7))


class Folds(unittest.TestCase):
    def test_no_writer_is_on_both_sides_of_any_fold(self):
        import trainer
        pages = [f"p-{i:03d}" for i in range(40)]; units = {p: f"W{i % 10}" for i, p in enumerate(pages)}                  # ten writers, four pages each
        rows = [{"page": p, "row": "pulse", "col": "c1", "label": "57"} for p in pages]
        folds = trainer.make_folds(rows, units=units)
        self.assertEqual(sorted(sum(folds, [])), pages)
        for f in folds:
            mine = {units[p] for p in f}; rest = {units[p] for g in folds if g is not f for p in g}; self.assertFalse(mine & rest)
        self.assertEqual(sorted(len({units[p] for p in f}) for f in folds), [2] * 5)
        by_page = trainer.make_folds(rows); self.assertEqual(sorted(len(f) for f in by_page), [8] * 5)                         # without units it is by page, as before


class Endpoint(ScanBase):
    def setUp(self):
        super().setUp()
        import http.client, threading
        for i in range(3): scans.add("alpha", self.scan("alpha", cells=((0, 0),), seed=i), f"p-{i:03d}.png")
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); self.port = self.httpd.server_address[1]; threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown(); self.httpd.server_close(); super().tearDown()

    def call(self, method, path, body=None, raw=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request(method, path, body=raw if raw is not None else (json.dumps(body).encode() if body is not None else b""), headers={"X-Card-Studio": "1"}); r = c.getresponse(); data = r.read(); c.close()
        return r.status, json.loads(data)

    def test_assign_and_read_back_over_http(self):
        st, r = self.call("POST", "/api/card/alpha/writers", {"writer": "W3", "first": "p-000", "last": "p-001"}); self.assertEqual((st, r["changed"], r["assigned"], r["split_unit"]), (200, 2, 2, "page"))
        st, r = self.call("GET", "/api/card/alpha/writers"); self.assertEqual((st, r["by_page"]), (200, {"p-000": "W3", "p-001": "W3"}))
        self.assertEqual(self.call("POST", "/api/card/alpha/writers", {"writer": "Dr Rahman", "pages": ["p-000"]})[0], 400)
        self.assertEqual(self.call("POST", "/api/card/nosuch/writers", {"writer": "W1", "pages": ["p-000"]})[0], 404)
        st, r = self.call("POST", "/api/scans?card=alpha&name=z.png&writer=W5", raw=self.scan("alpha", seed=77)); self.assertEqual(st, 200)
        self.assertEqual(cards.writers("alpha")["z"], "W5"); self.assertEqual(self.call("POST", "/api/scans?card=alpha&name=y.png&writer=bad%20name", raw=self.scan("alpha", seed=78))[0], 400)


if __name__ == "__main__":
    unittest.main()
