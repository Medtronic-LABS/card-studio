"""Labeling a card's cut cells and setting field kinds. Drawn cards only (no patient data). Run: python -m unittest discover -s tests"""
import csv, http.client, json, sys, threading, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, labeling, scans, server
from test_scans import Base as ScanBase, ROWS, COLS

F = [f"f1_{i}" for i in range(ROWS)]          # the drawn card's field keys (rows), top to bottom


class Base(ScanBase):
    def setUp(self):
        super().setUp()
        for i in range(3): scans.add("alpha", self.scan("alpha", cells=((0, 0), (1, 1), (2, 2)), seed=i), f"p-{i:03d}.png")
        for key, kind, rng in ((F[0], "number", [30, 150]), (F[1], "date", None), (F[2], "bp", None), (F[3], "mark", None)): labeling.set_field("alpha", key, kind, rng)


class Kinds(Base):
    def test_kinds_are_saved_and_checked(self):
        self.assertEqual([(f["key"], f["kind"], f["range"]) for f in cards.fields("alpha")],
                         [(F[0], "number", [30, 150]), (F[1], "date", None), (F[2], "bp", None), (F[3], "mark", None), (F[4], "text", None)])
        for bad in (("nosuch", "mark", None), ):
            with self.assertRaises(KeyError): labeling.set_field("alpha", *bad)
        for bad in ((F[0], "barcode", None), (F[3], "mark", [1, 2]), (F[0], "number", [9, 3]), (F[0], "number", [1]), (F[0], "number", "1-2")):
            with self.assertRaises(ValueError, msg=str(bad)): labeling.set_field("alpha", *bad)
        labeling.set_field("alpha", F[0], "number"); self.assertIsNone(cards.fields("alpha")[0]["range"])        # a range can be taken away again

    def test_unknown_cards(self):
        for c in ("anc", "nosuch", "../x"):
            with self.assertRaises(KeyError): labeling.overview(c)


class Queue(Base):
    def test_todo_is_the_inked_cells_in_field_page_visit_order(self):
        q = labeling.queue("alpha")
        self.assertEqual((q["total"], [(c["field"], c["page"], c["visit"]) for c in q["cells"]]),
                         (9, [(F[r], f"p-{i:03d}", f"c{r + 1}") for r in range(3) for i in range(3)]))
        self.assertEqual([c["page"] for c in labeling.queue("alpha", field=F[1])["cells"]], ["p-000", "p-001", "p-002"])
        self.assertEqual(labeling.queue("alpha", show="uninked")["total"], 3 * ROWS * COLS - 9)
        self.assertEqual(labeling.queue("alpha", limit=2, after=f"p-001|{F[0]}|c1")["cells"][0]["page"], "p-002")
        with self.assertRaises(KeyError): labeling.queue("alpha", field="nosuch")
        with self.assertRaises(ValueError): labeling.queue("alpha", show="everything")

    def test_a_recorded_cell_leaves_todo_and_shows_under_done(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57"); labeling.save("alpha", "p-001", F[0], "c1", note="blank")
        self.assertEqual([c["page"] for c in labeling.queue("alpha", field=F[0])["cells"]], ["p-002"])
        self.assertEqual([(c["page"], c["label"], c["note"]) for c in labeling.queue("alpha", field=F[0], show="done")["cells"]], [("p-000", "57", ""), ("p-001", "", "blank")])
        labeling.clear("alpha", "p-001", F[0], "c1"); self.assertEqual(labeling.queue("alpha", field=F[0])["total"], 2)
        with self.assertRaises(KeyError): labeling.clear("alpha", "p-001", F[0], "c1")


class Save(Base):
    def test_bangla_digits_are_stored_as_ascii_and_only_labels_are_trained_on(self):
        self.assertEqual(labeling.save("alpha", "p-000", F[0], "c1", " ৫৭ ")["label"], "57")
        labeling.save("alpha", "p-001", F[0], "c1", note="unreadable"); labeling.save("alpha", "p-002", F[0], "c1", note="blank")
        rows, bad = cards.labels("alpha")
        self.assertEqual(([(r["page"], r["row"], r["col"], r["label"]) for r in rows], bad), ([("p-000", F[0], "c1", "57")], {}))
        with open(cards.CARDS / "alpha" / "labels.csv", newline="") as fh: self.assertEqual(csv.DictReader(fh).fieldnames, labeling.COLS)

    def test_characters_the_field_cannot_emit_are_refused(self):
        for field, visit, label in ((F[0], "c1", "5y"), (F[1], "c2", "22-2-26x"), (F[2], "c3", "120.80"), (F[0], "c1", "x")):
            with self.assertRaises(ValueError, msg=label): labeling.save("alpha", "p-000", field, visit, label)
        self.assertEqual(labeling.queue("alpha", show="done")["total"], 0)

    def test_a_label_that_looks_wrong_needs_confirming(self):
        for field, visit, label, why in ((F[0], "c1", "570", "range"), (F[0], "c1", "5.7.1", "number"), (F[1], "c2", "22/26", "day, month, year"), (F[1], "c2", "40/13/26", "real day"), (F[2], "c3", "120", "systolic")):
            r = labeling.save("alpha", "p-000", field, visit, label); self.assertFalse(r["saved"], label); self.assertIn(why, r["warning"])
        self.assertEqual(labeling.queue("alpha", show="done")["total"], 0)
        self.assertTrue(labeling.save("alpha", "p-000", F[0], "c1", "570", confirm=True)["saved"])
        for field, visit, label in ((F[0], "c1", "57.5"), (F[0], "c1", "-"), (F[1], "c2", "22/2/26"), (F[1], "c2", "3.11.2026"), (F[2], "c3", "120/80")):
            self.assertEqual(labeling.save("alpha", "p-001", field, visit, label), {"saved": True, "label": label, "note": "", "warning": None})

    def test_bad_cells_notes_and_empty_labels(self):
        with self.assertRaises(KeyError): labeling.save("alpha", "p-999", F[0], "c1", "57")
        with self.assertRaises(KeyError): labeling.save("alpha", "p-000", "nosuch", "c1", "57")
        with self.assertRaises(ValueError): labeling.save("alpha", "p-000", F[0], "c1", "")
        with self.assertRaises(ValueError): labeling.save("alpha", "p-000", F[0], "c1", "57", note="maybe")
        with self.assertRaises(ValueError): labeling.save("alpha", "p-000", F[4], "c1", "a" * 41)

    def test_overview_counts_and_flags_labels_a_changed_kind_no_longer_allows(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57"); labeling.save("alpha", "p-001", F[0], "c1", note="blank")
        o = labeling.overview("alpha"); f = o["fields"][0]
        self.assertEqual((f["cells"], f["inked"], f["labeled"], f["blank"], f["unreadable"], f["todo"], f["not_allowed_now"]), (9, 3, 1, 1, 0, 1, 0))
        self.assertEqual((o["labeled_cells"], o["labeled_pages"], o["todo"], o["trainable"]), (1, 1, 7, False)); self.assertIn("needs at least", o["blocker"])
        labeling.set_field("alpha", F[0], "mark")
        self.assertEqual(labeling.overview("alpha")["fields"][0]["not_allowed_now"], 1)
        self.assertEqual(cards.labels("alpha"), ([], {"bad_characters": 1}))


class Endpoint(Base):
    def setUp(self):
        super().setUp(); self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown(); self.httpd.server_close(); super().tearDown()

    def call(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request(method, path, body=json.dumps(body).encode() if body is not None else b"", headers={"X-Card-Studio": "1"}); r = c.getresponse(); data = r.read(); c.close()
        return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)

    def test_label_a_cell_over_http(self):
        st, o = self.call("GET", "/api/card/alpha/labeling"); self.assertEqual((st, o["todo"], o["fields"][0]["kind"]), (200, 9, "number"))
        st, q = self.call("GET", f"/api/card/alpha/labeling/queue?field={F[0]}&limit=1"); c = q["cells"][0]; self.assertEqual((st, q["total"], c["page"]), (200, 3, "p-000"))
        st, img = self.call("GET", f"/api/card/alpha/crop/{c['field']}/{c['visit']}/{c['page']}.png"); self.assertEqual((st, img[:4]), (200, b"\x89PNG"))
        cell = {"page": c["page"], "field": c["field"], "visit": c["visit"]}
        self.assertEqual(self.call("POST", "/api/card/alpha/label", {**cell, "label": "5y"})[0], 400)
        st, r = self.call("POST", "/api/card/alpha/label", {**cell, "label": "570"}); self.assertEqual((st, r["saved"]), (200, False))
        st, r = self.call("POST", "/api/card/alpha/label", {**cell, "label": "57"}); self.assertEqual((st, r["saved"], r["label"]), (200, True, "57"))
        self.assertEqual(self.call("GET", f"/api/card/alpha/labeling/queue?field={F[0]}")[1]["total"], 2)
        self.assertEqual(self.call("POST", "/api/card/alpha/label/clear", cell), (200, {"cleared": True}))
        self.assertEqual(self.call("POST", "/api/card/alpha/field", {"key": F[4], "kind": "mark"})[0], 200)
        self.assertEqual(self.call("POST", "/api/card/alpha/field", {"key": F[4], "kind": "barcode"})[0], 400)
        for path in ("/api/card/nosuch/labeling", "/api/card/alpha/labeling/queue?field=nosuch", f"/api/card/alpha/crop/{F[0]}/c1/p-999.png", "/api/card/anc/labeling"):
            self.assertEqual(self.call("GET", path)[0], 404, path)
        self.assertEqual(self.call("POST", "/api/card/alpha/label", {"page": "p-999", "field": F[0], "visit": "c1", "label": "57"})[0], 404)


if __name__ == "__main__":
    unittest.main()
