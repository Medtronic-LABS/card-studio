"""Review status (studio/review.py): sync creates/resets rows, set_status records a decision, migrate seeds a sensible starting
state from `how`, and cards.labels(min_status=...) filters on it. Drawn cards only (no patient data)."""
import http.client, json, sys, threading, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, labeling, review, server
from test_labeling import Base, F


class Sync(Base):
    def test_adds_pending_rows_for_new_labels(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        out = review.sync("alpha")
        self.assertEqual(out["added"], 1)
        self.assertEqual(review.read("alpha")[("p-000", F[0], "c1")]["status"], "pending")

    def test_editing_a_reviewed_label_resets_it_to_pending(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.sync("alpha")
        review.set_status("alpha", "p-000", F[0], "c1", "approved", reviewer="r1")
        labeling.save("alpha", "p-000", F[0], "c1", "58", confirm=True)
        out = review.sync("alpha")
        self.assertEqual(out["reset"], 1)
        self.assertEqual(review.read("alpha")[("p-000", F[0], "c1")]["status"], "pending")

    def test_set_status_requires_a_labeled_cell(self):
        with self.assertRaises(KeyError): review.set_status("alpha", "p-000", F[0], "c1", "approved")

    def test_set_status_rejects_unknown_status(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        with self.assertRaises(ValueError): review.set_status("alpha", "p-000", F[0], "c1", "maybe")


class Migrate(Base):
    def test_typed_and_corrected_become_approved_accepted_depends_on_confidence(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")   # no suggestion existed -> how=typed
        out = review.migrate("alpha")
        self.assertEqual(out["approved"], 1)
        self.assertEqual(review.read("alpha")[("p-000", F[0], "c1")]["status"], "approved")

    def test_migrate_is_a_noop_once_review_csv_exists(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.migrate("alpha")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        self.assertIsNone(review.migrate("alpha"))
        self.assertNotIn(("p-001", F[0], "c1"), review.read("alpha"))


class Queue(Base):
    def test_needs_attention_lists_pending_and_flagged_not_approved_or_rejected(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        labeling.save("alpha", "p-002", F[0], "c1", "61")
        review.sync("alpha")
        review.set_status("alpha", "p-000", F[0], "c1", "approved")
        review.set_status("alpha", "p-001", F[0], "c1", "flagged", reason="looks odd")
        rows = review.queue("alpha")
        self.assertEqual({r["page"] for r in rows}, {"p-001", "p-002"})
        self.assertEqual(next(r for r in rows if r["page"] == "p-001")["reason"], "looks odd")

    def test_approved_show_lists_only_approved(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.sync("alpha")
        review.set_status("alpha", "p-000", F[0], "c1", "approved")
        rows = review.queue("alpha", "approved")
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["status"], "approved")

    def test_rows_are_sorted_by_field_then_page_then_col(self):
        labeling.save("alpha", "p-002", F[0], "c1", "57")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        rows = review.queue("alpha")
        self.assertEqual([r["page"] for r in rows], ["p-001", "p-002"])

    def test_unknown_show_value_is_refused(self):
        with self.assertRaises(ValueError): review.queue("alpha", "whenever")


class LabelsFilter(Base):
    def test_min_status_none_is_unchanged_behavior(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        rows, _ = cards.labels("alpha")
        self.assertEqual(len(rows), 1)

    def test_min_status_approved_excludes_pending(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        review.sync("alpha")
        rows, _ = cards.labels("alpha", min_status="approved")
        self.assertEqual(rows, [])
        review.set_status("alpha", "p-000", F[0], "c1", "approved")
        rows, _ = cards.labels("alpha", min_status="approved")
        self.assertEqual(len(rows), 1)

    def test_min_status_pending_excludes_only_rejected(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        review.sync("alpha")
        review.set_status("alpha", "p-001", F[0], "c1", "rejected")
        rows, _ = cards.labels("alpha", min_status="pending")
        self.assertEqual({(r["page"]) for r in rows}, {"p-000"})


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

    def test_review_over_http(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        labeling.save("alpha", "p-001", F[0], "c1", "60")
        st, o = self.call("GET", "/api/card/alpha/review"); self.assertEqual(st, 200)
        self.assertEqual({c["page"] for c in o["cells"]}, {"p-000", "p-001"})
        cell = o["cells"][0]
        st, r = self.call("POST", "/api/card/alpha/review/set", {"page": cell["page"], "field": cell["field"], "col": cell["col"], "status": "approved", "reviewer": "r1"})
        self.assertEqual((st, r["status"], r["reviewer"]), (200, "approved", "r1"))
        st, o = self.call("GET", "/api/card/alpha/review?show=approved"); self.assertEqual((st, len(o["cells"])), (200, 1))
        st, r = self.call("POST", "/api/card/alpha/review/set", {"page": cell["page"], "field": cell["field"], "col": cell["col"], "status": "nonsense"})
        self.assertEqual(st, 400)
        st, r = self.call("POST", "/api/card/alpha/review/set", {"page": "p-999", "field": cell["field"], "col": cell["col"], "status": "approved"})
        self.assertEqual(st, 404)

    def test_migrate_over_http(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57")
        st, r = self.call("POST", "/api/card/alpha/review/migrate", {}); self.assertEqual((st, r["approved"]), (200, 1))
        st, r = self.call("POST", "/api/card/alpha/review/migrate", {}); self.assertEqual((st, r.get("skipped") is not None), (200, True))


if __name__ == "__main__":
    unittest.main()
