"""Label Studio sync, against a stand-in server that speaks the few API calls Studio uses. No Label Studio needed. Run: python -m unittest discover -s tests"""
import http.server, json, re, sys, threading, unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, labeling, labelstudio as L, scans
from test_scans import Base as ScanBase, ROWS

F = [f"f1_{i}" for i in range(ROWS)]
STANDIN, WRONG = "stand-in-value", "another-value"      # not secrets: the stand-in server below accepts the first


class FakeLabelStudio(http.server.BaseHTTPRequestHandler):
    """Projects, a local-files storage entry, task import and task listing: only what labelstudio.py calls."""
    def log_message(self, *a): pass

    def _send(self, code, body):
        data = json.dumps(body).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def _go(self, method):
        S = self.server.state; u = urlparse(self.path); q = parse_qs(u.query)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"null")
        S["calls"].append(f"{method} {u.path}")
        if self.headers.get("Authorization") != f"Token {STANDIN}": return self._send(401, {"detail": "Invalid token."})
        if (method, u.path) == ("GET", "/api/projects"): return self._send(200, {"results": [{**p, "task_number": sum(t["project"] == p["id"] for t in S["tasks"])} for p in S["projects"]]})
        if (method, u.path) == ("POST", "/api/projects"):
            p = {"id": len(S["projects"]) + 1, **body}; S["projects"].append(p); return self._send(201, p)
        if (method, u.path) == ("POST", "/api/storages/localfiles"): S["storages"].append(body); return self._send(201, body)
        m = re.fullmatch(r"/api/projects/(\d+)/import", u.path)
        if method == "POST" and m:
            for t in body: S["tasks"].append({"id": len(S["tasks"]) + 1, "project": int(m[1]), "data": t["data"], "annotations": []})
            return self._send(201, {"task_count": len(body)})
        if (method, u.path) == ("GET", "/api/tasks"):
            mine = [t for t in S["tasks"] if t["project"] == int(q["project"][0])]; n, size = int(q["page"][0]), int(q["page_size"][0])
            return self._send(200, {"tasks": mine[(n - 1) * size:n * size]}) if (n - 1) * size < len(mine) or n == 1 else self._send(404, {"detail": "Invalid page."})
        self._send(404, {"detail": "not found"})

    def do_GET(self): self._go("GET")
    def do_POST(self): self._go("POST")


def answer(text=None, choice=None, cancelled=False, when="2026-11-11T10:00:00Z"):
    res = ([{"type": "textarea", "value": {"text": [text]}}] if text is not None else []) + ([{"type": "choices", "value": {"choices": [choice]}}] if choice else [])
    return {"result": res, "was_cancelled": cancelled, "updated_at": when}


class Base(ScanBase):
    def setUp(self):
        super().setUp()
        for i in range(2): scans.add("alpha", self.scan("alpha", cells=((0, 0), (1, 1)), seed=i), f"p-{i:03d}.png")
        labeling.set_field("alpha", F[0], "number", [30, 150]); labeling.set_field("alpha", F[1], "date")
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLabelStudio); self.httpd.state = self.S = {"projects": [], "storages": [], "tasks": [], "calls": []}
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.old_cfg = L.CONFIG; L.CONFIG = Path(self.tmp.name) / "labelstudio.json"; self.configure()

    def configure(self, url=None, token=STANDIN):
        L.CONFIG.write_text(json.dumps({"url": url or f"http://127.0.0.1:{self.httpd.server_address[1]}", "token": token}))

    def tearDown(self):
        L.CONFIG = self.old_cfg; self.httpd.shutdown(); self.httpd.server_close(); super().tearDown()

    def task(self, page, field, visit):
        return next(t for t in self.S["tasks"] if (t["data"]["page"], t["data"]["field"], t["data"]["visit"]) == (page, field, visit))


class Push(Base):
    def test_one_task_per_cell_to_do_and_a_second_push_adds_nothing(self):
        r = L.push("alpha"); self.assertEqual((r["added"], r["already"], len(self.S["tasks"])), (4, 0, 4))
        t = self.task("p-000", F[0], "1")
        self.assertEqual(t["data"], {"image": f"/data/local-files/?d=alpha/crops/{F[0]}/c1/p-000.png", "card": "alpha", "field": F[0], "kind": "number", "visit": "1", "page": "p-000"})
        self.assertTrue((cards.CARDS / "alpha" / "crops" / F[0] / "c1" / "p-000.png").exists())          # the path the task points at, under the document root
        self.assertEqual((self.S["projects"][0]["title"], self.S["storages"][0]["path"]), ("Card Studio: alpha", str((cards.CARDS / "alpha").resolve())))
        self.assertEqual(L.push("alpha")["added"], 0); self.assertEqual((len(self.S["tasks"]), len(self.S["projects"]), len(self.S["storages"])), (4, 1, 1))

    def test_cells_already_recorded_in_studio_are_not_sent_and_new_scans_are(self):
        labeling.save("alpha", "p-000", F[0], "c1", "57"); self.assertEqual(L.push("alpha")["added"], 3)
        scans.add("alpha", self.scan("alpha", cells=((0, 0),), seed=7), "p-007.png")
        r = L.push("alpha"); self.assertEqual((r["added"], r["already"]), (1, 3))


class Pull(Base):
    def test_answers_are_recorded_through_the_same_checks(self):
        L.push("alpha")
        self.task("p-000", F[0], "1")["annotations"] = [answer("৫৭")]
        self.task("p-001", F[0], "1")["annotations"] = [answer(choice="blank")]
        self.task("p-000", F[1], "2")["annotations"] = [answer("5y")]                                  # a date field cannot hold y
        self.task("p-001", F[1], "2")["annotations"] = [answer("22/2/26", cancelled=True)]              # skipped by the labeler: not an answer
        r = L.pull("alpha")
        self.assertEqual((r["annotated"], r["new"], r["changed"], r["same"], [(x["field"], x["label"]) for x in r["rejected"]]), (3, 2, 0, 0, [(F[1], "5y")]))
        self.assertEqual([(c["page"], c["field"], c["label"], c["note"]) for c in labeling.queue("alpha", show="done")["cells"]], [("p-000", F[0], "57", ""), ("p-001", F[0], "", "blank")])
        self.assertEqual(cards.labels("alpha")[0][0]["label"], "57")

    def test_newest_answer_wins_and_replaces_what_studio_had(self):
        L.push("alpha"); labeling.save("alpha", "p-000", F[0], "c1", "60")
        self.task("p-000", F[0], "1")["annotations"] = [answer("58", when="2026-11-11T09:00:00Z"), answer("59", when="2026-11-11T11:00:00Z")]
        r = L.pull("alpha"); self.assertEqual((r["new"], r["changed"], r["same"]), (0, 1, 0))
        self.assertEqual(labeling.queue("alpha", F[0], "done")["cells"][0]["label"], "59")
        self.assertEqual(L.pull("alpha")["same"], 1)

    def test_a_value_outside_the_range_is_still_recorded(self):
        L.push("alpha"); self.task("p-000", F[0], "1")["annotations"] = [answer("570")]                # a person there typed it: kept, as Studio keeps a confirmed warning
        self.assertEqual(L.pull("alpha")["new"], 1)

    def test_no_project_yet(self):
        with self.assertRaises(ValueError): L.pull("alpha")
        with self.assertRaises(KeyError): L.pull("nosuch")


class Availability(Base):
    def test_only_a_label_studio_on_this_machine_is_used(self):
        for url in ("http://203.0.113.9:8780", "https://127.0.0.1:8780", "http://labelstudio.example.com"):
            self.configure(url=url); n = len(self.S["calls"])
            with self.assertRaises(L.Unavailable): L.push("alpha")
            self.assertIn("not on this machine", L.status("alpha")["problem"]); self.assertEqual(len(self.S["calls"]), n)

    def test_wrong_token_missing_token_and_server_down(self):
        self.configure(token=WRONG); self.assertIn("refused the token", L.status("alpha")["problem"])
        with self.assertRaises(L.Unavailable): L.push("alpha")
        self.configure(token=""); s = L.status("alpha"); self.assertEqual((s["configured"], s["running"]), (False, False))
        self.configure(url="http://127.0.0.1:9"); s = L.status("alpha"); self.assertEqual((s["configured"], s["running"]), (True, False)); self.assertIn("not reachable", s["problem"])
        self.assertEqual(self.S["tasks"], [])

    def test_status_of_a_running_server(self):
        self.assertEqual((L.status("alpha")["running"], L.status("alpha")["project"]), (True, None))
        L.push("alpha"); p = L.status("alpha")["project"]; self.assertEqual((p["id"], p["tasks"]), (1, 4)); self.assertTrue(p["url"].endswith("/projects/1"))
        self.assertTrue(L.status()["running"]); self.assertIn("no such card", L.status("nosuch")["problem"])


if __name__ == "__main__":
    unittest.main()
