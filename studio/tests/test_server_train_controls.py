"""Stop / resume / curves / checklist routes of Card Studio, against a throwaway in-process server and a temporary runs directory."""
import http.client, json, sys, tempfile, threading, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class FakeProc:
    def __init__(self, alive=True): self.alive = alive
    def poll(self): return None if self.alive else 0


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.runs = Path(self.tmp.name)
        for p in (mock.patch.object(server, "RUNS", self.runs), mock.patch.object(server, "_proc", None), mock.patch.dict(server._cur, {"dir": None})):
            p.start(); self.addCleanup(p.stop)

    def call(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else b""
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True); c.putheader("Host", f"127.0.0.1:{self.port}")
        if method == "POST": c.putheader("X-Card-Studio", "1"); c.putheader("Content-Length", str(len(data)))
        c.endheaders(data); r = c.getresponse(); out = json.loads(r.read()); c.close()
        return r.status, out

    def run_dir(self, name="run-20260101-000000", stage="stopped", args=True, **extra):
        d = self.runs / name; d.mkdir(); (d / "status.json").write_text(json.dumps({"stage": stage, "detail": "x", "elapsed": 1, **extra}))
        if args: (d / "trainer_args.json").write_text("{}")
        return d

    def test_stop_with_nothing_running_is_400(self):
        st, b = self.call("POST", "/api/train/stop", {}); self.assertEqual(st, 400); self.assertIn("nothing to stop", b["error"])
        with mock.patch.object(server, "_proc", FakeProc(False)): self.assertEqual(self.call("POST", "/api/train/stop", {})[0], 400)

    def test_stop_writes_the_STOP_file_of_the_running_run(self):
        d = self.run_dir(stage="cv")
        with mock.patch.object(server, "_proc", FakeProc(True)), mock.patch.dict(server._cur, {"dir": d}):
            st, b = self.call("POST", "/api/train/stop", {})
        self.assertEqual((st, b["run"]), (200, d.name)); self.assertTrue((d / "STOP").exists())

    def test_resume_refuses_bad_names_and_missing_runs(self):
        for bad in ("", "../x", "run-1/../../x", "run1", "run-abc", 5):
            self.assertEqual(self.call("POST", "/api/train/resume", {"run": bad})[0], 400, bad)
        self.assertEqual(self.call("POST", "/api/train/resume", {"run": "run-1"})[0], 404)

    def test_resume_refusals(self):
        self.run_dir("run-1", args=False); self.run_dir("run-2", stage="done"); self.run_dir("run-3")
        st, b = self.call("POST", "/api/train/resume", {"run": "run-1"}); self.assertEqual(st, 400); self.assertIn("trainer_args", b["error"])
        st, b = self.call("POST", "/api/train/resume", {"run": "run-2"}); self.assertEqual(st, 400); self.assertIn("finished", b["error"])
        with mock.patch.object(server, "_proc", FakeProc(True)):
            st, b = self.call("POST", "/api/train/resume", {"run": "run-3"}); self.assertEqual(st, 400); self.assertIn("in progress", b["error"])

    def test_resume_starts_trainer_with_only_the_run_dir_and_resume(self):
        d = self.run_dir("run-3"); (d / "STOP").write_text("x")
        with mock.patch.object(server.subprocess, "Popen", return_value=FakeProc(True)) as P:
            st, b = self.call("POST", "/api/train/resume", {"run": "run-3"})
        self.assertEqual((st, b), (200, {"run": "run-3", "resumed": True}))
        cmd = P.call_args[0][0]; self.assertEqual(cmd[1:], [str(server.HERE / "trainer.py"), str(d), "--resume"])
        self.assertFalse((d / "STOP").exists()); self.assertEqual(json.loads((d / "status.json").read_text())["stage"], "labels")

    def test_status_exposes_stopped_and_resumable(self):
        self.run_dir("run-4", stage="stopped", resumable=True, fold_epochs={"0": 12})
        s = server.train_status(); self.assertEqual((s["state"], s["resumable"], s["run"]), ("stopped", True, "run-4")); self.assertEqual(s["fold_epochs"], {"0": 12})
        with mock.patch.object(server, "_proc", FakeProc(True)): self.assertFalse(server.train_status()["resumable"])

    def test_curves(self):
        d = self.run_dir("run-5", stage="cv"); (d / "curves").mkdir()
        (d / "curves" / "fold-0.json").write_text(json.dumps([{"epoch": 1, "loss": 2.0, "val_exact": 0.1, "val_cer": 0.9, "val_loss": 2.1, "lr": 0.001, "best": True}]))
        (d / "curves" / "fold-1.json").write_text("{not json"); (d / "curves" / "final.json").write_text(json.dumps([{"epoch": 1, "loss": 1.5, "lr": 0.001}]))
        st, b = self.call("GET", "/api/run/run-5/curves")
        self.assertEqual(st, 200); self.assertEqual(list(b["folds"]), ["0"]); self.assertEqual(b["folds"]["0"][0]["val_exact"], 0.1); self.assertEqual(b["final"][0]["loss"], 1.5)

    def test_curves_absent_and_bad_names(self):
        self.run_dir("run-6"); self.assertEqual(self.call("GET", "/api/run/run-6/curves"), (200, {"folds": {}, "final": []}))
        self.assertEqual(self.call("GET", "/api/run/run-nope/curves"), (200, {"folds": {}, "final": []}))
        for bad in ("..", "a.b", "x%2fy", "a%20b"): self.assertEqual(self.call("GET", f"/api/run/{bad}/curves")[0], 400, bad)

    def test_checklist_route(self):
        fake = {"checks": [{"id": "pages", "level": "ok", "message": "m", "hint": ""}], "ready": True}
        with mock.patch.object(server.datachecks, "checklist", return_value=fake), mock.patch.object(server.cards, "exists", lambda c: c == "anc"):
            self.assertEqual(self.call("GET", "/api/card/anc/checklist"), (200, fake))
            self.assertEqual(self.call("GET", "/api/card/nope/checklist")[0], 404)


if __name__ == "__main__":
    unittest.main()
