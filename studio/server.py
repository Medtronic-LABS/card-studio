"""Card Studio: layout/template workflow. Binds 127.0.0.1 only. Stdlib + OpenCV; no network calls.
Run: python server.py [--port 8770]"""
import argparse, json, re, subprocess, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import csv
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cardpack"))
import cardpack as cp
from detect import detect_layout, load_normalized, match_score
import gridconv, corrections, history, deskew, cleanup, mlstats, analysis, experiments, provenance, cards, scans, labeling, labelstudio, tflite_export, suggest, datasize, labruns, pretrained, external, extreader
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
TEMPLATES, UPLOADS = HERE / "templates", HERE / "uploads"
NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
VENDOR = {"/vendor/tabler.min.css", "/vendor/TABLER-LICENSE.txt", "/vendor/pico.min.css", "/vendor/PICO-LICENSE.md"}   # vendored, MIT; no CDN
HIGH, MID = 0.67, 0.25
MAX_BODY = 40 * 1024 * 1024      # a phone photo or PDF; anything bigger is refused before it is read
UPLOAD_TTL = 24 * 3600           # uploaded photos are patient data: deleted after a day


def request_problem(method, headers, port):
    """None if the request may proceed, else (status, message). The server listens on 127.0.0.1 only, but a web page
    open in the user's browser can still reach it (cross-site posts, DNS rebinding), so the origin is checked too."""
    ok_host = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if headers.get("Host", "") not in ok_host:
        return 421, "bad Host header (DNS rebinding guard)"
    if headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
        return 403, "cross-site request refused"
    if method == "POST":
        origin = headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in ok_host}:
            return 403, "cross-origin request refused"
        if headers.get("X-Card-Studio") != "1":
            return 403, "missing X-Card-Studio header"
        try:
            n = int(headers.get("Content-Length", "0"))
        except ValueError:
            return 400, "bad Content-Length"
        if n < 0 or n > MAX_BODY:
            return 413, f"body too large (limit {MAX_BODY // 2**20} MB)"
    return None


def purge_uploads(ttl=UPLOAD_TTL, now=None):
    """Delete uploaded photos older than ttl seconds. Returns how many were removed."""
    now = time.time() if now is None else now
    n = 0
    for f in UPLOADS.glob("*.png") if UPLOADS.exists() else []:
        if now - f.stat().st_mtime > ttl:
            f.unlink(); n += 1
    return n


def versions(name):
    d = TEMPLATES / name
    return sorted(int(p.name[1:]) for p in d.glob("v*") if p.name[1:].isdigit()) if d.exists() else []


def list_templates():
    out = []
    for d in sorted(TEMPLATES.glob("*")) if TEMPLATES.exists() else []:
        v = versions(d.name)
        if not v:
            continue
        meta = json.load(open(d / f"v{v[-1]}" / "meta.json"))
        gp = d / f"v{v[-1]}" / "grid.json"
        grid = None
        if gp.exists():
            g = json.loads(gp.read_text()); grid = {"rows": len(g["rows"]), "cols": len(g["col_lines_top"]) - 1}
        out.append({"name": d.name, "latest": v[-1], "versions": v, **meta, "grid": grid, "grid_opts": meta.get("grid") if gp.exists() else None})
    return out


PHASEA = HERE.parent / "work_phaseA"
RUNS = HERE / "runs"
_proc = None  # the one training run this server started
_curve = None  # the one learning-curve job this server started


def start_curve(card, target=85.0, start="auto"):
    """Start `datasize.py curve` for a card in the background. One heavy job at a time: not while a training run or another curve is going."""
    global _curve
    if (_proc is not None and _proc.poll() is None) or (_curve is not None and _curve.poll() is None):
        raise ValueError("a training run or a learning curve is already in progress")
    c = cards.describe(card)
    if not c["trainable"]: raise ValueError(f"{card} cannot be trained yet: {c['blocker']}")
    if not (50 <= float(target) <= 100): raise ValueError("target must be 50-100")
    pretrained.resolve(start)
    d = datasize.CURVES / card; d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"stage": "labels", "detail": "starting", "elapsed": 0, "card": card}))
    _curve = subprocess.Popen([sys.executable, str(HERE / "datasize.py"), "curve", card, "--target", str(float(target)), "--pretrained", start], stdout=open(d / "log.txt", "w"), stderr=subprocess.STDOUT, cwd=str(HERE))
    return {"card": card, "started": True}


def list_packs():
    out = []
    for p in sorted((HERE.parent / "cardpack" / "out").glob("*.cardpack")):
        try:
            m = cp.read(p); man = json.loads(m["manifest.json"]); fl = json.loads(m["fields.json"])["fields"]
            errs = cp.validate(p, check_reader=False)
            out.append({"file": p.name, "id": man["id"], "version": man["version"], "name": man["name"], "created": man["created"], "fields": len(fl),
                        "ship": sum(f["status"] == "ship" for f in fl), "review": sum(f["status"] != "ship" for f in fl),
                        "heldout": man["metrics"].get("heldout") is not None, "valid": not errs})
        except Exception as e:
            out.append({"file": p.name, "id": p.stem, "version": 0, "valid": False, "error": str(e)})
    return out


def labels_summary():
    work = HERE.parent / "work"; cells, new = {}, 0
    for f in ("labels.csv", "labels_new.csv"):
        fp = work / f
        if fp.exists():
            for r in csv.DictReader(open(fp, newline="")):
                if r["label"].strip():
                    cells[(r["page"], r["row"], r["col"])] = 1
                    if f == "labels_new.csv": new += 1
    return {"cells": len(cells), "pages": len({k[0] for k in cells}), "new": new}


def _grid_opts(get):
    """label_cols, data_cols, label_rows, data_rows, axis from a query or a JSON body (`get(key)` -> value or None)."""
    num = lambda k: int(get(k)) if get(k) not in (None, "") else None
    return {"label_cols": num("label_cols") or 0, "data_cols": num("data_cols"), "label_rows": num("label_rows") or 0, "data_rows": num("data_rows"), "axis": get("axis") or "rows"}


def template_grid(name, ver, label_cols=0, data_cols=None, label_rows=0, data_rows=None, axis="rows"):
    """Convert a saved Studio layout into a cardpack grid (see gridconv.py). Returns grid + warnings + errors."""
    f = TEMPLATES / name / f"v{int(ver)}" / "layout.json"
    if not NAME.match(name) or not f.exists():
        raise KeyError("no such template version")
    if not (0 <= label_cols <= 20) or (data_cols is not None and not (1 <= data_cols <= 60)):
        raise ValueError("label_cols 0-20, data_cols 1-60")
    if not (0 <= label_rows <= 20) or (data_rows is not None and not (1 <= data_rows <= 99)):
        raise ValueError("label_rows 0-20, data_rows 1-99")
    layout = json.loads(f.read_text())
    grid, warnings = gridconv.layout_to_grid(layout, label_cols=label_cols, data_cols=data_cols, label_rows=label_rows, data_rows=data_rows, axis=axis)
    return {"grid": grid, "warnings": warnings, "errors": gridconv.check_grid(grid), "image": layout["image"]}


PACKFILE = re.compile(r"^[A-Za-z0-9_.-]+\.cardpack$")


def pack_detail(name):
    p = HERE.parent / "cardpack" / "out" / name
    if not PACKFILE.match(name) or not p.exists():
        return None
    m = cp.read(p); man = json.loads(m["manifest.json"]); fl = json.loads(m["fields.json"])
    held = (man["metrics"].get("heldout") or {})
    return {"file": name, "manifest": {k: man[k] for k in ("id", "version", "name", "created", "template", "match", "reader", "metrics", "files", "preprocess")},
            "fields": fl["fields"], "grid": json.loads(m["grid.json"]), "heldout": held.get("per_field", {}),
            "errors": cp.validate(p)}


def latest_run():
    runs = [d for d in RUNS.glob("*") if (d / "status.json").exists()] if RUNS.exists() else []
    return max(runs, key=lambda d: (d / "status.json").stat().st_mtime) if runs else None


def train_status():
    d = latest_run()
    if d is None:
        return {"state": "none"}
    st = json.loads((d / "status.json").read_text())
    alive = _proc is not None and _proc.poll() is None
    stale = time.time() - (d / "status.json").stat().st_mtime > 900
    state = st["stage"] if st["stage"] in ("done", "error") else ("running" if alive or not stale else "stalled")
    out = {"state": state, "run": d.name, **st}
    if (d / "results.json").exists():
        out["results"] = json.loads((d / "results.json").read_text())
    return out


def start_training(req):
    global _proc
    if _proc is not None and _proc.poll() is None:
        raise ValueError("a training run is already in progress")
    target, min_n, card = float(req.get("target", 85)), int(req.get("min_n", 20)), str(req.get("card", cards.LEGACY))
    if not (50 <= target <= 100) or not (1 <= min_n <= 1000):
        raise ValueError("target must be 50-100 and min_n 1-1000")
    if not cards.exists(card):
        raise ValueError("no such card")
    c = cards.describe(card)
    if not c["trainable"]:
        raise ValueError(f"{card} cannot be trained yet: {c['blocker']}")
    start = str(req.get("pretrained", "auto"))
    pretrained.resolve(start)        # raises, saying why, when these starting weights cannot be used
    d = RUNS / time.strftime("run-%Y%m%d-%H%M%S"); d.mkdir(parents=True)
    json.dump({"stage": "labels", "detail": "starting", "elapsed": 0, "target": target, "min_n": min_n, "card": card}, open(d / "status.json", "w"))
    _proc = subprocess.Popen([sys.executable, str(HERE / "trainer.py"), str(d), "--card", card, "--target", str(target), "--min-n", str(min_n), "--pretrained", start],
                             stdout=open(d / "log.txt", "w"), stderr=subprocess.STDOUT, cwd=str(HERE))
    return {"run": d.name, "card": card}


MAX_EPOCHS = 300


def training_status():
    """Read-only view of the Phase A run: parses final.log, results.json. No patient data in these files."""
    log = PHASEA / "final.log"
    out = {"state": "none"}
    if not log.exists():
        return out
    last, pts = None, []
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"\s*ep (\d+) loss ([\d.]+) val_exact ([\d.]+) val_cer ([\d.]+) ([\d.]+)s/epoch", line)
        if m:
            last = m; pts.append({"epoch": int(m[1]), "loss": float(m[2]), "val_exact": float(m[3]), "val_cer": float(m[4])})
    out["history"] = pts; out["history_full"] = False
    if (PHASEA / "final_history.json").exists():
        out["history"] = [{"epoch": h["epoch"], "loss": h["loss"], "val_exact": h["val_exact"], "val_cer": h["val_cer"]}
                          for h in json.load(open(PHASEA / "final_history.json"))]
        out["history_full"] = True
    done = (PHASEA / "results.json").exists()
    age = time.time() - log.stat().st_mtime
    out["state"] = "done" if done else "running" if age < 600 else "stalled"
    if last:
        ep, sec = int(last[1]), float(last[5])
        out.update(epoch=ep, max_epochs=MAX_EPOCHS, loss=float(last[2]), val_exact=float(last[3]), val_cer=float(last[4]),
                   sec_per_epoch=sec, eta_sec_max=0 if done else round((MAX_EPOCHS - ep) * sec))
    if "Traceback" in log.read_text(errors="replace"):
        out["state"] = "error"
    if done:
        r = json.load(open(PHASEA / "results.json"))
        T = r["tables"]
        out["results"] = {"config": r["config"], "n_test": r["n_test"], "fields": [
            {"field": f, "n": d["n"], "exact": d["exact"], "lo": d["lo"], "hi": d["hi"], "cer": d["cer"],
             "cv_n": T["cv"].get(f, {}).get("n"), "cv_exact": T["cv"].get(f, {}).get("exact"),
             "cv_lo": T["cv"].get(f, {}).get("lo"), "cv_hi": T["cv"].get(f, {}).get("hi"),
             "cur_exact": T["cur_test"].get(f, {}).get("exact")} for f, d in T["new_test"].items()],
            "cv_all": T["cv"]["ALL"], "best_epoch": r["final"]["best_epoch"], "flags": r["flags"],
            "cv_info": r["cv_info"], "cutoffs": {"numbers": 0.7, "marks": 0.5}}
        pj = PHASEA / "predictions.json"
        if pj.exists():
            bins = [{"lo": i / 10, "correct": 0, "wrong": 0} for i in range(10)]
            for x in json.load(open(pj))["cv"]:
                if x["pred"] == "":
                    continue
                bins[min(9, int(x["conf"] * 10))]["correct" if x["pred"] == x["label"] else "wrong"] += 1
            out["results"]["conf_bins"] = bins
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json", headers=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _attach(self, body, ctype, filename):
        """A download. Names come from validated ids (letters, digits, _ - .), never from the request."""
        self._send(200, body, ctype, {"Content-Disposition": f'attachment; filename="{filename}"', "X-Content-Type-Options": "nosniff"})

    def _err(self, code, msg):
        print(f"{time.strftime('%H:%M:%S')} {self.command} {self.path.split('?')[0]} -> {code} {msg}", flush=True)   # no bodies, no query: nothing patient-related
        self._send(code, {"error": msg})

    def _guard(self, method):
        bad = request_problem(method, self.headers, self.server.server_address[1])
        if bad:
            self._err(*bad); return True
        return False

    def do_GET(self):
        if self._guard("GET"): return
        p = self.path.split("?")[0]
        if p == "/": return self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if p in VENDOR:
            return self._send(200, (HERE / "static" / p[1:]).read_bytes(), "text/css" if p.endswith(".css") else "text/plain")
        if p == "/idtest":
            f = RUNS / "idtest" / "report.html"
            if not f.exists(): return self._err(404, "no identification report yet: run python idtest.py build/score/report")
            return self._send(200, f.read_bytes().replace(b"../../static/vendor/pico.min.css", b"/vendor/pico.min.css"), "text/html; charset=utf-8")
        if p == "/api/idtest":
            f = RUNS / "idtest" / "summary.json"
            return self._send(200, json.loads(f.read_text()) if f.exists() else {"headline": []})
        if p == "/api/cleanup/packs":
            return self._send(200, [{"file": pk.path.name, "name": pk.name, "default": pk.default, "has_reference": pk.reference is not None} for pk in cleanup.installed()])
        m = re.fullmatch(r"/api/cleanup/([0-9a-f]{32})/([a-z_]+)\.png", p)
        if m:
            f = UPLOADS / f"{m[1]}_{m[2]}.png"
            return self._send(200, f.read_bytes(), "image/png") if f.exists() else self._err(404, "no such preview")
        if p == "/api/ml/overview":
            groups = mlstats.overview(); packs = {}
            for x in list_packs():
                if not x.get("error"): packs.setdefault(x["id"], []).append(x)
            def pack_name(v):
                newest = sorted(v, key=lambda x: -x["version"])[0]
                return json.loads(cp.read(HERE.parent / "cardpack" / "out" / newest["file"])["manifest.json"])["name"]
            card_types = [{"id": k, "name": pack_name(v), "versions": sorted(v, key=lambda x: -x["version"]), "runs": groups.get(k, [])} for k, v in sorted(packs.items())]
            templates = []
            for t in list_templates():
                linked = t["name"] if t["name"] in packs else None
                templates.append({"name": t["name"], "latest": t["latest"], "fields": t["fields"], "grid": t.get("grid"), "linked_pack": linked,
                                  "stages": {"layout": True, "grid": bool(t.get("grid")), "labels": 0, "run": bool(linked and groups.get(linked)), "pack": bool(linked)}})
            return self._send(200, {"card_types": card_types, "templates": templates})
        if p == "/api/ml":
            try: return self._send(200, mlstats.run_report((parse_qs(urlparse(self.path).query).get("run") or [None])[0]))
            except KeyError as e: return self._err(404, str(e).strip("'\""))
        if p.startswith("/api/crop/"):
            m = re.fullmatch(r"/api/crop/([A-Za-z0-9-]{1,30})/([A-Za-z0-9_]{1,40})/(c\d{1,2})\.png", p)
            f = analysis.crop_path(m[1], m[2], m[3]) if m else None
            return self._send(200, f.read_bytes(), "image/png", {"X-Content-Type-Options": "nosniff"}) if f else self._err(404, "no such crop")
        if p in ("/api/errors", "/api/compare/paired", "/api/ml/calibration", "/api/ml/policy", "/api/ml/provenance", "/api/ml/curves", "/api/experiments") or p.startswith("/api/experiments/"):
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                if p == "/api/errors": return self._send(200, analysis.errors(q.get("run", ""), q.get("field") or None, q.get("kind", "wrong"), min(int(q.get("limit", 200)), 500)))
                if p == "/api/compare/paired": return self._send(200, analysis.paired(q.get("a", ""), q.get("b", "")))
                if p == "/api/ml/calibration": return self._send(200, analysis.recalibrate(q.get("run", "")))
                if p == "/api/ml/policy": return self._send(200, analysis.policy_cells(q.get("run", "")))
                if p == "/api/ml/provenance":
                    analysis.load(q.get("run", "")); return self._send(200, provenance.describe(RUNS / q["run"]))
                if p == "/api/ml/curves":
                    analysis.load(q.get("run", "")); f = RUNS / q["run"] / "history.json"
                    return self._send(200, json.loads(f.read_text()) if f.exists() else {"folds": [], "note": "this run did not record learning curves (it predates them); new runs do"})
                if p == "/api/experiments": return self._send(200, experiments.list_all())
                return self._send(200, experiments.load(p.rsplit("/", 1)[1]))
            except KeyError as e: return self._err(404, str(e).strip("'\""))
            except ValueError as e: return self._err(400, str(e))
        if p == "/api/cards": return self._send(200, cards.list_cards())
        if p == "/api/jobs": return self._send(200, labruns.jobs())
        if p == "/api/pretrained": return self._send(200, pretrained.list_all())
        if p == "/api/external": return self._send(200, {"models": external.list_all(), "profiles": {k: v["label"] for k, v in extreader.PROFILES.items()}})
        if p == "/api/external/hf":
            repo = parse_qs(urlparse(self.path).query).get("repo", [""])[0]
            try: return self._send(200, external.hf_info(repo))
            except ValueError as e: return self._err(400, str(e))
        if p == "/api/scans": return self._send(200, scans.summaries())
        if p == "/api/labelstudio": return self._send(200, labelstudio.status(parse_qs(urlparse(self.path).query).get("card", [None])[0]))
        m = re.fullmatch(r"/api/card/([\w-]{1,40})/writers", p)
        if m:
            if not cards.exists(m[1]): return self._err(404, "no such card")
            w = cards.writers(m[1]); return self._send(200, {**cards.writer_status(m[1]), "by_page": w})
        m = re.fullmatch(r"/api/card/([\w-]{1,40})/datasize", p)
        if m:
            try: return self._send(200, datasize.summary(m[1], parse_qs(urlparse(self.path).query).get("target", [None])[0]))
            except KeyError as e: return self._err(404, str(e).strip("'\""))
            except ValueError as e: return self._err(400, str(e))
        m = re.fullmatch(r"/api/card/([\w-]{1,40})/(labeling|labeling/queue)", p)
        if m:
            try:
                if m[2] == "labeling": return self._send(200, labeling.overview(m[1]))
                q = parse_qs(urlparse(self.path).query); g = lambda k, d=None: q.get(k, [d])[0]
                return self._send(200, labeling.queue(m[1], g("field") or None, g("show", "todo"), int(g("limit", "60")), g("after"), g("order", "reading")))
            except KeyError as e: return self._err(404, str(e).strip("'\""))
            except ValueError as e: return self._err(400, str(e))
        m = re.fullmatch(r"/api/card/([\w-]{1,40})/crop/(\w{1,40})/(c\d{1,2})/([\w-]{1,40})\.png", p)
        if m:
            try: return self._send(200, labeling.crop_path(m[1], m[2], m[3], m[4]).read_bytes(), "image/png")
            except KeyError as e: return self._err(404, str(e).strip("'\""))
        m = re.fullmatch(r"/api/card/([\w-]{1,40})/scan/([\w-]{1,40})\.png", p)
        if m:
            try: return self._send(200, scans.overlay(m[1], m[2]), "image/png")
            except (KeyError, ValueError) as e: return self._err(404, str(e).strip("'\""))
        if p == "/api/runs": return self._send(200, history.list_runs())
        if p == "/api/runs/compare":
            q = parse_qs(urlparse(self.path).query)
            try: return self._send(200, history.compare(q.get("a", [""])[0], q.get("b", [""])[0]))
            except KeyError as e: return self._err(404, str(e).strip("'\""))
            except ValueError as e: return self._err(400, str(e))
        m = re.fullmatch(r"/api/run/([\w-]+)/(log|download/([\w.]+))", p)
        if m:
            try:
                if m[2] == "log": return self._send(200, history.run_log(m[1]))
                body, ctype, fn = history.run_file(m[1], m[3]); return self._attach(body, ctype, fn)
            except KeyError as e: return self._err(404, str(e).strip("'\""))
            except ValueError as e: return self._err(400, str(e))
        m = re.fullmatch(r"/api/pack/([\w.-]+)/reader\.tflite", p)
        if m:
            f = HERE.parent / "cardpack" / "out" / m[1]
            if not PACKFILE.match(m[1]) or not f.exists(): return self._err(404, "no such pack")
            blob = cp.read(f).get("reader.tflite")
            return self._attach(blob, "application/octet-stream", f"{f.stem}_reader.tflite") if blob else self._err(404, "this pack has no TFLite reader")
        m = re.fullmatch(r"/api/pack/([\w.-]+)/download", p)
        if m:
            f = HERE.parent / "cardpack" / "out" / m[1]
            if not PACKFILE.match(m[1]) or not f.exists(): return self._err(404, "no such pack")
            return self._attach(f.read_bytes(), "application/zip", m[1])
        m = re.fullmatch(r"/api/template/([\w-]+)/(\d+)/download", p)
        if m:
            try: body, fn = history.template_zip(TEMPLATES, m[1], m[2]); return self._attach(body, "application/zip", fn)
            except KeyError as e: return self._err(404, str(e).strip("'\""))
        if p == "/api/templates": return self._send(200, list_templates())
        if p == "/api/packs": return self._send(200, list_packs())
        m = re.fullmatch(r"/api/pack/([\w.-]+)/(template|detail)", p)
        if m:
            if m[2] == "detail":
                d = pack_detail(m[1])
                return self._send(200, d) if d else self._err(404, "no such pack")
            if not PACKFILE.match(m[1]) or not (HERE.parent / "cardpack" / "out" / m[1]).exists(): return self._err(404, "no such pack")
            return self._send(200, cp.read(HERE.parent / "cardpack" / "out" / m[1])["template.png"], "image/png")
        if p == "/api/labels/summary": return self._send(200, {**labels_summary(), "imported": corrections.summary()})
        if p == "/api/import/summary": return self._send(200, corrections.summary())
        if p == "/api/training": return self._send(200, training_status())
        if p == "/api/train/status": return self._send(200, train_status())
        m = re.fullmatch(r"/api/template/([\w-]+)/(\d+)/grid", p)
        if m:
            q = parse_qs(urlparse(self.path).query)
            try:
                return self._send(200, template_grid(m[1], m[2], **_grid_opts(lambda k: q.get(k, [None])[0])))
            except KeyError as e: return self._err(404, str(e).strip("'\""))
            except ValueError as e: return self._err(400, str(e))
        m = re.fullmatch(r"/api/template/([\w-]+)/(\d+)/(image|layout)", p)
        if m:
            f = TEMPLATES / m[1] / f"v{m[2]}"
            if not NAME.match(m[1]) or not f.exists(): return self._err(404, "no such template version")
            return self._send(200, (f / "template.png").read_bytes(), "image/png") if m[3] == "image" else self._send(200, (f / "layout.json").read_bytes())
        m = re.fullmatch(r"/api/upload/([0-9a-f]{32})/image", p)
        if m and (UPLOADS / f"{m[1]}.png").exists(): return self._send(200, (UPLOADS / f"{m[1]}.png").read_bytes(), "image/png")
        self._err(404, "not found")

    def do_POST(self):
        if self._guard("POST"): return
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        try:
            if self.path.split("?")[0] == "/api/analyze": return self._analyze(body, parse_qs(urlparse(self.path).query))
            if self.path == "/api/save": return self._save(json.loads(body))
            if self.path == "/api/train/start": return self._send(200, start_training(json.loads(body or b"{}")))
            if self.path.split("?")[0] == "/api/cleanup": return self._cleanup(body, parse_qs(urlparse(self.path).query))
            if self.path == "/api/template/grid": return self._save_grid(json.loads(body))
            if self.path == "/api/external/import":
                b = json.loads(body or b"{}")
                if not b.get("accept_licence"): raise ValueError("accept the licence first")
                keep = {k: b[k] for k in ("height", "channels", "max_width", "blank", "digit_map") if k in b}
                return self._send(200, external.hf_import(str(b.get("repo", "")), str(b.get("name", "")), str(b.get("onnx_file", "")), str(b.get("charset_file", "")), str(b.get("profile", "")),
                                                          True, {"yes": True, "no": False}.get(b.get("commercial")), str(b.get("note", ""))[:300], **keep))
            m = re.fullmatch(r"/api/external/([\w-]{1,40})/(benchmark|suggest|remove)", self.path.split("?")[0])
            if m:
                card = parse_qs(urlparse(self.path).query).get("card", [""])[0]
                if m[2] == "remove": external.remove(m[1]); return self._send(200, {"removed": m[1]})
                if not cards.exists(card): raise KeyError("no such card")
                return self._send(200, external.benchmark(card, m[1]) if m[2] == "benchmark" else external.suggest(card, m[1]))
            if self.path.split("?")[0] == "/api/scans":
                q = parse_qs(urlparse(self.path).query); card = q.get("card", [scans.AUTO])[0]
                if card != scans.AUTO and not cards.NAME.match(card): raise ValueError("bad card name")
                return self._send(200, {"results": scans.add(card, body, Path(q.get("name", ["scan"])[0]).name[:120], q.get("writer", [""])[0] or None)})
            m = re.fullmatch(r"/api/card/([\w-]{1,40})/datasize/curve", self.path)
            if m:
                if not cards.exists(m[1]): raise KeyError("no such card")
                b = json.loads(body or b"{}"); return self._send(200, start_curve(m[1], b.get("target", 85.0), str(b.get("pretrained", "auto"))))
            m = re.fullmatch(r"/api/card/([\w-]{1,40})/suggest", self.path)
            if m: return self._send(200, suggest.make(m[1]))
            m = re.fullmatch(r"/api/card/([\w-]{1,40})/writers", self.path)
            if m:
                if not cards.exists(m[1]): raise KeyError("no such card")
                r = json.loads(body or b"{}"); pages = list(r.get("pages") or [])
                if r.get("first") and r.get("last"): pages += cards.pages_between(m[1], str(r["first"]), str(r["last"]))
                n = cards.set_writers(m[1], str(r.get("writer", "")).strip(), sorted(set(pages)))
                return self._send(200, {"changed": n, **cards.writer_status(m[1])})
            m = re.fullmatch(r"/api/card/([\w-]{1,40})/(label|label/clear|field)", self.path)
            if m:
                r = json.loads(body or b"{}"); cell = (str(r.get("page", "")), str(r.get("field", "")), str(r.get("visit", "")))
                if m[2] == "label": return self._send(200, labeling.save(m[1], *cell, str(r.get("label", "")), str(r.get("note", "")), bool(r.get("confirm"))))
                if m[2] == "label/clear": return self._send(200, labeling.clear(m[1], *cell))
                return self._send(200, labeling.set_field(m[1], str(r.get("key", "")), str(r.get("kind", "")), r.get("range"), r.get("layout")))
            m = re.fullmatch(r"/api/pack/([\w.-]+)/tflite", self.path)
            if m:
                f = HERE.parent / "cardpack" / "out" / m[1]
                if not PACKFILE.match(m[1]) or not f.exists(): raise KeyError("no such pack")
                try: r = tflite_export.add_to_pack(f)
                except tflite_export.Unavailable as e: return self._err(503, str(e))
                return self._send(200, {"file": Path(r["pack"]).name, "version": r["version"], "errors": r["errors"], "tflite": r["tflite"]})
            m = re.fullmatch(r"/api/card/([\w-]{1,40})/labelstudio/(push|pull)", self.path)
            if m:
                try: return self._send(200, labelstudio.push(m[1]) if m[2] == "push" else labelstudio.pull(m[1]))
                except labelstudio.Unavailable as e: return self._err(503, str(e))
            m = re.fullmatch(r"/api/card/([\w-]{1,40})/(recut|scan/delete)", self.path)
            if m: return self._send(200, scans.recut(m[1]) if m[2] == "recut" else scans.remove(m[1], str(json.loads(body or b"{}").get("page", ""))))
            if self.path == "/api/import/corrections": return self._send(200, corrections.import_phone_zip(body))
            if self.path == "/api/import/label-studio": return self._send(200, corrections.import_label_studio(body))
        except KeyError as e:
            return self._err(404 if "no such" in str(e) else 400, str(e).strip("'\""))
        except (ValueError, OSError) as e:
            return self._err(400, str(e))
        except Exception as e:  # keep the server alive, tell the page
            return self._err(500, f"{type(e).__name__}: {e}")
        self._err(404, "not found")

    def _analyze(self, data, q=None):
        q = q or {}
        quarter = int(q.get("rotate", ["0"])[0])
        if quarter not in (0, 90, 180, 270): raise ValueError("rotate must be 0, 90, 180 or 270")
        bgr, orient = deskew.straighten(load_normalized(data), quarter, q.get("straighten", ["1"])[0] != "0")
        s = 2400 / max(bgr.shape[:2])
        if s < 1: bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        uid = uuid.uuid4().hex; UPLOADS.mkdir(exist_ok=True); purge_uploads()
        cv2.imwrite(str(UPLOADS / f"{uid}.png"), bgr)
        matches = []
        for t in list_templates():
            tb = cv2.imread(str(TEMPLATES / t["name"] / f"v{t['latest']}" / "template.png"))
            matches.append({"name": t["name"], "version": t["latest"], "score": round(match_score(bgr, tb), 3)})
        matches.sort(key=lambda m: -m["score"])
        best = matches[0]["score"] if matches else 0.0
        verdict = "known" if best >= HIGH else "similar" if best >= MID else "new"
        return self._send(200, {"upload_id": uid, "size": [bgr.shape[1], bgr.shape[0]], "matches": matches, "verdict": verdict,
                                "thresholds": {"high": HIGH, "mid": MID}, "layout": detect_layout(bgr), "orientation": orient})

    def _cleanup(self, data, q):
        """Clean-up lab: align one photo to a pack and read it with every photo clean-up option."""
        want = q.get("pack", ["auto"])[0]
        packs = cleanup.installed(); pack = None
        if want != "auto":
            if not PACKFILE.match(want): raise ValueError("bad pack name")
            pack = next((pk for pk in packs if pk.path.name == want), None)
            if pack is None: raise KeyError("no such pack")
        UPLOADS.mkdir(exist_ok=True); purge_uploads()
        self._send(200, cleanup.analyze(load_normalized(data), UPLOADS, pack=pack, packs=packs))

    def _save_grid(self, req):
        opts = _grid_opts(req.get); r = template_grid(req["name"], req["version"], **opts)
        if r["errors"]:
            raise ValueError("grid is not valid: " + "; ".join(r["errors"]))
        d = TEMPLATES / req["name"] / f"v{int(req['version'])}"
        (d / "grid.json").write_text(json.dumps(r["grid"], indent=1))
        meta = json.loads((d / "meta.json").read_text()); meta["grid"] = {"rows": len(r["grid"]["rows"]), "cols": len(r["grid"]["col_lines_top"]) - 1, **opts}
        (d / "meta.json").write_text(json.dumps(meta))
        self._send(200, {"saved": True, "rows": len(r["grid"]["rows"]), "cols": len(r["grid"]["col_lines_top"]) - 1, "warnings": r["warnings"]})

    def _save(self, req):
        name = req.get("name", "")
        if not NAME.match(name): raise ValueError("template name: letters, digits, _ or - (max 40)")
        fields = req.get("fields")
        if not isinstance(fields, list) or not fields: raise ValueError("no fields to save")
        for f in fields:
            if not all(isinstance(f.get(k), (int, float)) for k in "xywh") or f["w"] <= 0 or f["h"] <= 0: raise ValueError("bad field box")
            if not str(f.get("name", "")).strip(): raise ValueError("every field needs a name")
        if len({f["name"] for f in fields}) != len(fields): raise ValueError("field names must be unique")
        if req.get("upload_id"):
            if not re.fullmatch(r"[0-9a-f]{32}", req["upload_id"]): raise ValueError("bad upload id")
            img = (UPLOADS / f"{req['upload_id']}.png").read_bytes()
        elif req.get("base"):
            b = req["base"]; img = (TEMPLATES / b["name"] / f"v{int(b['version'])}" / "template.png").read_bytes()
        else: raise ValueError("need upload_id or base")
        v = (versions(name) or [0])[-1] + 1
        d = TEMPLATES / name / f"v{v}"; d.mkdir(parents=True)
        (d / "template.png").write_bytes(img)
        h, w = cv2.imdecode(__import__("numpy").frombuffer(img, "uint8"), 1).shape[:2]
        layout = {"image": [w, h], "fields": fields}
        (d / "layout.json").write_text(json.dumps(layout, indent=1))
        (d / "meta.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "fields": len(fields), "image": [w, h],
                                                 "base": req.get("base")}))
        self._send(200, {"name": name, "version": v})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8770); a = ap.parse_args()
    TEMPLATES.mkdir(exist_ok=True); UPLOADS.mkdir(exist_ok=True)
    print(f"Card Studio on http://127.0.0.1:{a.port} (local only); removed {purge_uploads()} old upload(s)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
