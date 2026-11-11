"""Run history, run-to-run comparison, logs and downloads for Card Studio. Local files only (studio/runs/<name>/).

A run folder holds status.json, results.json, cv_predictions.json, folds.json, merged_labels.csv, reader.onnx, reader.tflite (when made) and log.txt
(older hand-made runs keep it as runs_<name>.log next to runs/ or in studio/). Nothing here writes to a run.
"""
import csv, io, json, re, time, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNNAME = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
MAX_LOG = 200_000
FORMULA = re.compile(r"^(?:[=+@\t\r]|-[A-Za-z(@=+])")   # a bare '-' (a mark) or '-5' (a number) is data, not a formula


def _dir(name):
    if not RUNNAME.match(name) or not (RUNS / name).is_dir():
        raise KeyError("no such run")
    return RUNS / name


def _load(p, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def list_runs():
    """Newest first. Runs without status.json (e.g. the identification test folder) are skipped."""
    out = []
    for d in RUNS.glob("*") if RUNS.exists() else []:
        st = _load(d / "status.json")
        if not d.is_dir() or st is None:
            continue
        r = _load(d / "results.json") or {}
        fields = r.get("fields", [])
        cv = r.get("cv_all") or {}
        out.append({
            "name": d.name, "when": time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "status.json").stat().st_mtime)),
            "ts": (d / "status.json").stat().st_mtime, "state": st.get("stage", "?"), "detail": st.get("detail", ""),
            "config": r.get("config"), "target": r.get("target", st.get("target")), "min_n": r.get("min_n", st.get("min_n")),
            "n_cells": r.get("n_cells"), "n_pages": r.get("n_pages"), "minutes": r.get("minutes"),
            "exact": cv.get("exact"), "lo": cv.get("lo"), "hi": cv.get("hi"), "cer": cv.get("cer"),
            "shipped": sum(f.get("status") == "ship" for f in fields), "fields": len(fields),
            "pack": ({"id": r["pack"]["id"], "version": r["pack"]["version"]} if r.get("pack") else None),
            "has_log": log_path(d) is not None, "has_predictions": (d / "cv_predictions.json").exists(),
            "has_reader": (d / "reader.onnx").exists(), "has_tflite": (d / "reader.tflite").exists(),
        })
    return sorted(out, key=lambda x: -x["ts"])


def log_path(d):
    for p in (d / "log.txt", d.parent / f"runs_{d.name}.log", d.parent.parent / f"runs_{d.name}.log"):
        if p.exists():
            return p
    return None


def run_log(name, max_bytes=MAX_LOG):
    """Tail of the training log (the last max_bytes), plus whether it was cut."""
    p = log_path(_dir(name))
    if p is None:
        return {"text": "", "cut": False, "found": False}
    data = p.read_bytes()
    cut = len(data) > max_bytes
    return {"text": data[-max_bytes:].decode("utf-8", errors="replace"), "cut": cut, "found": True, "bytes": len(data)}


def compare(a, b):
    """Per-field difference between two runs (b minus a). Intervals are Wilson 95% from pooled page-level CV."""
    ra, rb = (_load(_dir(n) / "results.json") for n in (a, b))
    if ra is None or rb is None:
        raise ValueError("both runs need a results.json (finished runs only)")
    fa = {f["field"]: f for f in ra["fields"]}; fb = {f["field"]: f for f in rb["fields"]}
    rows = []
    for k in sorted(set(fa) | set(fb), key=lambda k: -(fb.get(k) or fa.get(k))["n"]):
        x, y = fa.get(k), fb.get(k)
        row = {"field": k, "a": x, "b": y}
        if x and y:
            row["delta"] = y["exact"] - x["exact"]
            row["delta_lo"] = y["lo"] - x["lo"]
            row["verdict"] = "better" if y["lo"] > x["hi"] else "worse" if y["hi"] < x["lo"] else "indistinguishable"
            row["status_change"] = None if x["status"] == y["status"] else f"{x['status']} -> {y['status']}"
        else:
            row.update(delta=None, delta_lo=None, verdict="only in " + ("b" if y else "a"), status_change=None)
        rows.append(row)
    caveats = []
    if ra.get("n_cells") != rb.get("n_cells") or ra.get("n_pages") != rb.get("n_pages"):
        caveats.append(f"different label sets: a has {ra.get('n_cells')} cells on {ra.get('n_pages')} pages, b has {rb.get('n_cells')} on {rb.get('n_pages')}. The held-out cells are not the same cells, so a difference may come from the data, not the model.")
    if (ra.get("target"), ra.get("min_n")) != (rb.get("target"), rb.get("min_n")):
        caveats.append(f"different gate settings (target {ra.get('target')}% / min {ra.get('min_n')} vs {rb.get('target')}% / {rb.get('min_n')}): ship decisions are not comparable.")
    if ra.get("config") != rb.get("config"):
        caveats.append(f"different training config ({ra.get('config')} vs {rb.get('config')}).")
    caveats.append("Cells on one page are correlated, so every interval is narrower than the truth: 'indistinguishable' is the safe reading unless the intervals are far apart.")
    cva, cvb = ra["cv_all"], rb["cv_all"]
    return {"a": a, "b": b, "rows": rows, "caveats": caveats,
            "overall": {"a": cva, "b": cvb, "delta": cvb["exact"] - cva["exact"],
                        "verdict": "better" if cvb["lo"] > cva["hi"] else "worse" if cvb["hi"] < cva["lo"] else "indistinguishable"},
            "shipped": {"a": sum(f["status"] == "ship" for f in ra["fields"]), "b": sum(f["status"] == "ship" for f in rb["fields"])}}


def _safe(v):
    v = "" if v is None else str(v)
    return "'" + v if FORMULA.match(v) else v


def predictions_csv(name):
    """Held-out (pooled CV) reads: one row per labeled cell. Contains patient values: handle as patient data."""
    preds = _load(_dir(name) / "cv_predictions.json")
    if preds is None:
        raise ValueError("this run has no predictions")
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["page", "field", "col", "label", "read", "confidence", "correct"])
    for p in preds:
        w.writerow([_safe(p["page"]), _safe(p["row"]), _safe(p["col"]), _safe(p["label"]), _safe(p["pred"]), f"{p['conf']:.4f}", int(p["pred"] == p["label"])])
    return buf.getvalue().encode("utf-8")


def run_file(name, which):
    """Bytes + download name for one run artifact."""
    d = _dir(name)
    files = {"reader.onnx": "application/octet-stream", "reader.tflite": "application/octet-stream", "results.json": "application/json", "folds.json": "application/json",
             "log.txt": "text/plain; charset=utf-8"}
    if which == "predictions.csv":
        return predictions_csv(name), "text/csv; charset=utf-8", f"{name}_predictions.csv"
    if which == "log.txt":
        p = log_path(d)
        if p is None: raise ValueError("this run has no log")
        return p.read_bytes(), files[which], f"{name}_log.txt"
    if which in files:
        if not (d / which).exists(): raise ValueError(f"this run has no {which}")
        return (d / which).read_bytes(), files[which], f"{name}_{which}"
    raise KeyError("unknown artifact")


def template_zip(templates_dir, name, version):
    """Template image, layout, grid (if saved) and meta for one template version, as a zip."""
    d = Path(templates_dir) / name / f"v{int(version)}"
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", name) or not (d / "template.png").exists():
        raise KeyError("no such template version")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("template.png", "layout.json", "grid.json", "meta.json"):
            if (d / f).exists():
                zi = zipfile.ZipInfo(f, (2026, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, (d / f).read_bytes())
    return buf.getvalue(), f"{name}_v{int(version)}_template.zip"
