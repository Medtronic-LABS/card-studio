"""Pretrained models from outside, kept in a registry and usable here. Local only.

An outside recogniser (an ONNX CTC model from Hugging Face or anywhere) is imported with the facts its file does not carry (extreader.py: its characters, blank and
picture preparation) and then used as it is:
  benchmark   read every labeled single-line cell of a card with it and score the reads against the labels (exact match, character error rate, per field)
  suggest     write its reads as label suggestions for a card (never labels: a person still decides)
It cannot start a training run: our trainer fine-tunes our own network, and outside weights in another shape are of no use to it. Weights in our shape are the
other registry, pretrained.py.

Each model lives in studio/external/<name>/ (gitignored: licences differ and some forbid redistribution):
    model.onnx    the file as imported (its sha256 is checked on every use)
    model.json    spec (extreader.make_spec), source, licence, whether the licence allows commercial use (null = unknown), note, hash, when
    bench-<card>.json   the last benchmark on that card

    python external.py list
    python external.py add NAME MODEL.onnx --profile paddle-rec --charset inference.yml [--source URL] [--licence apache-2.0] [--commercial yes|no] [--note TEXT]
    python external.py benchmark NAME CARD        python external.py suggest NAME CARD        python external.py remove NAME
"""
import csv, hashlib, json, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(ROOT / "cardpack")); sys.path.insert(0, str(HERE))
import extreader as X

DIR = HERE / "external"
_readers = {}
HF_FILE_LIMIT = 400 * 1024 * 1024          # one model file larger than this is refused (a recogniser a phone could run is far smaller)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def _record(name):
    try: return json.loads((DIR / name / "model.json").read_text())
    except (OSError, ValueError): return None


def list_all():
    """[{name, profile, classes, source, licence, commercial_use, note, added, bytes, ok, problem, benchmarks: [{card, n, exact, cer, when}]}]"""
    out = []
    for d in sorted(DIR.iterdir()) if DIR.exists() else []:
        r = _record(d.name)
        if r is None or not (d / "model.onnx").exists(): continue
        problem = "" if r.get("sha256") == _sha(d / "model.onnx") else "the model file does not match its record"
        bm = []
        for b in sorted(d.glob("bench-*.json")):
            try: j = json.loads(b.read_text()); bm.append({"card": j["card"], "n": j["overall"]["n"], "exact": j["overall"]["exact"], "cer": j["overall"]["cer"], "when": j["when"]})
            except (OSError, ValueError, KeyError): pass
        out.append({"name": d.name, "profile": r["spec"]["profile"], "classes": len(r["spec"]["classes"]), "source": r.get("source", ""), "licence": r.get("licence", ""),
                    "commercial_use": r.get("commercial_use"), "note": r.get("note", ""), "added": r.get("added"), "bytes": (d / "model.onnx").stat().st_size, "ok": not problem,
                    "problem": problem, "benchmarks": bm})
    return out


def add(name, onnx_path, profile, charset_text, source="", licence="", commercial_use=None, note="", **override):
    """Register a model. Refuses a taken name, and a model that does not run with the spec (wrong class count, wrong channels). -> the record."""
    if not X.NAME.match(name or ""): raise ValueError("model name: letters, digits, _ or - (max 40)")
    if (DIR / name).exists(): raise ValueError(f"a model called {name} is already registered")
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file(): raise ValueError("no such model file")
    spec = X.make_spec(profile, X.parse_charset(charset_text), **override)
    X.ExternalReader(onnx_path, spec).check()
    DIR.mkdir(exist_ok=True); tmp = DIR / f".{name}.part"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    try:
        shutil.copyfile(onnx_path, tmp / "model.onnx")
        rec = {"spec": spec, "source": source, "licence": licence, "commercial_use": commercial_use, "note": note, "sha256": _sha(tmp / "model.onnx"), "added": time.strftime("%Y-%m-%d %H:%M")}
        (tmp / "model.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False)); tmp.rename(DIR / name)
    finally: shutil.rmtree(tmp, ignore_errors=True)
    return rec


def remove(name):
    if not X.NAME.match(name or "") or not (DIR / name).is_dir(): raise KeyError(f"no such model: {name}")
    shutil.rmtree(DIR / name); _readers.pop(name, None)


def reader(name):
    r = _record(name)
    if r is None or not (DIR / name / "model.onnx").exists(): raise KeyError(f"no such model: {name}")
    key = (name, r["sha256"])
    if key not in _readers: _readers[key] = X.ExternalReader(DIR / name / "model.onnx", r["spec"])
    return _readers[key]


# ---------------------------------------------------------------- Hugging Face
def hf_info(repo):
    """What Hugging Face says about a model repo, before anything is downloaded. -> {repo, licence, gated, files: [{name, bytes}], url}. Uses your saved login only to see gated repos."""
    from huggingface_hub import HfApi
    api = HfApi()
    try: m = api.model_info(repo, files_metadata=True)
    except Exception as e: raise ValueError(f"cannot read {repo} on Hugging Face: {str(e)[:160]}")
    lic = next((t.split(":", 1)[1] for t in (m.tags or []) if t.startswith("license:")), "")
    return {"repo": repo, "licence": lic, "gated": bool(m.gated), "url": f"https://huggingface.co/{repo}",
            "files": [{"name": s.rfilename, "bytes": s.size or 0} for s in (m.siblings or []) if s.rfilename.lower().endswith((".onnx", ".yml", ".yaml", ".json", ".txt"))]}


def hf_import(repo, name, onnx_file, charset_file, profile, accept_licence=False, commercial_use=None, note="", **override):
    """Download the model file and its charset file from a Hugging Face repo and register them. The licence is read from the repo and recorded, and the caller
    must say it was shown to a person (accept_licence). A gated repo needs the access your saved login has. -> the record."""
    if not accept_licence: raise ValueError("the licence must be accepted first: show it to the person and pass accept_licence")
    info = hf_info(repo); size = {f["name"]: f["bytes"] for f in info["files"]}
    for f in (onnx_file, charset_file):
        if f not in size: raise ValueError(f"{repo} has no file {f}")
    if size[onnx_file] > HF_FILE_LIMIT: raise ValueError(f"{onnx_file} is {size[onnx_file] // 1048576} MB: too large for a phone-sized recogniser (limit {HF_FILE_LIMIT // 1048576} MB)")
    from huggingface_hub import hf_hub_download
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            o = hf_hub_download(repo, onnx_file, local_dir=td); c = hf_hub_download(repo, charset_file, local_dir=td)
        except Exception as e: raise ValueError(f"download failed: {str(e)[:200]}")
        return add(name, o, profile, Path(c).read_text(encoding="utf-8"), source=info["url"], licence=info["licence"] or "not stated", commercial_use=commercial_use, note=note, **override)


# ---------------------------------------------------------------- use
def _cells(card):
    """Labeled single-line cells of a card with their crops. -> ([(field, label, crop path, page, col)], {two_line: n, text: n})"""
    import cards
    rows, _ = cards.labels(card); legacy = {f["key"]: f for f in cards.fields(card)}; skipped = {"two_line_cells": 0, "free_text_cells": 0}; out = []
    for r in rows:
        f = legacy.get(r["row"]) or {}
        if f.get("layout", "single") != "single": skipped["two_line_cells"] += 1; continue
        if f.get("kind") == "text": skipped["free_text_cells"] += 1; continue
        out.append((r["row"], r["label"].strip(), r["crop"] or str(cards.crops_dir(card) / r["row"] / r["col"] / f"{r['page']}.png"), r["page"], r["col"]))
    return out, skipped


def _gray(path):
    import cv2
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def benchmark(card, name, limit=None):
    """Read every labeled single-line cell of the card with the model, no fine-tuning, and score the reads against the labels. Two-line cells and free text are not read
    (counted in `skipped`). Saved to external/<name>/bench-<card>.json. -> {card, model, when, overall, by_field, skipped, cells}"""
    rd = reader(name); cells, skipped = _cells(card)
    if limit: cells = cells[:int(limit)]
    if not cells: raise ValueError(f"{card} has no labeled single-line cells to score ({skipped['two_line_cells']} two-line cells are not read by an outside model)")
    t0 = time.time(); rows = []
    for field, label, crop, page, col in cells:
        im = _gray(crop)
        if im is not None: rows.append((field, label, rd.read(im)[0]))
    res = {"card": card, "model": name, "when": time.strftime("%Y-%m-%d %H:%M"), **X.score(rows), "skipped": skipped, "cells": len(rows), "seconds": round(time.time() - t0, 1),
           "examples": [{"field": f, "label": l, "read": p} for f, l, p in rows[:12]]}
    (DIR / name / f"bench-{card}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def suggest(card, name):
    """Write the model's reads of a card's cells with ink as label suggestions (suggestions.csv, source `external:<name>`), like the card's own reader does. A suggestion is never a
    label. -> {cells, reader, seconds}"""
    import cards, labeling, suggest as S
    labeling._card(card); rd = reader(name); t0 = time.time(); kinds = {f["key"]: f for f in cards.fields(card)}; rows = []
    for (page, row, col), ink in labeling._cells(card).items():
        f = kinds.get(row) or {}
        if not ink or f.get("kind") == "text" or f.get("layout", "single") != "single": continue
        im = _gray(cards.crops_dir(card) / row / col / f"{page}.png")
        if im is None: continue
        text, conf = rd.read(im); rows.append({"page": page, "row": row, "col": col, "text": text, "confidence": round(conf, 4), "source": f"external:{name}"})
    d = cards.CARDS / card; d.mkdir(exist_ok=True); tmp = d / "suggestions.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, S.COLS); w.writeheader(); w.writerows(rows)
    tmp.replace(d / "suggestions.csv")
    return {"card": card, "reader": f"external:{name}", "cells": len(rows), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add"); a.add_argument("name"); a.add_argument("onnx"); a.add_argument("--profile", required=True, choices=list(X.PROFILES)); a.add_argument("--charset", required=True)
    a.add_argument("--source", default=""); a.add_argument("--licence", default=""); a.add_argument("--commercial", choices=["yes", "no"]); a.add_argument("--note", default="")
    for c in ("benchmark", "suggest"): s = sub.add_parser(c); s.add_argument("name"); s.add_argument("card")
    r = sub.add_parser("remove"); r.add_argument("name")
    o = ap.parse_args()
    if o.cmd == "list": print(json.dumps(list_all(), indent=1, ensure_ascii=False))
    elif o.cmd == "add": add(o.name, o.onnx, o.profile, Path(o.charset).read_text(encoding="utf-8"), o.source, o.licence, {"yes": True, "no": False}.get(o.commercial), o.note); print("added", o.name)
    elif o.cmd == "benchmark": print(json.dumps(benchmark(o.card, o.name), indent=1, ensure_ascii=False))
    elif o.cmd == "suggest": print(json.dumps(suggest(o.card, o.name), indent=1))
    else: remove(o.name); print("removed", o.name)
