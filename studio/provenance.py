"""What a training run needs recorded to be reproduced or questioned later. There is no git repo here, so code is identified by file hashes.
`collect` is called by the trainer; `describe` reads a run and says which parts were recorded then and which can only be recomputed now."""
import hashlib, json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ["studio/trainer.py", "studio/cleanup.py", "studio/normalize.py", "work_phaseA/train.py", "work_phaseA/common.py", "cardpack/cardpack.py"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def crops_digest(rows, imports_dir=None):
    """One hash over the exact crop files a run trained on: sha256 of sorted (key, file hash). Rows carry page/row/col and optionally `crop`."""
    parts, missing = [], 0
    for r in sorted(rows, key=lambda r: (r["page"], r["row"], r["col"])):
        p = Path(r["crop"]) if r.get("crop") else ROOT / "work" / "crops" / r["row"] / r["col"] / f"{r['page']}.png"
        if p.exists():
            parts.append(f"{r['page']}/{r['row']}/{r['col']}:{sha256_file(p)}")
        else:
            missing += 1
    return {"n": len(parts), "missing": missing, "sha256": hashlib.sha256("\n".join(parts).encode()).hexdigest()}


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "torch", "onnxruntime", "cv2"):
        try:
            mod = __import__(m); out[m] = getattr(mod, "__version__", "?")
        except Exception:
            out[m] = None
    return out


def code_hashes():
    return {f: (sha256_file(ROOT / f) if (ROOT / f).exists() else None) for f in CODE}


def collect(merged_csv, rows, seed, config, split_sha256, preprocessing):
    return {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "labels_sha256": sha256_file(merged_csv), "n_cells": len(rows),
            "crops": crops_digest(rows), "split_sha256": split_sha256, "seed": seed, "config": config, "preprocessing": preprocessing,
            "code_sha256": code_hashes(), "versions": versions(), "git": None, "git_note": "not a git repository: code is identified by file hashes"}


def describe(run_dir):
    """For the run view: each item with where it came from. 'recorded' = written by the trainer at run time; 'recomputed now' = rebuilt today
    from files that may have changed since, so it does not prove what the run used."""
    run_dir = Path(run_dir); res = json.loads((run_dir / "results.json").read_text()); rec = res.get("provenance")
    items = []
    def add(name, value, source): items.append({"name": name, "value": value, "source": source})
    if rec:
        add("labels file hash", rec["labels_sha256"][:16], "recorded"); add("crops hash", f"{rec['crops']['sha256'][:16]} ({rec['crops']['n']} files, {rec['crops']['missing']} missing)", "recorded")
        add("split hash", (rec["split_sha256"] or "")[:16], "recorded"); add("seed / config", f"{rec['seed']} / {rec['config']}", "recorded")
        add("preprocessing", rec["preprocessing"], "recorded"); add("recorded at", rec["recorded_at"], "recorded")
        now = code_hashes(); changed = [f for f, h in rec["code_sha256"].items() if now.get(f) != h]
        add("code", "unchanged since the run" if not changed else "changed since the run: " + ", ".join(changed), "recorded vs now")
        add("packages", ", ".join(f"{k} {v}" for k, v in rec["versions"].items() if v and k != "platform"), "recorded")
    else:
        m = run_dir / "merged_labels.csv"
        if m.exists(): add("labels file hash", sha256_file(m)[:16], "recorded (the merged labels file is kept in the run folder)")
        f = run_dir / "folds.json"
        if f.exists(): add("split hash", hashlib.sha256(f.read_bytes()).hexdigest()[:16], "recorded (folds.json is kept in the run folder)")
        add("seed / config", f"0 (fixed) / {res.get('config')}", "inferred from the trainer's constants")
        add("crops, code, packages, preprocessing", "not recorded by this run", "unknown: this run predates provenance records")
    return {"items": items, "recorded_by_trainer": bool(rec)}
