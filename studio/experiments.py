"""Experiment records: anything run outside the Studio trainer (ablations, retraining tests, sweeps) kept as a small JSON file with the
question, the data, the seeds and the numbers, so it can sit next to the training runs instead of living in a log file.
Files: studio/experiments/<id>.json (gitignored: they describe patient-derived pages). Nothing is invented: `import_*` only reads result files
the experiment scripts wrote, and says where each number came from."""
import json, math, re, statistics, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIR = HERE / "experiments"
ID = re.compile(r"^[A-Za-z0-9_-]{1,60}$")


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def seed_stats(seeds):
    """Mean and spread across seeds of the same treatment on the same cells. sd is the sample sd (n-1)."""
    acc = [s["k"] / s["n"] for s in seeds]
    return {"seeds": len(seeds), "mean": statistics.fmean(acc), "sd": statistics.stdev(acc) if len(acc) > 1 else None, "min": min(acc), "max": max(acc),
            "mean_cer": statistics.fmean(s["cer"] for s in seeds) if all(s.get("cer") is not None for s in seeds) else None}


def save(exp):
    if not ID.match(exp.get("id", "")): raise ValueError("bad experiment id")
    DIR.mkdir(exist_ok=True); exp = {**exp, "saved": time.strftime("%Y-%m-%d %H:%M")}
    (DIR / f"{exp['id']}.json").write_text(json.dumps(exp, indent=1)); return exp


def load(eid):
    if not ID.match(eid) or not (DIR / f"{eid}.json").exists(): raise KeyError("no such experiment")
    exp = json.loads((DIR / f"{eid}.json").read_text())
    for t in exp.get("treatments", []):
        if t.get("seeds"): t["stats"] = seed_stats(t["seeds"])
    for b in exp.get("baselines", []):
        if b.get("n"): b["exact"] = b["k"] / b["n"]; b["lo"], b["hi"] = wilson(b["k"], b["n"])
    return exp


def list_all():
    out = []
    for p in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        try: e = load(p.stem)
        except (KeyError, ValueError): continue
        out.append({"id": e["id"], "title": e["title"], "kind": e.get("kind"), "saved": e.get("saved"), "question": e.get("question"),
                    "treatments": len(e.get("treatments", [])), "has_table": bool(e.get("table"))})
    return sorted(out, key=lambda x: x["saved"] or "", reverse=True)


# ---------------------------------------------------------------- importers: read what the experiment scripts already wrote
def import_apppath(results=ROOT / "work_phaseA" / "apppath_results.json"):
    """work_phaseA/apppath_train.py -> one record per page clean-up variant."""
    if not Path(results).exists(): return []
    r = json.loads(Path(results).read_text()); out = []
    for v, res in r["variants"].items():
        n = r["n"]["test"]; ref = r["phaseA_stored_crops"]
        out.append(save({
            "id": f"apppath-{v}", "kind": "retrain", "title": f"Retrain the reader on the app's preprocessing ({v.replace('_', ' ')} + line removal)",
            "question": "The app feeds the reader line-erased, cleaned-up crops; it was trained on stored crops. Does fine-tuning on what the app feeds it close the gap on unseen pages?",
            "data": {"eval": f"{n} held-out test pages cells (7 pages), read through the app path", "split": "work_phaseA/split.json (train 35 / val 7 / test 7 pages)",
                     "ink_gate_blanked_labeled_cells": res["gated"]},
            "recipe": f"same pretrained weights and recipe as Phase A ({r['cfg']}), fine-tuned on train pages only, early stopping on the val pages; TEST read once per model",
            "baselines": [{"name": "Phase A reader on stored crops (how it was trained)", "k": ref["k"], "n": ref["n"], "cer": ref["cer"]},
                          {"name": "Phase A reader through the app's preprocessing", "k": res["baseline_app_path"]["k"], "n": n, "cer": res["baseline_app_path"]["cer"],
                           "note": "test cells the ink gate blanked count as misses"}],
            "treatments": [{"name": "retrained on app-path crops", "seeds": [{"seed": s["seed"], "k": s["k"], "n": s["test_total"], "cer": s["cer"], "best_epoch": s["best_epoch"]} for s in res["retrained"]]}],
            "notes": ["62 cells on 7 pages: a one-cell difference is 1.6 points, so differences of a few points between seeds or variants are noise.",
                      "Results come from work_phaseA/apppath_results.json written by apppath_train.py."],
            "source_files": ["work_phaseA/apppath_train.py", "work_phaseA/apppath_results.json", "work_phaseA/apppath_train.log"]}))
    return out


def import_preproc(val=ROOT / "work_phaseA" / "preproc_val.json", logs=(ROOT / "work_phaseA" / "preproc_test_reinhard.log", ROOT / "work_phaseA" / "preproc_test_stretch.log")):
    """work_phaseA/preproc_test.py -> one record: the validation sweep table plus the two single test confirmations."""
    if not Path(val).exists(): return []
    v = json.loads(Path(val).read_text()); conds = ["scan", "cast", "shadow", "lowcontrast", "combined"]; variants = []
    for key in v:
        name = key.split("|")[0]
        if name not in variants: variants.append(name)
    rows = [{"variant": n, **{c: {"acc": v[f"{n}|{c}"][0], "lo": v[f"{n}|{c}"][1], "hi": v[f"{n}|{c}"][2]} for c in conds}} for n in variants]
    tests = []
    for lg in logs:
        if not Path(lg).exists(): continue
        txt = Path(lg).read_text(); m = re.search(r"baseline vs '([^']+)' \((\d+) cells", txt)
        if not m: continue
        pairs = re.findall(r"^\s+(\w+)\s+baseline\s+([\d.]+) \[(\d+)-(\d+)\]\s+.+?\s+([\d.]+) \[(\d+)-(\d+)\]\s+diff\s+([+-][\d.]+) pp", txt, re.M)
        tests.append({"variant": m[1], "n": int(m[2]), "conditions": {p[0]: {"baseline": float(p[1]), "treated": float(p[4]), "diff_pp": float(p[7])} for p in pairs}})
    return [save({
        "id": "preproc-sweep", "kind": "ablation", "title": "Photo clean-up options on simulated bad photos",
        "question": "Which page/cell normalization keeps a fixed reader accurate when the photo is shadowed, washed out or colour-cast?",
        "data": {"eval": "validation sweep: 57 cells on 7 validation pages; confirmation of two finalists: 62 cells on 7 test pages, read once",
                 "conditions": "scan, warm cast, shadow, low contrast, all combined (simulated, 3 random draws each except scan)"},
        "recipe": "fixed model (final_model.pt), colour aligned scans degraded, variant applied to the page, cells cut with the recorded boxes",
        "table": {"title": "Validation, exact match (%)", "columns": conds, "rows": [{"label": r["variant"], "cells": [{"v": 100 * r[c]["acc"], "lo": 100 * r[c]["lo"], "hi": 100 * r[c]["hi"]} for c in conds]} for r in rows]},
        "test_confirmations": tests,
        "baselines": [], "treatments": [],
        "notes": ["Degradations are simulated, not real phone photos. The reader was not retrained with any option.",
                  "'Seeds' here are degradation draws on the same model, not training seeds."],
        "source_files": ["work_phaseA/preproc_test.py", "work_phaseA/preproc_val.json", "work_phaseA/preproc_test_reinhard.log", "work_phaseA/preproc_test_stretch.log"]})]


if __name__ == "__main__":
    import sys
    done = import_apppath() + import_preproc()
    print("saved:", [e["id"] for e in done])
