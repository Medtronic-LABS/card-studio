"""ML-engineer view of a training run: error analysis, calibration, risk-coverage, confusions, model facts. numpy only (+ onnxruntime for latency).

Input is what the trainer already writes: runs/<name>/cv_predictions.json, one row per labeled cell, each predicted by a model that never
saw that cell's page (pooled 5-fold CV by page). So every number here is held-out. Cells on one page are correlated, so intervals are
narrower than the truth. Nothing here is written back to a run.
"""
import json, math, time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BINS = 10


def align_ops(ref, hyp):
    """Levenshtein alignment. Returns ops [('ok'|'sub'|'ins'|'del', ref_char, hyp_char)]; ties prefer substitutions (so a swap is one sub)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], "")); i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1])); j -= 1
    return ops[::-1]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def field_stats(rows, min_conf):
    """Per field: n, exact, CER, edit breakdown, confidence, flagged."""
    by = defaultdict(list)
    for r in rows: by[r["row"]].append(r)
    out = []
    for f, rs in by.items():
        sub = ins = dele = chars = hit = 0; confs = []; flagged = 0
        for r in rs:
            ops = align_ops(r["label"], r["pred"]); chars += len(r["label"]); hit += r["pred"] == r["label"]
            sub += sum(o[0] == "sub" for o in ops); ins += sum(o[0] == "ins" for o in ops); dele += sum(o[0] == "del" for o in ops)
            if r["pred"]:
                confs.append(r["conf"]); flagged += r["conf"] < (min_conf.get(f) or 0.0)
        lo, hi = wilson(hit, len(rs))
        out.append({"field": f, "n": len(rs), "exact": hit / len(rs), "lo": lo, "hi": hi, "cer": (sub + ins + dele) / max(chars, 1),
                    "sub": sub, "ins": ins, "del": dele, "chars": chars, "mean_conf": sum(confs) / len(confs) if confs else None,
                    "no_read": sum(1 for r in rs if not r["pred"]), "flagged": flagged})
    return sorted(out, key=lambda x: -x["n"])


def calibration(rows):
    """Reliability by confidence bin over cells that were read (a blank read has no confidence). ECE = size-weighted |accuracy - confidence|."""
    read = [r for r in rows if r["pred"]]
    bins = [{"lo": i / BINS, "hi": (i + 1) / BINS, "n": 0, "correct": 0, "conf_sum": 0.0} for i in range(BINS)]
    for r in read:
        b = bins[min(BINS - 1, int(r["conf"] * BINS))]; b["n"] += 1; b["correct"] += r["pred"] == r["label"]; b["conf_sum"] += r["conf"]
    ece = sum(b["n"] / len(read) * abs(b["correct"] / b["n"] - b["conf_sum"] / b["n"]) for b in bins if b["n"]) if read else 0.0
    return {"bins": [{"lo": b["lo"], "hi": b["hi"], "n": b["n"], "acc": b["correct"] / b["n"] if b["n"] else None,
                      "conf": b["conf_sum"] / b["n"] if b["n"] else None} for b in bins], "ece": ece, "read": len(read), "no_read": len(rows) - len(read)}


def risk_coverage(rows, min_conf):
    """Accept a read when confidence >= t. coverage = accepted / all cells; accuracy among accepted; silent errors = accepted but wrong, per all cells.
    A blank read (nothing emitted) is never accepted as a value here: the app does not flag blanks, so those are counted as silent misses
    separately in `policy`."""
    n = len(rows); curve = []
    for t in [i / 20 for i in range(0, 21)]:
        acc = [r for r in rows if r["pred"] and r["conf"] >= t]; ok = sum(r["pred"] == r["label"] for r in acc)
        curve.append({"t": t, "coverage": len(acc) / n, "accuracy": ok / len(acc) if acc else None, "silent_error_rate": (len(acc) - ok) / n})
    # the policy the app uses: a read is flagged when confidence < the field's minimum
    accepted = [r for r in rows if r["pred"] and r["conf"] >= (min_conf.get(r["row"]) or 0.0)]
    ok = sum(r["pred"] == r["label"] for r in accepted)
    blank_misses = sum(1 for r in rows if not r["pred"] and r["label"])
    policy = {"accepted": len(accepted), "coverage": len(accepted) / n, "accuracy": ok / len(accepted) if accepted else None,
              "silent_errors": len(accepted) - ok, "silent_error_rate": (len(accepted) - ok) / n, "blank_reads_of_filled_cells": blank_misses,
              "flagged_for_review": sum(1 for r in rows if r["pred"] and r["conf"] < (min_conf.get(r["row"]) or 0.0))}
    return {"curve": curve, "policy": policy}


def confusions(rows, top=12):
    sub, ins, dele = Counter(), Counter(), Counter()
    for r in rows:
        for op, a, b in align_ops(r["label"], r["pred"]):
            if op == "sub": sub[(a, b)] += 1
            elif op == "ins": ins[b] += 1
            elif op == "del": dele[a] += 1
    chars = sorted({c for (a, b) in sub for c in (a, b)} | set(ins) | set(dele))
    return {"top_substitutions": [{"true": a, "read": b, "n": n} for (a, b), n in sub.most_common(top)],
            "most_deleted": [{"char": c, "n": n} for c, n in dele.most_common(6)], "most_inserted": [{"char": c, "n": n} for c, n in ins.most_common(6)],
            "matrix": {"chars": chars, "sub": [[sub.get((a, b), 0) for b in chars] for a in chars]},
            "totals": {"sub": sum(sub.values()), "ins": sum(ins.values()), "del": sum(dele.values())}}


def positions(rows):
    """Where in the label the first mistake falls, for labels of 2+ characters. An even spread would give each of first / middle / last
    roughly its share of the positions; a pile-up on the first character points at the crop's left edge or the model's left padding."""
    wrong = first = middle = last = first_del = multi = 0
    for r in rows:
        if len(r["label"]) < 2: continue
        multi += 1
        if r["pred"] == r["label"]: continue
        wrong += 1; i = 0; at = None; kind = None
        for op, a, b in align_ops(r["label"], r["pred"]):
            if op != "ok" and at is None: at, kind = i, op
            if op != "ins": i += 1
        if at == 0: first += 1; first_del += kind == "del"
        elif at == len(r["label"]) - 1: last += 1
        else: middle += 1
    return {"multi_cells": multi, "wrong": wrong, "first": first, "middle": middle, "last": last, "first_deleted": first_del}


def label_stats(rows):
    cnt = Counter(c for r in rows for c in r["label"])
    return {"chars": [{"char": c, "n": n} for c, n in sorted(cnt.items(), key=lambda kv: -kv[1])], "cells": len(rows),
            "pages": len({r["page"] for r in rows}), "mean_len": sum(len(r["label"]) for r in rows) / max(len(rows), 1)}


def model_facts(pack_path):
    """Size, parameter count (if the onnx package is installed) and CPU latency of a pack's reader."""
    import numpy as np, onnxruntime as ort
    sys_path = str(HERE.parent / "cardpack")
    import sys
    if sys_path not in sys.path: sys.path.insert(0, sys_path)
    import cardpack as cp
    m = cp.read(pack_path); man = json.loads(m["manifest.json"]); r = man["reader"]
    out = {"onnx_bytes": len(m["reader.onnx"]), "input": [r["input"]["height"], r["input"]["width"]], "decode": r["decode"], "params": None}
    try:
        import onnx
        from onnx import numpy_helper
        out["params"] = int(sum(int(np.prod(numpy_helper.to_array(t).shape)) for t in onnx.load_from_string(m["reader.onnx"]).graph.initializer))
    except Exception:
        pass
    s = ort.InferenceSession(m["reader.onnx"], providers=["CPUExecutionProvider"]); x = np.random.RandomState(0).rand(1, 1, r["input"]["height"], r["input"]["width"]).astype(np.float32)
    for _ in range(5): s.run(None, {r["input"]["name"]: x})
    ts = []
    for _ in range(40):
        t0 = time.perf_counter(); s.run(None, {r["input"]["name"]: x}); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); out["latency_ms_median"] = round(ts[len(ts) // 2], 2); out["latency_ms_p95"] = round(ts[int(len(ts) * 0.95)], 2)
    return out


_cache = {}


def run_summary(d):
    """One line of quality numbers for a finished run folder (what the per-card-type run table shows)."""
    res = json.loads((d / "results.json").read_text()); rows = json.loads((d / "cv_predictions.json").read_text())
    import sys
    sys.path.insert(0, str(HERE.parent / "cardpack"))
    import cardpack as cp
    pack = Path(res["pack"]["path"]); min_conf = {}
    if pack.exists():
        min_conf = {f["key"]: f.get("min_confidence") for f in json.loads(cp.read(pack)["fields.json"])["fields"]}
    chars = sum(len(r["label"]) for r in rows); errs = sum(o[0] != "ok" for r in rows for o in align_ops(r["label"], r["pred"]))
    hit = sum(r["pred"] == r["label"] for r in rows); lo, hi = wilson(hit, len(rows)); pol = risk_coverage(rows, min_conf)["policy"]
    fl = res.get("fields", [])
    return {"name": d.name, "ts": (d / "results.json").stat().st_mtime, "when": time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "results.json").stat().st_mtime)),
            "pack_id": res["pack"]["id"], "pack": f"{res['pack']['id']} v{res['pack']['version']}", "config": res.get("config"), "minutes": res.get("minutes"),
            "n_cells": len(rows), "n_pages": res.get("n_pages"), "exact": hit / len(rows), "lo": lo, "hi": hi, "cer": errs / max(chars, 1),
            "ece": calibration(rows)["ece"], "silent_error_rate": pol["silent_error_rate"], "coverage": pol["coverage"], "policy_accuracy": pol["accuracy"],
            "shipped": sum(f.get("status") == "ship" for f in fl), "fields": len(fl)}


def overview():
    """Finished runs grouped by card type (pack id), oldest first inside a group. Runs belong to a pack, not to a Studio template."""
    groups = defaultdict(list)
    for d in sorted(RUNS.glob("*")):
        if (d / "results.json").exists() and (d / "cv_predictions.json").exists():
            try:
                r = run_summary(d); groups[r["pack_id"]].append(r)
            except (KeyError, ValueError, OSError):
                continue
    return {k: sorted(v, key=lambda r: r["ts"]) for k, v in groups.items()}


def run_report(name=None, pack_path=None):
    """Everything for the Model quality tab. name = a run folder under runs/ (default: the newest finished one)."""
    import sys
    sys.path.insert(0, str(HERE.parent / "cardpack"))
    runs = sorted([d for d in RUNS.glob("*") if (d / "results.json").exists() and (d / "cv_predictions.json").exists()], key=lambda d: (d / "results.json").stat().st_mtime)
    if not runs: raise KeyError("no finished run with predictions yet")
    d = next((x for x in runs if x.name == name), None) if name else runs[-1]
    if d is None: raise KeyError("no such run")
    res = json.loads((d / "results.json").read_text()); rows = json.loads((d / "cv_predictions.json").read_text())
    import cardpack as cp
    pack = Path(pack_path or res["pack"]["path"]); min_conf = {}
    if pack.exists():
        min_conf = {f["key"]: f.get("min_confidence") for f in json.loads(cp.read(pack)["fields.json"])["fields"]}
    key = (str(pack), pack.stat().st_mtime if pack.exists() else 0)
    if key not in _cache and pack.exists(): _cache[key] = model_facts(pack)
    all_ops = [o for r in rows for o in align_ops(r["label"], r["pred"])]; chars = sum(len(r["label"]) for r in rows)
    errs = sum(o[0] != "ok" for o in all_ops); hit = sum(r["pred"] == r["label"] for r in rows); lo, hi = wilson(hit, len(rows))
    return {"run": d.name, "config": res.get("config"), "pack": f"{res['pack']['id']} v{res['pack']['version']}", "minutes": res.get("minutes"),
            "seeds_epochs": {"folds_best_epoch": res.get("best_epochs"), "final_epochs": res.get("final_epochs")},
            "data": label_stats(rows), "overall": {"n": len(rows), "exact": hit / len(rows), "lo": lo, "hi": hi, "cer": errs / max(chars, 1),
                                                    "sub": sum(o[0] == "sub" for o in all_ops), "ins": sum(o[0] == "ins" for o in all_ops), "del": sum(o[0] == "del" for o in all_ops)},
            "fields": field_stats(rows, min_conf), "calibration": calibration(rows), "risk_coverage": risk_coverage(rows, min_conf),
            "confusions": confusions(rows), "positions": positions(rows), "model": _cache.get(key), "min_confidence": min_conf,
            "caveat": "Held-out: every cell was predicted by a model that never saw its page (pooled 5-fold CV). Cells on a page are correlated, so intervals are optimistic; page writers are unknown."}
