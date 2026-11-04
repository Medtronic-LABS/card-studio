"""Error browsing, paired run comparison, recalibration and policy data for the Model quality tab. numpy only; reads runs/<name>/ files, writes nothing.

Every number is held-out (pooled 5-fold CV by page). Cells on one page are correlated, so anything that resamples uses PAGES as the unit."""
import json, math
from pathlib import Path
import numpy as np
import mlstats as M

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ROOT = HERE.parent


def load(name):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name):
        raise KeyError("no such run")
    d = RUNS / name
    if not (d / "cv_predictions.json").exists():
        raise KeyError("no such run")
    rows = json.loads((d / "cv_predictions.json").read_text())
    folds = json.loads((d / "folds.json").read_text())["folds"] if (d / "folds.json").exists() else []
    fold_of = {p: i for i, f in enumerate(folds) for p in f}
    for r in rows:
        r["fold"] = fold_of.get(r["page"], -1); r["ok"] = r["pred"] == r["label"]
    return rows, d


def min_conf(d):
    import sys
    sys.path.insert(0, str(ROOT / "cardpack"))
    import cardpack as cp
    res = json.loads((d / "results.json").read_text()); p = Path(res["pack"]["path"])
    return {f["key"]: f.get("min_confidence") for f in json.loads(cp.read(p)["fields.json"])["fields"]} if p.exists() else {}


# ---------------------------------------------------------------- failures and label audit
KINDS = {
    "wrong": "all wrong reads",
    "silent": "wrong but accepted without review (the dangerous ones)",
    "confident_wrong": "label audit: wrong at confidence 0.8 or more (a label error or a systematic confusion)",
    "blank_read": "labeled cell read as blank",
    "flagged_right": "right, but flagged for review (needless work)",
}


def errors(name, field=None, kind="wrong", limit=200):
    rows, d = load(name); mc = min_conf(d)
    def accepted(r): return bool(r["pred"]) and r["conf"] >= (mc.get(r["row"]) or 0.0)
    pick = {"wrong": lambda r: not r["ok"], "silent": lambda r: not r["ok"] and accepted(r), "confident_wrong": lambda r: not r["ok"] and r["pred"] and r["conf"] >= 0.8,
            "blank_read": lambda r: not r["pred"] and r["label"], "flagged_right": lambda r: r["ok"] and r["pred"] and not accepted(r)}
    if kind not in pick:
        raise ValueError("unknown kind")
    sel = [r for r in rows if pick[kind](r) and (field is None or r["row"] == field)]
    sel.sort(key=lambda r: -r["conf"] if kind in ("silent", "confident_wrong") else r["conf"])
    fields = sorted({r["row"] for r in rows if pick[kind](r)})
    return {"kind": kind, "kinds": KINDS, "total": len(sel), "fields": fields,
            "items": [{"page": r["page"], "field": r["row"], "col": r["col"], "label": r["label"], "read": r["pred"], "conf": round(r["conf"], 3),
                       "ops": [{"op": o, "a": a, "b": b} for o, a, b in M.align_ops(r["label"], r["pred"])], "crop": f"/api/crop/{r['page']}/{r['row']}/{r['col']}.png"} for r in sel[:limit]]}


def crop_path(page, field, col):
    """Where a cell's crop lives: work/crops for scanned pages, studio/imports/crops for phone corrections."""
    for base in (ROOT / "work" / "crops", HERE / "imports" / "crops"):
        p = base / field / col / f"{page}.png"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- paired comparison
def mcnemar_exact(b01, b10):
    """Two-sided exact binomial test on the discordant cells (b01: only B right, b10: only A right)."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b, boot=2000, seed=0):
    ra, _ = load(a); rb, _ = load(b)
    ka = {(r["page"], r["row"], r["col"]): r for r in ra}; kb = {(r["page"], r["row"], r["col"]): r for r in rb}
    keys = sorted(set(ka) & set(kb))
    if len(keys) < 10:
        raise ValueError("the two runs share fewer than 10 labeled cells, so a paired comparison is not meaningful")
    A = np.array([ka[k]["ok"] for k in keys]); B = np.array([kb[k]["ok"] for k in keys]); pages = np.array([k[0] for k in keys])
    only_b, only_a = int((B & ~A).sum()), int((A & ~B).sum())
    up = sorted(set(pages)); idx = {p: np.where(pages == p)[0] for p in up}; rng = np.random.RandomState(seed); diffs = []
    for _ in range(boot):
        take = np.concatenate([idx[up[i]] for i in rng.randint(0, len(up), len(up))]); diffs.append(B[take].mean() - A[take].mean())
    byf = {}
    for i, k in enumerate(keys):
        d = byf.setdefault(k[1], {"n": 0, "only_a": 0, "only_b": 0, "both": 0, "neither": 0}); d["n"] += 1
        d["only_a" if A[i] and not B[i] else "only_b" if B[i] and not A[i] else "both" if A[i] else "neither"] += 1
    fields = [{"field": f, **d, "p": mcnemar_exact(d["only_b"], d["only_a"]), "delta": (d["only_b"] - d["only_a"]) / d["n"]} for f, d in sorted(byf.items(), key=lambda kv: -kv[1]["n"])]
    ca, cb = len(set(ka)) , len(set(kb))
    return {"a": a, "b": b, "common": len(keys), "cells_a": ca, "cells_b": cb, "pages": len(up), "acc_a": float(A.mean()), "acc_b": float(B.mean()),
            "diff": float(B.mean() - A.mean()), "only_a": only_a, "only_b": only_b, "both": int((A & B).sum()), "neither": int((~A & ~B).sum()),
            "p_mcnemar": mcnemar_exact(only_b, only_a), "ci_page_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "bootstrap": boot,
            "fields": fields,
            "notes": ["McNemar looks only at the cells where the runs disagree; it treats cells as independent, which they are not (cells on a page are correlated), so its p-value is optimistic.",
                      "The bootstrap resamples whole pages, which respects that correlation; read the interval, not the p-value.",
                      "Only cells labeled in both runs are compared." + ("" if ca == cb == len(keys) else f" Run A has {ca} cells, run B {cb}, {len(keys)} in common.")]}


# ---------------------------------------------------------------- recalibration (cross-fitted)
def pav(x, y):
    """Isotonic regression (pool adjacent violators). Returns block upper-x edges and block means."""
    order = np.argsort(x, kind="stable"); xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks = [[ys[i], 1.0, xs[i]] for i in range(len(xs))]      # mean, weight, max x
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2, x2 = out.pop(); m1, w1, x1 = out.pop(); out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, x2])
    return np.array([b[2] for b in out]), np.array([b[0] for b in out])


def apply_iso(edges, means, x):
    i = np.searchsorted(edges, x, side="left")
    return means[np.clip(i, 0, len(means) - 1)]


def recalibrate(name):
    """Fit isotonic confidence -> P(correct) on four page-folds, apply to the fifth. ECE and reliability after, against before."""
    rows, _ = load(name); read = [r for r in rows if r["pred"] and r["fold"] >= 0]
    if len(read) < 50:
        raise ValueError("not enough reads to recalibrate")
    x = np.array([r["conf"] for r in read]); y = np.array([r["ok"] for r in read], float); f = np.array([r["fold"] for r in read]); cal = np.zeros_like(x)
    for k in sorted(set(f)):
        tr = f != k
        if tr.sum() < 10 or (~tr).sum() == 0:
            cal[~tr] = x[~tr]; continue
        e, m = pav(x[tr], y[tr]); cal[~tr] = apply_iso(e, m, x[~tr])
    def bins(p):
        out = []
        for i in range(10):
            sel = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1.0)
            out.append({"lo": i / 10, "n": int(sel.sum()), "acc": float(y[sel].mean()) if sel.any() else None, "conf": float(p[sel].mean()) if sel.any() else None})
        return out
    def ece(p): return float(sum(b["n"] / len(p) * abs(b["acc"] - b["conf"]) for b in bins(p) if b["n"]))
    e, m = pav(x, y)
    return {"n": len(read), "ece_before": ece(x), "ece_after": ece(cal), "bins_before": bins(x), "bins_after": bins(cal),
            "curve": [{"conf": float(c), "p_correct": float(apply_iso(e, m, np.array([c]))[0])} for c in np.linspace(0, 1, 21)],
            "note": "Isotonic recalibration, cross-fitted: each page-fold is recalibrated by a map learned on the other four, so 'after' is a fair estimate, not a fit to the cells it scores."}


def policy_cells(name):
    """Compact per-cell data so the page can recompute coverage / accuracy / silent errors for any cutoff without a round trip."""
    rows, d = load(name); mc = min_conf(d); read = [r for r in rows if r["pred"] and r["fold"] >= 0]
    x = np.array([r["conf"] for r in read]); y = np.array([r["ok"] for r in read], float); f = np.array([r["fold"] for r in read]); cal = np.zeros_like(x)
    for k in sorted(set(f)):
        tr = f != k
        if tr.sum() >= 10 and (~tr).sum():
            e, m = pav(x[tr], y[tr]); cal[~tr] = apply_iso(e, m, x[~tr])
        else:
            cal[~tr] = x[~tr]
    calmap = {id(r): float(c) for r, c in zip(read, cal)}
    return {"n": len(rows), "min_confidence": {k: v for k, v in mc.items() if v is not None},
            "cells": [{"f": r["row"], "c": round(r["conf"], 4), "cc": round(calmap.get(id(r), r["conf"]), 4), "ok": int(r["ok"]), "r": int(bool(r["pred"]))} for r in rows]}
