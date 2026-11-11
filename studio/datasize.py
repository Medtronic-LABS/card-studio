"""Is there enough labeled data? Three views of one card, cheapest first. Local only.

1. Evidence (free, from the latest training run). A field ships when the low end of the 95% interval of its held-out accuracy meets the
   target. A field can miss that for two different reasons, and they need different work:
     - it reads well but too few cells have been checked: at the accuracy seen so far, N more labeled cells would prove it;
     - it does not read well enough: checking more cells will not help, the reader has to get better (more varied data, or a change).
2. History (free). Every finished run of the card: how many cells it had and what held-out accuracy it reached. Shows whether adding labels
   has been paying off.
3. Learning curve (costs training time: `python datasize.py curve CARD`). Pages are set aside for testing; readers are trained on a
   quarter, half, three quarters and all of the remaining pages; each is scored on the pages set aside; this is repeated with different
   splits. A power law, error = a * cells^-b, is fitted to the mean error at each size and extended to say roughly how many labeled cells
   the target would take. Treat that number as an order of magnitude: it assumes more data of the same kind (same writers, same cards),
   and the fit rests on four points. If error is not falling as data is added, no number is given.

Curve results are in studio/curves/<card>/ (curve.json holds counts and accuracies only, no cell values).
"""
import json, math, random, statistics, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(ROOT / "cardpack")); sys.path.insert(0, str(HERE))
import cards, history

CURVES = HERE / "curves"
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
REPEATS, TEST_SHARE, SEED = 3, 0.2, 20261111
MIN_TRAIN_PAGES = 8          # the smallest subset still has to give training and validation pages
GROUPS = {"marks": ("mark",), "numbers, dates, bp": ("number", "date", "bp"), "text": ("text",)}
MAX_NEED = 200_000
FAR = 10                     # an extension beyond this many times the largest measured size is reported as out of reach, not as a plan


def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n)) / d - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)


def evidence(k, n, target):
    """What one field's held-out result (k right of n) says against a target for the interval's low end (a fraction).
    -> {verdict: ship | more-cells | not-accurate | no-data, need: total held-out cells that would prove it at this accuracy, or None}."""
    if n == 0: return {"verdict": "no-data", "need": None}
    if wilson_lo(k, n) >= target: return {"verdict": "ship", "need": n}
    p = k / n
    if p <= target: return {"verdict": "not-accurate", "need": None}
    lo, hi = n, MAX_NEED          # smallest N with the same share right whose interval clears the target (it only narrows as N grows)
    if wilson_lo(math.floor(p * hi), hi) < target: return {"verdict": "more-cells", "need": None}
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_lo(math.floor(p * mid), mid) >= target: hi = mid
        else: lo = mid + 1
    return {"verdict": "more-cells", "need": lo}


def fit_power(sizes, errors):
    """Least squares of log(error) on log(size). -> (a, b) for error = a * size^-b, or None when there are under three usable points."""
    pts = [(math.log(s), math.log(e)) for s, e in zip(sizes, errors) if s > 0 and e > 0]
    if len(pts) < 3: return None
    mx = statistics.mean(x for x, _ in pts); my = statistics.mean(y for _, y in pts); sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0: return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return math.exp(my - slope * mx), -slope


def extrapolate(sizes, errors, target_error):
    """-> {a, b, need, note}: labeled cells at which the fitted curve reaches target_error. need None with a note when it cannot be said."""
    if errors and min(errors) <= target_error:
        return {"a": None, "b": None, "need": int(round(min(s for s, e in zip(sizes, errors) if e <= target_error))), "note": "already reached at a size that was measured"}
    f = fit_power(sizes, errors)
    if f is None: return {"a": None, "b": None, "need": None, "note": "too few points to fit a curve"}
    a, b = f
    if b < 0.1: return {"a": a, "b": b, "need": None, "note": "error is not falling as cells are added: more labels of the same kind are unlikely to be enough"}
    need = (a / target_error) ** (1 / b)
    if need > MAX_NEED: return {"a": a, "b": b, "need": None, "note": f"the curve falls too slowly: over {MAX_NEED:,} cells by this fit"}
    need = int(math.ceil(need - 1e-6))
    if need > FAR * max(sizes):       # so far past what was measured that the number means "not this way", and is rounded so it is not read as precise
        mag = 10 ** (len(str(need)) - 2)
        return {"a": a, "b": b, "need": int(round(need / mag) * mag), "note": f"more than {FAR} times the largest size measured: read this as 'more labels of the same kind will not get there', the reader or the data has to change"}
    return {"a": a, "b": b, "need": need, "note": "a rough extension of four measured points, for more data of the same kind"}


def runs_of(card):
    """Finished training runs of this card, oldest first: cells, pages, held-out exact, fields shipped. Runs from before cards had ids are anc."""
    out = []
    for r in history.list_runs():
        res = history._load(history.RUNS / r["name"] / "results.json") or {}
        if r["state"] != "done" or (res.get("card") or cards.LEGACY) != card or r["exact"] is None: continue
        out.append({"run": r["name"], "when": r["when"], "cells": r["n_cells"], "pages": r["n_pages"], "exact": r["exact"], "lo": r["lo"], "hi": r["hi"], "shipped": r["shipped"], "fields": r["fields"]})
    return out[::-1]


def read_curve(card):
    d = CURVES / card; st = history._load(d / "status.json"); cv = history._load(d / "curve.json")
    if st and st.get("stage") not in ("done", "error") and time.time() - (d / "status.json").stat().st_mtime > 1800: st = {**st, "stage": "stalled"}
    return {"status": st, "curve": cv}


def summary(card, target=None):
    """Everything the Train tab shows about data sufficiency for one card."""
    if not cards.exists(card): raise KeyError("no such card")
    runs = runs_of(card); latest = history._load(history.RUNS / runs[-1]["run"] / "results.json") if runs else None
    target = float(target) if target else float((latest or {}).get("target", 85.0)); t = target / 100.0
    kinds = {f["key"]: f["kind"] for f in cards.fields(card)}; fields = []
    for f in (latest or {}).get("fields", []):
        e = evidence(f["k"], f["n"], t)
        fields.append({"field": f["field"], "kind": kinds.get(f["field"], "?"), "n": f["n"], "k": f["k"], "exact": f["exact"], "lo": f["lo"], **e,
                       "more": (e["need"] - f["n"]) if e["verdict"] == "more-cells" and e["need"] else None})
    for key in (latest or {}).get("unlabeled_fields", []):
        fields.append({"field": key, "kind": kinds.get(key, "?"), "n": 0, "k": 0, "exact": None, "lo": None, "verdict": "no-data", "need": None, "more": None})
    c = cards.describe(card)
    return {"card": card, "target": target, "latest_run": runs[-1]["run"] if runs else None, "labeled": c["labels"], "writers": cards.writer_status(card), "fields": fields, "history": runs,
            "counts": {v: sum(f["verdict"] == v for f in fields) for v in ("ship", "more-cells", "not-accurate", "no-data")},
            "more_cells_total": sum(f["more"] or 0 for f in fields), **read_curve(card)}


# ---------------------------------------------------------------- learning curve (needs torch)
def _job(a):
    card, merged, rep, frac, max_epochs, units = a
    import torch, common as C, train as T, trainer
    torch.set_num_threads(2); C.use_fields(cards.masks(card))
    rows = C.read_labels(merged); pages = sorted({r["page"] for r in rows})
    # the split is made over units: a writer's pages (when writers are known) or single pages; a unit is never on both sides
    uo = sorted({units[p] for p in pages}); random.Random(SEED + rep).shuffle(uo)
    n_test = max(2, round(len(uo) * TEST_SHARE)); tu, pu = uo[:n_test], uo[n_test:]
    test = {p for p in pages if units[p] in set(tu)}; use_units = pu[:max(MIN_TRAIN_PAGES, round(len(pu) * frac))]
    use = [p for p in pages if units[p] in set(use_units)]
    vu = set(random.Random(1000 + rep).sample(use_units, max(2, round(len(use_units) / 5)))); val = {p for p in use if units[p] in vu}
    tr = [r for r in rows if r["page"] in use and r["page"] not in val]; va = [r for r in rows if r["page"] in val]; te = [r for r in rows if r["page"] in test]
    assert not ({r["page"] for r in tr + va} & test)
    m, info = T.finetune(torch.load(ROOT / "work_phaseA" / "pretrained.pt"), tr, va, T.CONFIGS[trainer.CFG], seed=rep, max_epochs=max_epochs, log=lambda *_: None)
    hits = [(r["row"], p == r["label"]) for (p, _), r in zip(T.predict_torch(m, te), te)]
    return {"repeat": rep, "fraction": frac, "train_pages": len(use), "train_cells": len(tr) + len(va), "test_cells": len(te), "best_epoch": info["best_epoch"], "hits": hits}


def curve(card, target=85.0, max_epochs=300, workers=4):
    """Run the learning-curve experiment and write studio/curves/<card>/curve.json. Progress in status.json beside it."""
    import trainer
    d = CURVES / card; d.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def status(stage, detail="", **kw): (d / "status.json").write_text(json.dumps(dict(stage=stage, detail=detail, elapsed=round(time.time() - t0), card=card, **kw)))
    try:
        status("labels", "merging labels"); merged = d / "merged_labels.csv"; rows = trainer.merge_labels(merged, card)
        pages = sorted({r["page"] for r in rows})
        if len(pages) < cards.MIN_PAGES + 2: raise ValueError(f"{len(pages)} labeled pages: a learning curve needs at least {cards.MIN_PAGES + 2}")
        kinds = {f["key"]: f["kind"] for f in cards.fields(card)}; units, unit, n_units = cards.split_units(card, pages)
        if n_units < cards.MIN_PAGES + 2: raise ValueError(f"{n_units} {unit}(s): a learning curve needs at least {cards.MIN_PAGES + 2}")
        jobs = [(card, str(merged), r, f, max_epochs, units) for r in range(REPEATS) for f in FRACTIONS]
        status("training", f"0 of {len(jobs)} readers trained", done=0, total=len(jobs)); res = []
        with Pool(workers) as pool:
            for r in pool.imap_unordered(_job, jobs):
                res.append(r); status("training", f"{len(res)} of {len(jobs)} readers trained", done=len(res), total=len(jobs))
        merged.unlink(missing_ok=True)                    # the labels were only needed by the workers; do not keep a second copy of patient values
        groups = {"all fields": None, **GROUPS}; out = {}
        for g, ks in groups.items():
            pts = []
            for f in FRACTIONS:
                reps = [r for r in res if r["fraction"] == f]; accs = []
                for r in reps:
                    h = [ok for row, ok in r["hits"] if ks is None or kinds.get(row) in ks]
                    if h: accs.append(sum(h) / len(h))
                if not accs: continue
                n_g = statistics.mean(r["train_cells"] for r in reps) if ks is None else None
                pts.append({"fraction": f, "train_pages": round(statistics.mean(r["train_pages"] for r in reps), 1), "train_cells": round(statistics.mean(r["train_cells"] for r in reps), 1),
                            "exact": round(statistics.mean(accs), 4), "sd": round(statistics.pstdev(accs), 4), "min": round(min(accs), 4), "max": round(max(accs), 4), "repeats": len(accs),
                            "test_cells": round(statistics.mean(sum(1 for row, _ in r["hits"] if ks is None or kinds.get(row) in ks) for r in reps), 1)})
            if len(pts) >= 2:
                out[g] = {"points": pts, "fit": extrapolate([p["train_cells"] for p in pts], [1 - p["exact"] for p in pts], 1 - target / 100.0)}
        doc = {"card": card, "made": time.strftime("%Y-%m-%d %H:%M"), "target": target, "labeled_cells": len(rows), "labeled_pages": len(pages), "split_unit": unit, "split_units": n_units, "repeats": REPEATS, "fractions": list(FRACTIONS),
               "test_share": TEST_SHARE, "max_epochs": max_epochs, "minutes": round((time.time() - t0) / 60, 1), "groups": out,
               "notes": ["Sizes are all labeled cells in the training share (every field), so a group's curve says how its accuracy moves as the whole labeled set grows.",
                         "Each point is the mean over repeats with different pages set aside; with few test cells the spread between repeats is wide.",
                         "The target here is accuracy itself. Shipping needs the low end of the interval to clear the target, which takes somewhat more."]}
        (d / "curve.json").write_text(json.dumps(doc, indent=1)); status("done", "curve written")
        return doc
    except Exception:
        status("error", traceback.format_exc()[-1200:]); raise


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "curve":
        opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
        doc = curve(a[1], opt("--target", 85.0), opt("--max-epochs", 300))
        for g, v in doc["groups"].items(): print(g, [(p["train_cells"], p["exact"]) for p in v["points"]], v["fit"])
    elif len(a) == 2 and a[0] == "summary": print(json.dumps(summary(a[1]), indent=1))
    else: sys.exit("usage: python datasize.py summary CARD | curve CARD [--target 85] [--max-epochs 300]")
