"""Three ways to read the ANC card's numbers better, measured on held-out cells. Local only; `collect` needs torch, `evaluate` needs only numpy.

constrained.py showed that decoding a number against the values its field can hold lifted numbers from 62% to 66%. The mistakes left are mostly
valid-looking wrong values, and the decimals (glucose, hemoglobin) fail most. Three changes are measured here, on the same held-out logits:

  1 triage      a read the app is unsure about is sent to a person. Measured as a curve: share of cells the person must look at against the
                accuracy of the cells they do not. Uncertainty is the grammar margin (how far the winner beat the reader's own best path).
  2 shape       a field only holds one shape: glucose and hemoglobin are always one-decimal (6.4), the rest are whole numbers. Decoding against
                that shape alone, instead of against both, removes the readings that were never possible for the field.
  3 row         a card is read down a row of visits and weight, height, hemoglobin... move slowly between visits. Every cell keeps its few best
                candidates with their likelihood; a Viterbi pass along the visits adds a penalty for a jump between neighbouring reads.

`python numfix.py collect`   fine-tune per fold from the v2 pretraining (page-split 5-fold, as constrained.py), keep the held-out logits (capacity/numfix/logits.npz)
`python numfix.py evaluate`  score greedy, current grammar, 2, 3 and 2+3 on those logits and write capacity/numfix/summary.json
"""
import collections, json, math, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
OUT = HERE / "capacity" / "numfix"
RANGES = {"weight_kg": (30, 150), "pulse": (40, 180), "fundal_height_cm": (5, 45), "hemoglobin": (4, 18), "glucose": (2, 30), "gestational_month": (1, 10),
          "height_cm": (120, 200), "temp_f": (94, 106), "fetal_movement": (0, 100)}
DECIMAL = {"glucose", "hemoglobin"}              # the fields written with one decimal on the card (checked against the labels in evaluate())
TOPK = 12


def collect(weights=None):
    import random
    import cards, trainer, constrained
    from multiprocessing import Pool
    weights = str(weights or HERE / "capacity" / "v2" / "full" / "pre_v2.pt"); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    merged = OUT / "merged_labels.csv"; rows = trainer.merge_labels(merged, cards.LEGACY); folds = trainer.make_folds(rows)
    cells, logits = [], []
    with Pool(5) as pool:
        for fold, rs, lg, best in pool.imap_unordered(constrained.fold_job, [(f, folds, str(merged), weights, 300, 25) for f in range(5)]):
            cells += rs; logits.append(lg)
    merged.unlink(missing_ok=True)
    np.savez_compressed(OUT / "logits.npz", logits=np.concatenate(logits).astype(np.float16)); (OUT / "cells.json").write_text(json.dumps(cells))
    print(f"{len(cells)} held-out cells in {(time.time() - t0) / 60:.1f} min -> {OUT}")


# ---------------------------------------------------------------- evaluation
def _setup():
    import cards, grammar
    m = cards.meta(); masks = cards.masks(cards.LEGACY)
    return m, masks, grammar


def _scores(grammar, m, masks, lg, row, cands):
    idx = list(masks[row]); chars = [m["classes"][i] for i in idx]; blank = idx.index(m["blank_idx"])
    logp = grammar.log_softmax(np.asarray(lg, np.float32)[:, idx])
    return grammar.scores(logp, chars, blank, cands), float(logp.max(1).sum())


def _cands(grammar, row, shape):
    """shape 'both': whole numbers and one-decimal values; 'strict': only the shape the field really has."""
    lo, hi = RANGES[row]; allc = grammar.candidates("number", (lo, hi))
    if shape == "both": return allc
    return [c for c in allc if ("." in c) == (row in DECIMAL)]


def _greedy(m, masks, lg, row):
    idx = list(masks[row]); z = np.asarray(lg, np.float32)[:, idx]; best = z.argmax(1); blank = idx.index(m["blank_idx"]); out = []; prev = -1
    for b in best:
        if b != blank and b != prev: out.append(m["classes"][idx[b]])
        prev = b
    return "".join(out)


def _wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _acc(pairs):
    k = sum(a == b for a, b in pairs); n = len(pairs); lo, hi = _wilson(k, n)
    return {"n": n, "correct": k, "exact": round(k / max(1, n), 3), "lo": round(lo, 3), "hi": round(hi, 3)}


def _viterbi(per_visit, scale):
    """per_visit: list (in visit order) of [(candidate string, loglik), ...] or None for an empty visit. Chooses one candidate per visit maximising the
    sum of loglik minus |difference between neighbouring chosen values| / scale (a Laplace prior on the step). -> list of chosen strings."""
    vis = [i for i, c in enumerate(per_visit) if c]
    if not vis: return {}
    prev = [(s, ll, None) for s, ll in per_visit[vis[0]]]; back = []; tab = [prev]
    for a, b in zip(vis, vis[1:]):
        cur = []
        for s, ll in per_visit[b]:
            v = float(s); best = max(range(len(tab[-1])), key=lambda j: tab[-1][j][1] - abs(float(tab[-1][j][0]) - v) / scale)
            cur.append((s, ll + tab[-1][best][1] - abs(float(tab[-1][best][0]) - v) / scale, best))
        tab.append(cur)
    j = max(range(len(tab[-1])), key=lambda j: tab[-1][j][1]); chosen = {}
    for t in range(len(vis) - 1, -1, -1):
        chosen[vis[t]] = tab[t][j][0]; j = tab[t][j][2] if tab[t][j][2] is not None else 0
    return chosen


def evaluate():
    m, masks, grammar = _setup(); cells = json.loads((OUT / "cells.json").read_text()); lg = np.load(OUT / "logits.npz")["logits"].astype(np.float32)
    num = [i for i, c in enumerate(cells) if c["row"] in RANGES]
    shape_ok = {r: sum(("." in cells[i]["label"]) == (r in DECIMAL) for i in num if cells[i]["row"] == r) / max(1, sum(cells[i]["row"] == r for i in num)) for r in RANGES}
    res = {"cells": len(num), "labels_fit_assumed_shape": {r: round(v, 2) for r, v in shape_ok.items()}}
    cache = {}
    for i in num:
        row = cells[i]["row"]
        for shape in ("both", "strict"):
            sc, best_path = _scores(grammar, m, masks, lg[i], row, _cands(grammar, row, shape)); cache[i, shape] = (sc, best_path)
    def top(i, shape):
        sc, bp = cache[i, shape]; return sorted(sc.items(), key=lambda kv: -kv[1])[:TOPK], bp
    # reads
    reads = collections.defaultdict(dict)
    for i in num:
        row = cells[i]["row"]; reads["greedy"][i] = _greedy(m, masks, lg[i], row)
        for shape, name in (("both", "grammar (current)"), ("strict", "2 shape")):
            t, bp = top(i, shape); reads[name][i] = t[0][0] if t else ""; reads[name + " margin"][i] = (t[0][1] - bp) if t else -99
    # step scale per field from the labels (median absolute step between neighbouring visits)
    steps = collections.defaultdict(list); byrow = collections.defaultdict(list)
    for i in num: byrow[cells[i]["page"], cells[i]["row"]].append(i)
    for (page, row), ids in byrow.items():
        ids.sort(key=lambda i: int(cells[i]["col"][1:])); vals = [float(cells[i]["label"]) for i in ids if cells[i]["label"].replace(".", "").isdigit()]
        steps[row] += [abs(a - b) for a, b in zip(vals, vals[1:])]
    scale = {r: max(0.5, float(np.median(s)) if s else 1.0) for r, s in steps.items()}
    for shape, name in (("both", "3 row (current shape)"), ("strict", "2+3 shape and row")):
        for (page, row), ids in byrow.items():
            ids.sort(key=lambda i: int(cells[i]["col"][1:])); per = []
            for i in ids:
                t, bp = top(i, shape); per.append(t or None)
            chosen = _viterbi(per, scale.get(row, 1.0))
            for k, i in enumerate(ids): reads[name][i] = chosen.get(k, "")
    methods = ["greedy", "grammar (current)", "2 shape", "3 row (current shape)", "2+3 shape and row"]
    res["overall"] = {n: _acc([(reads[n][i], cells[i]["label"]) for i in num]) for n in methods}
    res["by_field"] = {r: {n: _acc([(reads[n][i], cells[i]["label"]) for i in num if cells[i]["row"] == r])["correct"] for n in methods} | {"n": sum(cells[i]["row"] == r for i in num)} for r in RANGES if any(cells[i]["row"] == r for i in num)}
    res["step_scale"] = {r: round(v, 2) for r, v in scale.items()}
    # 1 triage on the best method: order cells by margin, show accuracy of what is not sent to a person
    best = "2+3 shape and row"; mkey = "2 shape margin"
    order = sorted(num, key=lambda i: reads[mkey][i])            # least sure first
    curve = []
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        cut = int(len(order) * frac); keep = order[cut:]; curve.append({"sent_to_person": f"{int(frac * 100)}%", "n_read_by_app": len(keep), "app_exact": round(sum(reads[best][i] == cells[i]["label"] for i in keep) / max(1, len(keep)), 3)})
    res["triage_on_2+3_reads_by_margin_of_2"] = curve
    # how many wrong reads sit in the least-sure 30%
    wrong = [i for i in num if reads[best][i] != cells[i]["label"]]; low = set(order[:int(len(order) * 0.3)])
    res["wrong_reads_caught_by_asking_about_least_sure_30%"] = f"{sum(i in low for i in wrong)}/{len(wrong)}"
    (OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect": collect(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "evaluate": evaluate()
    else: sys.exit("usage: python numfix.py collect [WEIGHTS.pt] | evaluate")
