"""Which confidence tells a right read from a wrong one? Measured on held-out cells (cross-validation logits kept by numfix.py and twolinefix.py).

A grammar read picks the likeliest string among everything a field can hold. How sure the app should be is a separate question, and three answers are compared here:
  margin     the current one: how far the winner's likelihood is above the reader's own best path (grammar.decode's margin; confidence = min(1, exp(margin)))
  top        the winner's share of the probability of ALL candidate strings (a posterior over the grammar)
  class      the share of the winning MEANING: every spelling of the same day and month (5/3, 05.03, 5.3, ...) is the same read, so their probabilities are added
             before dividing (the idea of the Bangladeshi prescription reader's "ingredient-set posterior")
For each, on the same cells: AUROC (does a higher confidence mean a right read?), accuracy of the most confident 80/60/40% of cells, and the share of cells that can be
accepted without a person at a wrong-accept rate of at most 2% and 5%. The cutoff is chosen on one half of the pages and the share and the wrong-accept rate measured on the
other half (and the halves swapped), so the numbers are not flattered by choosing the cutoff on the cells they are measured on.

    python confidence.py                   needs the logits of numfix.py collect and twolinefix.py collect (capacity/numfix, capacity/twoline/epi_scan)
"""
import collections, json, math, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
OUT = HERE / "capacity" / "confidence"


def softmax_shares(scores):
    """{cand: log-likelihood} -> {cand: share of the total probability over the candidates}"""
    m = max(scores.values()); z = {c: math.exp(v - m) for c, v in scores.items()}; s = sum(z.values())
    return {c: v / s for c, v in z.items()}


def date_class(text):
    """'5.3', '05/03', '5.3.26' -> (5, 3): the day and month a top line means; None when it is not one"""
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})", (text or "").strip())
    return (int(m[1]), int(m[2])) if m else None


def auroc(conf, ok):
    """probability that a random right read has a higher confidence than a random wrong one (ties count half)"""
    pos = [c for c, o in zip(conf, ok) if o]; neg = [c for c, o in zip(conf, ok) if not o]
    if not pos or not neg: return None
    allv = sorted(conf); rank = {}
    for i, v in enumerate(allv): rank.setdefault(v, []).append(i + 1)
    r = {v: sum(x) / len(x) for v, x in rank.items()}
    return (sum(r[c] for c in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def at_coverage(conf, ok, frac):
    order = sorted(range(len(conf)), key=lambda i: -conf[i]); k = max(1, int(len(order) * frac)); return sum(ok[i] for i in order[:k]) / k


def accepted_at_cap(conf, ok, pages, cap):
    """Choose the lowest cutoff whose wrong-accept rate on half A is <= cap, apply it to half B (and swap). -> (share of cells accepted, wrong-accept rate among them) pooled over both directions"""
    half = {p: i % 2 for i, p in enumerate(sorted(set(pages)))}; acc = wrong = 0
    for sel in (0, 1):
        A = [i for i in range(len(conf)) if half[pages[i]] == sel]; B = [i for i in range(len(conf)) if half[pages[i]] != sel]
        cuts = sorted(set(conf[i] for i in A), reverse=True); best = None
        for c in cuts:
            taken = [i for i in A if conf[i] >= c]
            if taken and sum(not ok[i] for i in taken) / len(taken) <= cap: best = c
            else: break
        if best is None: continue
        t = [i for i in B if conf[i] >= best]; acc += len(t); wrong += sum(not ok[i] for i in t)
    return (acc / len(conf), wrong / acc if acc else None)


def report(name, rows):
    """rows: [{margin, top, cls, ok, page}]"""
    res = {"cells": len(rows), "right": sum(r["ok"] for r in rows)}
    ok = [r["ok"] for r in rows]; pages = [r["page"] for r in rows]
    for m in ("margin", "top", "cls"):
        c = [r[m] for r in rows]
        res[m] = {"auroc": round(auroc(c, ok) or 0, 3), **{f"acc_top{int(f * 100)}": round(at_coverage(c, ok, f), 3) for f in (0.8, 0.6, 0.4)}}
        for cap in (0.02, 0.05):
            share, wa = accepted_at_cap(c, ok, pages, cap); res[m][f"accepted_at_{int(cap * 100)}pct_wrong"] = {"share": round(share, 3), "wrong_rate": None if wa is None else round(wa, 3)}
    return res


def dates():
    import cards, grammar
    d = HERE / "capacity" / "twoline" / "epi_scan"; meta = json.loads((d / "cells.json").read_text()); cells = meta["cells"]; lg = np.load(d / "logits.npz")["logits"].astype(np.float32)
    m = cards.meta(); classes, blank = m["classes"], m["blank_idx"]; masks = cards.masks("epi_scan"); cands = grammar.candidates("daymonth"); rows = []
    for i, c in enumerate(cells):
        if not c["row"].endswith("#1"): continue
        idx = sorted(masks[c["row"]]); chars = [classes[j] for j in idx]; b = idx.index(blank)
        logp = grammar.log_softmax(lg[i][:, idx]); sc = grammar.scores(logp, chars, b, cands)
        if not sc: continue
        want = date_class(c["label"])
        if want is None: continue
        best = max(sc, key=sc.get); sh = softmax_shares(sc)
        cls = collections.defaultdict(float)
        for k, v in sh.items(): cls[date_class(k)] += v
        top_cls = date_class(best); margin = min(1.0, math.exp(sc[best] - float(logp.max(1).sum())))
        rows.append({"margin": margin, "top": sh[best], "cls": cls[top_cls], "ok": top_cls == want, "page": c["page"]})
    return report("EPI top line (day.month)", rows)


def numbers():
    import cards, grammar, numfix
    d = numfix.OUT; cells = json.loads((d / "cells.json").read_text()); lg = np.load(d / "logits.npz")["logits"].astype(np.float32); m, masks, _ = numfix._setup(); rows = []
    for i, c in enumerate(cells):
        row = c["row"]
        if row not in numfix.RANGES: continue
        sc, bp = numfix._scores(grammar, m, masks, lg[i], row, numfix._cands(grammar, row, "both"))
        if not sc: continue
        best = max(sc, key=sc.get); sh = softmax_shares(sc); cls = collections.defaultdict(float)
        for k, v in sh.items(): cls[float(k)] += v
        rows.append({"margin": min(1.0, math.exp(sc[best] - bp)), "top": sh[best], "cls": cls[float(best)], "ok": best == c["label"], "page": c["page"]})
    return report("ANC numbers", rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); res = {}
    for name, f in (("epi_top_line_day_month", dates), ("anc_numbers", numbers)):
        try: res[name] = f()
        except FileNotFoundError as e: res[name] = {"error": f"missing logits: {e.filename}"}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
