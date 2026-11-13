"""Does reading a two-line date against the days each month has, and checking the two lines together, read EPI dates better? Measured on held-out cells.

A card's labeled cells are read by models that never saw their page (page-split 5-fold cross-validation, the way trainer.py does it, from the same starting
weights). For every held-out stacked date we keep the logits of both lines and read the cell four ways:

  greedy     the top line (day.month) and the year line each decoded greedily: what the phone does
  daymonth   the top line decoded against every valid day.month (grammar.py `daymonth`), the year line greedy
  + valid    either of the above, plus whether the two lines name a real calendar day (grammar.date_ok); an unreal date is flagged for a person
and, on the greedy reads, what the same-day check (cardlogic.py) would suggest in each dose column.

`python twolinefix.py collect [CARD]`   fine-tune per fold, keep held-out logits (capacity/twoline/<card>/logits.npz, cells.json). Needs torch.
`python twolinefix.py evaluate [CARD]`  score the four reads and the same-day check; write capacity/twoline/<card>/summary.json. Needs only numpy.

Labels are whatever the card's labels.csv holds. When every label is an `ai-read` (a suggestion accepted without a person typing it), the exact-match
numbers are agreement with the suggesting reader, not accuracy; evaluate() says so.
"""
import collections, csv, json, math, random, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
CARD = "epi_scan"


def out_dir(card): return HERE / "capacity" / "twoline" / card


def fold_job(a):
    fold, folds, merged, card, weights, max_epochs = a
    import torch, common as C, train as T, cards, trainer
    torch.set_num_threads(2); C.use_fields(cards.masks(card))
    rows = C.read_labels(merged); pre = torch.load(weights)
    test_pages = set(folds[fold]); rest = sorted({r["page"] for r in rows} - test_pages)
    vp = set(random.Random(1000 + fold).sample(rest, max(4, round(len(rest) / 7))))
    tr = [r for r in rows if r["page"] in rest and r["page"] not in vp]; va = [r for r in rows if r["page"] in vp]; te = [r for r in rows if r["page"] in test_pages]
    m, info = T.finetune(pre, tr, va, T.CONFIGS[trainer.CFG], seed=0, max_epochs=max_epochs, log=lambda *_: None); m.eval()
    with torch.no_grad(): lg = m(torch.from_numpy(np.stack([C.to_input(C.cell_image(r)) for r in te])).unsqueeze(1)).numpy()       # [T, B, V]
    return fold, [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"]) for r in te], lg.transpose(1, 0, 2).astype(np.float32), info["best_epoch"]


def collect(card=CARD, max_epochs=300):
    import cards, pretrained, trainer
    from multiprocessing import Pool
    t0 = time.time(); o = out_dir(card); o.mkdir(parents=True, exist_ok=True)
    weights, wrec = pretrained.resolve("auto"); merged = o / "merged_labels.csv"
    rows = trainer.merge_labels(merged, card); layouts = trainer.merge_labels.layouts
    if not layouts: raise SystemExit(f"{card} has no two-line fields")
    pages = sorted({r["page"] for r in rows}); units, unit, n_units = cards.split_units(card, pages); folds = trainer.make_folds(rows, units=units)
    cells, logits, bests = [], [], []
    with Pool(5) as pool:
        for fold, rs, lg, best in pool.imap_unordered(fold_job, [(f, folds, str(merged), card, str(weights), max_epochs) for f in range(5)]):
            cells += rs; logits.append(lg); bests.append(best)
    merged.unlink(missing_ok=True)
    np.savez_compressed(o / "logits.npz", logits=np.concatenate(logits).astype(np.float16))
    (o / "cells.json").write_text(json.dumps({"cells": cells, "layouts": layouts, "split_unit": unit, "split_units": n_units, "pretrained": wrec["name"], "best_epochs": sorted(bests)}))
    print(f"{len(cells)} held-out pieces, split by {unit} ({n_units}), in {(time.time() - t0) / 60:.1f} min -> {o}")


# ---------------------------------------------------------------- evaluation
def _wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _acc(k, n):
    lo, hi = _wilson(k, n); return {"n": n, "correct": k, "exact": round(k / max(1, n), 3), "lo": round(lo, 3), "hi": round(hi, 3)}


def _greedy(lg, idx, classes, blank):
    z = np.asarray(lg, np.float32)[:, idx]; best = z.argmax(1); b = idx.index(blank); out = []; prev = -1
    for k in best:
        if k != b and k != prev: out.append(classes[idx[k]])
        prev = k
    return "".join(out)


def evaluate(card=CARD):
    import cards, grammar, cardlogic
    o = out_dir(card); meta = json.loads((o / "cells.json").read_text()); cells = meta["cells"]; lg = np.load(o / "logits.npz")["logits"].astype(np.float32)
    m = cards.meta(); classes, blank = m["classes"], m["blank_idx"]; masks = cards.masks(card)
    by = {(c["page"], c["row"], c["col"]): i for i, c in enumerate(cells)}
    wholes = []        # (page, field, col, label) for every labeled stacked cell whose two pieces were held out
    for c in cells:
        if c["row"].endswith("#1") and (c["page"], c["row"][:-2] + "#2", c["col"]) in by:
            wholes.append((c["page"], c["row"][:-2], c["col"]))
    label_how = collections.Counter()
    with open(cards.CARDS / card / "labels.csv", newline="") as fh:
        for r in csv.DictReader(fh): label_how[r.get("how") or "typed"] += 1
    reads = {"greedy": {}, "daymonth": {}}; truth = {}
    for page, f, col in wholes:
        i1, i2 = by[page, f + "#1", col], by[page, f + "#2", col]; top_lab, yr_lab = cells[i1]["label"], cells[i2]["label"]
        truth[page, f, col] = cardlogic_join(top_lab, yr_lab)
        idx1, idx2 = sorted(masks[f + "#1"]), sorted(masks[f + "#2"]); chars1 = [classes[i] for i in idx1]
        yr = _greedy(lg[i2], idx2, classes, blank); g_top = _greedy(lg[i1], idx1, classes, blank)
        text, margin = grammar.decode(lg[i1][:, idx1], chars1, idx1.index(blank), "daymonth"); d_top = text or g_top
        for name, top in (("greedy", g_top), ("daymonth", d_top)):
            reads[name][page, f, col] = {"top": top, "year": yr, "text": cardlogic_join(top, yr), "valid": grammar.date_ok(top, yr)}
    keys = sorted(truth); n = len(keys)
    res = {"cells": n, "pages": len({k[0] for k in keys}), "split": f"{meta['split_unit']} ({meta['split_units']})", "pretrained": meta["pretrained"], "labels_by_how": dict(label_how),
           "labels_are_all_ai_read": set(label_how) == {"ai-read"}, "labels_that_are_real_dates": sum(cardlogic.date_key(truth[k]) is not None for k in keys)}
    ok = lambda t, k: cardlogic.canon(t) == cardlogic.canon(truth[k])          # noqa: E731
    for name in ("greedy", "daymonth"):
        rd = reads[name]
        res[name] = {"exact": _acc(sum(ok(rd[k]["text"], k) for k in keys), n), "top_line_exact": _acc(sum(rd[k]["top"].replace(".", "/") == cells[by[k[0], k[1] + "#1", k[2]]]["label"].replace(".", "/") for k in keys), n),
                     "a_real_date": _acc(sum(rd[k]["valid"] for k in keys), n), "an_impossible_top_line": sum(not _top_ok(rd[k]["top"]) for k in keys)}
        wrong = [k for k in keys if not ok(rd[k]["text"], k)]; flagged = [k for k in keys if not rd[k]["valid"]]
        res[name]["flag_unreal_date"] = {"flagged": len(flagged), "of_them_wrong": sum(k in set(wrong) for k in flagged), "wrong_reads": len(wrong), "wrong_reads_caught": sum(k in set(flagged) for k in wrong)}
    g, d = reads["greedy"], reads["daymonth"]
    res["daymonth_vs_greedy"] = {"only_daymonth_right": sum(ok(d[k]["text"], k) and not ok(g[k]["text"], k) for k in keys), "only_greedy_right": sum(ok(g[k]["text"], k) and not ok(d[k]["text"], k) for k in keys)}
    # same-day check on the greedy reads, per page and visit
    asked = cards.checks(card).get("same_day") or []
    if asked:
        for name in ("greedy", "daymonth"):
            texts = {(k[0], k[1], k[2]): reads[name][k]["text"] for k in keys}; found = []
            for page in sorted({k[0] for k in keys}):
                per = {(f, int(c[1:])): t for (p, f, c), t in texts.items() if p == page}
                for f in cardlogic.same_day(per, asked): found.append((page, f["field"], f"c{f['visit']}", f["suggestion"]))
            right = sum(cardlogic.canon(s) == cardlogic.canon(truth[(p, fl, c)]) for p, fl, c, s in found if (p, fl, c) in truth)
            fixes = sum(not ok(texts[(p, fl, c)], (p, fl, c)) and cardlogic.canon(s) == cardlogic.canon(truth[(p, fl, c)]) for p, fl, c, s in found if (p, fl, c) in truth)
            harm = sum(ok(texts[(p, fl, c)], (p, fl, c)) for p, fl, c, s in found if (p, fl, c) in truth)
            res.setdefault("same_day", {})[name] = {"suggestions": len(found), "suggestion_equals_label": right, "would_fix_a_wrong_read": fixes, "on_a_read_that_was_right": harm, "wrong_reads_in_these_fields": sum(not ok(texts[k], k) for k in texts if k[1] in asked)}
        agree = collections.Counter()
        for page in sorted({k[0] for k in keys}):
            for col in sorted({k[2] for k in keys if k[0] == page}):
                ts = [cardlogic.date_key(truth[(page, f, col)]) for f in asked if (page, f, col) in truth]; ts = [t for t in ts if t]
                if len(ts) >= 2: agree["columns_with_2_plus_labeled_dates"] += 1; agree["columns_where_the_labels_all_agree"] += len(set(ts)) == 1
        res["same_day_labels"] = dict(agree)
    (o / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def cardlogic_join(top, yr):
    return (top.replace(".", "/") + "/" + yr) if top and yr else ""


def _top_ok(top):
    import grammar
    return top in set(grammar.candidates("daymonth"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""; card = sys.argv[2] if len(sys.argv) > 2 else CARD
    if cmd == "collect": collect(card)
    elif cmd == "evaluate": evaluate(card)
    else: sys.exit("usage: python twolinefix.py collect|evaluate [CARD]")
