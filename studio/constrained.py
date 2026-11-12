"""Reading a field against what it can be: decoding with a grammar. Local only; needs torch.

Greedy CTC decoding takes the best class at every step, so a date can come out as `/.6` or a weight as `.8`: nothing says a date has a day, a month
and a year. Here the reader's own output is scored against every value the field can hold, and the best-scoring one wins:
  date     day 1-31, month 1-12 (each with or without a leading zero), a two-digit year from 20 to 27, `/` or `.` between (about 13,000 strings)
  bp       systolic 70-220 over diastolic 40-130
  number   whole numbers in the field's range, and one-decimal values (about 1,000 to 15,000 strings, from cards.py ranges)
The score of a string is the CTC log-likelihood of that exact string under the reader's masked output, the same quantity training maximizes. Nothing
is retrained: this changes only how the output is read.

A value outside the grammar (a weight of 160, a year of 2031) can never be read, so the grammar is a bet that the card's rules hold. `margin` says how
far the winning string beat the best the reader would have said unconstrained; a person can be asked about a read with a small margin.

`python constrained.py run [--pretrained PATH]` measures it on the ANC card: page-split 5-fold cross-validation as capacity.py, with each fold's test
logits kept, greedy against constrained on the same cells, results in studio/capacity/constrained/.
"""
import argparse, json, random, sys, time, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
import cards

OUT = HERE / "capacity" / "constrained"
YEARS = (20, 27)          # default two-digit years a date may carry; a pack field can say otherwise (`date.years`)
_classes = {}


def _index(char):
    """Class index of a character in the shared class list (no torch needed)."""
    if not _classes:
        m = cards.meta(); _classes.update({c: i for i, c in enumerate(m["classes"])}); _classes["<blank>"] = m["blank_idx"]
    return _classes[char]


def encode(label): return [_index(c) for c in label]
RANGES = {"weight_kg": (30, 150), "pulse": (40, 180), "fundal_height_cm": (5, 45), "hemoglobin": (4, 18), "glucose": (2, 30), "gestational_month": (1, 10),
          "height_cm": (120, 200), "temp_f": (94, 106), "fetal_movement": (0, 100)}      # the ANC card's ranges (cardpack/export_anc.py)
_cache = {}


def candidates(kind, rng=None, years=None):
    """Every string the grammar allows for this kind of field (rng: a number's [low, high]; years: a date's [first, last] two-digit year). Cached."""
    import grammar
    return grammar.candidates(kind, rng, years)


def _candidates_unsorted(kind, rng=None, years=None):
    """The original enumeration, kept so a test can check grammar.candidates holds the same strings."""
    years = tuple(years) if years else YEARS; key = (kind, tuple(rng) if rng else None, years if kind == "date" else None)
    if key in _cache: return _cache[key]
    if kind == "date":
        days = [str(d) for d in range(1, 32)] + [f"{d:02d}" for d in range(1, 10)]; mons = [str(m) for m in range(1, 13)] + [f"{m:02d}" for m in range(1, 10)]
        out = [f"{d}{s}{m}{s}{y}" for d in days for m in mons for y in range(years[0], years[1] + 1) for s in "/."]
    elif kind == "bp":
        out = [f"{s}/{d}" for s in range(70, 221) for d in range(40, 131)]
    elif kind == "number":
        lo, hi = int(rng[0]), int(rng[1]); out = [str(v) for v in range(lo, hi + 1)] + [f"{v}.{t}" for v in range(lo, hi) for t in range(10)]
    else: raise ValueError(f"no grammar for kind {kind!r}")
    _cache[key] = out; return out


def _masked_logp(logits, allowed):
    """logits [T, V] -> log-softmax over the allowed classes only (others get -inf), as the decoder does."""
    z = np.full(logits.shape, -np.inf, np.float64); z[:, allowed] = logits[:, allowed]
    z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def score(logits, allowed, cands, batch=3000):
    """CTC log-likelihood of each candidate string for one cell. -> numpy [len(cands)]. Needs torch (its CTC loss does the sum over alignments)."""
    import torch, torch.nn.functional as F
    blank = _index("<blank>"); lp = torch.from_numpy(_masked_logp(logits, allowed).astype(np.float32)); lp = torch.where(torch.isinf(lp), torch.full_like(lp, -1e4), lp); T = lp.shape[0]; out = []
    for i in range(0, len(cands), batch):
        chunk = cands[i:i + batch]; tg = [encode(c) for c in chunk]; n = len(chunk)
        l = F.ctc_loss(lp[:, None, :].expand(T, n, lp.shape[1]), torch.tensor([x for t in tg for x in t]), torch.full((n,), T, dtype=torch.long), torch.tensor([len(t) for t in tg]), blank=blank, reduction="none", zero_infinity=True)
        out.append(-l.numpy())
    return np.concatenate(out)


def decode(logits, allowed, kind, rng=None, years=None):
    """-> (best string, margin). margin = log-likelihood of the winner minus the log-probability of the single best unconstrained path (a string's likelihood sums over
    every alignment, so margin can be above 0 when many alignments agree; the more negative, the more the grammar had to overrule the reader). logits: [T, classes], allowed: class indices the field may emit.
    Runs grammar.py, the reference implementation (numpy only: no torch, no model); `score` below is the torch cross-check of it."""
    import grammar
    m = cards.meta(); idx = list(allowed); chars = [m["classes"][i] for i in idx]; blank = idx.index(m["blank_idx"])
    return grammar.decode(np.asarray(logits)[:, idx], chars, blank, kind, rng, years)


# ---------------------------------------------------------------- experiment
def fold_job(a):
    """One fold: fine-tune from the given pretrained weights and return the test cells with their logits."""
    fold, folds, merged, weights, max_epochs, patience = a
    import torch, common as C, capacity as K, train as T
    torch.set_num_threads(2); rows = C.read_labels(merged); pre = torch.load(weights)
    test_pages = set(folds[fold]); rest = sorted({r["page"] for r in rows} - test_pages)
    vp = set(random.Random(1000 + fold).sample(rest, max(4, round(len(rest) / 7))))
    tr = [r for r in rows if r["page"] in rest and r["page"] not in vp]; va = [r for r in rows if r["page"] in vp]; te = [r for r in rows if r["page"] in test_pages]
    m, info = T.finetune(pre, tr, va, T.CONFIGS[K.CFG], seed=0, max_epochs=max_epochs, patience=patience, log=lambda *_: None)
    m.eval()
    with torch.no_grad(): lg = m(torch.from_numpy(np.stack([C.to_input(C.cell_image(r)) for r in te])).unsqueeze(1)).numpy()       # [T, B, V]
    return fold, [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"]) for r in te], lg.transpose(1, 0, 2).astype(np.float32), info["best_epoch"]


def run(weights, max_epochs, patience, quick=False):
    import common as C, capacity as K, trainer
    from multiprocessing import Pool
    out = OUT / ("quick" if quick else "full"); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def status(stage, detail=""): (out / "status.json").write_text(json.dumps({"stage": stage, "detail": detail, "elapsed": round(time.time() - t0)}))
    try:
        merged = out / "merged_labels.csv"; rows = trainer.merge_labels(merged, cards.LEGACY); folds = trainer.make_folds(rows); use = list(range(2 if quick else 5))
        kinds = {f["key"]: f["kind"] for f in cards.fields(cards.LEGACY)}; meta = cards.meta(); masks = cards.masks(cards.LEGACY)
        status("finetune", f"{len(use)} folds from {Path(weights).name}"); cells, logits = [], []
        with Pool(len(use)) as pool:
            for fold, rs, lg, best in pool.imap_unordered(fold_job, [(f, folds, str(merged), str(weights), max_epochs, patience) for f in use]): cells += rs; logits.append(lg)
        merged.unlink(missing_ok=True); lg = np.concatenate(logits)
        status("decode", f"{len(cells)} cells"); recs = []
        for i, c in enumerate(cells):
            k = kinds[c["row"]]; g = C.decode_app(lg[i].reshape(lg.shape[1], -1), c["row"])[0]
            rec = {**c, "kind": k, "greedy": g, "constrained": None, "margin": None}
            if k in ("date", "bp") or (k == "number" and c["row"] in RANGES):
                rec["constrained"], rec["margin"] = decode(lg[i], masks[c["row"]], k, RANGES.get(c["row"]))
            recs.append(rec)
        def acc(rs, key):
            k = sum((r[key] or "") == r["label"] for r in rs); return {"n": len(rs), "correct": k, "exact": round(k / max(1, len(rs)), 4)}
        groups = {"dates": [r for r in recs if r["kind"] == "date"], "bp": [r for r in recs if r["kind"] == "bp"], "numbers (with a range)": [r for r in recs if r["kind"] == "number" and r["constrained"] is not None]}
        res = {g: {"greedy": acc(rs, "greedy"), "constrained": acc(rs, "constrained")} for g, rs in groups.items()}
        res["all grammar fields"] = {"greedy": acc(sum(groups.values(), []), "greedy"), "constrained": acc(sum(groups.values(), []), "constrained")}
        # how often is the label something the grammar cannot produce at all?
        res["label_outside_grammar"] = {g: sum(r["label"] not in set(candidates(r["kind"], RANGES.get(r["row"]))) for r in rs) for g, rs in groups.items()}
        doc = {"made": time.strftime("%Y-%m-%d %H:%M"), "quick": quick, "weights": str(Path(weights).name), "folds": len(use), "results": res, "minutes": round((time.time() - t0) / 60, 1)}
        (out / "summary.json").write_text(json.dumps(doc, indent=1)); (out / "reads.json").write_text(json.dumps(recs)); status("done", "summary written"); return doc
    except Exception:
        status("error", traceback.format_exc()[-1500:]); raise


def show(doc):
    print(f"{doc['folds']} folds from {doc['weights']}, {doc['minutes']} min{'  (QUICK RUN: not a result)' if doc['quick'] else ''}")
    for g, v in doc["results"].items():
        if g == "label_outside_grammar": print("  labels the grammar cannot produce:", v); continue
        print(f"  {g:26s} greedy {v['greedy']['correct']}/{v['greedy']['n']} ({v['greedy']['exact'] * 100:.0f}%)   constrained {v['constrained']['correct']}/{v['constrained']['n']} ({v['constrained']['exact'] * 100:.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "report"]); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pretrained", default=str(HERE / "capacity" / "v2" / "full" / "pre_v2.pt")); a = ap.parse_args()
    if a.cmd == "run": show(run(a.pretrained, 3 if a.quick else 150, 3 if a.quick else 40, a.quick))
    else: show(json.loads((OUT / ("quick" if a.quick else "full") / "summary.json").read_text()))
