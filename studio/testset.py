"""Score a finished run's reader on the card's held-out test set, and count every look.

The test set (cards.holdout) is reserved once and nothing else reads its pages: training, cross-validation, tuning, learning curves and comparisons all use `cards.labels(card)`, which leaves
those pages out. A reader trained and tuned without them gets one honest number here. Every look is logged, because each extra look at the same test set while choosing between settings
turns it into a tuning set: the panel shows how many looks a card has had.

A run is only scored if its folds did not contain a test page (a run started before the test set was reserved is refused).

    python testset.py reserve CARD [--fraction 0.2]      draw the test set (once)
    python testset.py score CARD [RUN]                   score the reader of RUN (default: the newest finished run of the card) on it
    python testset.py status CARD
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
import cards

RUNS = HERE / "runs"


def log_path(card):
    return cards.holdout_path(card).with_name("holdout_looks.json")


def looks(card):
    try: return json.loads(log_path(card).read_text())
    except (OSError, ValueError): return []


def newest_run(card):
    best = None
    for d in sorted(RUNS.glob("run-*")):
        try: r = json.loads((d / "results.json").read_text())
        except (OSError, ValueError): continue
        if r.get("card") == card and (d / "reader.onnx").exists(): best = d.name
    return best


def score(card, run=None):
    """-> {run, n, exact, lo, hi, cer, by_field, looks, warning}. Reads the held-out cells with the run's ONNX reader, the same masked decoding (and field grammars) the pack uses, scores
    whole cells, and appends the look to the log."""
    h = cards.holdout(card)
    if not h: raise ValueError("this card has no test set: reserve one first")
    run = run or newest_run(card)
    if not run: raise ValueError("no finished run of this card to score")
    d = RUNS / run
    try: res = json.loads((d / "results.json").read_text()); onnx = d / "reader.onnx"; folds = json.loads((d / "folds.json").read_text())["folds"]
    except (OSError, ValueError): raise ValueError(f"{run} is not a finished run with a reader")
    if res.get("card") != card or not onnx.exists(): raise ValueError(f"{run} is not a finished run of {card}")
    seen = {p for f in folds for p in f} & set(h["pages"])
    if seen: raise ValueError(f"{run} was trained on {len(seen)} page(s) of the test set (it started before the test set was reserved): train again, then score")
    rows, _ = cards.labels(card, holdout=True)
    if not rows: raise ValueError("the test set has no labeled cells")
    import numpy as np, onnxruntime as ort                         # heavy imports only once the checks above have passed
    import common as C, constrained, subcells, trainer
    C.use_fields(cards.masks(card)); masks = cards.masks(card); gram = trainer.grammar_spec(card); decode = {k: v.get("decode") for k, v in res.get("decode", {}).items()}
    pieces, layouts = subcells.expand(card, rows)
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]); preds = []
    for r in pieces:
        x = C.to_input(C.cell_image(r))[None, None].astype(np.float32); lg = sess.run(None, {"image": x})[0][:, 0, :]
        pred, conf = C.decode_app(lg, r["row"])
        g = gram.get(r["row"])
        if g and decode.get(r["row"]) == "grammar":
            text, _ = constrained.decode(lg, masks[r["row"]], g["kind"], g["range"], g["years"]); pred = text or pred
        preds.append(dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=pred, conf=conf))
    whole = {(r["page"], r["row"], r["col"]): r["label"] for r in rows}
    preds = subcells.compose(preds, layouts, whole) if layouts else preds
    by = trainer.metrics(preds); allm = trainer.metrics([dict(p, row="ALL") for p in preds])["ALL"]
    log = looks(card); log.append({"run": run, "when": time.strftime("%Y-%m-%d %H:%M"), "n": allm["n"], "exact": round(allm["exact"], 4), "lo": round(allm["lo"], 4), "hi": round(allm["hi"], 4)})
    p = log_path(card); p.write_text(json.dumps(log, indent=1))
    warn = None if len(log) == 1 else f"This test set has been looked at {len(log)} times. Each extra look while choosing between settings makes this number less trustworthy: treat it as a tuning set from now on, or reserve a new one."
    return {"run": run, "n": allm["n"], "pages": len({p["page"] for p in preds}), "exact": allm["exact"], "lo": allm["lo"], "hi": allm["hi"], "cer": allm["cer"],
            "by_field": {k: {x: v[x] for x in ("n", "k", "exact", "lo", "hi", "cer")} for k, v in sorted(by.items())}, "looks": len(log), "warning": warn}


def status(card):
    return {**cards.holdout_status(card), "looks": looks(card)}


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        if a[:1] == ["reserve"] and len(a) >= 2: print(json.dumps(cards.make_holdout(a[1], float(a[a.index("--fraction") + 1]) if "--fraction" in a else cards.HOLDOUT_FRACTION), indent=1))
        elif a[:1] == ["score"] and len(a) >= 2: print(json.dumps(score(a[1], a[2] if len(a) > 2 else None), indent=1))
        elif a[:1] == ["status"] and len(a) == 2: print(json.dumps(status(a[1]), indent=1))
        else: sys.exit("usage: python testset.py reserve CARD [--fraction 0.2] | score CARD [RUN] | status CARD")
    except ValueError as e:
        sys.exit(str(e))
