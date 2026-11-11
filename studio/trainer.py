"""Studio train / gate / export for one card type (cards.py says where its labels, crops and template are). Local only.
Labels (for anc: labels.csv + labels_new.csv, new wins) -> page-split 5-fold CV (every labeled cell predicted once by a model that never
saw its page) -> gate per field -> final reader trained on ALL pages (epochs fixed from the CV) -> ONNX -> new pack version.
Usage: python trainer.py RUN_DIR [--card anc] [--target 85] [--min-n 20] [--max-epochs 300]   (progress in RUN_DIR/status.json, result in RUN_DIR/results.json)"""
import argparse, collections, csv, hashlib, json, random, statistics, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(ROOT / "cardpack")); sys.path.insert(0, str(HERE))
import cards, provenance
CFG = "light_lr3e-4"          # chosen on val in Phase A (selection.json)
SEED = 20261103


def merge_labels(out, card=cards.LEGACY, dropped=None):
    """All labels for the card (cards.labels: later sources win, imports from phones / Label Studio included) written to `out`.
    A row with a `crop` path is read from there; anc rows without one use work/crops. `dropped` receives the counts of rows left out."""
    rows, problems = cards.labels(card)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["page", "row", "col", "label", "crop"]); w.writeheader(); w.writerows(rows)
    if dropped is not None: dropped.update(problems)
    return rows


def make_folds(rows, k=5, tries=4000):
    pages = sorted({r["page"] for r in rows}); by = collections.defaultdict(collections.Counter)
    for r in rows: by[r["page"]][r["row"]] += 1
    tot = collections.Counter(r["row"] for r in rows); fields = [f for f, n in tot.items() if n >= 10]
    rng = random.Random(SEED); best = None
    for _ in range(tries):
        s = pages[:]; rng.shuffle(s); folds = [sorted(s[i::k]) for i in range(k)]
        cnt = [collections.Counter() for _ in folds]
        for i, f in enumerate(folds):
            for p in f: cnt[i].update(by[p])
        key = (sum(min(c[x] for c in cnt) for x in fields), -(max(sum(c.values()) for c in cnt) - min(sum(c.values()) for c in cnt)))
        if best is None or key > best[0]: best = (key, folds)
    return best[1]


def cv_job(a):
    fold, folds, merged, card, max_epochs = a
    import torch; import common as C, train as T
    torch.set_num_threads(2); C.use_fields(cards.masks(card))
    rows = C.read_labels(merged); pre = torch.load(ROOT / "work_phaseA" / "pretrained.pt")
    test_pages = set(folds[fold]); rest = sorted({r["page"] for r in rows} - test_pages)
    vp = set(random.Random(1000 + fold).sample(rest, max(4, round(len(rest) / 7))))
    tr = [r for r in rows if r["page"] in rest and r["page"] not in vp]; va = [r for r in rows if r["page"] in vp]; te = [r for r in rows if r["page"] in test_pages]
    assert not ({r["page"] for r in tr + va} & test_pages)
    m, info = T.finetune(pre, tr, va, T.CONFIGS[CFG], seed=0, max_epochs=max_epochs, log=lambda *_: None)
    out = [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=p, conf=c) for (p, c), r in zip(T.predict_torch(m, te), te)]
    return fold, out, info["best_epoch"], info.get("hist", [])


def train_fixed(pre, rows, epochs, seed=0):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    import common as C, train as T
    torch.set_num_threads(6); torch.manual_seed(seed)
    model = C.CRNN(len(C.CLASSES)); model.load_state_dict(pre)
    fc = collections.Counter(r["row"] for r in rows)
    sampler = torch.utils.data.WeightedRandomSampler([1 / fc[r["row"]] for r in rows], len(rows), replacement=True, generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(T.DS(rows, T.CONFIGS[CFG]["aug"], seed), batch_size=64, sampler=sampler, collate_fn=T.collate)
    opt = torch.optim.Adam(model.parameters(), lr=T.CONFIGS[CFG]["lr"]); ctc = nn.CTCLoss(blank=C.BLANK_IDX, zero_infinity=True)
    for ep in range(epochs):
        model.train()
        for x, tg, tlen, fields in dl:
            lg = model(x).clone()
            for k, f in enumerate(fields): lg[:, k, ~T.mask_t(f)] = -1e9
            lp = F.log_softmax(lg, -1)
            loss = ctc(lp, tg, torch.full((x.size(0),), lp.size(0), dtype=torch.long), tlen)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def metrics(recs):
    import common as C
    by = collections.defaultdict(list)
    for r in recs: by[r["row"]].append(r)
    out = {}
    for f, rs in by.items():
        n = len(rs); k = sum(r["pred"] == r["label"] for r in rs); lo, hi = C.wilson(k, n)
        ed = sum(C.edit_distance(r["pred"], r["label"]) for r in rs); ln = sum(len(r["label"]) for r in rs)
        out[f] = dict(n=n, k=k, exact=k / n, lo=lo, hi=hi, cer=ed / ln)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--target", type=float, default=85.0); ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--card", default=cards.LEGACY); ap.add_argument("--max-epochs", type=int, default=300, help="cap per CV fold (lower only for a quick trial)")
    a = ap.parse_args(); run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def status(stage, detail="", **kw):
        json.dump(dict(stage=stage, detail=detail, elapsed=round(time.time() - t0), target=a.target, min_n=a.min_n, card=a.card, **kw), open(run / "status.json", "w"))
    try:
        status("labels", "merging labels")
        if not cards.exists(a.card): raise ValueError(f"no such card: {a.card}")
        merged = run / "merged_labels.csv"; dropped = {}; rows = merge_labels(merged, a.card, dropped)
        pages = sorted({r["page"] for r in rows})
        if len(pages) < cards.MIN_PAGES: raise ValueError(f"{len(rows)} labeled cells on {len(pages)} page(s): training needs at least {cards.MIN_PAGES} labeled pages")
        folds = make_folds(rows)
        assert sorted(sum(folds, [])) == pages and all(not set(folds[i]) & set(folds[j]) for i in range(5) for j in range(i))
        split_bytes = json.dumps({"seed": SEED, "unit": "page", "folds": folds}, sort_keys=True).encode()
        (run / "folds.json").write_bytes(split_bytes)
        status("cv", f"{len(rows)} labeled cells, {len(pages)} pages; 0 of 5 folds done", cv_done=0)
        preds, bests, curves = [], [], []
        with Pool(5) as pool:
            for fold, out, best, hist in pool.imap_unordered(cv_job, [(f, folds, str(merged), a.card, a.max_epochs) for f in range(5)]):
                preds += out; bests.append(best); curves.append({"fold": fold, "best_epoch": best, "history": hist})
                status("cv", f"{len(bests)} of 5 folds done", cv_done=len(bests))
        assert len(preds) == len(rows) and len({(p["page"], p["row"], p["col"]) for p in preds}) == len(rows)
        cv = metrics(preds); overall = metrics([dict(p, row="ALL") for p in preds])["ALL"]
        epochs = int(statistics.median(bests))
        status("final", f"training the reader on all {len(pages)} pages for {epochs} epochs (median best epoch of the folds)", cv_done=5)
        import torch, common as C, train as T
        C.use_fields(cards.masks(a.card))
        pre = torch.load(ROOT / "work_phaseA" / "pretrained.pt")
        model = train_fixed(pre, C.read_labels(merged), epochs); model.eval()
        status("export", "ONNX export and parity check")
        import numpy as np, onnxruntime as ort
        onnx_path = run / "reader.onnx"; torch.set_num_threads(2)
        torch.onnx.export(model, torch.zeros(1, 1, C.IMG_H, C.IMG_W), str(onnx_path), input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}}, opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(onnx_path)); sample = C.read_labels(merged)[::4]
        x = np.stack([C.to_input(C.cell_image(r)) for r in sample])[:, None]
        with torch.no_grad(): lt = model(torch.from_numpy(x)).numpy()
        lo = sess.run(None, {"image": x})[0]
        same = all(C.decode_app(lt[:, i], r["row"])[0] == C.decode_app(lo[:, i], r["row"])[0] for i, r in enumerate(sample))
        parity = dict(cells=len(sample), max_abs_diff=float(np.abs(lt - lo).max()), decode_equal=bool(same))
        assert parity["decode_equal"] and parity["max_abs_diff"] < 1e-3, f"ONNX parity failed: {parity}"

        # ---- gate + pack
        import cardpack as cp
        mem = cards.base_members(a.card); man = json.loads(mem["manifest.json"]); fld = json.loads(mem["fields.json"])
        gate = {}
        for f in fld["fields"]:
            m = cv.get(f["key"]); lo_pct = m["lo"] * 100 if m else None
            if m is None: st, why = "needs-review", "no labeled cells for this field"
            elif m["n"] < a.min_n: st, why = "needs-review", f"only {m['n']} held-out cells (need {a.min_n})"
            elif lo_pct < a.target: st, why = "needs-review", f"held-out lower bound {lo_pct:.0f}% is under the {a.target:.0f}% target ({m['k']}/{m['n']})"
            else: st, why = "ship", f"held-out {m['k']}/{m['n']}, lower bound {lo_pct:.0f}% meets the {a.target:.0f}% target"
            f["status"], f["status_reason"] = st, why; gate[f["key"]] = (st, why)
        man["version"] = man["version"] + 1; man["created"] = time.strftime("%Y-%m-%d")
        # photo normalization: gray-level reference from the scans the reader was just trained on. contrast_match was the best
        # option on held-out pages for the phone-photo conditions tested (see CARDPACK.md); without enough scans keep today's flatten.
        scans = [p for p in (cards.aligned_dir(a.card) / f"{pg}.png" for pg in sorted(pages)) if p.exists()]
        if len(scans) >= 10:
            mean, std = cp.scan_reference(scans)
            man["preprocess"]["normalize"] = {"default": "contrast_match", "reference": {"mean": mean, "std": std, "source": f"gray mean/std over {len(scans)} aligned scans the reader was trained on"}}
        else:
            man["preprocess"]["normalize"] = {"default": "flatten"}
        man["metrics"] = {"heldout": {"method": "pooled 5-fold CV by page of the same recipe; every labeled cell predicted once by a model that never saw its page",
                                      "split_unit": "page (writer unknown)", "split_file_sha256": hashlib.sha256(split_bytes).hexdigest(),
                                      "reader_trained_on": f"all {len(pages)} labeled pages for {epochs} epochs (the shipped weights have no held-out number of their own)",
                                      "gate": {"target_lower_bound_pct": a.target, "min_n": a.min_n},
                                      "per_field": {k: dict(n=v["n"], correct=v["k"], ci95=[round(v["lo"], 4), round(v["hi"], 4)]) for k, v in cv.items()}}}
        mem["reader.onnx"] = onnx_path.read_bytes(); mem["fields.json"] = json.dumps(fld, indent=1, ensure_ascii=False).encode()
        man["files"] = {n: cp.sha256(mem[n]) for n in cp.HASHED}; mem["manifest.json"] = json.dumps(man, indent=1).encode()
        out_pack = cards.PACKS / f"{man['id']}_v{man['version']}.cardpack"; out_pack.parent.mkdir(parents=True, exist_ok=True); cp.build(mem, out_pack)
        errs = cp.validate(out_pack)
        crops_note = "stored gray crops (work/crops) or imported crops" if a.card == cards.LEGACY else f"stored gray crops (studio/cards/{a.card}/crops) or imported crops"
        res = dict(card=a.card, dropped_labels=dropped, config=CFG, target=a.target, min_n=a.min_n, n_cells=len(rows), n_pages=len(pages), folds=[len(f) for f in folds], best_epochs=sorted(bests),
                   final_epochs=epochs, cv_all=overall, parity=parity, minutes=round((time.time() - t0) / 60, 1),
                   fields=[dict(field=k, **v, status=gate.get(k, ("needs-review", ""))[0], reason=gate.get(k, ("", ""))[1]) for k, v in sorted(cv.items(), key=lambda kv: -kv[1]["n"])],
                   unlabeled_fields=[f["key"] for f in fld["fields"] if f["key"] not in cv],
                   pack=dict(path=str(out_pack), id=man["id"], version=man["version"], validation_errors=errs, shipped=[k for k, v in gate.items() if v[0] == "ship"]),
                   provenance=provenance.collect(merged, rows, SEED, CFG, hashlib.sha256(split_bytes).hexdigest(), crops_note + ", no line removal; page normalization not applied at training time"))
        json.dump(res, open(run / "results.json", "w"), indent=1)
        json.dump(preds, open(run / "cv_predictions.json", "w"))
        json.dump({"note": "per-epoch history of each CV fold (loss, validation exact match and CER on that fold's 7 or so validation pages)", "folds": sorted(curves, key=lambda c: c["fold"])}, open(run / "history.json", "w"))
        status("done" if not errs else "error", "pack written" if not errs else f"pack failed validation: {errs}")
    except Exception:
        status("error", traceback.format_exc()[-1500:])
        raise


if __name__ == "__main__":
    main()
