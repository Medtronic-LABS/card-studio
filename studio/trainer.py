"""Studio train / gate / export for one card type (cards.py says where its labels, crops and template are). Local only.
Labels (for anc: labels.csv + labels_new.csv, new wins) -> page-split 5-fold CV (every labeled cell predicted once by a model that never
saw its page) -> gate per field -> final reader trained on ALL pages (epochs fixed from the CV) -> ONNX -> new pack version.
Usage: python trainer.py RUN_DIR [--card anc] [--target 85] [--min-n 20] [--max-epochs 300] [--pretrained auto|legacy|NAME] [--config light_lr3e-4|light_lr1e-4|nbaug_lr1e-4]   (progress in RUN_DIR/status.json, result in RUN_DIR/results.json)
Checkpoints, curves and STOP (a file RUN_DIR/STOP makes every worker finish its epoch, save and exit; stage "stopped"), continued with: python trainer.py RUN_DIR --resume"""
import argparse, collections, csv, hashlib, json, os, random, statistics, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(ROOT / "cardpack")); sys.path.insert(0, str(HERE))
import cards, constrained, pretrained, provenance, subcells
CFG = "light_lr3e-4"          # chosen on val in Phase A (selection.json); the default of --config
CONFIG_NAMES = ("light_lr3e-4", "light_lr1e-4", "nbaug_lr1e-4")     # the keys of train.CONFIGS: learning rate and how hard the cells are distorted while training
SEED = 20261103


def merge_labels(out, card=cards.LEGACY, dropped=None):
    """All labels for the card (cards.labels: later sources win, imports from phones / Label Studio included) written to `out`.
    A row with a `crop` path is read from there; anc rows without one use work/crops. `dropped` receives the counts of rows left out."""
    rows, problems = cards.labels(card)
    if dropped is not None: dropped.update(problems)
    train_rows, merge_labels.layouts = subcells.expand(card, rows, dropped)       # a two-line cell is trained as its two pieces
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["page", "row", "col", "label", "crop"]); w.writeheader(); w.writerows(train_rows)
    return rows


def make_folds(rows, k=5, tries=4000, units=None):
    """k folds of pages. `units` ({page: writer or page}, cards.split_units) keeps every page of one unit in one fold: with writers known, a
    fold is tested on people the other folds never saw. -> lists of pages."""
    pages = sorted({r["page"] for r in rows}); units = units or {p: p for p in pages}; uo = sorted({units[p] for p in pages}); by = collections.defaultdict(collections.Counter)
    for r in rows: by[units[r["page"]]][r["row"]] += 1
    tot = collections.Counter(r["row"] for r in rows); fields = [f for f, n in tot.items() if n >= 10]
    rng = random.Random(SEED); best = None
    for _ in range(tries):
        s = uo[:]; rng.shuffle(s); groups = [sorted(s[i::k]) for i in range(k)]
        cnt = [collections.Counter() for _ in groups]
        for i, g in enumerate(groups):
            for u in g: cnt[i].update(by[u])
        key = (sum(min(c[x] for c in cnt) for x in fields), -(max(sum(c.values()) for c in cnt) - min(sum(c.values()) for c in cnt)))
        if best is None or key > best[0]: best = (key, groups)
    return [sorted(p for p in pages if units[p] in set(g)) for g in best[1]]


def grammar_spec(card):
    """{field: {kind, range, years}} for the fields whose output can also be read against a grammar (constrained.py): dates, blood pressures, ranged
    numbers, one line each. Taken from the pack this run starts from."""
    fld = json.loads(cards.base_members(card)["fields.json"])["fields"]
    return {f["key"]: {"kind": f["kind"], "range": f["range"], "years": (f.get("date") or {}).get("years")} for f in fld
            if f.get("layout", "single") == "single" and (f["kind"] in ("date", "bp") or (f["kind"] == "number" and f["range"]))}


def cv_job(a):
    fold, folds, merged, card, max_epochs, gspec, weights, cfg, arch, seed = a[:10]; ck = a[10] if len(a) > 10 else None          # ck: where this run keeps checkpoints, curves, STOP (None = none of that)
    import torch; import common as C, train as T
    torch.set_num_threads(2); C.use_fields(cards.masks(card))
    rows = C.read_labels(merged); pre = torch.load(weights) if weights else None           # no weights: this architecture starts from scratch
    test_pages = set(folds[fold]); rest = sorted({r["page"] for r in rows} - test_pages)
    vp = set(random.Random(1000 + fold).sample(rest, max(4, round(len(rest) / 7))))
    tr = [r for r in rows if r["page"] in rest and r["page"] not in vp]; va = [r for r in rows if r["page"] in vp]; te = [r for r in rows if r["page"] in test_pages]
    assert not ({r["page"] for r in tr + va} & test_pages)
    fck = dict(ck, name=f"fold-{fold}", curves=str(Path(ck["curves"]) / f"fold-{fold}.json")) if ck else None
    m, info = T.finetune(pre, tr, va, cfg if isinstance(cfg, dict) else T.resolve_cfg(cfg), seed=seed, max_epochs=max_epochs, log=lambda *_: None, arch=arch, ck=fck)
    if info.get("stopped"): return fold, None, info["best_epoch"], info.get("hist", [])          # STOP: the -last checkpoint is written, this fold is not finished
    out = [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=p, conf=c) for (p, c), r in zip(T.predict_torch(m, te), te)]
    gi = [i for i, r in enumerate(te) if r["row"] in gspec]          # the same cells read again against the field's grammar, from the same model
    if gi:
        import numpy as np, constrained
        m.eval(); masks = cards.masks(card)
        with torch.no_grad(): lg = m(torch.from_numpy(np.stack([C.to_input(C.cell_image(te[i])) for i in gi])).unsqueeze(1)).numpy()
        for j, i in enumerate(gi):
            g = gspec[te[i]["row"]]; out[i]["pred_grammar"], out[i]["margin"] = constrained.decode(lg[:, j, :], masks[te[i]["row"]], g["kind"], g["range"], g["years"])
    if ck:
        import train as _T; _T.write_json_atomic(Path(ck["done"]) / f"fold-{fold}.json", {"fold": fold, "out": out, "best": info["best_epoch"], "hist": info.get("hist", [])})
    return fold, out, info["best_epoch"], info.get("hist", [])


def train_fixed(pre, rows, epochs, seed=0, cfg=CFG, arch=None, ck=None):
    """The final reader: all labeled pages, a fixed number of epochs (the median best epoch of the folds). cfg: a preset name or a resolved config; arch: None for the shipped CRNN;
    pre: starting weights or None. ck: a Checkpointer config (train.py) for the final reader's checkpoints / curves / STOP / resume; -> None when STOP ended it before the last epoch."""
    import torch, torch.nn as nn
    import common as C, train as T
    torch.set_num_threads(6); torch.manual_seed(seed)
    cfg = cfg if isinstance(cfg, dict) else T.resolve_cfg(cfg)
    model = T.new_model(arch)
    if pre is not None: model.load_state_dict(pre)
    dl, opt, sch = T.training_setup(model, rows, cfg, seed, epochs, mode="min"); ctc = nn.CTCLoss(blank=C.BLANK_IDX, zero_infinity=True)      # no validation pages here: a plateau schedule watches the training loss
    plateau = cfg.get("schedule", "none") == "plateau"; cp = T.Checkpointer(ck) if ck else None; hist, start = [], 1
    st = cp.load_last() if cp else None
    if st and st.get("epochs") == epochs:
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); hist, start = st["hist"], st["epoch"] + 1; T._rng_set(dl, st["rng"])
        if sch and st.get("sched") is not None: sch.load_state_dict(st["sched"])
    for ep in range(start, epochs + 1):
        lr = opt.param_groups[0]["lr"]; loss = T.train_epoch(model, dl, opt, ctc, cfg.get("grad_clip", 0.0))
        if sch: sch.step(loss) if plateau else sch.step()
        hist.append({"epoch": ep, "loss": loss, "lr": lr})
        if cp:
            cp.write_curves(hist)
            def full(): return {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sch.state_dict() if sch else None, "epoch": ep, "epochs": epochs, "hist": hist, "rng": T._rng_get(dl)}
            if cp.due(ep): cp.save_numbered(ep, model.state_dict()); cp.save_last(full())
            if ep == epochs: cp.save_last(full())
            elif cp.stop_requested(): cp.save_last(full()); return None
    return model


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj)); os.replace(tmp, path)


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
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--target", type=float, default=85.0); ap.add_argument("--min-n", type=int, default=20); ap.add_argument("--config", default=CFG, choices=CONFIG_NAMES, help="training preset: learning rate and augmentation strength")
    ap.add_argument("--seed", type=int, default=0, help="training seed (the folds do not change with it): run several to see how much a result moves by chance")
    ap.add_argument("--cv-only", action="store_true", help="stop after cross-validation: write cv_predictions.json and a short results.json, train no final reader and write no pack (what seed experiments use)")
    ap.add_argument("--settings", default=None, help="JSON file: {\"preset\": ..., \"hp\": {every training value}, \"arch\": a preset name or an architecture spec (archnet.py)}; see settings.py")
    ap.add_argument("--card", default=cards.LEGACY); ap.add_argument("--pretrained", default="auto", help="starting weights: auto (studio/pretrained/DEFAULT, else legacy), legacy, or a name from pretrained.py list"); ap.add_argument("--max-epochs", type=int, default=300, help="cap per CV fold (lower only for a quick trial)")
    ap.add_argument("--resume", action="store_true", help="continue a stopped or failed run from RUN_DIR/checkpoints (RUN_DIR/trainer_args.json holds everything else; no other option may be given)")
    a = ap.parse_args(); run = Path(a.run_dir); settings_raw = None
    if a.resume:
        clash = sorted(k for k, v in vars(a).items() if k not in ("run_dir", "resume") and v != ap.get_default(k))
        if clash: ap.error(f"--resume takes only RUN_DIR: the settings come from trainer_args.json (also given: {', '.join('--' + k.replace('_', '-') for k in clash)})")
        if not (run / "trainer_args.json").is_file(): ap.error(f"cannot resume: {run / 'trainer_args.json'} not found (not a run directory this trainer started)")
        saved = json.loads((run / "trainer_args.json").read_text()); settings_raw = saved.get("settings_raw")
        for k, v in saved["args"].items(): setattr(a, k, v)
        a.resume = True; (run / "STOP").unlink(missing_ok=True)          # the STOP that ended the last run must not end this one
    run.mkdir(parents=True, exist_ok=True); t0 = time.time(); fold_epochs = {}
    def status(stage, detail="", **kw):
        if "resumable" not in kw and stage == "error": kw["resumable"] = any((run / "checkpoints").glob("*.pt"))
        if fold_epochs: kw.setdefault("fold_epochs", fold_epochs)
        write_json(run / "status.json", dict(stage=stage, detail=detail, elapsed=round(time.time() - t0), target=a.target, min_n=a.min_n, card=a.card, **kw))
    try:
        status("labels", "merging labels")
        if not cards.exists(a.card): raise ValueError(f"no such card: {a.card}")
        import settings as S
        if not a.resume and a.settings:
            try: settings_raw = json.loads(Path(a.settings).read_text())
            except (OSError, ValueError): settings_raw = None
        sett = S.load(settings_raw if a.resume or settings_raw is not None else a.settings, a.config); a.config = sett["preset"]; arch, cfg = sett["arch"], sett["cfg"]          # fails early, with the reason, when a value or the architecture is wrong
        ck_run = sett["run"]; early = dict(monitor=cfg["monitor"], min_delta=cfg["min_delta"], restore=cfg["restore"], patience=cfg["patience"], plateau_patience=cfg["plateau_patience"])
        ck_rec = dict(every=ck_run["checkpoint_every"], keep_last=ck_run["keep_last"])
        if not a.resume: write_json(run / "trainer_args.json", {"args": {k: v for k, v in vars(a).items() if k != "resume"}, "settings_raw": settings_raw, "settings": sett["record"]})
        weights, wrec = pretrained.resolve(a.pretrained, arch)      # fails early, with the reason, when the weights cannot be used (or were made for another network)
        status("labels", f"starting weights: {wrec['name']}" + (f" ({wrec['note']})" if wrec.get("note") else "") + f" · network: {sett['arch_label']}")
        merged = run / "merged_labels.csv"; dropped = {}; old_sha = hashlib.sha256(merged.read_bytes()).hexdigest() if a.resume and merged.exists() else None
        rows = merge_labels(merged, a.card, dropped); layouts = merge_labels.layouts
        if old_sha and old_sha != hashlib.sha256(merged.read_bytes()).hexdigest(): raise ValueError("the labels changed since this run stopped: its checkpoints no longer match; start a new run")
        whole = {(r["page"], r["row"], r["col"]): r["label"] for r in rows}
        with open(merged, newline="") as fh: n_train = sum(1 for _ in csv.DictReader(fh))
        pages = sorted({r["page"] for r in rows})
        if len(pages) < cards.MIN_PAGES: raise ValueError(f"{len(rows)} labeled cells on {len(pages)} page(s): training needs at least {cards.MIN_PAGES} labeled pages")
        units, unit, n_units = cards.split_units(a.card, pages)
        if len(pages) < 5 or n_units < 5: raise ValueError(f"{n_units} {unit}(s): 5-fold cross-validation needs at least 5")
        folds = make_folds(rows, units=units)
        assert sorted(sum(folds, [])) == pages and all(not set(folds[i]) & set(folds[j]) for i in range(5) for j in range(i))
        assert all(len({units[p] for p in f} & {units[p] for g in folds if g is not f for p in g}) == 0 for f in folds), "a writer appears in two folds"
        split_bytes = json.dumps({"seed": SEED, "unit": unit, "folds": folds}, sort_keys=True).encode()
        (run / "folds.json").write_bytes(split_bytes)
        status("cv", f"{len(rows)} labeled cells, {len(pages)} pages; 0 of 5 folds done", cv_done=0)
        preds, bests, curves = [], [], []; gspec = grammar_spec(a.card); stopped_folds = []
        ck = dict(dir=str(run / "checkpoints"), every=ck_run["checkpoint_every"], keep_last=ck_run["keep_last"], stop=str(run / "STOP"), curves=str(run / "curves"), done=str(run / "folds_done"), resume=a.resume)
        def read_epochs(fs):
            for f in fs:
                try: fold_epochs[str(f)] = len(json.loads((run / "curves" / f"fold-{f}.json").read_text()))
                except (OSError, ValueError): pass
        def finish(fold, out, best, hist):
            if out is None: stopped_folds.append(fold); return
            preds.extend(out); bests.append(best); curves.append({"fold": fold, "best_epoch": best, "history": hist})
        todo = []
        for f in range(5):
            dn = run / "folds_done" / f"fold-{f}.json"
            if a.resume and dn.exists(): d = json.loads(dn.read_text()); finish(d["fold"], d["out"], d["best"], d["hist"]); read_epochs([f])
            else: todo.append(f)
        status("cv", f"{len(rows)} labeled cells, {len(pages)} pages; {len(bests)} of 5 folds done", cv_done=len(bests))
        if todo:
            with Pool(5) as pool:
                pending = {f: pool.apply_async(cv_job, ((f, folds, str(merged), a.card, a.max_epochs, gspec, str(weights) if weights else None, cfg, arch, a.seed, ck),)) for f in todo}
                while pending:
                    for f, r in list(pending.items()):
                        if r.ready(): finish(*r.get()); read_epochs([f]); del pending[f]
                    read_epochs(pending); status("cv", f"{len(bests)} of 5 folds done", cv_done=len(bests))
                    if pending: next(iter(pending.values())).wait(2)
        if stopped_folds:
            status("stopped", f"stopped during cross-validation: {len(bests)} of 5 folds finished, folds {sorted(stopped_folds)} saved at epochs " + ", ".join(f"{f}: {fold_epochs.get(str(f), '?')}" for f in sorted(stopped_folds)) + "; resume with --resume", cv_done=len(bests), resumable=True)
            return
        assert len(preds) == n_train and len({(p["page"], p["row"], p["col"]) for p in preds}) == n_train
        parts_preds = preds; preds = subcells.compose(preds, layouts, whole) if layouts else preds       # two-line cells are scored as whole cells
        assert len(preds) == len(rows) or layouts, "every labeled cell must have exactly one cross-validation read"
        cv = metrics(preds); overall = metrics([dict(p, row="ALL") for p in preds])["ALL"]
        epochs = int(statistics.median(bests))
        if a.cv_only:
            json.dump(preds, open(run / "cv_predictions.json", "w"))
            json.dump(dict(card=a.card, cv_only=True, seed=a.seed, settings=sett["record"], early_stopping=early, checkpoints=ck_rec, cv_all=overall, best_epochs=sorted(bests), n_cells=len(rows), n_pages=len(pages), split_unit=unit, minutes=round((time.time() - t0) / 60, 1),
                           fields=[dict(field=k, **v) for k, v in sorted(cv.items(), key=lambda kv: -kv[1]["n"])]), open(run / "results.json", "w"), indent=1)
            status("done", "cross-validation only: no final reader and no pack"); return
        status("final", f"training the reader on all {len(pages)} pages for {epochs} epochs (median best epoch of the folds)", cv_done=5)
        import torch, common as C, train as T
        C.use_fields(cards.masks(a.card))
        pre = torch.load(weights) if weights else None
        model = train_fixed(pre, C.read_labels(merged), epochs, seed=a.seed, cfg=cfg, arch=arch, ck=dict(ck, name="final", curves=str(run / "curves" / "final.json")))
        if model is None:
            try: fe = len(json.loads((run / "curves" / "final.json").read_text()))
            except (OSError, ValueError): fe = "?"
            status("stopped", f"stopped during the final training at epoch {fe} of {epochs}; resume with --resume", cv_done=5, resumable=True); return
        model.eval()
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
        # the same weights for TensorFlow Lite (phones). Optional: without a TensorFlow environment the pack ships the ONNX reader only.
        status("export", "TFLite conversion and parity check")
        import tflite_export
        try:
            tfl_bytes, tfl = tflite_export.export(onnx_path.read_bytes(), samples=x[:, 0]); (run / "reader.tflite").write_bytes(tfl_bytes); tfl = {"built": True, **tfl}
        except (tflite_export.Unavailable, ValueError, OSError) as e:
            tfl_bytes, tfl = None, {"built": False, "reason": str(e)}

        # ---- gate + pack
        import cardpack as cp
        mem = cards.base_members(a.card); man = json.loads(mem["manifest.json"]); fld = json.loads(mem["fields.json"])
        gate = {}; dec = {}; cv_g = metrics([dict(p, pred=p["pred_grammar"]) for p in preds if "pred_grammar" in p])
        for f in fld["fields"]:
            m = cv.get(f["key"]); g = cv_g.get(f["key"])
            if m and g and g["n"] >= 10 and g["k"] >= m["k"]:       # grammar decoding only where held-out cells show it at least as good as greedy
                f["decode"] = "grammar"; dec[f["key"]] = dict(decode="grammar", greedy_k=m["k"], grammar_k=g["k"]); m = cv[f["key"]] = g
                if f["kind"] == "date": f.setdefault("date", {"years": list(constrained.YEARS)})
            elif m and g: dec[f["key"]] = dict(decode="greedy", greedy_k=m["k"], grammar_k=g["k"]); f.pop("decode", None)
            lo_pct = m["lo"] * 100 if m else None
            if m is None: st, why = "needs-review", "no labeled cells for this field"
            elif m["n"] < a.min_n: st, why = "needs-review", f"only {m['n']} held-out cells (need {a.min_n})"
            elif lo_pct < a.target: st, why = "needs-review", f"held-out lower bound {lo_pct:.0f}% is under the {a.target:.0f}% target ({m['k']}/{m['n']})"
            else: st, why = "ship", f"held-out {m['k']}/{m['n']}, lower bound {lo_pct:.0f}% meets the {a.target:.0f}% target"
            if f["key"] in dec: why += f" · read with {'the field grammar' if dec[f['key']]['decode'] == 'grammar' else 'greedy decoding'}: held-out {dec[f['key']]['grammar_k']} right with the grammar, {dec[f['key']]['greedy_k']} greedy"
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
                                      "split_unit": f"writer ({n_units} writers)" if unit == "writer" else "page (writer unknown)", "split_file_sha256": hashlib.sha256(split_bytes).hexdigest(),
                                      "reader_trained_on": f"all {len(pages)} labeled pages for {epochs} epochs (the shipped weights have no held-out number of their own)",
                                      "gate": {"target_lower_bound_pct": a.target, "min_n": a.min_n},
                                      "per_field": {k: dict(n=v["n"], correct=v["k"], ci95=[round(v["lo"], 4), round(v["hi"], 4)]) for k, v in cv.items()}}}
        mem.pop("reader.tflite", None); man["reader"].pop("tflite", None)
        if tfl_bytes: mem["reader.tflite"] = tfl_bytes; man["reader"]["tflite"] = tflite_export.manifest_entry(tfl, man["reader"])
        mem["reader.onnx"] = onnx_path.read_bytes(); mem["fields.json"] = json.dumps(fld, indent=1, ensure_ascii=False).encode()
        man["files"] = {n: cp.sha256(mem[n]) for n in cp.hashed(mem)}; mem["manifest.json"] = json.dumps(man, indent=1).encode()
        out_pack = cards.PACKS / f"{man['id']}_v{man['version']}.cardpack"; out_pack.parent.mkdir(parents=True, exist_ok=True); cp.build(mem, out_pack)
        errs = cp.validate(out_pack)
        crops_note = "stored gray crops (work/crops) or imported crops" if a.card == cards.LEGACY else f"stored gray crops (studio/cards/{a.card}/crops) or imported crops"
        overall_greedy = overall; overall = metrics([dict(p, row="ALL", pred=p["pred_grammar"] if dec.get(p["row"], {}).get("decode") == "grammar" else p["pred"]) for p in preds])["ALL"]
        res = dict(card=a.card, pretrained={"name": wrec["name"], "sha256": pretrained.sha256(weights)[:16] if weights else None, "images": wrec.get("images"), "epochs": wrec.get("epochs"), "generator": wrec.get("generator")}, cv_all_greedy=overall_greedy, decode=dec, split_unit=unit, split_units=n_units, dropped_labels=dropped, config=a.config, settings=sett["record"], early_stopping=early, checkpoints=ck_rec, target=a.target, min_n=a.min_n, n_cells=len(rows), n_pages=len(pages), folds=[len(f) for f in folds], best_epochs=sorted(bests),
                   final_epochs=epochs, cv_all=overall, parity=parity, tflite=tfl, minutes=round((time.time() - t0) / 60, 1),
                   fields=[dict(field=k, **v, decode=dec.get(k, {}).get("decode", "greedy"), status=gate.get(k, ("needs-review", ""))[0], reason=gate.get(k, ("", ""))[1]) for k, v in sorted(cv.items(), key=lambda kv: -kv[1]["n"])],
                   unlabeled_fields=[f["key"] for f in fld["fields"] if f["key"] not in cv],
                   pack=dict(path=str(out_pack), id=man["id"], version=man["version"], validation_errors=errs, shipped=[k for k, v in gate.items() if v[0] == "ship"]),
                   provenance=provenance.collect(merged, rows, SEED, sett["config_label"], hashlib.sha256(split_bytes).hexdigest(), crops_note + ", no line removal; page normalization not applied at training time"))
        json.dump(res, open(run / "results.json", "w"), indent=1)
        json.dump(preds, open(run / "cv_predictions.json", "w"))
        if layouts: json.dump(parts_preds, open(run / "cv_predictions_pieces.json", "w"))
        json.dump({"note": "per-epoch history of each CV fold (loss, validation exact match and CER on that fold's 7 or so validation pages)", "folds": sorted(curves, key=lambda c: c["fold"])}, open(run / "history.json", "w"))
        status("done" if not errs else "error", "pack written" if not errs else f"pack failed validation: {errs}")
    except Exception:
        status("error", traceback.format_exc()[-1500:])
        raise


if __name__ == "__main__":
    main()
