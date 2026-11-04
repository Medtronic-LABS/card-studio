"""A4-A6: final train with the VAL-chosen config, ONE test evaluation, pooled 5-fold page CV, current-model eval,
plausibility flags, report.md. Run after step2_select.py."""
import json, random, collections
from multiprocessing import Pool
import numpy as np, onnxruntime as ort
from common import *
import plaus

def load():
    S = json.load(open(HERE / "split.json")); rows = read_labels(HERE / "merged_labels.csv")
    return S, rows

def cv_job(args):
    fold, cfgname = args
    import torch; from train import CONFIGS, finetune, predict_torch, torch as _t
    _t.set_num_threads(2)
    S, rows = load(); pre = torch.load(HERE / "pretrained.pt")
    test_pages = set(S["cv_folds"][fold]); rest = sorted({r["page"] for r in rows} - test_pages)
    rng = random.Random(1000 + fold); vp = set(rng.sample(rest, max(4, round(len(rest) / 7))))
    tr = [r for r in rows if r["page"] in rest and r["page"] not in vp]; va = [r for r in rows if r["page"] in vp]
    te = [r for r in rows if r["page"] in test_pages]
    assert not ({r["page"] for r in tr} & {r["page"] for r in te}) and not ({r["page"] for r in va} & {r["page"] for r in te})
    m, info = finetune(pre, tr, va, CONFIGS[cfgname], seed=0, log=lambda *_: None)
    out = [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=p, conf=c, fold=fold)
           for (p, c), r in zip(predict_torch(m, te), te)]
    return out, dict(fold=fold, best_epoch=info["best_epoch"], val_exact=info["val_exact"], n_test=len(te), n_train=len(tr), n_val=len(va))

def current_model_preds(rows):
    s = ort.InferenceSession(str(WORK / "crnn.onnx"))
    x = np.stack([to_input(cell_image(r)) for r in rows])[:, None]
    lg = s.run(None, {"image": x})[0]
    return [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=p, conf=c)
            for i, r in enumerate(rows) for p, c in [decode_app(lg[:, i, :], r["row"])]]

def table(recs, order=None):
    by = collections.defaultdict(list)
    for r in recs: by[r["row"]].append(r)
    out = {}
    for f, rs in by.items():
        n = len(rs); k = sum(r["pred"] == r["label"] for r in rs)
        ed = sum(edit_distance(r["pred"], r["label"]) for r in rs); ln = sum(len(r["label"]) for r in rs)
        lo, hi = wilson(k, n); out[f] = dict(n=n, k=k, exact=k / n, cer=ed / ln, lo=lo, hi=hi)
    n = len(recs); k = sum(r["pred"] == r["label"] for r in recs)
    ed = sum(edit_distance(r["pred"], r["label"]) for r in recs); ln = sum(len(r["label"]) for r in recs)
    lo, hi = wilson(k, n); out["ALL"] = dict(n=n, k=k, exact=k / n, cer=ed / ln, lo=lo, hi=hi)
    return out

def flag_stats(recs):
    def go(rs):
        n = len(rs); fl = [plaus.check_cell(r["row"], r["pred"], r["conf"]) is not None for r in rs]
        wrong = [r["pred"] != r["label"] for r in rs]
        unf = [i for i in range(n) if not fl[i]]
        caught = sum(1 for i in range(n) if wrong[i] and fl[i]); nw = sum(wrong)
        k = sum(1 for i in unf if not wrong[i])
        lo, hi = wilson(k, len(unf))
        return dict(n=n, flagged=sum(fl), flag_rate=sum(fl) / n if n else float("nan"), wrong=nw, wrong_caught=caught,
                    unflagged=len(unf), unflagged_correct=k, unflagged_acc=k / len(unf) if unf else float("nan"), ulo=lo, uhi=hi,
                    wrong_unflagged=nw - caught)
    marks = [r for r in recs if r["row"] in plaus.MARKS]; nums = [r for r in recs if r["row"] not in plaus.MARKS]
    return dict(all=go(recs), marks=go(marks), numbers_dates=go(nums))

def main():
    S, rows = load(); sel = json.load(open(HERE / "selection.json")); cfgname = sel["chosen"]
    import torch; from train import CONFIGS, finetune, predict_torch
    torch.set_num_threads(4)
    pre = torch.load(HERE / "pretrained.pt")
    tr = [r for r in rows if r["page"] in S["train"]]; va = [r for r in rows if r["page"] in S["val"]]
    te = [r for r in rows if r["page"] in S["test"]]
    assert not (set(S["train"]) & set(S["test"])) and not (set(S["val"]) & set(S["test"]))
    # start CV in parallel with the final model
    pool = Pool(5); cv_async = pool.map_async(cv_job, [(f, cfgname) for f in range(5)])
    model, info = finetune(pre, tr, va, CONFIGS[cfgname], seed=0)
    torch.save(model.state_dict(), HERE / "final_model.pt")
    new_test = [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=p, conf=c)
                for (p, c), r in zip(predict_torch(model, te), te)]   # the single TEST evaluation
    cv_res = cv_async.get(); pool.close()
    cv = [x for o, _ in cv_res for x in o]; cv_info = [i for _, i in cv_res]
    assert len(cv) == len(rows) and len({(x["page"], x["row"], x["col"]) for x in cv}) == len(rows)
    cur_all = current_model_preds(rows); te_keys = {(r["page"], r["row"], r["col"]) for r in te}
    cur_test = [x for x in cur_all if (x["page"], x["row"], x["col"]) in te_keys]
    res = dict(config=cfgname, final=info | {"hist": None}, cv_info=cv_info,
               tables=dict(new_test=table(new_test), cur_test=table(cur_test), cv=table(cv), cur_all=table(cur_all)),
               flags=dict(new_test=flag_stats(new_test), cur_test=flag_stats(cur_test), cv=flag_stats(cv), cur_all=flag_stats(cur_all)),
               split_sizes={k: len(S[k]) for k in ("train", "val", "test")}, n_test=len(te), n_train=len(tr), n_val=len(va))
    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    json.dump(dict(new_test=new_test, cv=cv, cur_all=cur_all), open(HERE / "predictions.json", "w"), indent=1)
    json.dump(info["hist"], open(HERE / "final_history.json", "w"))
    write_report(res, S, sel)

def fmt(t, f):
    if f not in t: return "-"
    d = t[f]; return f"{d['k']}/{d['n']} {d['exact']*100:.0f}% [{d['lo']*100:.0f}-{d['hi']*100:.0f}]"

def write_report(res, S, sel):
    T = res["tables"]; order = ["visit_date", "bp", "weight_kg", "pulse", "fundal_height_cm", "hemoglobin", "glucose", "fetal_movement",
                                "edema", "urine_albumin", "urine_bilirubin", "usg", "tt_vaccine", "iron_tablet", "calcium_tablet",
                                "height_cm", "temp_f", "gestational_month"]
    L = ["# Phase A: held-out retrain test", ""]
    L += ["## Setup", "",
          "- Labels: `merged_labels.csv` = copy of `work/labels.csv` (`work/labels_new.csv` absent, no conflicts). 369 labeled cells, 49 pages.",
          "- **Split unit is the PAGE.** Nothing in the data identifies the writer, so pages from the same writer may sit in different splits "
          "(writer leakage possible; the held-out numbers could be optimistic for new writers). Cells on one page are correlated, so the Wilson CIs below "
          "(which treat cells as independent) are narrower than the true uncertainty.",
          f"- Split (seed {S['seed']}, `split.json`): train {len(S['train'])} pages / {res['n_train']} cells, val {len(S['val'])} / {res['n_val']}, "
          f"test {len(S['test'])} / {res['n_test']}. Asserted no page in two splits.",
          f"- Same architecture, charset, field masks and NumtaDB synthetic pretraining (15 epochs, 4000 sequences) as `train_crnn.ipynb`. "
          f"Fine-tune: Adam, batch 64, field-balanced sampler, early stop on val (exact, then CER; patience 80, max 300).",
          f"- Settings chosen on VAL only (3 configs x 3 seeds, `selection.json`): chosen **{res['config']}**. "
          + "; ".join(f"{n}: val exact {v['val_exact']*100:.1f}%, CER {v['val_cer']:.3f}" for n, v in sel['summary'].items()) + ".",
          f"- Final model: seed 0, best epoch {res['final']['best_epoch']} (ran {res['final']['epochs_run']}), "
          f"{res['final']['mean_epoch_sec']:.2f} s/epoch (CPU, 4 threads), val exact {res['final']['val_exact']*100:.1f}%.",
          "- TEST was evaluated once, after the settings were fixed. Exact = full-string match after masked greedy decode. CER = total edit distance / total label length. "
          "CI = Wilson 95%, shown as [lo-hi]%.",
          "- Decoding and confidence follow `anc_reader.dart` (masked greedy CTC; confidence = lowest per-character masked-softmax probability).",
          "- Current model (`work/crnn.onnx`): `model_meta.json` has no training-page list, so I assume it trained on **all** labeled pages (notebook used a random 10% of cells as val, so ~90% of cells). "
          "Its numbers are **training-writer numbers, not accuracy**. Same stored crops, same preprocessing as training (not the app's line-removal/flatten path).", ""]
    L += ["## A4. Per-field result on the held-out TEST pages", "",
          "| field | NEW (held-out) exact [95% CI] | NEW CER | CURRENT on same cells (trained on them) | n |", "|---|---|---|---|---|"]
    for f in order + ["ALL"]:
        if f in T["new_test"]:
            n = T["new_test"][f]
            L.append(f"| {f} | {fmt(T['new_test'], f)} | {n['cer']:.3f} | {fmt(T['cur_test'], f)} (CER {T['cur_test'][f]['cer']:.3f}) | {n['n']}{' **n<10: no conclusion**' if f != 'ALL' and n['n'] < 10 else ''} |")
        else:
            L.append(f"| {f} | no test cells | | | 0 |")
    L += ["", f"Overall (all {res['n_test']} test cells, micro): NEW {T['new_test']['ALL']['exact']*100:.1f}%. "
          "Every field has n<10 on test; per-field rows show direction only. The overall row is the most reliable number here.", ""]
    L += ["## Pooled 5-fold CV by page (supplement)", "",
          "Every labeled cell gets exactly one held-out prediction (model trained on the other 4 folds' pages, minus ~1/7 of them as early-stop val). "
          "Settings reused from the val choice above, one seed per fold. Pooled numbers; not a TEST result.", "",
          "| field | pooled CV exact [95% CI] | CER | CURRENT on all cells (training writers) |", "|---|---|---|---|"]
    for f in order + ["ALL"]:
        if f in T["cv"]:
            L.append(f"| {f} | {fmt(T['cv'], f)}{' (n<10)' if f != 'ALL' and T['cv'][f]['n'] < 10 else ''} | {T['cv'][f]['cer']:.3f} | {fmt(T['cur_all'], f)} |")
    L += ["", "Fold detail: " + "; ".join(f"fold{i['fold']}: n_test {i['n_test']}, best epoch {i['best_epoch']}, val exact {i['val_exact']*100:.0f}%" for i in res["cv_info"]), ""]
    L += ["## A5. Plausibility flags (port of `plausibility.dart`)", "",
          "Confidence cutoffs 0.7 numbers/dates/BP, 0.5 marks; ranges, BP order, date validity, mark set as in the Dart file. Blank reads are never flagged. "
          "'Wrong reads caught' = wrong cells that were flagged / all wrong cells. Empty reads of non-empty labels count as wrong and unflagged.", "",
          "| set | cells | flagged (rate) | wrong | wrong caught | unflagged | unflagged accuracy [95% CI] |", "|---|---|---|---|---|---|---|"]
    names = {"new_test": "NEW, TEST (held-out)", "cv": "NEW, pooled CV (held-out)", "cur_test": "CURRENT, same TEST cells (trained on)", "cur_all": "CURRENT, all cells (trained on)"}
    for key in ["new_test", "cv", "cur_test", "cur_all"]:
        for g, gn in [("all", "all"), ("marks", "marks"), ("numbers_dates", "numbers+dates+BP")]:
            s = res["flags"][key][g]
            L.append(f"| {names[key]} / {gn} | {s['n']} | {s['flagged']} ({s['flag_rate']*100:.0f}%) | {s['wrong']} | {s['wrong_caught']}/{s['wrong']} | {s['unflagged']} | "
                     f"{s['unflagged_correct']}/{s['unflagged']} {s['unflagged_acc']*100:.0f}% [{s['ulo']*100:.0f}-{s['uhi']*100:.0f}] |")
    L += ["", "## What this does and does not show", "",
          "- Only the NEW rows on TEST and in pooled CV are held-out accuracy. CURRENT rows are training-writer numbers.",
          f"- TEST is {res['n_test']} cells from {len(S['test'])} pages; every field has n<10. No per-field conclusion is possible for any field, and none at all for bp, height_cm, temp_f, gestational_month "
          "(bp has 0 test cells: all 7 BP cells are on train pages; CV is the only held-out BP evidence, n=7).",
          "- Writer identity unknown, so held-out here means held-out pages, not held-out writers.",
          "- Not verified: the app's real photo path (line removal, flatten, SIFT alignment); the current model's true training set; variance across seeds on TEST (one seed, by design).",
          "- Seeds/config were chosen on a 7-page val set (57 cells); that choice is itself noisy.", ""]
    open(HERE / "report.md", "w").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    main()
