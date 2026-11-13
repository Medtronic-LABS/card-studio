"""Does a larger reader read better? A controlled comparison on one card's labeled cells. Local only; needs torch.

Three readers, same input (48x192 gray), same 48 output steps, same CTC loss, same per-field class masks, same decoding:
  base      the shipped reader: 4 conv blocks, one BiLSTM of 128 per direction (0.54M parameters, 2.2 MB as float32)
  big_crnn  wider convolutions and two BiLSTM layers of 288 per direction
  attn      the same kind of convolution stem, then a transformer encoder (attention across the 48 steps) with a CTC head
Each is pretrained on synthetic digit strings, dates and blood pressures stitched from handwritten NumtaDB digits, under two conditions:
  small     4,000 sequences, 15 epochs: what the shipped reader got
  large     LARGE_N sequences, LARGE_EPOCHS epochs: more of the same synthetic data
then fine-tuned and scored with the Studio trainer's page-split 5-fold cross-validation, on the SAME folds for every reader, so any two can
be compared cell by cell. Every reader is trained with the same recipe (including the baseline, re-run here), because this recipe adds a
learning-rate warm-up and gradient clipping to what the original reader used, which a larger model needs.

What this can and cannot say. It uses the ANC card's 369 labeled cells from few writers; the folds are by page, not by writer. A reader that
gains here may be fitting those writers. Differences of a few points are inside the noise (see the intervals); `paired` counts the cells one
reader gets right and the other wrong, with a sign-test p-value. Latency is PyTorch on this machine's CPU, one cell at a time: it is not a
phone number. The larger readers here (BigCRNN, the attention encoder) are not converted to TensorFlow Lite; networks described by archnet.py specs are (tflite_export.py).

    python capacity.py run [--quick] [--variants base,big_crnn,attn] [--conds small,large]     results in studio/capacity/ (gitignored)
    python capacity.py report                                                                 print the table from stored results
"""
import argparse, collections, json, math, random, statistics, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "capacity"
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(ROOT / "cardpack")); sys.path.insert(0, str(HERE))
import common as C
import cards

CFG = "light_lr3e-4"
SEED = 20261111
LARGE_N, LARGE_EPOCHS, SMALL_N, SMALL_EPOCHS = 30000, 4, 4000, 15
PRE_BATCH, PRE_LR, WARMUP, CLIP = 64, 1e-3, 100, 5.0
FT_MAX_EPOCHS, FT_PATIENCE = 150, 40
POOLS = [(2, 2), (2, 2), (2, 1), (6, 1)]          # width 192 -> 48 steps, height 48 -> 1, as in the shipped reader


class BigCRNN(nn.Module):
    def __init__(self, n, ch=(64, 128, 256, 256), hid=288, layers=2, drop=0.3):
        super().__init__(); L, i = [], 1
        for c, p in zip(ch, POOLS): L += [nn.Conv2d(i, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(p)]; i = c
        self.cnn = nn.Sequential(*L); self.rnn = nn.LSTM(i, hid, num_layers=layers, bidirectional=True, batch_first=True, dropout=0.2)
        self.do = nn.Dropout(drop); self.fc = nn.Linear(2 * hid, n)

    def forward(self, x):
        o, _ = self.rnn(self.cnn(x).squeeze(2).permute(0, 2, 1)); return self.fc(self.do(o)).permute(1, 0, 2)


class AttnCTC(nn.Module):
    def __init__(self, n, ch=(64, 128, 256, 256), d=288, layers=4, heads=6, drop=0.1):
        super().__init__(); L, i = [], 1
        for c, p in zip(ch, POOLS): L += [nn.Conv2d(i, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(p)]; i = c
        self.cnn = nn.Sequential(*L); self.proj = nn.Linear(i, d); self.pos = nn.Parameter(torch.randn(1, 48, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d, n)

    def forward(self, x):
        f = self.proj(self.cnn(x).squeeze(2).permute(0, 2, 1)); return self.fc(self.enc(f + self.pos[:, :f.shape[1]])).permute(1, 0, 2)


def make(variant, n=None):
    n = n or len(C.CLASSES)
    return {"base": lambda: C.CRNN(n), "big_crnn": lambda: BigCRNN(n), "attn": lambda: AttnCTC(n)}[variant]()


def facts(variant):
    m = make(variant).eval(); p = sum(q.numel() for q in m.parameters()); x = torch.randn(1, 1, 48, 192); torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(5): m(x)
        t = time.perf_counter()
        for _ in range(40): m(x)
        ms = (time.perf_counter() - t) / 40 * 1000
    return {"params_m": round(p / 1e6, 2), "fp32_mb": round(p * 4 / 1e6, 1), "int8_mb_if_quantized": round(p / 1e6, 1), "cpu_ms_per_cell_pytorch_1thread": round(ms, 1)}


# ---------------------------------------------------------------- synthetic pretraining
def synth(n):
    """n labeled synthetic cells as (uint8 images [n,48,192], label index lists, field names). Same generator as the shipped reader's pretraining."""
    import train as T
    imgs, pool = T._glyphs(); g = random.Random(42); fmap = {"numeric": "pulse", "date": "visit_date", "bp": "bp"}
    X = np.empty((n, C.IMG_H, C.IMG_W), np.uint8); Y, Fd = [], []
    for i in range(n):
        ft = g.choice(["numeric"] * 3 + ["date", "bp"]); lab = T._rand_label(ft, g); im = T._compose(lab, g, imgs, pool); h, w = im.shape
        im = cv2.resize(im, (min(C.IMG_W, max(1, int(w * C.IMG_H / h))), C.IMG_H), interpolation=cv2.INTER_AREA)
        c = np.full((C.IMG_H, C.IMG_W), 255, np.uint8); c[:, : im.shape[1]] = im; X[i] = c; Y.append(C.encode_label(lab)); Fd.append(fmap[ft])
    return X, Y, Fd


def pretrain(variant, n, epochs, seed=0, log=print):
    torch.set_num_threads(10); torch.manual_seed(seed); X, Y, Fd = synth(n); model = make(variant); ctc = nn.CTCLoss(blank=C.BLANK_IDX, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=PRE_LR); idx = list(range(n)); step = 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); random.Random(ep).shuffle(idx); tl = nb = 0
        for i in range(0, n, PRE_BATCH):
            b = idx[i:i + PRE_BATCH]
            for g in opt.param_groups: g["lr"] = PRE_LR * min(1.0, (step + 1) / WARMUP)
            x = torch.from_numpy(C.to_input(X[b]).astype(np.float32)).unsqueeze(1); tg = torch.cat([torch.tensor(Y[j]) for j in b]); tlen = torch.tensor([len(Y[j]) for j in b])
            lg = model(x).clone()
            for k, j in enumerate(b): lg[:, k, ~C.mask_t(Fd[j])] = -1e9
            lp = F.log_softmax(lg, -1); loss = ctc(lp, tg, torch.full((len(b),), lp.size(0), dtype=torch.long), tlen)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); tl += loss.item(); nb += 1; step += 1
        log(f"  pretrain {variant} epoch {ep}/{epochs} loss {tl / nb:.4f} {time.time() - t0:.0f}s")
    return model.state_dict()


# ---------------------------------------------------------------- cross-validation (the Studio trainer's folds and fold recipe)
def cv_job(a):
    variant, cond, fold, folds, merged, max_epochs, patience, weights = a
    import train as T
    torch.set_num_threads(2); T.CRNN = lambda n: make(variant, n)               # train.finetune builds its model through this name
    rows = C.read_labels(merged); pre = torch.load(weights)
    test_pages = set(folds[fold]); rest = sorted({r["page"] for r in rows} - test_pages)
    vp = set(random.Random(1000 + fold).sample(rest, max(4, round(len(rest) / 7))))
    tr = [r for r in rows if r["page"] in rest and r["page"] not in vp]; va = [r for r in rows if r["page"] in vp]; te = [r for r in rows if r["page"] in test_pages]
    assert not ({r["page"] for r in tr + va} & test_pages)
    m, info = T.finetune(pre, tr, va, T.CONFIGS[CFG], seed=0, max_epochs=max_epochs, patience=patience, log=lambda *_: None)
    return fold, [dict(page=r["page"], row=r["row"], col=r["col"], label=r["label"], pred=p, conf=c) for (p, c), r in zip(T.predict_torch(m, te), te)], info["best_epoch"]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(preds, kinds):
    """Exact match and character error rate overall and per field kind. Intervals are Wilson and optimistic (cells on a page are correlated)."""
    out = {}
    for name, ks in (("all", None), ("marks", ("mark",)), ("numbers", ("number",)), ("dates", ("date",)), ("bp", ("bp",)), ("numbers+dates+bp", ("number", "date", "bp"))):
        rs = [r for r in preds if ks is None or kinds.get(r["row"]) in ks]
        if not rs: continue
        k = sum(r["pred"] == r["label"] for r in rs); lo, hi = wilson(k, len(rs)); ed = sum(C.edit_distance(r["pred"], r["label"]) for r in rs); ln = sum(len(r["label"]) for r in rs)
        out[name] = {"n": len(rs), "correct": k, "exact": round(k / len(rs), 4), "lo": round(lo, 4), "hi": round(hi, 4), "cer": round(ed / max(1, ln), 4)}
    return out


def sign_test(a, b):
    """Cells reader a gets right and b wrong (and the reverse); two-sided exact sign-test p-value on the disagreements."""
    ka = {(r["page"], r["row"], r["col"]): r["pred"] == r["label"] for r in a}; kb = {(r["page"], r["row"], r["col"]): r["pred"] == r["label"] for r in b}
    only_a = sum(ka[k] and not kb[k] for k in ka); only_b = sum(kb[k] and not ka[k] for k in ka); n = only_a + only_b
    if n == 0: return {"only_first": 0, "only_second": 0, "p": 1.0}
    k = min(only_a, only_b); p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_first": only_a, "only_second": only_b, "p": round(p, 4)}


def run(variants, conds, quick=False):
    import trainer
    out = OUT / ("quick" if quick else "full"); OUT.mkdir(exist_ok=True); out.mkdir(exist_ok=True); t0 = time.time()
    def status(stage, detail=""): (out / "status.json").write_text(json.dumps({"stage": stage, "detail": detail, "elapsed": round(time.time() - t0)}))
    try:
        merged = out / "merged_labels.csv"; rows = trainer.merge_labels(merged, cards.LEGACY); pages = sorted({r["page"] for r in rows}); folds = trainer.make_folds(rows)
        kinds = {f["key"]: f["kind"] for f in cards.fields(cards.LEGACY)}; use_folds = list(range(2 if quick else 5))
        n_small, ep_small, n_large, ep_large, mx, pat = (200, 1, 400, 1, 3, 3) if quick else (SMALL_N, SMALL_EPOCHS, LARGE_N, LARGE_EPOCHS, FT_MAX_EPOCHS, FT_PATIENCE)
        sizes = {"small": (n_small, ep_small), "large": (n_large, ep_large)}; results = {}
        for variant in variants:
            for cond in conds:
                key = f"{variant}_{cond}"; rf = out / f"cv_{key}.json"
                if rf.exists(): results[key] = json.loads(rf.read_text()); continue
                pw = out / f"pre_{key}.pt"          # in the run's own folder, so a quick trial can never leak its weights into the full run
                if not pw.exists():
                    status("pretrain", f"{key}: {sizes[cond][0]} sequences x {sizes[cond][1]} epochs"); tp = time.time()
                    torch.save(pretrain(variant, *sizes[cond], log=lambda m: status("pretrain", f"{key}: {m.strip()}")), pw); ptime = time.time() - tp
                else: ptime = None
                status("finetune", f"{key}: cross-validation, {len(use_folds)} folds"); tf = time.time(); preds, best = [], []
                with Pool(len(use_folds)) as pool:
                    for fold, p, b in pool.imap_unordered(cv_job, [(variant, cond, f, folds, str(merged), mx, pat, str(pw)) for f in use_folds]): preds += p; best.append(b)
                results[key] = {"variant": variant, "cond": cond, "synthetic": {"sequences": sizes[cond][0], "epochs": sizes[cond][1]}, "folds": len(use_folds), "best_epochs": sorted(best),
                                "score": score(preds, kinds), "preds": preds, "pretrain_minutes": None if ptime is None else round(ptime / 60, 1), "finetune_minutes": round((time.time() - tf) / 60, 1)}
                rf.write_text(json.dumps(results[key]))
        merged.unlink(missing_ok=True)          # labels were only needed by the workers
        doc = {"made": time.strftime("%Y-%m-%d %H:%M"), "quick": quick, "card": cards.LEGACY, "labeled_cells": len(rows), "labeled_pages": len(pages), "folds": len(use_folds),
               "recipe": {"finetune_config": CFG, "finetune_max_epochs": mx, "finetune_patience": pat, "pretrain_lr": PRE_LR, "warmup_steps": WARMUP, "grad_clip": CLIP, "synthetic": sizes},
               "models": {v: facts(v) for v in variants}, "results": {k: {a: b for a, b in v.items() if a != "preds"} for k, v in results.items()}, "paired": {}, "minutes": round((time.time() - t0) / 60, 1)}
        keys = list(results)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]: doc["paired"][f"{a} vs {b}"] = sign_test(results[a]["preds"], results[b]["preds"])
        (out / "summary.json").write_text(json.dumps(doc, indent=1)); status("done", "summary written"); return doc
    except Exception:
        status("error", traceback.format_exc()[-1500:]); raise


def show(doc):
    print(f"{doc['labeled_cells']} cells, {doc['labeled_pages']} pages, {doc['folds']} folds, {doc['minutes']} min{'  (QUICK RUN: not a result)' if doc['quick'] else ''}")
    for v, f in doc["models"].items(): print(f"  {v:9s} {f['params_m']}M params, {f['fp32_mb']} MB fp32, {f['cpu_ms_per_cell_pytorch_1thread']} ms/cell (PyTorch CPU, 1 thread)")
    print(f"{'reader':22s} {'all':>14s} {'marks':>14s} {'numbers+dates+bp':>18s} {'dates':>10s} {'CER':>6s}")
    for k, r in doc["results"].items():
        s = r["score"]; cell = lambda g: f"{s[g]['exact'] * 100:4.0f}% [{s[g]['lo'] * 100:.0f}-{s[g]['hi'] * 100:.0f}]" if g in s else "-"
        print(f"{k:22s} {cell('all'):>14s} {cell('marks'):>14s} {cell('numbers+dates+bp'):>18s} {s.get('dates', {}).get('correct', '-')}/{s.get('dates', {}).get('n', '-'):>3} {s['all']['cer']:>6.2f}")
    print("paired (cells only the first / only the second gets right, sign-test p):")
    for k, v in doc["paired"].items(): print(f"  {k}: {v['only_first']} / {v['only_second']}  p={v['p']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "report"]); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", default="base,big_crnn,attn"); ap.add_argument("--conds", default="small,large"); a = ap.parse_args()
    if a.cmd == "run": show(run(a.variants.split(","), a.conds.split(","), a.quick))
    else: show(json.loads((OUT / ("quick" if a.quick else "full") / "summary.json").read_text()))
