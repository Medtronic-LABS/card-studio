"""Does the second synthetic generator (synthgen.py) pretrain a better reader than the first? One controlled comparison. Local only; needs torch.

Same reader (the shipped 0.54M-parameter CRNN), same recipe, same folds, same fine-tuning as capacity.py. Only the synthetic pretraining
data differs:
  small   4,000 digit strings stitched on white squares, 15 epochs      (what the shipped reader got; from capacity.py)
  large   30,000 of the same kind, 4 epochs                             (from capacity.py: more of the same)
  v2      30,000 images from synthgen: field grammars, real empty-cell backgrounds, two-line layouts cut into their pieces, photocopy and
          phone damage, 4 epochs
`small` and `large` come from the stored capacity.py results, so this needs `python capacity.py run` to have finished for them (it still
runs and reports v2 alone when it has not). The comparison is paired: cells one reader gets right and the other wrong, with a sign-test p.

The ANC card has no two-line cells, so what is measured here is grammars, backgrounds and damage; the two-line layouts are in the
generator for the NCD and EPI cards and are not tested by this experiment. The empty-cell backgrounds are cut from a blank card estimated from
all of the card's scans, including the pages a fold tests on; that blank card contains no handwriting and no labels.

    python synthexp.py run [--quick] [--images 30000] [--epochs 4]       results in studio/capacity/v2/
    python synthexp.py report
"""
import argparse, json, random, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
import common as C
import capacity as K
import cards, subcells, synthgen as G

OUT = K.OUT / "v2"
BG_CARDS = ("anc", "epi_scan", "ncd_scan")


def register_masks():
    """Mask names for synthetic cells: the shipped reader's pretraining only needs "what may this kind of cell contain"."""
    m = cards.meta(); cl, bl = m["classes"], m["blank_idx"]
    for kind in G.KINDS: C.FIELD_ALLOWED[f"syn_{kind}"] = cards.allowed_idx(kind, cl, bl)
    for layout in ("stacked", "diagonal"):
        for suffix, idx in subcells.piece_masks(layout, cl, bl).items(): C.FIELD_ALLOWED[f"syn_{subcells.KIND_FOR[layout]}{suffix}"] = idx
    C._masks.clear()


def build(n_images, seed=7, log=print):
    """n_images labeled 48x192 training images (uint8), their label index lists and mask names, from synthgen over the cards' empty cells."""
    register_masks(); rng = random.Random(seed); pool = G.load_pool(); bgs = []
    for c in BG_CARDS:
        b = G.backgrounds(c); log(f"  backgrounds from {c}: {len(b)} clean cells"); bgs += b
    X = np.empty((n_images, C.IMG_H, C.IMG_W), np.uint8); Y, M = [], []; i = 0; t0 = time.time()
    while i < n_images:
        s = G.sample(rng, pool, bgs)
        for img, lab, suffix in G.pieces(s):
            if i >= n_images: break
            h, w = img.shape; im = cv2.resize(img, (min(C.IMG_W, max(1, int(w * C.IMG_H / h))), C.IMG_H), interpolation=cv2.INTER_AREA)
            c = np.full((C.IMG_H, C.IMG_W), 255, np.uint8); c[:, : im.shape[1]] = im; X[i] = c; Y.append(C.encode_label(lab))
            M.append(f"syn_{s['kind']}{suffix}"); i += 1
        if i % 5000 < 2 and i: log(f"  drew {i} of {n_images} images, {time.time() - t0:.0f}s")
    return X, Y, M


def pretrain(path, n_images, epochs, log=print):
    torch.set_num_threads(10); torch.manual_seed(0); X, Y, M = build(n_images, log=log); model = K.make("base"); ctc = nn.CTCLoss(blank=C.BLANK_IDX, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=K.PRE_LR); idx = list(range(len(Y))); step = 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); random.Random(ep).shuffle(idx); tl = nb = 0
        for i in range(0, len(idx), K.PRE_BATCH):
            b = idx[i:i + K.PRE_BATCH]
            for g in opt.param_groups: g["lr"] = K.PRE_LR * min(1.0, (step + 1) / K.WARMUP)
            x = torch.from_numpy(C.to_input(X[b]).astype(np.float32)).unsqueeze(1); tg = torch.cat([torch.tensor(Y[j]) for j in b]); tlen = torch.tensor([len(Y[j]) for j in b])
            lg = model(x).clone()
            for k, j in enumerate(b): lg[:, k, ~C.mask_t(M[j])] = -1e9
            lp = F.log_softmax(lg, -1); loss = ctc(lp, tg, torch.full((len(b),), lp.size(0), dtype=torch.long), tlen)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), K.CLIP); opt.step(); tl += loss.item(); nb += 1; step += 1
        log(f"  pretrain v2 epoch {ep}/{epochs} loss {tl / nb:.4f} {time.time() - t0:.0f}s")
    torch.save(model.state_dict(), path)


def run(n_images, epochs, quick=False):
    import trainer
    out = OUT / ("quick" if quick else "full"); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def status(stage, detail=""): (out / "status.json").write_text(json.dumps({"stage": stage, "detail": detail, "elapsed": round(time.time() - t0)}))
    try:
        merged = out / "merged_labels.csv"; rows = trainer.merge_labels(merged, cards.LEGACY); folds = trainer.make_folds(rows)
        kinds = {f["key"]: f["kind"] for f in cards.fields(cards.LEGACY)}; use = list(range(2 if quick else 5)); mx, pat = (3, 3) if quick else (K.FT_MAX_EPOCHS, K.FT_PATIENCE)
        pw = out / "pre_v2.pt"
        if not pw.exists(): status("pretrain", f"{n_images} images x {epochs} epochs"); pretrain(pw, n_images, epochs, log=lambda m: status("pretrain", m.strip()))
        status("finetune", f"cross-validation, {len(use)} folds"); preds, best = [], []; tf = time.time()
        with Pool(len(use)) as pool:
            for fold, p, b in pool.imap_unordered(K.cv_job, [("base", "v2", f, folds, str(merged), mx, pat, str(pw)) for f in use]): preds += p; best.append(b)
        merged.unlink(missing_ok=True)
        res = {"images": n_images, "epochs": epochs, "folds": len(use), "best_epochs": sorted(best), "score": K.score(preds, kinds), "finetune_minutes": round((time.time() - tf) / 60, 1)}
        doc = {"made": time.strftime("%Y-%m-%d %H:%M"), "quick": quick, "labeled_cells": len(rows), "v2": res, "paired": {}, "minutes": round((time.time() - t0) / 60, 1)}
        base = {c: json.loads((K.OUT / "full" / f"cv_base_{c}.json").read_text()) for c in ("small", "large") if (K.OUT / "full" / f"cv_base_{c}.json").exists()}
        doc["others"] = {f"base_{c}": {"score": r["score"], "synthetic": r["synthetic"]} for c, r in base.items()}
        if not quick:
            for c, r in base.items(): doc["paired"][f"v2 vs base_{c}"] = K.sign_test(preds, r["preds"])
        (out / "summary.json").write_text(json.dumps(doc, indent=1)); status("done", "summary written"); return doc
    except Exception:
        status("error", traceback.format_exc()[-1500:]); raise


def show(doc):
    print(f"{doc['labeled_cells']} cells, {doc['v2']['folds']} folds, {doc['minutes']} min{'  (QUICK RUN: not a result)' if doc['quick'] else ''}")
    print(f"{'pretraining':28s} {'all':>14s} {'marks':>14s} {'numbers+dates+bp':>18s} {'dates':>8s}")
    for name, s in [(f"v2 ({doc['v2']['images']} images)", doc["v2"]["score"])] + [(f"{k} ({v['synthetic']['sequences']} strings)", v["score"]) for k, v in doc["others"].items()]:
        cell = lambda g: f"{s[g]['exact'] * 100:4.0f}% [{s[g]['lo'] * 100:.0f}-{s[g]['hi'] * 100:.0f}]" if g in s else "-"
        print(f"{name:28s} {cell('all'):>14s} {cell('marks'):>14s} {cell('numbers+dates+bp'):>18s} {s.get('dates', {}).get('correct', '-')}/{s.get('dates', {}).get('n', '-')}")
    for k, v in doc["paired"].items(): print(f"  {k}: only v2 right {v['only_first']}, only the other right {v['only_second']}, p={v['p']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "report"]); ap.add_argument("--quick", action="store_true"); ap.add_argument("--images", type=int, default=30000); ap.add_argument("--epochs", type=int, default=4); a = ap.parse_args()
    if a.cmd == "run": show(run(300 if a.quick else a.images, 1 if a.quick else a.epochs, a.quick))
    else: show(json.loads((OUT / ("quick" if a.quick else "full") / "summary.json").read_text()))
