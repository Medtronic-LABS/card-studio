"""Does one digit reader serve several scripts? A controlled cross-script test on synthetic digit strings. Local only; needs torch.

Every label Studio stores is ASCII 0-9, so a Hindi, Kannada or Telugu digit glyph maps to the same ten classes as a Bangla one. If digits
from one script teach the reader something about another's, one reader can read the digits, dates and blood pressures of several scripts.
This tests that on isolated digit glyphs from public datasets, composed into strings of 1 to 4 digits:

  scripts     bangla      NumtaDB, 55k glyphs (data/bangla_digits), split 90/10 by position
              kannada     Kannada-MNIST (Zenodo 3359691, CC BY 4.0): 60,000 train glyphs; test = its 10,000 test glyphs
              devanagari  CMATERdb 3.2.1 numerals via MNIST-MIX: 2,400 train, 600 test
              telugu      CMATERdb 3.4.1 numerals via MNIST-MIX: 2,400 train, 600 test
              kannada_dig Dig-MNIST (10,240 glyphs by volunteers who do not write Kannada): a harder, out-of-domain test for Kannada only
  readers     the shipped 0.54M-parameter CRNN, trained from scratch on synthetic strings of the glyphs of the scripts in each condition
  conditions  bangla | kannada | bangla+kannada | bangla+devanagari+telugu | all four
  test        strings composed from the TEST glyphs of every script, so a glyph is never both trained and tested on; for each reader the
              scripts it never trained on show how far digits transfer, and the scripts it did train on show what it can reach

What this cannot say. Glyphs are isolated, 28x28, clean; real digits are written in strings and cells, by people, on forms. The glyph split is not
a writer split (these datasets do not give writers), so a script's own score is optimistic. Tamil has no public digit set that was found, and
real Hindi, Kannada and Tamil forms may use Latin digits anyway. A transfer result here is a reason to try real cells, not a replacement.

    python xscript.py run [--quick] [--strings 20000] [--epochs 4]     results in studio/capacity/xscript/
    python xscript.py report
"""
import argparse, json, math, random, sys, time, traceback
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(HERE))
import common as C
import capacity as K
import synthgen as G

OUT = K.OUT / "xscript"
DATA = ROOT / "data" / "digits_xscript"
SCRIPTS = ("bangla", "kannada", "devanagari", "telugu")
CONDITIONS = {"bangla": ("bangla",), "kannada": ("kannada",), "bangla+kannada": ("bangla", "kannada"), "bangla+devanagari+telugu": ("bangla", "devanagari", "telugu"),
              "all": ("bangla", "kannada", "devanagari", "telugu")}
TESTS = ("bangla", "kannada", "kannada_dig", "devanagari", "telugu")
MASK = "digits_only"


def _glyphs(X, y):
    """Digit images (any polarity, 28x28) -> dark strokes on white, tight to the digit, by digit."""
    X = np.asarray(X, np.uint8)
    if X.reshape(len(X), -1)[: 2000].mean() < 127: X = 255 - X          # MNIST-style sets are white digits on black
    by = {d: [] for d in range(10)}
    for im, lab in zip(X, np.asarray(y).reshape(-1)):
        ys, xs = np.where(im < 160)
        if len(ys) < 8: continue
        by[int(lab)].append(np.ascontiguousarray(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return G.Pool("", by)


def load_pools():
    """{name: (train pool, test pool)}. Raises OSError with what is missing."""
    pools = {}
    X = np.load(ROOT / "data" / "bangla_digits" / "numtaX_train.npy", allow_pickle=False); Y = np.load(ROOT / "data" / "bangla_digits" / "numtaY_train.npy", allow_pickle=False).argmax(1)
    imgs = (255 - np.clip(X.astype(np.int32) * 13, 0, 255)).astype(np.uint8); cut = int(len(imgs) * 0.9)
    pools["bangla"] = (_glyphs(imgs[:cut], Y[:cut]), _glyphs(imgs[cut:], Y[cut:]))
    mix = DATA / "mnist_mix"
    for name, fn in (("devanagari", "Devanagari_train_test.npz"), ("telugu", "Telugu_train_test.npz")):
        d = np.load(mix / fn, allow_pickle=False); pools[name] = (_glyphs(d["X_train"], d["y_train"]), _glyphs(d["X_test"], d["y_test"]))
    kd = DATA / "kannada_mnist" / "Kannada_MNIST_npz"
    ld = lambda p: np.load(kd / p, allow_pickle=False)["arr_0"]
    pools["kannada"] = (_glyphs(ld("Kannada_MNIST/X_kannada_MNIST_train.npz"), ld("Kannada_MNIST/y_kannada_MNIST_train.npz")), _glyphs(ld("Kannada_MNIST/X_kannada_MNIST_test.npz"), ld("Kannada_MNIST/y_kannada_MNIST_test.npz")))
    pools["kannada_dig"] = (None, _glyphs(ld("Dig_MNIST/X_dig_MNIST.npz"), ld("Dig_MNIST/y_dig_MNIST.npz")))
    return pools


def strings(n, pool_list, seed):
    """n synthetic digit strings (48x192 uint8 images, label text), each in the script of a pool drawn uniformly from pool_list."""
    rng = random.Random(seed); X = np.empty((n, C.IMG_H, C.IMG_W), np.uint8); labs = []
    for i in range(n):
        pool = pool_list[rng.randrange(len(pool_list))]; text = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([1, 2, 2, 3, 3, 4])))
        line = G._stroke(G._shear_turn(G.string_image(text, pool, rng, rng.randint(34, 46)), rng, .2, 3), rng); h = 80; w = max(line.shape[1] + 20, 120)
        cell = np.full((h, w), 248, np.uint8); G.lay(cell, line, w * rng.uniform(.45, .55), h / 2)
        cell = G.damage(cell, rng); hh, ww = cell.shape; im = cv2.resize(cell, (min(C.IMG_W, max(1, int(ww * C.IMG_H / hh))), C.IMG_H), interpolation=cv2.INTER_AREA)
        c = np.full((C.IMG_H, C.IMG_W), 255, np.uint8); c[:, : im.shape[1]] = im; X[i] = c; labs.append(text)
    return X, labs


def train_reader(X, labs, epochs, log=print):
    C.FIELD_ALLOWED[MASK] = sorted({C.BLANK_IDX} | {C.CHAR2IDX[c] for c in "0123456789"}); C._masks.clear()
    torch.set_num_threads(10); torch.manual_seed(0); model = K.make("base"); ctc = nn.CTCLoss(blank=C.BLANK_IDX, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=K.PRE_LR); idx = list(range(len(labs))); Y = [C.encode_label(t) for t in labs]; step = 0; mask = C.mask_t(MASK)
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); random.Random(ep).shuffle(idx); tl = nb = 0
        for i in range(0, len(idx), K.PRE_BATCH):
            b = idx[i:i + K.PRE_BATCH]
            for g in opt.param_groups: g["lr"] = K.PRE_LR * min(1.0, (step + 1) / K.WARMUP)
            x = torch.from_numpy(C.to_input(X[b]).astype(np.float32)).unsqueeze(1); tg = torch.cat([torch.tensor(Y[j]) for j in b]); tlen = torch.tensor([len(Y[j]) for j in b])
            lg = model(x).clone(); lg[:, :, ~mask] = -1e9
            lp = F.log_softmax(lg, -1); loss = ctc(lp, tg, torch.full((len(b),), lp.size(0), dtype=torch.long), tlen)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), K.CLIP); opt.step(); tl += loss.item(); nb += 1; step += 1
        log(f"    epoch {ep}/{epochs} loss {tl / nb:.4f} {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def evaluate(model, X, labs):
    model.eval(); preds = []
    for i in range(0, len(X), 256):
        lg = model(torch.from_numpy(C.to_input(X[i:i + 256]).astype(np.float32)).unsqueeze(1)).numpy()
        preds += [C.decode_app(lg[:, j], MASK)[0] for j in range(lg.shape[1])]
    k = sum(p == t for p, t in zip(preds, labs)); ed = sum(C.edit_distance(p, t) for p, t in zip(preds, labs)); ln = sum(len(t) for t in labs); lo, hi = C.wilson(k, len(labs))
    by_len = {L: [p == t for p, t in zip(preds, labs) if len(t) == L] for L in (1, 2, 3, 4)}
    return {"n": len(labs), "correct": k, "exact": round(k / len(labs), 4), "lo": round(lo, 4), "hi": round(hi, 4), "cer": round(ed / ln, 4),
            "by_length": {str(L): round(sum(v) / len(v), 3) for L, v in by_len.items() if v}}


def run(n_train, epochs, n_test, quick=False):
    out = OUT / ("quick" if quick else "full"); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    def status(stage, detail=""): (out / "status.json").write_text(json.dumps({"stage": stage, "detail": detail, "elapsed": round(time.time() - t0)}))
    try:
        status("data", "loading glyph pools"); pools = load_pools(); counts = {n: {"train": None if tr is None else sum(len(v) for v in tr.by_digit.values()), "test": sum(len(v) for v in te.by_digit.values())} for n, (tr, te) in pools.items()}
        tests = {t: strings(n_test, [pools[t][1]], seed=900 + i) for i, t in enumerate(TESTS)}; res = {}
        for cond, scripts in CONDITIONS.items():
            rf = out / f"{cond}.json"
            if rf.exists(): res[cond] = json.loads(rf.read_text()); continue
            status("train", f"{cond}: {n_train} strings x {epochs} epochs"); X, labs = strings(n_train, [pools[s][0] for s in scripts], seed=11)
            model = train_reader(X, labs, epochs, log=lambda m: status("train", f"{cond}: {m.strip()}"))
            res[cond] = {"scripts": list(scripts), "tests": {t: evaluate(model, *tests[t]) for t in TESTS}}; rf.write_text(json.dumps(res[cond]))
        doc = {"made": time.strftime("%Y-%m-%d %H:%M"), "quick": quick, "train_strings": n_train, "epochs": epochs, "test_strings": n_test, "glyphs": counts, "conditions": res, "minutes": round((time.time() - t0) / 60, 1)}
        (out / "summary.json").write_text(json.dumps(doc, indent=1)); status("done", "summary written"); return doc
    except Exception:
        status("error", traceback.format_exc()[-1500:]); raise


def show(doc):
    print(f"{doc['train_strings']} training strings x {doc['epochs']} epochs per reader, {doc['test_strings']} test strings per script, {doc['minutes']} min{'  (QUICK RUN: not a result)' if doc['quick'] else ''}")
    print("glyphs (train/test):", {k: f"{v['train']}/{v['test']}" for k, v in doc["glyphs"].items()})
    print(f"{'reader trained on':28s}" + "".join(f"{t:>14s}" for t in TESTS) + "   (exact match on strings; * = script not in training)")
    for cond, r in doc["conditions"].items():
        row = ""
        for t in TESTS:
            s = r["tests"][t]; seen = t.split("_")[0] in r["scripts"]; row += f"{s['exact'] * 100:>11.0f}%{'' if seen else ' *'}".rjust(14)
        print(f"{cond:28s}{row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "report"]); ap.add_argument("--quick", action="store_true"); ap.add_argument("--strings", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--test-strings", type=int, default=800); a = ap.parse_args()
    if a.cmd == "run": show(run(400 if a.quick else a.strings, 1 if a.quick else a.epochs, 100 if a.quick else a.test_strings, a.quick))
    else: show(json.loads((OUT / ("quick" if a.quick else "full") / "summary.json").read_text()))
