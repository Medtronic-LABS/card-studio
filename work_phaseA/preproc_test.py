"""Does colour/gray preprocessing change reading accuracy? Fixed model (final_model.pt), colour aligned pages, simulated bad photos.

Variants are applied to the whole colour page (as the app does for its flatten step), cells are cut with the same boxes
work/manifest.csv recorded, and the stored model reads them. Nothing is retrained and nothing in work/ is written.
Protocol: sweep every variant on the VAL pages; then confirm baseline vs the best variant on the TEST pages once.
Run: python preproc_test.py sweep     |     python preproc_test.py confirm <variant>
"""
import csv, json, sys, zipfile
from pathlib import Path
import cv2, numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

PACK = HERE.parent / "cardpack" / "out" / "anc_v4.cardpack"
SEEDS = (0, 1, 2)
S = json.load(open(HERE / "split.json"))
LABELS = C.read_labels(HERE / "merged_labels.csv")
MAN = {(r["page"], r["row"], r["col"]): r for r in csv.DictReader(open(C.WORK / "manifest.csv"))}
FLAT_K = json.loads(zipfile.ZipFile(PACK).read("manifest.json"))["preprocess"]["flatten_kernel"]


def load_model():
    m = C.CRNN(len(C.CLASSES)); m.load_state_dict(torch.load(HERE / "final_model.pt")); m.eval(); return m


# ---------- simulated bad photos (applied to the colour page) ----------
def degrade(bgr, cond, seed):
    rng = np.random.RandomState(seed); im = bgr.astype(np.float32); h, w = im.shape[:2]
    if cond in ("cast", "combined"):                       # warm / yellow light: blue down, red up
        im = im * np.float32([0.78, 0.92, 1.08])
    if cond in ("shadow", "combined"):                     # uneven light, one side darker
        th = rng.uniform(0, 2 * np.pi); gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        t = np.clip((np.cos(th) * gx + np.sin(th) * gy + 1) / 2, 0, 1)          # 0 on one side of the page, 1 on the other
        im = im * (0.55 + 0.45 * t)[..., None]
        im = cv2.GaussianBlur(im, (0, 0), 1.0)
    if cond in ("lowcontrast", "combined"):                # washed out: compress towards grey, add sensor noise
        im = 150 + (im - 128) * 0.55 + rng.normal(0, 7, im.shape).astype(np.float32)
    return np.clip(im, 0, 255).astype(np.uint8)


CONDS = ["scan", "cast", "shadow", "lowcontrast", "combined"]


# ---------- preprocessing variants: colour page -> gray page ----------
def gray(b): return cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)


def flatten(g):                                           # the app's anc_reader._flatten
    k = FLAT_K
    bg = cv2.GaussianBlur(cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))), (k, k), 0)
    return cv2.divide(g, bg, scale=255 / 0.8)


_REF = None
def reinhard_ref():
    """Mean/std per Lab channel over 10 clean TRAIN pages: the reference Reinhard matches to."""
    global _REF
    if _REF is None:
        labs = [cv2.cvtColor(cv2.imread(str(C.WORK / "aligned" / f"{p}.png")), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64) for p in S["train"][:10]]
        a = np.concatenate(labs); _REF = (a.mean(0), a.std(0))
    return _REF


def reinhard(b):
    lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float64); mt, st = reinhard_ref()
    ms, ss = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    return gray(cv2.cvtColor(np.clip((lab - ms) / ss * st + mt, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))


VARIANTS = {
    "gray (as trained)":       lambda b: gray(b),
    "gray + flatten (app)":    lambda b: flatten(gray(b)),
    "min channel (ink)":       lambda b: b.min(axis=2),
    "min channel + flatten":   lambda b: flatten(b.min(axis=2)),
    "Reinhard -> gray":        lambda b: reinhard(b),
    "Reinhard + flatten":      lambda b: flatten(reinhard(b)),
    "CLAHE":                   lambda b: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray(b)),
    "gray + cell stretch":     lambda b: gray(b),            # stretch happens per cell, below
    "flatten + cell stretch":  lambda b: flatten(gray(b)),
}
STRETCH = {"gray + cell stretch", "flatten + cell stretch"}


def stretch(crop):
    lo, hi = np.percentile(crop, 2), np.percentile(crop, 98)
    return crop if hi - lo < 20 else np.clip((crop.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def to_model_image(crop):
    h, w = crop.shape
    im = cv2.resize(crop, (min(C.IMG_W, max(1, int(w * C.IMG_H / h))), C.IMG_H), interpolation=cv2.INTER_AREA)
    out = np.full((C.IMG_H, C.IMG_W), 255, np.uint8); out[:, : im.shape[1]] = im
    return out


def evaluate(model, pages, variant, cond, seed, rows):
    imgs = {}
    for p in pages:
        b = degrade(cv2.imread(str(C.WORK / "aligned" / f"{p}.png")), cond, seed) if cond != "scan" else cv2.imread(str(C.WORK / "aligned" / f"{p}.png"))
        imgs[p] = VARIANTS[variant](b)
    C._cache.clear()
    for r in rows:
        m = MAN[(r["page"], r["row"], r["col"])]; crop = imgs[r["page"]][int(m["y0"]):int(m["y1"]), int(m["x0"]):int(m["x1"])]
        C._cache[(r["page"], r["row"], r["col"])] = to_model_image(stretch(crop) if variant in STRETCH else crop)
    out = C.predict_torch(model, rows); C._cache.clear()
    return [p == r["label"] for (p, _), r in zip(out, rows)]


def table(model, pages, title):
    rows = [r for r in LABELS if r["page"] in pages]; n = len(rows)
    print(f"\n{title}: {n} cells on {len(pages)} pages, {len(SEEDS)} degradation seeds (mean exact; cells are the unit; CI = Wilson 95% on the mean)", flush=True)
    res = {}
    for v in VARIANTS:
        line = []
        for c in CONDS:
            seeds = (0,) if c == "scan" else SEEDS
            acc = np.mean([np.mean(evaluate(model, pages, v, c, sd, rows)) for sd in seeds])
            lo, hi = C.wilson(int(round(acc * n)), n); res[(v, c)] = (acc, lo, hi)
            line.append(f"{100 * acc:5.1f} [{100 * lo:2.0f}-{100 * hi:2.0f}]")
        print(f"  {v:26s} " + "  ".join(line), flush=True)
    print(f"  {'':26s} " + "  ".join(f"{c:>12s}" for c in CONDS))
    return res, n


def check_baseline(model):
    """Gray page + stored-crop geometry must reproduce the stored crops and the reported numbers."""
    p = S["val"][0]; im = gray(cv2.imread(str(C.WORK / "aligned" / f"{p}.png"))); diffs = []
    for (pg, row, col), m in list(MAN.items()):
        if pg == p and len(diffs) < 40:
            a = im[int(m["y0"]):int(m["y1"]), int(m["x0"]):int(m["x1"])]; b = cv2.imread(str(C.CROPS / row / col / f"{p}.png"), 0)
            if a.shape == b.shape: diffs.append(float(np.abs(a.astype(int) - b.astype(int)).mean()))
    print(f"sanity: re-cut crops vs stored crops, mean abs pixel difference {np.mean(diffs):.3f} over {len(diffs)} cells (0 = identical)")


if __name__ == "__main__":
    torch.set_num_threads(4); model = load_model(); check_baseline(model)
    if sys.argv[1] == "sweep":
        res, n = table(model, set(S["val"]), "VAL sweep")
        json.dump({f"{v}|{c}": r for (v, c), r in res.items()}, open(HERE / "preproc_val.json", "w"))
    else:
        v = sys.argv[2]; pages = set(S["test"]); rows = [r for r in LABELS if r["page"] in pages]; n = len(rows)
        print(f"\nTEST confirmation, once: baseline vs {v!r} ({n} cells, {len(pages)} pages)")
        for c in CONDS:
            seeds = (0,) if c == "scan" else SEEDS
            out = {}
            for name in ("gray (as trained)", v):
                acc = np.mean([np.mean(evaluate(model, pages, name, c, sd, rows)) for sd in seeds]); lo, hi = C.wilson(int(round(acc * n)), n); out[name] = (acc, lo, hi)
            a, b = out["gray (as trained)"], out[v]
            print(f"  {c:12s} baseline {100*a[0]:5.1f} [{100*a[1]:2.0f}-{100*a[2]:2.0f}]   {v} {100*b[0]:5.1f} [{100*b[1]:2.0f}-{100*b[2]:2.0f}]   diff {100*(b[0]-a[0]):+5.1f} pp", flush=True)
