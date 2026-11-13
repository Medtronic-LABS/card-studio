"""A3: same architecture / charset / masking / NumtaDB synthetic pretraining as train_crnn.ipynb,
but fine-tuning data and early-stopping data are page-disjoint (split.json)."""
import collections, copy, json, random, time, sys
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from common import *

torch.set_num_threads(4)
DEVICE = "cpu"
NUMTA = ROOT / "data" / "bangla_digits"

# ---------- synthetic pretraining (copied logic from notebook cells 14) ----------
def _glyphs():
    X = np.load(NUMTA / "numtaX_train.npy"); Y = np.load(NUMTA / "numtaY_train.npy").argmax(1)
    imgs = (255 - np.clip(X.astype(np.int32) * 13, 0, 255)).astype(np.uint8)
    return imgs, {d: np.where(Y == d)[0] for d in range(10)}

def _symbol(ch, size):
    g = np.full((size, size), 255, np.uint8)
    if ch == ".": cv2.circle(g, (size // 2, int(size * .82)), max(1, size // 10), 0, -1)
    elif ch == "/": cv2.line(g, (int(size * .15), size - 1), (int(size * .85), 0), 0, max(2, size // 12))
    elif ch == "-": cv2.line(g, (int(size * .1), size // 2), (int(size * .9), size // 2), 0, max(2, size // 10))
    return g

def _compose(label, rng, imgs, pool):
    gh = rng.randint(34, 44); gl = []
    for ch in label:
        if ch.isdigit():
            g = imgs[rng.choice(pool[int(ch)])]; h = max(8, int(gh * rng.uniform(.85, 1.15)))
            gl.append(cv2.resize(g, (h, h), interpolation=cv2.INTER_AREA))
        else:
            gl.append(_symbol(ch, max(10, gh // 2)))
    gap = max(1, gh // 10); hm = max(g.shape[0] for g in gl); pads = []
    for g in gl:
        t = (hm - g.shape[0]) // 2; pads.append(np.pad(g, ((t, hm - g.shape[0] - t), (0, 0)), constant_values=255))
    row = pads[0]
    for g in pads[1:]:
        row = np.hstack([row, np.full((hm, rng.randint(gap, gap * 2)), 255, np.uint8), g])
    return row

def _rand_label(ft, rng):
    d = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))
    if ft == "numeric":
        s = rng.choice(["DD", "DDD", "D.D", "DD.D"])
        return {"DD": d(2), "DDD": d(3), "D.D": f"{d(1)}.{d(1)}"}.get(s) or f"{d(2)}.{d(1)}"
    if ft == "date":
        sep = rng.choice(["/", "."]); return f"{d(rng.choice([1, 2]))}{sep}{d(rng.choice([1, 2]))}{sep}{d(2)}"
    return f"{d(rng.choice([2, 3]))}/{d(2)}"

def pretrain(epochs=15, n=4000, batch=64, lr=1e-3, seed=0, log=print, arch=None):
    torch.manual_seed(seed)
    model = new_model(arch).to(DEVICE)
    imgs, pool = _glyphs(); g = random.Random(42); X, T, Fd = [], [], []
    fmap = {"numeric": "pulse", "date": "visit_date", "bp": "bp"}
    for _ in range(n):
        ft = g.choice(["numeric"] * 3 + ["date", "bp"]); lab = _rand_label(ft, g)
        im = _compose(lab, g, imgs, pool); h, w = im.shape
        im = cv2.resize(im, (min(IMG_W, max(1, int(w * IMG_H / h))), IMG_H), interpolation=cv2.INTER_AREA)
        c = np.full((IMG_H, IMG_W), 255, np.uint8); c[:, : im.shape[1]] = im
        X.append(torch.from_numpy(to_input(c)).unsqueeze(0)); T.append(torch.tensor(encode_label(lab))); Fd.append(fmap[ft])
    opt = torch.optim.Adam(model.parameters(), lr=lr); ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    idx = list(range(n))
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); random.Random(ep).shuffle(idx); tl = 0; nb = 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            x = torch.stack([X[j] for j in b]); tg = torch.cat([T[j] for j in b]); tl_ = torch.tensor([len(T[j]) for j in b])
            lg = model(x).clone()
            for k, j in enumerate(b): lg[:, k, ~mask_t(Fd[j])] = -1e9
            lp = F.log_softmax(lg, -1)
            loss = ctc(lp, tg, torch.full((len(b),), lp.size(0), dtype=torch.long), tl_)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        log(f"pretrain epoch {ep}/{epochs} loss {tl / nb:.4f} {time.time() - t0:.1f}s")
    return model.state_dict()

# ---------- fine-tuning ----------
CONFIGS = {
    # light aug = half the notebook's ranges; notebook aug kept as a reference config
    "light_lr1e-4": dict(lr=1e-4, aug=dict(rot=2, tx=2, ty=1, bright=.08, shift=8, blur=.15)),
    "light_lr3e-4": dict(lr=3e-4, aug=dict(rot=2, tx=2, ty=1, bright=.08, shift=8, blur=.15)),
    "nbaug_lr1e-4": dict(lr=1e-4, aug=dict(rot=3, tx=3, ty=2, bright=.15, shift=15, blur=.30)),
}

# ---------- every training value, settable (a preset is only the starting point) ----------
HP_DEFAULTS = dict(optimizer="adam", weight_decay=0.0, batch=64, schedule="none", step_every=50, step_gamma=0.5, grad_clip=0.0, balance=True, patience=80)
AUG_DEFAULTS = dict(rot=0.0, tx=0.0, ty=0.0, bright=0.0, shift=0.0, blur=0.0, noise=0.0, erase=0.0)
HP_LIMITS = dict(lr=(1e-6, 1e-1), weight_decay=(0.0, 0.1), batch=(4, 256), step_every=(1, 500), step_gamma=(0.05, 0.95), grad_clip=(0.0, 100.0), patience=(3, 500))
AUG_LIMITS = dict(rot=(0.0, 15.0), tx=(0.0, 20.0), ty=(0.0, 10.0), bright=(0.0, 0.6), shift=(0.0, 60.0), blur=(0.0, 1.0), noise=(0.0, 40.0), erase=(0.0, 0.8))
OPTIMIZERS, SCHEDULES = ("adam", "adamw", "sgd"), ("none", "cosine", "step")

def resolve_cfg(base, overrides=None):
    """A preset name (a key of CONFIGS) or a config dict, plus overrides -> the full, validated training config. Raises ValueError naming the bad value.
    With no overrides it is exactly the preset, so a run that sets nothing trains as it always did."""
    cfg = copy.deepcopy(CONFIGS[base] if isinstance(base, str) else base)
    if isinstance(base, str) and base not in CONFIGS: raise ValueError(f"unknown preset {base!r}: {', '.join(CONFIGS)}")
    o = dict(overrides or {}); oaug = o.pop("aug", None) or {}
    unknown = sorted(set(o) - set(HP_DEFAULTS) - {"lr"})
    if unknown: raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
    unknown = sorted(set(oaug) - set(AUG_DEFAULTS))
    if unknown: raise ValueError(f"unknown augmentation(s): {', '.join(unknown)}")
    full = {**HP_DEFAULTS, **{k: v for k, v in cfg.items() if k != "aug"}, **o}
    aug = {**AUG_DEFAULTS, **cfg.get("aug", {}), **oaug}
    def num(name, v, lo, hi, kind=float):
        try: x = kind(v)
        except (TypeError, ValueError): raise ValueError(f"{name} must be a number")
        if not lo <= x <= hi: raise ValueError(f"{name} must be between {lo} and {hi}")
        return x
    for k in ("lr", "weight_decay", "step_gamma", "grad_clip"): full[k] = num(k, full[k], *HP_LIMITS[k])
    for k in ("batch", "step_every", "patience"): full[k] = num(k, full[k], *HP_LIMITS[k], int)
    if full["optimizer"] not in OPTIMIZERS: raise ValueError(f"optimizer is one of {', '.join(OPTIMIZERS)}")
    if full["schedule"] not in SCHEDULES: raise ValueError(f"schedule is one of {', '.join(SCHEDULES)}")
    full["balance"] = bool(full["balance"])
    for k in aug: aug[k] = num(f"augmentation {k}", aug[k], *AUG_LIMITS[k])
    full["aug"] = aug
    return full

def new_model(arch=None):
    """The network to train: the shipped CRNN when arch is None, else the architecture spec built by archnet.py."""
    if arch is None: return CRNN(len(CLASSES))
    import archnet
    return archnet.build(arch, len(CLASSES))

def training_setup(model, rows, cfg, seed, max_epochs):
    """-> (data loader, optimizer, scheduler or None) for a resolved config. The default config is the original recipe: balanced sampling by field, batch 64, Adam."""
    fc = collections.Counter(r["row"] for r in rows)
    if cfg.get("balance", True):
        sampler = torch.utils.data.WeightedRandomSampler([1 / fc[r["row"]] for r in rows], len(rows), replacement=True, generator=torch.Generator().manual_seed(seed))
        dl = DataLoader(DS(rows, cfg["aug"], seed), batch_size=cfg.get("batch", 64), sampler=sampler, collate_fn=collate)
    else:
        dl = DataLoader(DS(rows, cfg["aug"], seed), batch_size=cfg.get("batch", 64), shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    wd, kind = cfg.get("weight_decay", 0.0), cfg.get("optimizer", "adam")
    opt = (torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=wd) if kind == "adam" else torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=wd) if kind == "adamw"
           else torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=wd))
    sch = {"none": None, "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs)),
           "step": lambda: torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("step_every", 50), gamma=cfg.get("step_gamma", 0.5))}[cfg.get("schedule", "none")]
    return dl, opt, (sch() if sch else None)

def train_epoch(model, dl, opt, ctc, grad_clip=0.0):
    """One epoch of CTC training with each cell's field mask. -> mean loss"""
    model.train(); tl = 0; nb = 0
    for x, tg, tlen, fields in dl:
        lg = model(x).clone()
        for k, f in enumerate(fields): lg[:, k, ~mask_t(f)] = -1e9
        lp = F.log_softmax(lg, -1)
        loss = ctc(lp, tg, torch.full((x.size(0),), lp.size(0), dtype=torch.long), tlen)
        opt.zero_grad(); loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); tl += loss.item(); nb += 1
    return tl / max(1, nb)

def augment(im, rng, a):
    h, w = im.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-a["rot"], a["rot"]), 1.0)
    M[0, 2] += rng.uniform(-a["tx"], a["tx"]); M[1, 2] += rng.uniform(-a["ty"], a["ty"])
    im = cv2.warpAffine(im, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR).astype(np.float32)
    im = im * rng.uniform(1 - a["bright"], 1 + a["bright"]) + rng.uniform(-a["shift"], a["shift"])
    im = np.clip(im, 0, 255).astype(np.uint8)
    if rng.random() < a["blur"]: im = cv2.GaussianBlur(im, (3, 3), 0)
    if a.get("noise"): im = np.clip(im.astype(np.float32) + np.random.default_rng(rng.randrange(2 ** 32)).normal(0, a["noise"], im.shape), 0, 255).astype(np.uint8)
    if a.get("erase") and rng.random() < a["erase"]:                       # a random patch of paper-white: the reader must cope with a missing piece
        h, w = im.shape; ph, pw = rng.randint(3, max(4, h // 3)), rng.randint(3, max(4, w // 8)); y, x = rng.randint(0, h - ph), rng.randint(0, w - pw); im = im.copy(); im[y:y + ph, x:x + pw] = 255
    return im

class DS(Dataset):
    def __init__(self, rows, aug=None, seed=0):
        self.rows, self.aug, self.rng = rows, aug, random.Random(seed)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; im = cell_image(r)
        if self.aug: im = augment(im, self.rng, self.aug)
        return torch.from_numpy(to_input(im)).unsqueeze(0), torch.tensor(encode_label(r["label"])), r["row"]

def collate(b):
    x, t, f = zip(*b)
    return torch.stack(x), torch.cat(t), torch.tensor([len(i) for i in t]), f

def score(model, rows):
    preds = predict_torch(model, rows)
    ex = sum(p == r["label"] for (p, _), r in zip(preds, rows)) / max(1, len(rows))
    ed = sum(edit_distance(p, r["label"]) for (p, _), r in zip(preds, rows)); ln = sum(len(r["label"]) for r in rows)
    return ex, ed / max(1, ln)

def finetune(pre_state, train_rows, val_rows, cfg, seed=0, max_epochs=300, patience=80, log=print, arch=None):
    """Train one reader on train_rows, keep the epoch that did best on val_rows. pre_state: weights to start from, or None to start from scratch. arch: an architecture spec
    (archnet.py), None for the shipped CRNN. cfg: a config from resolve_cfg (a bare preset dict from CONFIGS also works and trains exactly as before)."""
    torch.manual_seed(seed)
    model = new_model(arch).to(DEVICE)
    if pre_state is not None: model.load_state_dict(pre_state)
    patience = cfg.get("patience", patience)
    dl, opt, sch = training_setup(model, train_rows, cfg, seed, max_epochs); ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    best, best_key, best_ep, hist = None, (-1.0, 9e9), 0, []
    for ep in range(1, max_epochs + 1):
        t0 = time.time(); loss = train_epoch(model, dl, opt, ctc, cfg.get("grad_clip", 0.0))
        if sch: sch.step()
        ex, c = score(model, val_rows); dt = time.time() - t0
        hist.append(dict(epoch=ep, loss=loss, val_exact=ex, val_cer=c, sec=dt))
        key = (ex, -c)
        if key > (best_key[0], -best_key[1]):
            best_key, best_ep = (ex, c), ep; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep <= 3 or ep % 25 == 0: log(f"  ep {ep} loss {loss:.4f} val_exact {ex:.3f} val_cer {c:.3f} {dt:.2f}s/epoch")
        if ep - best_ep >= patience: break
    model.load_state_dict(best)
    return model, dict(best_epoch=best_ep, val_exact=best_key[0], val_cer=best_key[1], epochs_run=ep, hist=hist,
                       mean_epoch_sec=float(np.mean([h["sec"] for h in hist])))
