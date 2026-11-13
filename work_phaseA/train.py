"""A3: same architecture / charset / masking / NumtaDB synthetic pretraining as train_crnn.ipynb,
but fine-tuning data and early-stopping data are page-disjoint (split.json)."""
import collections, copy, json, os, random, re, time, sys
from pathlib import Path
import hpspec
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
CONFIGS = copy.deepcopy(hpspec.PRESETS)          # light aug = half the notebook's ranges; the notebook's strength is kept as a reference config

# ---------- every training value, settable (hpspec.py holds the values, defaults and limits) ----------
HP_DEFAULTS, AUG_DEFAULTS, HP_LIMITS, AUG_LIMITS, OPTIMIZERS, SCHEDULES = hpspec.HP_DEFAULTS, hpspec.AUG_DEFAULTS, hpspec.HP_LIMITS, hpspec.AUG_LIMITS, hpspec.OPTIMIZERS, hpspec.SCHEDULES

def resolve_cfg(base, overrides=None):
    """A preset name (a key of CONFIGS) or a config dict, plus overrides -> the full, validated training config (hpspec.resolve). With no overrides it is exactly the preset."""
    return hpspec.resolve(base, overrides, CONFIGS)

def new_model(arch=None):
    """The network to train: the shipped CRNN when arch is None, else the architecture spec built by archnet.py."""
    if arch is None: return CRNN(len(CLASSES))
    import archnet
    return archnet.build(arch, len(CLASSES))

def training_setup(model, rows, cfg, seed, max_epochs, mode=None):
    """-> (data loader, optimizer, scheduler or None) for a resolved config. The default config is the original recipe: balanced sampling by field, batch 64, Adam.
    mode: the direction of a plateau schedule's metric ("min"/"max"); None = from cfg monitor (exact: max, else min)."""
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
           "step": lambda: torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("step_every", 50), gamma=cfg.get("step_gamma", 0.5)),
           "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode or ("max" if cfg.get("monitor", "exact") == "exact" else "min"), factor=cfg.get("step_gamma", 0.5),
                                                                           patience=cfg.get("plateau_patience", 10), threshold=cfg.get("min_delta", 0.0), threshold_mode="abs")}[cfg.get("schedule", "none")]
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

def score_full(model, rows):
    """-> (exact, CER, CTC loss) on rows, from one forward pass. exact and CER are exactly score(); the loss is the training loss (field masks, mean CTC) in eval mode."""
    if not rows: return 0.0, 0.0, 0.0
    model.eval()
    x = torch.from_numpy(np.stack([to_input(cell_image(r)) for r in rows])).unsqueeze(1).to(DEVICE)
    with torch.no_grad(): lg = model(x)
    lgn = lg.cpu().numpy(); preds = [decode_app(lgn[:, i, :], r["row"]) for i, r in enumerate(rows)]
    ex = sum(p == r["label"] for (p, _), r in zip(preds, rows)) / len(rows)
    ed = sum(edit_distance(p, r["label"]) for (p, _), r in zip(preds, rows)); ln = sum(len(r["label"]) for r in rows)
    with torch.no_grad():
        lg = lg.clone()
        for k, r in enumerate(rows): lg[:, k, ~mask_t(r["row"])] = -1e9
        lp = F.log_softmax(lg, -1); tg = [torch.tensor(encode_label(r["label"])) for r in rows]
        vl = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)(lp, torch.cat(tg), torch.full((len(rows),), lp.size(0), dtype=torch.long), torch.tensor([len(t) for t in tg])).item()
    return ex, ed / max(1, ln), vl

# ---------- early stopping (pure) ----------
class EarlyStop:
    """Which epoch is the best, and when to stop. monitor: "exact" (validation exact match, higher; ties by lower CER: the original rule), "cer" (lower; ties by higher exact),
    "loss" (lower validation CTC loss). An epoch is an improvement when it beats the best by more than min_delta, in the monitor's own units (min_delta 0 = strictly better,
    ties broken by the second metric). Patience counts epochs since the last improvement."""
    def __init__(self, monitor="exact", min_delta=0.0, patience=80):
        assert monitor in ("exact", "cer", "loss"), monitor
        self.monitor, self.min_delta, self.patience = monitor, float(min_delta), patience
        self.best_ep, self.best = 0, None            # best: {"exact","cer","loss"} of the best epoch
    def value(self, ex, c, vl):
        """The monitored metric in its own units."""
        return {"exact": ex, "cer": c, "loss": vl}[self.monitor]
    def _key(self, ex, c, vl):         # higher is better: (primary, tie-break)
        return {"exact": (ex, -c), "cer": (-c, ex), "loss": (-vl, ex)}[self.monitor]
    def update(self, ep, ex, c, vl):
        """Record epoch ep. -> True when it is a new best."""
        k = self._key(ex, c, vl)
        if self.best is None: better = True
        else:
            b = self._key(self.best["exact"], self.best["cer"], self.best["loss"])
            better = (k[0] > b[0] + self.min_delta) if self.min_delta > 0 else (k > b)
        if better: self.best_ep, self.best = ep, {"exact": ex, "cer": c, "loss": vl}
        return better
    def should_stop(self, ep): return ep - self.best_ep >= self.patience
    def state(self): return {"best_ep": self.best_ep, "best": self.best}
    def load(self, st): self.best_ep, self.best = st["best_ep"], st["best"]

# ---------- checkpoints, curves, STOP ----------
def write_json_atomic(path, obj):
    """A temp file in the same folder, then a rename: a reader (the web UI polling) never sees half a file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh: json.dump(obj, fh)
    os.replace(tmp, path)

def prune_numbered(dirpath, name, keep_last):
    """Delete all but the newest keep_last numbered checkpoints <name>-eNNN.pt in dirpath (the -best and -last files never match). -> the names deleted."""
    pat = re.compile(re.escape(name) + r"-e(\d+)\.pt$"); found = sorted(((int(m.group(1)), f) for f in Path(dirpath).iterdir() if (m := pat.match(f.name))), reverse=True)
    gone = []
    for _, f in found[max(1, keep_last):]: f.unlink(); gone.append(f.name)
    return gone

class Checkpointer:
    """The files of one training (a CV fold, name "fold-3", or the final reader, name "final"). cfg: {"dir","name","every","keep_last","stop","curves","resume"}; see trainer.py."""
    def __init__(self, ck):
        self.dir, self.name, self.every, self.keep = Path(ck["dir"]), ck["name"], ck.get("every", 0), ck.get("keep_last", 2)
        self.stop_file, self.curves, self.resume = ck.get("stop"), ck.get("curves"), bool(ck.get("resume")); self.dir.mkdir(parents=True, exist_ok=True)
    def path(self, kind): return self.dir / f"{self.name}-{kind}.pt"
    def _save(self, obj, path):
        tmp = path.with_name(path.name + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)
    def save_best(self, state): self._save(state, self.path("best"))
    def save_last(self, state): self._save(state, self.path("last"))
    def save_numbered(self, ep, state):
        self._save(state, self.path(f"e{ep:03d}")); prune_numbered(self.dir, self.name, self.keep)
    def due(self, ep): return self.every > 0 and ep % self.every == 0
    def stop_requested(self): return bool(self.stop_file) and Path(self.stop_file).exists()
    def write_curves(self, rows):
        if self.curves: write_json_atomic(self.curves, rows)
    def load_last(self):
        p = self.path("last")
        return torch.load(p, map_location="cpu", weights_only=False) if self.resume and p.exists() else None

def _rng_get(dl):
    g = dl.sampler.generator if dl.sampler.__class__.__name__ in ("WeightedRandomSampler", "RandomSampler") and dl.sampler.generator is not None else None
    return {"torch": torch.get_rng_state(), "sampler": g.get_state() if g is not None else None, "ds": dl.dataset.rng.getstate()}

def _rng_set(dl, st):
    torch.set_rng_state(st["torch"]); dl.dataset.rng.setstate(st["ds"])
    g = dl.sampler.generator if st["sampler"] is not None else None
    if g is not None: g.set_state(st["sampler"])

def _clone(sd): return {k: v.detach().clone() for k, v in sd.items()}

def finetune(pre_state, train_rows, val_rows, cfg, seed=0, max_epochs=300, patience=80, log=print, arch=None, ck=None):
    """Train one reader on train_rows, keep the epoch that did best on val_rows (cfg monitor / min_delta / restore decide which and what is kept). pre_state: weights to start from,
    or None to start from scratch. arch: an architecture spec (archnet.py), None for the shipped CRNN. cfg: a config from resolve_cfg (a bare preset dict from CONFIGS also works
    and trains exactly as before). ck: a Checkpointer config (see Checkpointer) to write checkpoints and curves, honour STOP and resume; None = none of that.
    info["stopped"] is True when STOP ended it (the model is then the current, not the best, one)."""
    torch.manual_seed(seed)
    model = new_model(arch).to(DEVICE)
    if pre_state is not None: model.load_state_dict(pre_state)
    patience = cfg.get("patience", patience)
    dl, opt, sch = training_setup(model, train_rows, cfg, seed, max_epochs); ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    plateau = cfg.get("schedule", "none") == "plateau"
    es = EarlyStop(cfg.get("monitor", "exact"), cfg.get("min_delta", 0.0), patience)
    cp = Checkpointer(ck) if ck else None
    best, hist, start, stopped = None, [], 1, False
    st = cp.load_last() if cp else None
    if st:
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        if sch and st.get("sched") is not None: sch.load_state_dict(st["sched"])
        hist, start = st["hist"], st["epoch"] + 1; es.load(st["es"]); _rng_set(dl, st["rng"])
        bp = cp.path("best"); best = torch.load(bp, map_location="cpu") if bp.exists() else _clone(model.state_dict())
        log(f"  resumed from epoch {st['epoch']}")
    ep = start - 1
    while ep < max_epochs and not (ep >= 1 and es.should_stop(ep)):
        ep += 1
        t0 = time.time(); lr = opt.param_groups[0]["lr"]; loss = train_epoch(model, dl, opt, ctc, cfg.get("grad_clip", 0.0))
        if sch and not plateau: sch.step()
        ex, c, vl = score_full(model, val_rows); dt = time.time() - t0
        hist.append(dict(epoch=ep, loss=loss, val_exact=ex, val_cer=c, val_loss=vl, lr=lr, sec=dt))
        if es.update(ep, ex, c, vl):
            best = _clone(model.state_dict())
            if cp: cp.save_best(best)
        if sch and plateau: sch.step(es.value(ex, c, vl))
        if ep <= 3 or ep % 25 == 0: log(f"  ep {ep} loss {loss:.4f} val_exact {ex:.3f} val_cer {c:.3f} {dt:.2f}s/epoch")
        if cp:
            cp.write_curves([dict(epoch=h["epoch"], loss=h["loss"], val_exact=h["val_exact"], val_cer=h["val_cer"], val_loss=h["val_loss"], lr=h["lr"], best=h["epoch"] == es.best_ep) for h in hist])
            def full(): return {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sch.state_dict() if sch else None, "epoch": ep, "hist": hist, "es": es.state(), "rng": _rng_get(dl), "best_path": str(cp.path("best"))}
            if cp.due(ep): cp.save_numbered(ep, model.state_dict()); cp.save_last(full())
            if not es.should_stop(ep) and ep < max_epochs and cp.stop_requested():
                cp.save_last(full()); stopped = True; break
    if best is not None and not stopped and cfg.get("restore", "best") == "best": model.load_state_dict(best)
    b = es.best or {"exact": 0.0, "cer": 0.0, "loss": 0.0}
    return model, dict(best_epoch=es.best_ep, val_exact=b["exact"], val_cer=b["cer"], val_loss=b["loss"], epochs_run=ep, hist=hist, stopped=stopped, restored=cfg.get("restore", "best"),
                       mean_epoch_sec=float(np.mean([h["sec"] for h in hist])) if hist else 0.0)
