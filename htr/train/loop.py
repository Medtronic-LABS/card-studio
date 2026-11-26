"""The one training entry every architecture goes through: masked CTC loss (a cell's field kind restricts which classes it may emit,
same idea as the legacy engine's per-field mask, now from `vocab.mask(kind)` instead of a fixed `model_meta.json`), dynamic-width
`input_lengths`/`out_lengths` instead of a fixed 192px canvas, early stopping and checkpointing via `tracking.py`.

Needs torch; not imported by `htr/__init__.py`."""
import torch
import torch.nn.functional as F

from . import config as C
from . import tracking as TR
from .. import text as T


def _mask_tensor(vocab, kind, device):
    return torch.tensor(vocab.mask(kind), dtype=torch.bool, device=device)


def train_epoch(model, dl, opt, vocab, grad_clip=0.0, device="cpu"):
    """One epoch of masked CTC training. -> mean training loss."""
    model.train()
    ctc = torch.nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    total, n = 0.0, 0
    for batch in dl:
        images, input_lengths = batch["images"].to(device), batch["input_lengths"].to(device)
        targets, target_lengths = batch["targets"].to(device), batch["target_lengths"].to(device)
        logits, out_lengths = model(images, input_lengths)
        masked = logits.clone()
        for k, kind in enumerate(batch["kinds"]): masked[:, k, ~_mask_tensor(vocab, kind, logits.device)] = -1e9
        logp = F.log_softmax(masked, -1)
        loss = ctc(logp, targets, out_lengths, target_lengths)
        opt.zero_grad(); loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); total += loss.item(); n += 1
    return total / max(1, n)


def _greedy_decode(logits_tv, length, mask, blank):
    """logits_tv: (T, V) tensor for one sample. -> list of class ids (CTC greedy, restricted to `mask`, de-duplicated, blank dropped)."""
    seq = logits_tv[:length].clone()
    seq[:, ~mask] = -1e9
    best = seq.argmax(-1).tolist()
    out, prev = [], -1
    for b in best:
        if b != blank and b != prev: out.append(b)
        prev = b
    return out


@torch.no_grad()
def evaluate(model, dl, vocab, device="cpu"):
    """-> {loss, exact, cer_grapheme, cer_codepoint, wer, n}. `loss`: masked CTC loss, same objective as training. The rest: decode
    every sample (masked greedy CTC, truncated to its own `out_lengths`) and score against its normalized reference text."""
    from ..eval import metrics as M
    model.eval()
    ctc = torch.nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    total_loss, n_batches, pairs = 0.0, 0, []
    for batch in dl:
        images, input_lengths = batch["images"].to(device), batch["input_lengths"].to(device)
        targets, target_lengths = batch["targets"].to(device), batch["target_lengths"].to(device)
        logits, out_lengths = model(images, input_lengths)
        masked = logits.clone()
        masks = [_mask_tensor(vocab, kind, logits.device) for kind in batch["kinds"]]
        for k, m in enumerate(masks): masked[:, k, ~m] = -1e9
        logp = F.log_softmax(masked, -1)
        total_loss += ctc(logp, targets, out_lengths, target_lengths).item(); n_batches += 1
        for i, (row, kind) in enumerate(zip(batch["rows"], batch["kinds"])):
            ids = _greedy_decode(logits[:, i, :], int(out_lengths[i]), masks[i], vocab.blank)
            hyp = vocab.decode(ids)
            ref = T.normalize(row["text"], kind=kind)
            pairs.append((ref, hyp))
    return {"loss": total_loss / max(1, n_batches), "exact": M.exact(pairs), "cer_grapheme": M.cer_grapheme(pairs),
            "cer_codepoint": M.cer_codepoint(pairs), "wer": M.wer(pairs), "n": len(pairs)}


def _build_optimizer(model, optim_cfg):
    kind, lr, wd = optim_cfg["optimizer"], optim_cfg["lr"], optim_cfg["weight_decay"]
    if kind == "adam": return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "adamw": return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)


def _build_scheduler(opt, optim_cfg, epochs, monitor):
    sched = optim_cfg["schedule"]
    if sched == "none": return None
    if sched == "cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if sched == "step": return torch.optim.lr_scheduler.StepLR(opt, step_size=optim_cfg["step_every"], gamma=optim_cfg["step_gamma"])
    mode = "max" if monitor == "exact" else "min"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=optim_cfg["step_gamma"], patience=optim_cfg["plateau_patience"])


def train(config, model, train_loader, val_loader, vocab, ckpt=None, log=print, device="cpu"):
    """config: overlaid onto `htr.train.config.DEFAULTS` (see `config.resolve`). -> {history: [{epoch, train_loss, loss, exact,
    cer_grapheme, cer_codepoint, wer, n}, ...], best_epoch, best, stopped}. `stopped`: True when a `ckpt["stop"]` file ended the run
    early (the model holds the last, not the best, epoch's weights in that case - as with the legacy engine)."""
    cfg = C.resolve(config)
    model = model.to(device)
    opt = _build_optimizer(model, cfg["optim"])
    monitor = cfg["loop"]["monitor"]
    sched = _build_scheduler(opt, cfg["optim"], cfg["loop"]["epochs"], monitor)
    es = TR.EarlyStop(monitor, cfg["loop"]["min_delta"], cfg["loop"]["patience"])
    cp = TR.Checkpointer(ckpt) if ckpt else None
    history, start, stopped = [], 1, False
    if cp:
        st = cp.load_last()
        if st:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            if sched is not None and st.get("sched") is not None: sched.load_state_dict(st["sched"])
            es.load(st["early_stop"]); history = st.get("history", []); start = st["epoch"] + 1
    for ep in range(start, cfg["loop"]["epochs"] + 1):
        if cp and cp.stop_requested(): stopped = True; break
        train_loss = train_epoch(model, train_loader, opt, vocab, cfg["loop"]["grad_clip"], device)
        val = evaluate(model, val_loader, vocab, device)
        improved = es.update(ep, val)
        row = {"epoch": ep, "train_loss": train_loss, **val}
        history.append(row)
        log(f"epoch {ep}/{cfg['loop']['epochs']} train_loss {train_loss:.4f} val_{monitor} {val[monitor]:.4f}{' *' if improved else ''}")
        if cp:
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                     "epoch": ep, "early_stop": es.state(), "history": history, "vocab_sha": vocab.sha256}
            cp.save_last(state)
            if improved: cp.save_best(state)
            if cp.due(ep): cp.save_numbered(ep, state)
            cp.write_curves(history)
        if sched is not None:
            if cfg["optim"]["schedule"] == "plateau": sched.step(val[monitor])
            else: sched.step()
        if es.should_stop(ep): break
    return {"history": history, "best_epoch": es.best_ep, "best": es.best, "stopped": stopped}
