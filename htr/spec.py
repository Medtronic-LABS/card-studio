"""The reader's architecture as a spec you can change: convolution blocks, optional 1-D context convolutions, an optional recurrent layer,
and the head. This module is the direct successor to `work_phaseA/archnet.py`: every function here behaves exactly as that one did for a
legacy-shaped spec (the four keys below), so a spec, a hash or a registered weights file made before this package existed still normalizes,
hashes and builds the same way. `htr/catalog.py` describes the richer schema-2 layer catalog (CNN-Transformer, SVTR, custom layers); this
file is kept small and stable because packs, `studio/pretrained.py` and old `results.json` records depend on its hash never moving.

Every reader takes one gray cell [batch, 1, 48, 192] and gives CTC scores [T, batch, classes]; that contract is all the pack, the phone's
ONNX reader and the TFLite conversion rely on, so any spec that keeps it works as a reader. The shipped reader is the preset `crnn`.

A legacy spec (JSON):
    {"conv":    [{"out": 32, "k": 3, "pool": [2, 2]}, ...],     1-8 blocks of Conv, BatchNorm, ReLU, then a max pool; k is 3, 5 or 7; pool is [height, width] with each 1, 2 or 4,
                                                                 or "collapse" (all remaining height, width kept: the last block always collapses)
     "context": [{"out": 256, "k": 5}, ...],                    0-4 convolutions along the time axis, after the height has collapsed (what a CNN-only reader uses instead of an RNN)
     "rnn":     {"type": "lstm" | "gru", "hidden": 128, "layers": 1, "bidirectional": true, "dropout": 0.0} or null,
     "dropout": 0.3}                                             before the final linear layer

Pure Python until a network is built; `normalize` and `describe` need no torch.
"""
import copy, hashlib, json

IMG_H, IMG_W = 48, 192
MIN_T = 24                       # a reader needs enough time steps to spell its longest value with the repeats CTC needs
LIMITS = {"conv_blocks": (1, 8), "conv_out": (4, 512), "ctx_blocks": (0, 4), "ctx_out": (16, 1024), "hidden": (16, 512), "layers": (1, 3), "dropout": (0.0, 0.7), "params": 20_000_000}

PRESETS = {
    "crnn": {"label": "CRNN (the shipped reader): 4 convolutions, one bidirectional LSTM", "spec": {
        "conv": [{"out": 32, "k": 3, "pool": [2, 2]}, {"out": 64, "k": 3, "pool": [2, 2]}, {"out": 128, "k": 3, "pool": [2, 1]}, {"out": 128, "k": 3, "pool": "collapse"}],
        "context": [], "rnn": {"type": "lstm", "hidden": 128, "layers": 1, "bidirectional": True, "dropout": 0.0}, "dropout": 0.3}},
    "crnn-small": {"label": "Small CRNN: narrower convolutions and LSTM (about a third of the parameters)", "spec": {
        "conv": [{"out": 24, "k": 3, "pool": [2, 2]}, {"out": 48, "k": 3, "pool": [2, 2]}, {"out": 96, "k": 3, "pool": [2, 1]}, {"out": 96, "k": 3, "pool": "collapse"}],
        "context": [], "rnn": {"type": "lstm", "hidden": 96, "layers": 1, "bidirectional": True, "dropout": 0.0}, "dropout": 0.3}},
    "crnn-wide": {"label": "Wide CRNN: wider convolutions and a two-layer LSTM (about 8x the parameters)", "spec": {
        "conv": [{"out": 64, "k": 3, "pool": [2, 2]}, {"out": 128, "k": 3, "pool": [2, 2]}, {"out": 256, "k": 3, "pool": [2, 1]}, {"out": 256, "k": 3, "pool": "collapse"}],
        "context": [], "rnn": {"type": "lstm", "hidden": 192, "layers": 2, "bidirectional": True, "dropout": 0.2}, "dropout": 0.3}},
    "gru-crnn": {"label": "CRNN with a bidirectional GRU instead of the LSTM", "spec": {
        "conv": [{"out": 32, "k": 3, "pool": [2, 2]}, {"out": 64, "k": 3, "pool": [2, 2]}, {"out": 128, "k": 3, "pool": [2, 1]}, {"out": 128, "k": 3, "pool": "collapse"}],
        "context": [], "rnn": {"type": "gru", "hidden": 128, "layers": 1, "bidirectional": True, "dropout": 0.0}, "dropout": 0.3}},
    "cnn-ctc": {"label": "CNN only: the same convolutions, then two 1-D convolutions along the line instead of an RNN", "spec": {
        "conv": [{"out": 32, "k": 3, "pool": [2, 2]}, {"out": 64, "k": 3, "pool": [2, 2]}, {"out": 128, "k": 3, "pool": [2, 1]}, {"out": 128, "k": 3, "pool": "collapse"}],
        "context": [{"out": 256, "k": 5}, {"out": 256, "k": 5}], "rnn": None, "dropout": 0.3}},
}
DEFAULT = "crnn"


def _num(v, name, lo, hi, kind=float):
    try: x = kind(v)
    except (TypeError, ValueError): raise ValueError(f"{name} must be a number")
    if not lo <= x <= hi: raise ValueError(f"{name} must be between {lo} and {hi}")
    return x


def normalize(spec):
    """A preset name or a spec dict -> the canonical spec (every default filled in). Raises ValueError, naming what is wrong, when it cannot be a reader."""
    if isinstance(spec, str):
        if spec not in PRESETS: raise ValueError(f"unknown architecture preset {spec!r}: {', '.join(PRESETS)}")
        spec = PRESETS[spec]["spec"]
    if not isinstance(spec, dict): raise ValueError("an architecture is a preset name or an object with conv, context, rnn and dropout")
    unknown = sorted(set(spec) - {"conv", "context", "rnn", "dropout"})
    if unknown: raise ValueError(f"unknown architecture key(s): {', '.join(unknown)}")
    conv = spec.get("conv")
    if not isinstance(conv, list) or not LIMITS["conv_blocks"][0] <= len(conv) <= LIMITS["conv_blocks"][1]:
        raise ValueError(f"conv needs {LIMITS['conv_blocks'][0]} to {LIMITS['conv_blocks'][1]} blocks")
    blocks, height, width = [], IMG_H, IMG_W
    for i, b in enumerate(conv, 1):
        last = i == len(conv)
        if not isinstance(b, dict) or set(b) - {"out", "k", "pool"}: raise ValueError(f"conv block {i}: use out, k and pool")
        out = _num(b.get("out"), f"conv block {i} out", *LIMITS["conv_out"], int); k = _num(b.get("k", 3), f"conv block {i} k", 3, 7, int)
        if k % 2 == 0: raise ValueError(f"conv block {i}: k must be 3, 5 or 7")
        pool = b.get("pool", "collapse" if last else [1, 1])
        if pool == "collapse": ph, pw = "collapse", 1
        elif isinstance(pool, (list, tuple)) and len(pool) == 2 and pool[0] in (1, 2, 4, "collapse") and pool[1] in (1, 2, 4): ph, pw = pool[0], int(pool[1])
        else: raise ValueError(f"conv block {i}: pool is [height, width] with each 1, 2 or 4, or \"collapse\"")
        if ph == "collapse" and not last: raise ValueError(f"conv block {i}: only the last block can collapse the height")
        if last: ph = "collapse"                                   # the last block always collapses the height that is left
        height = 1 if ph == "collapse" else height // ph; width //= pw
        if height < 1 or width < 1: raise ValueError(f"conv block {i}: the pools shrink the picture to nothing")
        blocks.append({"out": out, "k": k, "pool": "collapse" if (ph == "collapse" and pw == 1) else [ph, pw]})
    T = width
    if T < MIN_T: raise ValueError(f"the pools leave {T} time steps; a reader needs at least {MIN_T} (use less width pooling)")
    ctx = spec.get("context") or []
    if not isinstance(ctx, list) or len(ctx) > LIMITS["ctx_blocks"][1]: raise ValueError(f"context has at most {LIMITS['ctx_blocks'][1]} blocks")
    cblocks = []
    for i, b in enumerate(ctx, 1):
        if not isinstance(b, dict) or set(b) - {"out", "k"}: raise ValueError(f"context block {i}: use out and k")
        k = _num(b.get("k", 3), f"context block {i} k", 3, 7, int)
        if k % 2 == 0: raise ValueError(f"context block {i}: k must be 3, 5 or 7")
        cblocks.append({"out": _num(b.get("out"), f"context block {i} out", *LIMITS["ctx_out"], int), "k": k})
    rnn = spec.get("rnn")
    if rnn is not None:
        if not isinstance(rnn, dict) or set(rnn) - {"type", "hidden", "layers", "bidirectional", "dropout"}: raise ValueError("rnn: use type, hidden, layers, bidirectional and dropout")
        if rnn.get("type") not in ("lstm", "gru"): raise ValueError("rnn type is lstm or gru (or rnn: null for none)")
        layers = _num(rnn.get("layers", 1), "rnn layers", *LIMITS["layers"], int)
        rnn = {"type": rnn["type"], "hidden": _num(rnn.get("hidden"), "rnn hidden", *LIMITS["hidden"], int), "layers": layers, "bidirectional": bool(rnn.get("bidirectional", True)),
               "dropout": _num(rnn.get("dropout", 0.0), "rnn dropout", *LIMITS["dropout"]) if layers > 1 else 0.0}
    if rnn is None and not cblocks: raise ValueError("with no rnn, add at least one context convolution so the reader sees along the line")
    out = {"conv": blocks, "context": cblocks, "rnn": rnn, "dropout": _num(spec.get("dropout", 0.3), "dropout", *LIMITS["dropout"])}
    n = describe(out)["params"]
    if n > LIMITS["params"]: raise ValueError(f"{n:,} parameters is over the {LIMITS['params']:,} limit")
    return out


def describe(spec, n_classes=129):
    """-> {params, time_steps, feature_size, blocks}: counted from the spec without torch."""
    h, w, c, p = IMG_H, IMG_W, 1, 0
    for b in spec["conv"]:
        p += c * b["out"] * b["k"] * b["k"] + b["out"] + 2 * b["out"]; c = b["out"]
        if b["pool"] == "collapse": h = 1
        elif b["pool"][0] == "collapse": h, w = 1, w // b["pool"][1]
        else: h, w = h // b["pool"][0], w // b["pool"][1]
    for b in spec["context"]:
        p += c * b["out"] * b["k"] + b["out"] + 2 * b["out"]; c = b["out"]
    feat = c
    if spec["rnn"]:
        r = spec["rnn"]; d = 2 if r["bidirectional"] else 1; g = 4 if r["type"] == "lstm" else 3; inp = c
        for _ in range(r["layers"]):
            p += d * (g * r["hidden"] * (inp + r["hidden"]) + 2 * g * r["hidden"]); inp = d * r["hidden"]
        feat = inp
    p += feat * n_classes + n_classes
    return {"params": p, "time_steps": w, "feature_size": feat, "blocks": len(spec["conv"]) + len(spec["context"])}


def spec_hash(spec):
    """Short stable id of a normalized spec; weights are only for the spec they were trained as."""
    return hashlib.sha256(json.dumps(normalize(spec), sort_keys=True).encode()).hexdigest()[:12]


def is_default(spec):
    return normalize(spec) == normalize(DEFAULT)


def flow(spec, n_classes=129):
    """The picture's journey through the network, stage by stage, for a diagram: [{"label", "shape"}]. shape is [channels, height, width] for the convolutions and [time steps, features]
    after the height has collapsed. Pure Python."""
    s = normalize(spec); h, w, c = IMG_H, IMG_W, 1
    out = [{"label": "cell (gray)", "shape": [1, h, w]}]
    for b in s["conv"]:
        pool = b["pool"]
        if pool == "collapse": h = 1; pl = "collapse height"
        elif pool[0] == "collapse": h, w = 1, w // pool[1]; pl = f"collapse height, width /{pool[1]}"
        else: h, w = h // pool[0], w // pool[1]; pl = "no pool" if tuple(pool) == (1, 1) else f"pool {pool[0]}x{pool[1]}"
        c = b["out"]; out.append({"label": f"conv {b['k']}x{b['k']} to {c}, {pl}", "shape": [c, h, w]})
    for b in s["context"]:
        c = b["out"]; out.append({"label": f"conv along the line, k {b['k']}, to {c}", "shape": [w, c]})
    r = s["rnn"]; feat = c
    if r:
        feat = r["hidden"] * (2 if r["bidirectional"] else 1)
        out.append({"label": f"{r['layers']}x {'bi' if r['bidirectional'] else ''}{r['type'].upper()} {r['hidden']}", "shape": [w, feat]})
    out.append({"label": f"linear to {n_classes} classes", "shape": [w, n_classes]})
    return out


def summary(spec, n_classes=129):
    """Everything a builder shows while a spec is edited, without torch: {ok, spec, hash, params, time_steps, flow} or {ok: False, problems}."""
    try: s = normalize(spec)
    except ValueError as e: return {"ok": False, "problems": [str(e)]}
    d = describe(s, n_classes)
    return {"ok": True, "spec": s, "hash": spec_hash(s), "params": d["params"], "time_steps": d["time_steps"], "flow": flow(s, n_classes)}


def tflite_support(spec):
    """Whether the legacy TFLite rebuild backend (`htr/export/rebuild_backend.py`) can rebuild this spec. Kept in step with that converter:
    it rebuilds every spec `normalize` accepts, so anything that normalizes is supported."""
    try: normalize(spec)
    except ValueError as e: return {"supported": False, "reason": f"not a valid architecture: {e}"}
    return {"supported": True, "reason": ""}
