"""The reader's network as a spec you can change: convolution blocks, optional 1-D context convolutions, an optional recurrent layer, and the head.

Every reader takes one gray cell [batch, 1, 48, 192] and gives CTC scores [T, batch, classes]; that contract is all the pack, the phone's ONNX reader and the
TFLite conversion rely on, so any spec that keeps it works as a reader. The shipped reader is the preset `crnn`; `ArchNet(PRESETS["crnn"])` has exactly the
parameters of `common.CRNN`, so weights trained for one load into the other.

A spec (JSON):
    {"conv":    [{"out": 32, "k": 3, "pool": [2, 2]}, ...],     1-8 blocks of Conv, BatchNorm, ReLU, then a max pool; k is 3, 5 or 7; pool is [height, width] with each 1, 2 or 4,
                                                                 or "collapse" (all remaining height, width kept: the last block always collapses)
     "context": [{"out": 256, "k": 5}, ...],                    0-4 convolutions along the time axis, after the height has collapsed (what a CNN-only reader uses instead of an RNN)
     "rnn":     {"type": "lstm" | "gru", "hidden": 128, "layers": 1, "bidirectional": true, "dropout": 0.0} or null,
     "dropout": 0.3}                                             before the final linear layer

Pure Python until a network is built; `normalize` and `describe` need no torch.
"""
import copy, hashlib, json, math

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


# ---------------------------------------------------------------- the network (needs torch)
def build(spec, n_classes=129):
    import torch.nn as nn

    s = normalize(spec)

    class ArchNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c, h = [], 1, IMG_H
            for b in s["conv"]:
                layers += [nn.Conv2d(c, b["out"], b["k"], 1, b["k"] // 2), nn.BatchNorm2d(b["out"]), nn.ReLU()]; c = b["out"]
                if b["pool"] == "collapse" or isinstance(b["pool"], list) and b["pool"][0] == "collapse":
                    pw = b["pool"][1] if isinstance(b["pool"], list) else 1; layers.append(nn.MaxPool2d((h, pw))); h = 1
                elif tuple(b["pool"]) != (1, 1): layers.append(nn.MaxPool2d(tuple(b["pool"]))); h //= b["pool"][0]
            self.cnn = nn.Sequential(*layers)
            ctx = []
            for b in s["context"]: ctx += [nn.Conv2d(c, b["out"], (1, b["k"]), 1, (0, b["k"] // 2)), nn.BatchNorm2d(b["out"]), nn.ReLU()]; c = b["out"]
            self.ctx = nn.Sequential(*ctx)
            r = s["rnn"]; feat = c
            if r:
                cell = nn.LSTM if r["type"] == "lstm" else nn.GRU
                self.rnn = cell(c, r["hidden"], num_layers=r["layers"], bidirectional=r["bidirectional"], batch_first=True, dropout=r["dropout"] if r["layers"] > 1 else 0.0)
                feat = r["hidden"] * (2 if r["bidirectional"] else 1)
            self.dropout = nn.Dropout(s["dropout"]); self.fc = nn.Linear(feat, n_classes)

        def forward(self, x):
            f = self.ctx(self.cnn(x)).squeeze(2).permute(0, 2, 1)         # [B, T, C]
            if hasattr(self, "rnn"): f, _ = self.rnn(f)
            return self.fc(self.dropout(f)).permute(1, 0, 2)               # [T, B, V]

    net = ArchNet(); net.spec = s
    return net


def check(spec, n_classes=129, runs=20):
    """A dry run of a spec before any training: builds it, runs one cell, exports ONNX and compares it with the PyTorch output, and times both.
    -> {ok, spec, params, time_steps, torch_ms, onnx_ms, onnx_ok, onnx_bytes, tflite, shipped_weights_fit, problems}. Never raises for a bad spec: the reason is in `problems`."""
    import io, time
    out = {"ok": False, "problems": []}
    try: s = normalize(spec)
    except ValueError as e: out["problems"].append(str(e)); return out
    import numpy as np, torch
    out["spec"] = s; out["hash"] = spec_hash(s); d = describe(s, n_classes); out.update(params=d["params"], time_steps=d["time_steps"])
    net = build(s, n_classes).eval(); x = torch.zeros(1, 1, IMG_H, IMG_W)
    with torch.no_grad():
        y = net(x)
        if tuple(y.shape) != (d["time_steps"], 1, n_classes): out["problems"].append(f"output shape {tuple(y.shape)}, expected {(d['time_steps'], 1, n_classes)}"); return out
        t0 = time.time(); [net(x) for _ in range(runs)]; out["torch_ms"] = round((time.time() - t0) * 1000 / runs, 2)
    out["params_built"] = sum(p.numel() for p in net.parameters())
    if out["params_built"] != d["params"]: out["problems"].append(f"counted {d['params']:,} parameters but built {out['params_built']:,}")
    try:
        import onnxruntime as ort
        buf = io.BytesIO(); torch.onnx.export(net, x, buf, input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}}, opset_version=17, dynamo=False)
        sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"]); xr = np.random.RandomState(0).randn(1, 1, IMG_H, IMG_W).astype(np.float32)
        with torch.no_grad(): ref = net(torch.from_numpy(xr)).numpy()
        got = sess.run(None, {"image": xr})[0]; diff = float(np.abs(ref - got).max())
        t0 = time.time(); [sess.run(None, {"image": xr}) for _ in range(runs)]; out.update(onnx_ms=round((time.time() - t0) * 1000 / runs, 2), onnx_bytes=len(buf.getvalue()), onnx_max_diff=diff, onnx_ok=diff < 1e-3)
        if diff >= 1e-3: out["problems"].append(f"ONNX output differs from PyTorch by {diff:.4f}")
    except Exception as e:
        out["onnx_ok"] = False; out["problems"].append(f"ONNX export failed: {str(e)[:160]}")
    out["tflite"] = tflite_support(s); out["shipped_weights_fit"] = is_default(s)
    out["ok"] = not out["problems"]
    return out


def tflite_support(spec):
    """Whether tflite_export.py can rebuild this spec (so a pack can carry reader.tflite). -> {supported, reason}. Kept in step with that converter:
    it rebuilds every spec `normalize` accepts (1-8 conv blocks with k 3/5/7 and any pools, 0-4 context convs, no rnn or 1-3 layers of LSTM/GRU, either
    direction, then the linear layer), so anything that normalizes is supported and anything that does not is refused with normalize's reason."""
    try: normalize(spec)
    except ValueError as e: return {"supported": False, "reason": f"not a valid architecture: {e}"}
    return {"supported": True, "reason": ""}
