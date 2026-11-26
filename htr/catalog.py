"""The schema-2 layer catalog: every block type a configurable or custom architecture can use, what keys it takes, how it changes the
picture's shape, and how many parameters it adds. Pure Python (no torch) so the UI and a bundle check can validate and describe a
config without the torch dependency; `htr/nn/layers.py` builds the matching `nn.Module` for each type.

Implemented today: conv2d, dwsep (depthwise-separable conv), maxpool, avgpool, collapse (collapse all remaining height to 1), rnn
(lstm/gru, uni- or bidirectional), transformer (a standard post-norm encoder stack), dropout. Deferred (M2+, not yet in the catalog):
residual, mobilenet_block, attention (a non-CTC attention decoder), temporal_conv, batchnorm/layernorm as standalone blocks,
custom Python layers (`htr/nn/custom.py`).

A schema-2 config (see `htr/presets.py`) is {"schema": 2, "input": {...}, "backbone": [block, ...], "sequence": [block, ...],
"head": {...}, "decoder": {...}}. `backbone` runs on the [B, C, H, W] picture and must collapse height to 1 by its last block (its
own block list, mixing conv2d/dwsep/pool/collapse freely); `sequence` then runs on the [B, T, C] sequence that leaves (rnn and/or
transformer blocks, freely mixed and repeated)."""
IMG_H = 48
IMG_W_REF = 192          # a nominal width used only to describe/validate a config (actual width is dynamic at train and inference time)
MIN_T = 24
LIMITS = {
    "conv_out": (4, 1024), "k": (1, 7), "rnn_hidden": (16, 512), "rnn_layers": (1, 4), "transformer_dim": (16, 1024),
    "transformer_heads": (1, 16), "transformer_layers": (1, 8), "dropout": (0.0, 0.7), "params": 20_000_000,
}
POOL_FACTORS = (1, 2, 4)


def _pool(pool, h, w, last):
    """pool: [height_factor, width_factor] (each 1/2/4) or "collapse" -> (h2, w2). The last backbone block always collapses height
    regardless of what it asks for (a reader needs height fully reduced before the sequence stage)."""
    if pool == "collapse": return 1, w
    if not (isinstance(pool, (list, tuple)) and len(pool) == 2): raise ValueError('pool is [height, width] (each 1, 2 or 4) or "collapse"')
    ph, pw = pool
    if ph == "collapse": return 1, w // pw if pw != 1 else w
    if ph not in POOL_FACTORS or pw not in POOL_FACTORS: raise ValueError(f"pool factors must be one of {POOL_FACTORS}, or height \"collapse\"")
    h2 = 1 if last else max(1, h // ph)
    return h2, max(1, w // pw)


def _conv2d_shape(b, h, w, c, last):
    k = b.get("k", 3)
    if k % 2 == 0 or not LIMITS["k"][0] <= k <= LIMITS["k"][1]: raise ValueError("conv2d k must be odd, 3 to 7")
    out = b["out"]
    if not LIMITS["conv_out"][0] <= out <= LIMITS["conv_out"][1]: raise ValueError(f"conv2d out must be {LIMITS['conv_out']}")
    h2, w2 = _pool(b.get("pool", [1, 1]), h, w, last)
    return h2, w2, out


def _conv2d_params(b, c, bn=True):
    """Matches `htr/nn/layers.py:conv2d`: the conv always has its own bias (even with a BatchNorm right after it, same redundancy
    the legacy engine has always had - see `htr.spec.describe`)."""
    k, out = b.get("k", 3), b["out"]
    p = c * out * k * k + out
    if bn: p += 2 * out
    return p


def _dwsep_shape(b, h, w, c, last): return _conv2d_shape(b, h, w, c, last)


def _dwsep_params(b, c):
    k, out = b.get("k", 3), b["out"]
    return (c * k * k + c + 2 * c) + (c * out + out + 2 * out)   # depthwise (+bias+bn) then pointwise (+bias+bn)


def _pool_only_shape(b, h, w, c, last): return (*_pool(b.get("pool", [2, 2]), h, w, last), c)
def _collapse_shape(b, h, w, c, last): return 1, w, c
def _rnn_shape(b, h, w, c, last):
    if h != 1: raise ValueError("a sequence block (rnn/transformer) needs height already collapsed to 1 by the backbone")
    hidden = b.get("hidden"); layers = b.get("layers", 1)
    if not LIMITS["rnn_hidden"][0] <= hidden <= LIMITS["rnn_hidden"][1]: raise ValueError(f"rnn hidden must be {LIMITS['rnn_hidden']}")
    if not LIMITS["rnn_layers"][0] <= layers <= LIMITS["rnn_layers"][1]: raise ValueError(f"rnn layers must be {LIMITS['rnn_layers']}")
    if b.get("cell", "lstm") not in ("lstm", "gru"): raise ValueError("rnn cell is lstm or gru")
    return 1, w, hidden * (2 if b.get("bidirectional", True) else 1)


def _rnn_params(b, c):
    hidden, layers, bidir = b.get("hidden"), b.get("layers", 1), b.get("bidirectional", True)
    gate = 4 if b.get("cell", "lstm") == "lstm" else 3
    d = 2 if bidir else 1
    p, inp = 0, c
    for _ in range(layers):
        p += d * (gate * hidden * (inp + hidden) + 2 * gate * hidden); inp = d * hidden
    return p


def _transformer_shape(b, h, w, c, last):
    if h != 1: raise ValueError("a sequence block (rnn/transformer) needs height already collapsed to 1 by the backbone")
    dim, heads, layers = b.get("dim"), b.get("heads", 4), b.get("layers", 1)
    if not LIMITS["transformer_dim"][0] <= dim <= LIMITS["transformer_dim"][1]: raise ValueError(f"transformer dim must be {LIMITS['transformer_dim']}")
    if dim % heads != 0: raise ValueError("transformer dim must divide evenly by heads")
    if not LIMITS["transformer_heads"][0] <= heads <= LIMITS["transformer_heads"][1]: raise ValueError(f"transformer heads must be {LIMITS['transformer_heads']}")
    if not LIMITS["transformer_layers"][0] <= layers <= LIMITS["transformer_layers"][1]: raise ValueError(f"transformer layers must be {LIMITS['transformer_layers']}")
    return 1, w, dim


def _transformer_params(b, c):
    """Matches `nn.TransformerEncoderLayer` exactly: per layer, in_proj (3*dim^2+3*dim) + out_proj (dim^2+dim) + a 2-layer FFN of
    width dim*ff_mult (dim*ff+ff then ff*dim+dim) + two LayerNorms (4*dim) = 4*dim^2 + 2*dim*ff + 9*dim + ff."""
    dim, layers, ff_mult = b.get("dim"), b.get("layers", 1), b.get("ff_mult", 4)
    ff = dim * ff_mult
    proj = c * dim + dim if c != dim else 0
    per_layer = 4 * dim * dim + 2 * dim * ff + 9 * dim + ff
    return proj + layers * per_layer


def _dropout_shape(b, h, w, c, last):
    if not LIMITS["dropout"][0] <= b.get("p", 0.3) <= LIMITS["dropout"][1]: raise ValueError(f"dropout p must be {LIMITS['dropout']}")
    return h, w, c


LAYER_TYPES = {
    "conv2d": {"stage": "backbone", "required": ("out",), "shape": _conv2d_shape, "params": lambda b, h, c: _conv2d_params(b, c, b.get("bn", True)), "backends": ("rebuild", "onnx", "ai_edge_torch", "onnx2tf")},
    "dwsep": {"stage": "backbone", "required": ("out",), "shape": _dwsep_shape, "params": lambda b, h, c: _dwsep_params(b, c), "backends": ("onnx", "ai_edge_torch", "onnx2tf")},
    "maxpool": {"stage": "backbone", "required": (), "shape": _pool_only_shape, "params": lambda b, h, c: 0, "backends": ("rebuild", "onnx", "ai_edge_torch", "onnx2tf")},
    "avgpool": {"stage": "backbone", "required": (), "shape": _pool_only_shape, "params": lambda b, h, c: 0, "backends": ("rebuild", "onnx", "ai_edge_torch", "onnx2tf")},
    "collapse": {"stage": "backbone", "required": (), "shape": _collapse_shape, "params": lambda b, h, c: 0, "backends": ("rebuild", "onnx", "ai_edge_torch", "onnx2tf")},
    "rnn": {"stage": "sequence", "required": ("hidden",), "shape": _rnn_shape, "params": lambda b, h, c: _rnn_params(b, c), "backends": ("rebuild", "onnx", "ai_edge_torch")},
    "transformer": {"stage": "sequence", "required": ("dim",), "shape": _transformer_shape, "params": lambda b, h, c: _transformer_params(b, c), "backends": ("onnx", "ai_edge_torch")},
    "dropout": {"stage": "sequence", "required": (), "shape": _dropout_shape, "params": lambda b, h, c: 0, "backends": ("rebuild", "onnx", "ai_edge_torch", "onnx2tf")},
}
STAGES = ("backbone", "sequence")


def validate_block(kind, block, h, w, c, last=False):
    """-> (h2, w2, c2, params_added). Raises ValueError (naming the problem) for an unknown type, a missing required key, or a block
    whose own limits are violated."""
    if kind not in LAYER_TYPES: raise ValueError(f"unknown layer type {kind!r} (use one of {', '.join(LAYER_TYPES)})")
    entry = LAYER_TYPES[kind]
    missing = [k for k in entry["required"] if k not in block]
    if missing: raise ValueError(f"{kind} block needs {', '.join(missing)}")
    h2, w2, c2 = entry["shape"](block, h, w, c, last)
    if h2 < 1 or w2 < 1: raise ValueError(f"{kind} block shrinks the picture to nothing")
    return h2, w2, c2, entry["params"](block, h, c)


def backends_for(blocks):
    """-> {backend: supported (bool)} across every (type, block) pair given: which mobile export backend each layer type is even
    plausible for (an RNN tagged out for onnx2tf, say). This is per-layer-type only, not a real compatibility check - it does not
    know whether the whole assembled graph matches a specific backend's accepted shapes. `htr/export/compat.py` makes the actual
    rebuild-backend call (schema-2 is always refused there, regardless of which layer types a config happens to use), since
    `rebuild_backend.read_onnx` is strict about the exact graph it accepts."""
    tags = [set(LAYER_TYPES[kind]["backends"]) for kind, _ in blocks if kind in LAYER_TYPES]
    all_backends = {"rebuild", "onnx", "ai_edge_torch", "onnx2tf"}
    supported = set.intersection(*tags) if tags else all_backends
    return {b: b in supported for b in sorted(all_backends)}
