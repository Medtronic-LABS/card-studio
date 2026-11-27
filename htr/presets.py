"""Schema-2 architectures: a `backbone` of catalog blocks (`htr/catalog.py`) that collapses height to 1, then a `sequence` of
rnn/transformer blocks, then a linear head and a CTC decoder. This is the configurable/custom path; the 5 original presets
(`crnn`, `crnn-small`, `crnn-wide`, `gru-crnn`, `cnn-ctc`) stay in `htr.spec` under the legacy 4-key schema because their hash must
never move. Everything here is new, so there is no hash to keep stable - only internal consistency (`spec_hash` changes if and only
if the normalized config changes).

A schema-2 config: {"schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1}, "backbone": [block, ...],
"sequence": [block, ...], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}. Pure Python until a network is built."""
import copy, hashlib, json

from . import catalog as K

PRESETS = {
    "crnn-tiny": {"label": "Tiny CRNN: half crnn-small's width, for the smallest devices", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 16, "k": 3, "pool": [2, 2]}, {"type": "conv2d", "out": 32, "k": 3, "pool": [2, 2]},
                     {"type": "conv2d", "out": 64, "k": 3, "pool": [2, 1]}, {"type": "conv2d", "out": 64, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "rnn", "hidden": 64, "layers": 1, "bidirectional": True}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
    "crnn-base": {"label": "Base CRNN: depthwise-separable convolutions, about crnn's size", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 32, "k": 3, "pool": [2, 2]}, {"type": "dwsep", "out": 64, "k": 3, "pool": [2, 2]},
                     {"type": "dwsep", "out": 128, "k": 3, "pool": [2, 1]}, {"type": "dwsep", "out": 128, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "rnn", "hidden": 128, "layers": 1, "bidirectional": True}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
    "crnn-large": {"label": "Large CRNN: wider depthwise-separable convolutions, a deeper two-layer BiLSTM", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 64, "k": 3, "pool": [2, 2]}, {"type": "dwsep", "out": 128, "k": 3, "pool": [2, 2]},
                     {"type": "dwsep", "out": 256, "k": 3, "pool": [2, 1]}, {"type": "dwsep", "out": 256, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "rnn", "hidden": 256, "layers": 2, "bidirectional": True, "dropout": 0.2}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
    "big-crnn": {"label": "Big CRNN (studio/capacity.py's big_crnn): four plain 64-128-256-256 convolutions, a two-layer 288-unit BiLSTM", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 64, "k": 3, "pool": [2, 2]}, {"type": "conv2d", "out": 128, "k": 3, "pool": [2, 2]},
                     {"type": "conv2d", "out": 256, "k": 3, "pool": [2, 1]}, {"type": "conv2d", "out": 256, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "rnn", "hidden": 288, "layers": 2, "bidirectional": True, "dropout": 0.2}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
    "attn": {"label": "Attention CTC (studio/capacity.py's attn): the same four convolutions, a four-layer 288-wide Transformer encoder", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 64, "k": 3, "pool": [2, 2]}, {"type": "conv2d", "out": 128, "k": 3, "pool": [2, 2]},
                     {"type": "conv2d", "out": 256, "k": 3, "pool": [2, 1]}, {"type": "conv2d", "out": 256, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "transformer", "dim": 288, "heads": 6, "layers": 4, "ff_mult": 4, "dropout": 0.1}], "head": {"dropout": 0.1}, "decoder": {"type": "ctc"}}},
    "cnn-transformer": {"label": "CNN backbone, a small Transformer encoder instead of an RNN", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 32, "k": 3, "pool": [2, 2]}, {"type": "conv2d", "out": 64, "k": 3, "pool": [2, 2]},
                     {"type": "conv2d", "out": 128, "k": 3, "pool": [2, 1]}, {"type": "conv2d", "out": 128, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "transformer", "dim": 128, "heads": 4, "layers": 2, "ff_mult": 4, "dropout": 0.1}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
    "svtr-tiny": {"label": "Approximate SVTR-Tiny: a strided conv patch embedding, then a small Transformer (not the full SVTR local/global mixing blocks)", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 48, "k": 3, "pool": [2, 2]}, {"type": "conv2d", "out": 96, "k": 3, "pool": [2, 2]},
                     {"type": "conv2d", "out": 192, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "transformer", "dim": 192, "heads": 6, "layers": 3, "ff_mult": 4, "dropout": 0.1}], "head": {"dropout": 0.1}, "decoder": {"type": "ctc"}}},
    "mobilenetv3-small": {"label": "Approximate MobileNetV3-Small: a depthwise-separable backbone narrowing 16-24-40-48-96 (no squeeze-excite or h-swish, neither is in the catalog yet), a BiGRU, CTC", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 16, "k": 3, "pool": [2, 2]}, {"type": "dwsep", "out": 24, "k": 3, "pool": [2, 2]},
                     {"type": "dwsep", "out": 40, "k": 3, "pool": [2, 1]}, {"type": "dwsep", "out": 48, "k": 3, "pool": [2, 1]},
                     {"type": "dwsep", "out": 96, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "rnn", "hidden": 96, "layers": 1, "bidirectional": True, "cell": "gru"}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
    "vgg7-bigru": {"label": "Custom VGG-7: seven plain convolutions (channel schedule 64-128-256-256-512-512-512, no depthwise separation), a BiGRU, CTC", "spec": {
        "schema": 2, "input": {"height": 48, "width": "dynamic", "channels": 1},
        "backbone": [{"type": "conv2d", "out": 64, "k": 3, "pool": [2, 2]}, {"type": "conv2d", "out": 128, "k": 3, "pool": [2, 2]},
                     {"type": "conv2d", "out": 256, "k": 3, "pool": [2, 1]}, {"type": "conv2d", "out": 256, "k": 3, "pool": [1, 1]},
                     {"type": "conv2d", "out": 512, "k": 3, "pool": [2, 1]}, {"type": "conv2d", "out": 512, "k": 3, "pool": [1, 1]},
                     {"type": "conv2d", "out": 512, "k": 3, "pool": "collapse"}],
        "sequence": [{"type": "rnn", "hidden": 256, "layers": 1, "bidirectional": True, "cell": "gru"}], "head": {"dropout": 0.3}, "decoder": {"type": "ctc"}}},
}
DEFAULT = "crnn-base"


def normalize(spec):
    """A preset name or a schema-2 config dict -> the canonical config (defaults filled in). Raises ValueError, naming what is wrong."""
    if isinstance(spec, str):
        if spec not in PRESETS: raise ValueError(f"unknown schema-2 preset {spec!r}: {', '.join(PRESETS)}")
        spec = PRESETS[spec]["spec"]
    spec = copy.deepcopy(spec)
    if not isinstance(spec, dict) or spec.get("schema") != 2: raise ValueError('a schema-2 config needs "schema": 2')
    unknown = sorted(set(spec) - {"schema", "name", "input", "backbone", "sequence", "head", "decoder", "reg"})
    if unknown: raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    inp = spec.get("input", {})
    if inp.get("height", K.IMG_H) != K.IMG_H or inp.get("width", "dynamic") != "dynamic" or inp.get("channels", 1) != 1:
        raise ValueError(f"input must be height {K.IMG_H}, width \"dynamic\", channels 1")
    backbone, sequence = spec.get("backbone"), spec.get("sequence") or []
    if not isinstance(backbone, list) or not backbone: raise ValueError("backbone needs at least one block")
    h, w, c = K.IMG_H, K.IMG_W_REF, 1
    norm_backbone = []
    for i, b in enumerate(backbone, 1):
        kind = b.get("type")
        try: h, w, c, _ = K.validate_block(kind, b, h, w, c, last=(i == len(backbone)))
        except ValueError as e: raise ValueError(f"backbone block {i} ({kind}): {e}")
        norm_backbone.append(dict(b))
    if h != 1: raise ValueError("the backbone's last block must collapse height to 1")
    norm_sequence = []
    for i, b in enumerate(sequence, 1):
        kind = b.get("type")
        try: h, w, c, _ = K.validate_block(kind, b, h, w, c)
        except ValueError as e: raise ValueError(f"sequence block {i} ({kind}): {e}")
        norm_sequence.append(dict(b))
    if w < K.MIN_T: raise ValueError(f"the backbone's pools leave {w} time steps (at a reference width of {K.IMG_W_REF}); a reader needs at least {K.MIN_T}")
    head = spec.get("head") or {}
    dropout = head.get("dropout", 0.3)
    if not K.LIMITS["dropout"][0] <= dropout <= K.LIMITS["dropout"][1]: raise ValueError(f"head dropout must be {K.LIMITS['dropout']}")
    decoder = spec.get("decoder") or {"type": "ctc"}
    if decoder.get("type", "ctc") != "ctc": raise ValueError("decoder type must be ctc (an attention decoder is not supported yet)")
    out = {"schema": 2, "input": {"height": K.IMG_H, "width": "dynamic", "channels": 1}, "backbone": norm_backbone, "sequence": norm_sequence,
           "head": {"dropout": dropout}, "decoder": {"type": "ctc"}}
    n = describe(out)["params"]
    if n > K.LIMITS["params"]: raise ValueError(f"{n:,} parameters is over the {K.LIMITS['params']:,} limit")
    return out


def describe(spec, n_classes=129):
    """-> {params, time_steps, feature_size, blocks}, computed at the reference width (`htr.catalog.IMG_W_REF`); the real network
    accepts any width at train/inference time."""
    s = normalize(spec) if not _is_canonical(spec) else spec
    h, w, c, params = K.IMG_H, K.IMG_W_REF, 1, 0
    for i, b in enumerate(s["backbone"], 1):
        h, w, c, p = K.validate_block(b["type"], b, h, w, c, last=(i == len(s["backbone"]))); params += p
    for b in s["sequence"]:
        h, w, c, p = K.validate_block(b["type"], b, h, w, c); params += p
    params += c * n_classes + n_classes
    return {"params": params, "time_steps": w, "feature_size": c, "blocks": len(s["backbone"]) + len(s["sequence"])}


def _is_canonical(spec):
    return isinstance(spec, dict) and spec.get("schema") == 2 and "head" in spec and set(spec["head"]) == {"dropout"}


def spec_hash(spec):
    return hashlib.sha256(json.dumps(normalize(spec), sort_keys=True).encode()).hexdigest()[:12]


def flow(spec, n_classes=129):
    s = normalize(spec); h, w, c = K.IMG_H, K.IMG_W_REF, 1
    out = [{"label": "cell (gray)", "shape": [1, h, w]}]
    for i, b in enumerate(s["backbone"], 1):
        h, w, c, _ = K.validate_block(b["type"], b, h, w, c, last=(i == len(s["backbone"])))
        out.append({"label": f"{b['type']} to {c}" + (f", k{b['k']}" if "k" in b else ""), "shape": [c, h, w]})
    for b in s["sequence"]:
        h, w, c, _ = K.validate_block(b["type"], b, h, w, c)
        label = f"{b.get('layers', 1)}x {'bi' if b.get('bidirectional', True) else ''}{b.get('cell', 'lstm').upper()}" if b["type"] == "rnn" else f"{b.get('layers', 1)}x transformer, dim {c}, {b.get('heads')} heads"
        out.append({"label": label, "shape": [w, c]})
    out.append({"label": f"linear to {n_classes} classes", "shape": [w, n_classes]})
    return out


def summary(spec, n_classes=129):
    try: s = normalize(spec)
    except ValueError as e: return {"ok": False, "problems": [str(e)]}
    d = describe(s, n_classes)
    return {"ok": True, "spec": s, "hash": spec_hash(s), "params": d["params"], "time_steps": d["time_steps"], "flow": flow(s, n_classes),
            "backends": K.backends_for([(b["type"], b) for b in s["backbone"] + s["sequence"]])}
