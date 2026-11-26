"""Load a custom architecture: a plain Python module under `htr/custom/<name>.py` defining `build(n_classes) -> nn.Module`, whose
network follows the same contract every other HTRArchitecture does - `forward(images, input_lengths=None) -> (logits [T,B,V],
output_lengths [B] or None)`. `load` imports it by name only (never a path or uploaded source - the plan's "no code upload over
HTTP" rule: the file has to already be on disk, the same as any other code in this repo) and dry-runs it on two different widths
before handing it back, so a contract violation surfaces here, not three training epochs later.

A custom architecture's "config" is `{"custom": "<name>"}` - the whole point is that there is no declarative spec to normalize,
only a name pointing at code already on disk - so `normalize`/`describe`/`spec_hash`/`is_default`/`flow`/`summary` here mirror
`htr.spec`'s and `htr.presets`' interface (the same ones `work_phaseA/archnet.py`'s shim dispatches to) but work by loading and
dry-running the actual module rather than computing from JSON: `describe`'s params and `flow`'s shapes come from the built network
itself, and `spec_hash` hashes the module's own source bytes (so a code change changes the hash the same way a JSON edit would -
weights stay tied to the code they were trained against).

Needs torch."""
import re

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
IMG_H, IMG_W_REF = 48, 192


def available():
    """-> sorted module names under htr/custom/ (minus __init__), what `load` will accept."""
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "custom"
    return sorted(p.stem for p in d.glob("*.py") if p.stem != "__init__")


def _dry_run(net, n_classes, img_h=48):
    import torch
    net.eval()
    widths = (96, 64)
    x = torch.zeros(2, 1, img_h, max(widths))
    lengths = torch.tensor(list(widths), dtype=torch.long)
    with torch.no_grad():
        try:
            out = net(x, lengths)
        except Exception as e:
            raise ValueError(f"forward(images, input_lengths) raised: {type(e).__name__}: {e}") from e
    if not (isinstance(out, tuple) and len(out) == 2):
        raise ValueError(f"forward must return (logits, output_lengths); got {type(out)}")
    logits, out_lengths = out
    if logits.dim() != 3 or logits.shape[1] != 2 or logits.shape[2] != n_classes:
        raise ValueError(f"logits must be [T, batch, {n_classes}] with batch 2; got {tuple(logits.shape)}")
    T = logits.shape[0]
    if out_lengths is None: raise ValueError("output_lengths must not be None when input_lengths is given")
    if tuple(out_lengths.shape) != (2,): raise ValueError(f"output_lengths must be shape [2]; got {tuple(out_lengths.shape)}")
    ol = out_lengths.tolist()
    if any(not (0 <= v <= T) for v in ol): raise ValueError(f"output_lengths {ol} must each be between 0 and T={T}")
    if ol[0] < ol[1]: raise ValueError(f"a shorter input_length ({widths[1]}) produced a longer output_length ({ol[1]}) than a longer input ({widths[0]}) did ({ol[0]}): output_lengths must be monotonic in input_lengths")
    with torch.no_grad():
        logits_no_lengths = net(x, None)
    if not isinstance(logits_no_lengths, tuple) or logits_no_lengths[1] is not None:
        raise ValueError("forward(images, None) must return (logits, None) - output_lengths only when input_lengths is given")


def load(name, n_classes=129, img_h=48):
    """-> the built, dry-run-validated nn.Module. Raises ValueError (naming the problem) for an unknown module, a missing/broken
    `build`, or a forward that does not follow the HTRArchitecture contract."""
    if not NAME_RE.match(name): raise ValueError(f"custom architecture name {name!r} must be lowercase letters, digits and underscores, starting with a letter")
    if name not in available(): raise ValueError(f"no custom architecture {name!r} under htr/custom/ (have: {', '.join(available()) or 'none'})")
    import importlib
    mod = importlib.import_module(f"htr.custom.{name}")
    if not hasattr(mod, "build"): raise ValueError(f"htr/custom/{name}.py has no build(n_classes) function")
    try:
        net = mod.build(n_classes)
    except Exception as e:
        raise ValueError(f"htr.custom.{name}.build({n_classes}) raised: {type(e).__name__}: {e}") from e
    import torch.nn as nn
    if not isinstance(net, nn.Module): raise ValueError(f"htr.custom.{name}.build must return an nn.Module, got {type(net)}")
    _dry_run(net, n_classes, img_h)
    net.n_classes, net.custom_name = n_classes, name
    return net


def is_custom_config(config):
    return isinstance(config, dict) and set(config) == {"custom"} and isinstance(config.get("custom"), str)


def normalize(config):
    """{"custom": name} -> itself, after checking the name exists and is well-formed. Raises ValueError otherwise (same as
    `htr.spec.normalize`/`htr.presets.normalize` do for their own configs)."""
    if not is_custom_config(config): raise ValueError('a custom architecture config is {"custom": "<name>"}')
    name = config["custom"]
    if not NAME_RE.match(name): raise ValueError(f"custom architecture name {name!r} must be lowercase letters, digits and underscores, starting with a letter")
    if name not in available(): raise ValueError(f"no custom architecture {name!r} under htr/custom/ (have: {', '.join(available()) or 'none'})")
    return {"custom": name}


def describe(config, n_classes=129):
    """-> {params, time_steps, feature_size, blocks}, from actually building and dry-running the module at the reference width
    (`IMG_W_REF`) - there is no spec to compute this from."""
    s = normalize(config)
    net = load(s["custom"], n_classes, IMG_H)
    import torch
    with torch.no_grad():
        logits, _ = net(torch.zeros(1, 1, IMG_H, IMG_W_REF), torch.tensor([IMG_W_REF]))
    return {"params": sum(p.numel() for p in net.parameters()), "time_steps": logits.shape[0], "feature_size": logits.shape[2], "blocks": len(list(net.children()))}


def spec_hash(config):
    """Hash of the module's own source file, not of the config dict (there is nothing in `{"custom": name}` itself worth hashing -
    the code is the spec). Changes exactly when `htr/custom/<name>.py` changes."""
    import hashlib
    from pathlib import Path
    s = normalize(config)
    path = Path(__file__).resolve().parent.parent / "custom" / f"{s['custom']}.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def is_default(config):
    return False   # a custom architecture is never "the shipped reader"


def flow(config, n_classes=129):
    """The picture's journey through the network, one entry per direct child module, each a forward hook's output shape - a
    best-effort trace (it is arbitrary Python, not a declarative spec, so this is what can be observed, not computed)."""
    s = normalize(config)
    net = load(s["custom"], n_classes, IMG_H)
    import torch
    shapes = []
    hooks = []
    for name, child in net.named_children():
        def make(name):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                shapes.append({"label": name, "shape": list(t.shape) if hasattr(t, "shape") else None})
            return hook
        hooks.append(child.register_forward_hook(make(name)))
    try:
        with torch.no_grad(): net(torch.zeros(1, 1, IMG_H, IMG_W_REF), torch.tensor([IMG_W_REF]))
    finally:
        for h in hooks: h.remove()
    return [{"label": "cell (gray)", "shape": [1, IMG_H, IMG_W_REF]}] + shapes


def summary(config, n_classes=129):
    try: s = normalize(config)
    except ValueError as e: return {"ok": False, "problems": [str(e)]}
    d = describe(s, n_classes)
    return {"ok": True, "spec": s, "hash": spec_hash(s), "params": d["params"], "time_steps": d["time_steps"], "flow": flow(s, n_classes)}
