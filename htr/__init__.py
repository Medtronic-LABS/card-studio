"""Bengali handwritten-text-recognition engine: vocabulary, architecture lab, data, training, evaluation and export.

Pure-Python modules (text, vocab, spec, catalog, presets, legacy, eval.metrics, nn.custom's config-shape helpers) need no torch, so
the UI, a bundle check and the Frappe app can use them. `htr.nn.layers`/`compile`/`check`, `htr.train` and `htr.export` need torch
(and, for LiteRT, their own environment).

normalize/describe/spec_hash/is_default/flow/summary/PRESETS/tflite_support dispatch across all three architecture engines: a legacy
preset/spec (`htr.spec`, hash-stable - the shipped `crnn` weights depend on it never moving), a schema-2 preset/config (`htr.presets`,
the configurable layer-list engine), or `{"custom": name}` (`htr.nn.custom`, a Python module under `htr/custom/`). This module is the
direct successor to `work_phaseA/archnet.py` (deleted once every caller imported this one instead); every function here behaves
exactly as that shim's did for a legacy spec, hash or the shipped `crnn` weights."""
from . import spec as _S
from . import presets as _P
from .nn import custom as _CUST
from .vocab import Vocabulary  # noqa: F401

IMG_H, IMG_W, MIN_T = _S.IMG_H, _S.IMG_W, _S.MIN_T
LIMITS = _S.LIMITS                      # the legacy 4-key schema's own limits; a schema-2 config is checked against htr.catalog.LIMITS instead
PRESETS = {**_S.PRESETS, **_P.PRESETS}  # legacy presets (hash-stable) plus the lab's schema-2 presets, one dict for every caller that iterates it
DEFAULT = _S.DEFAULT


def _is_schema2(config):
    return config in _P.PRESETS if isinstance(config, str) else isinstance(config, dict) and config.get("schema") == 2


def _is_custom(config):
    return _CUST.is_custom_config(config)


def _engine(spec):
    if _is_custom(spec): return _CUST
    return _P if _is_schema2(spec) else _S


def normalize(spec):
    return _engine(spec).normalize(spec)


def describe(spec, n_classes=129):
    return _engine(spec).describe(spec, n_classes)


def spec_hash(spec):
    return _engine(spec).spec_hash(spec)


def is_default(spec):
    return (not _is_schema2(spec)) and (not _is_custom(spec)) and _S.is_default(spec)


def flow(spec, n_classes=129):
    return _engine(spec).flow(spec, n_classes)


def summary(spec, n_classes=129):
    return _engine(spec).summary(spec, n_classes)


def tflite_support(spec):
    """Whether `htr/export/rebuild_backend.py` can rebuild this spec. A schema-2 or custom config never is (that converter only
    knows the legacy conv/context/rnn/linear family, and a custom architecture's ONNX graph is not checked against it either way);
    see `htr/export/compat.py` for the full per-backend picture (ONNX always works)."""
    if _is_custom(spec): return {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks; a custom architecture is not checked against that family"}
    if _is_schema2(spec): return {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks, not a schema-2 config"}
    return _S.tflite_support(spec)


def build(config, n_classes=129):
    """-> an eval-ready-shaped `nn.Module` whose `forward(x)` takes one `[batch, 1, 48, 192]` cell and returns bare `[T, batch,
    classes]` logits (the pre-`htr` reader contract, unchanged) - built underneath by `htr.nn.compile` (the richer dynamic-length
    contract), unwrapped."""
    from .nn.compile import compile as _compile
    net = _compile(config, n_classes)
    lengths_forward = net.forward

    def forward(x):
        out = lengths_forward(x)
        return out[0] if isinstance(out, tuple) else out

    net.forward = forward
    return net


def check(config, n_classes=129, runs=20):
    """Same report shape as `work_phaseA/archnet.py`'s used to be: {ok, spec, hash, params, time_steps, torch_ms, onnx_ms, onnx_ok,
    onnx_bytes, onnx_max_diff, tflite, shipped_weights_fit, problems, params_built}."""
    from .nn.check import check as _check
    return _check(config, n_classes, runs)
