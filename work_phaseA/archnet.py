"""The reader's architecture as a spec you can change. This module is now a thin shim over `htr.spec`/`htr.presets`/`htr.nn.compile`/
`htr.nn.check` (the Bengali HTR lab's architecture engine, M1-M2): every legacy spec, hash, parameter count and the shipped `crnn`
weights behave exactly as they always did (htr.spec is a verbatim port, tested byte-for-byte against what this file used to compute),
and `PRESETS` now also offers the lab's schema-2 presets (crnn-tiny/base/large, cnn-transformer, svtr-tiny) to every caller that
already iterates `archnet.PRESETS` - the training settings UI, `pretrained.py`, `studio/server.py`'s architecture check endpoint.

`build()` wraps `htr.nn.compile`'s network (which returns `(logits, output_lengths)`, the lab's richer dynamic-length contract) back
to the bare `logits` tensor every existing caller here expects (`common.CRNN`, `work_phaseA/train.py`, `studio/synthexp.py`): this
file's own reader contract - one gray cell in, `[T, batch, classes]` logits out, nothing else - has not changed.

Planned (M3): this file is deleted once `studio/trainer.py` and the Frappe bridge call `htr.nn.compile`/`htr.train.loop` directly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import htr.catalog as _K
import htr.presets as _P
import htr.spec as _S
import htr.nn.custom as _CUST
from htr.nn.compile import _is_custom, _is_schema2, compile as _compile
from htr.nn.check import check as _check

IMG_H, IMG_W, MIN_T = _S.IMG_H, _S.IMG_W, _S.MIN_T
LIMITS = _S.LIMITS                      # the legacy 4-key schema's own limits; a schema-2 config is checked by htr.catalog.LIMITS instead
PRESETS = {**_S.PRESETS, **_P.PRESETS}   # legacy presets (hash-stable) plus the lab's schema-2 presets, one list for every caller
DEFAULT = _S.DEFAULT


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
    """Whether `studio/tflite_export.py` (now `htr/export/rebuild_backend.py`) can rebuild this spec. A schema-2 or custom config
    never is (that converter only knows the legacy conv/context/rnn/linear family, and a custom architecture's ONNX graph is not
    checked against it either way); see `htr/export/compat.py` for the full per-backend picture (ONNX always works)."""
    if _is_custom(spec): return {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks; a custom architecture is not checked against that family"}
    if _is_schema2(spec): return {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks, not a schema-2 config"}
    return _S.tflite_support(spec)


def build(spec, n_classes=129):
    """-> an eval-ready-shaped `nn.Module` whose `forward(x)` takes one `[batch, 1, 48, 192]` cell and returns bare `[T, batch,
    classes]` logits (this file's own contract, unchanged) - built underneath by `htr.nn.compile` (dynamic-length contract), unwrapped."""
    net = _compile(spec, n_classes)
    lengths_forward = net.forward

    def forward(x):
        out = lengths_forward(x)
        return out[0] if isinstance(out, tuple) else out

    net.forward = forward
    return net


def check(spec, n_classes=129, runs=20):
    """Same report shape as before: {ok, spec, hash, params, time_steps, torch_ms, onnx_ms, onnx_ok, onnx_bytes, onnx_max_diff, tflite,
    shipped_weights_fit, problems, params_built}. Delegates to `htr.nn.check` (ported from this function originally)."""
    return _check(spec, n_classes, runs)
