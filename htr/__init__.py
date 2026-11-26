"""Bengali handwritten-text-recognition engine: vocabulary, architecture lab, data, training, evaluation and export.

Pure-Python modules (text, vocab, spec, catalog, presets, legacy, eval.metrics) need no torch, so the UI, a bundle check and the Frappe
app can use them. `htr.nn`, `htr.train` and `htr.export` need torch (and, for LiteRT, their own environment).
"""
from .spec import normalize, describe, flow, summary, spec_hash, is_default, PRESETS, LIMITS, DEFAULT, IMG_H  # noqa: F401
from .vocab import Vocabulary  # noqa: F401


def build(config, n_classes=129):
    """config -> nn.Module (torch). Kept here so `import htr; htr.build(...)` mirrors the old archnet API."""
    from .nn.compile import compile as _compile
    return _compile(config, n_classes)


def check(config, n_classes=129, runs=20):
    from .nn.check import check as _check
    return _check(config, n_classes, runs)
