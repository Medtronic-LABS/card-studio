"""Torch side of the architecture lab: build a legacy (`htr.spec`) config into an `nn.Module`, and check one before training.

`compile` and `check` are deliberately not re-exported here under those names: `htr.nn.compile` and `htr.nn.check` are submodules
(`htr/nn/compile.py`, `htr/nn/check.py`), and binding a same-named function on the package would shadow the submodule attribute.
Use `from htr.nn.compile import compile` (or `htr.build`/`htr.check` at the top of the package) instead."""
from .compile import HTRArchitecture  # noqa: F401
