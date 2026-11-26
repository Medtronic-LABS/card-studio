"""Torch side of the architecture lab: build a legacy (`htr.spec`) config into an `nn.Module`, and check one before training.

`compile` and `check` are deliberately not re-exported here under those names: `htr.nn.compile` and `htr.nn.check` are submodules
(`htr/nn/compile.py`, `htr/nn/check.py`), and binding a same-named function on the package would shadow the submodule attribute.
Use `from htr.nn.compile import compile` (or `htr.build`/`htr.check` at the top of the package) instead.

`HTRArchitecture` is loaded lazily (module `__getattr__`, PEP 562): `htr.nn.custom` is pure Python (no torch) and `htr/__init__.py`
imports it unconditionally so the Frappe app and a bundle check can list/describe architectures without torch installed; eagerly
importing `.compile` here (which imports `.layers`, which imports torch) would force torch on every `import htr.nn.custom` too,
since Python runs a package's `__init__.py` before any of its submodules."""


def __getattr__(name):
    if name == "HTRArchitecture":
        from .compile import HTRArchitecture
        return HTRArchitecture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
