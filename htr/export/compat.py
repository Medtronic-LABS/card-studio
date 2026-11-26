"""Which export backends a given architecture can even attempt, before spending time trying. Pure Python (no torch): a UI can grey
out an unsupported export target as soon as a config is edited."""
from .. import spec as S

BACKENDS = ("onnx", "rebuild", "ai_edge_torch", "onnx2tf")


def check(config):
    """config: a legacy preset/spec, a schema-2 preset/config, or {"custom": name}. -> {backend: {supported, reason}}. "supported"
    means both the config's layer types are the kind this backend can take at all, AND (for ai_edge_torch/onnx2tf) the package is
    actually installed in this environment - a backend being "supported" here is never a promise it has been exercised against the
    real package; `ai_edge_backend.py`/`onnx2tf_backend.py` say that plainly in their own docstrings.
    - "onnx" is always supported (every architecture this lab builds is a plain torch.onnx.export away from an ONNX file).
    - "rebuild" (htr/export/rebuild_backend.py): only a legacy-shaped network (conv/context/optional-LSTM-or-GRU/linear) - the
      family `read_onnx` knows how to parse back out of the ONNX graph. A schema-2 config is never supported here (dwsep, transformer
      and multi-stage sequences are not in that family), even when it happens to look similar; neither is a custom architecture (its
      ONNX graph is never checked against that family - it is refused on principle, not because it was tried and failed).
    - "ai_edge_torch": every legacy/schema-2 architecture, layer-type-wise, and a custom one optimistically (it traces whatever
      torch.export can trace, which is most models); refused only when the package itself is not installed.
    - "onnx2tf": refuses an rnn block (a plain LSTM/GRU ONNX subgraph does not convert cleanly through it) layer-type-wise for a
      legacy/schema-2 config; refused for a custom one too (its ops are not inspected, so this stays conservative), and when the
      package is not installed."""
    from ..nn.compile import _is_custom, _is_schema2
    from . import ai_edge_backend as AE
    from . import onnx2tf_backend as O2T
    out = {"onnx": {"supported": True, "reason": ""}}
    custom = _is_custom(config)
    if custom:
        out["rebuild"] = {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks; a custom architecture is refused on principle, not checked against that family"}
        has_rnn = True   # unknown ops: conservative
    elif _is_schema2(config):
        out["rebuild"] = {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks (htr.spec), not a schema-2 config"}
        has_rnn = any(b["type"] == "rnn" for b in __import__("htr.presets", fromlist=["x"]).normalize(config)["sequence"])
    else:
        try:
            S.normalize(config)
            out["rebuild"] = {"supported": True, "reason": ""}
        except ValueError as e:
            out["rebuild"] = {"supported": False, "reason": str(e)}
        has_rnn = True   # every legacy spec either has an rnn block or needs one unless context convs replace it; conservative
    if AE.available(): out["ai_edge_torch"] = {"supported": True, "reason": ""}
    else: out["ai_edge_torch"] = {"supported": False, "reason": "ai-edge-torch is not installed in this environment (its own venv: studio/.venv-litert)"}
    if not O2T.available(): out["onnx2tf"] = {"supported": False, "reason": "onnx2tf is not installed in this environment (its own venv: studio/.venv-onnx2tf)"}
    elif custom: out["onnx2tf"] = {"supported": False, "reason": "a custom architecture's ops are not inspected, so onnx2tf support is refused conservatively rather than guessed at"}
    elif has_rnn: out["onnx2tf"] = {"supported": False, "reason": "onnx2tf refuses a recurrent (LSTM/GRU) layer; use ai_edge_torch or rebuild for this architecture"}
    else: out["onnx2tf"] = {"supported": True, "reason": ""}
    return out
