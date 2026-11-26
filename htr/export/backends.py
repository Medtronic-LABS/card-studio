"""Pick an export backend and run it. `auto` tries ai_edge_torch, then the legacy rebuild backend, then onnx2tf, keeping the first
one that is both compatible with the config (`compat.py`) and actually installed, with INT8 when asked and the backend supports it.
ai-edge-torch first because it is meant to eventually reach every architecture (schema-2 included) with INT8; until it (or onnx2tf)
is installed in this environment, `auto` on a legacy-shaped config falls through to `rebuild`, and a schema-2 config gets ONNX
only."""
from pathlib import Path

from . import ai_edge_backend as AE
from . import compat
from . import onnx2tf_backend as O2T
from . import onnx_backend as ONNX
from . import rebuild_backend as REBUILD

ORDER = ("ai_edge_torch", "rebuild", "onnx2tf")


class NoBackend(Exception):
    """No backend in `ORDER` (or the one named) can export this config; `.attempts` has what each one said."""
    def __init__(self, attempts): self.attempts = attempts; super().__init__("; ".join(f"{b}: {r}" for b, r in attempts.items()))


def export(model, config, img_h, img_w, onnx_path, tflite_path=None, backend="auto", samples=None, quantize=False):
    """Always writes the ONNX file at `onnx_path` (every backend needs it as a starting point, and it is what every pack ships
    regardless). backend: "auto", or one of ("ai_edge_torch", "rebuild", "onnx2tf") to require exactly that one. quantize: INT8
    where the chosen backend supports it (rebuild never does: it is float32 only). -> {onnx: {...}, tflite: {...} or None,
    backend: which one produced the tflite file (or None)}. Raises NoBackend when a specific backend was named and it cannot
    handle this config, or (backend="auto") when none of them can yet."""
    onnx_bytes = ONNX.export(model, img_h, img_w, onnx_path)
    par = ONNX.parity(model, onnx_path, img_h, img_w)
    report = {"onnx": {"bytes": len(onnx_bytes), "parity": par}, "tflite": None, "backend": None}
    if tflite_path is None: return report
    compatible = compat.check(config)
    order = ORDER if backend == "auto" else (backend,)
    attempts = {}
    for name in order:
        ok, reason = compatible.get(name, {"supported": False, "reason": "unknown backend"}).values()
        if not ok: attempts[name] = reason; continue
        try:
            if name == "rebuild":
                import sys
                studio_dir = str(Path(__file__).resolve().parent.parent.parent / "studio")
                if studio_dir not in sys.path: sys.path.insert(0, studio_dir)
                import tflite_export as T   # the venv-subprocess plumbing; stays in studio/ (see rebuild_backend.py's docstring)
                blob, tfl_report = T.export(onnx_bytes, samples=samples)
                Path(tflite_path).write_bytes(blob)
            elif name == "ai_edge_torch":
                tfl_report = AE.convert(model, img_h, img_w, tflite_path, quantize=quantize, calibration_loader=samples)
            else:
                tfl_report = O2T.convert(onnx_path, Path(tflite_path).parent, quantize=quantize, representative_data=samples)
        except (REBUILD.Unavailable, AE.Unavailable, O2T.Unavailable, ValueError, ImportError) as e:
            attempts[name] = str(e); continue
        report["tflite"] = tfl_report; report["backend"] = name
        return report
    raise NoBackend(attempts)
