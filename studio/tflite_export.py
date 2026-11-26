"""reader.onnx -> reader.tflite, for running the reader on a phone with TensorFlow Lite (LiteRT). Local only.

The reader is trained in PyTorch and exported to ONNX (trainer.py). The actual rebuild-and-convert logic (`read_onnx`, `build`,
`convert`) moved to `htr/export/rebuild_backend.py` (M1: the lab's export package); this file keeps the pieces that are its own -
running that conversion in its separate TensorFlow venv (`export`), and folding the result into a pack (`add_to_pack`,
`manifest_entry`) - and re-exports the moved names so nothing that imports `tflite_export.read_onnx` etc. needs to change.

TensorFlow is only needed for the conversion step. It lives in its own environment so that the training environment is left alone:
    python -m venv studio/.venv-tflite && studio/.venv-tflite/bin/pip install tensorflow onnx onnxruntime numpy
`export()` (used by trainer.py and by `add`) runs this file with that environment's Python; set STUDIO_TFLITE_PYTHON to use another one.

CLI: python tflite_export.py convert READER.onnx OUT.tflite [--samples CELLS.npy]     (needs TensorFlow: run it with the Python above)
     python tflite_export.py add PACK.cardpack                                        (writes the next pack version with reader.tflite added)
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from htr.export.rebuild_backend import MAX_DIFF, RANDOM_CELLS, Unavailable, build, convert, read_onnx   # noqa: F401  re-exported

VENV_PY = HERE / ".venv-tflite" / "bin" / "python"


# ---------------------------------------------------------------- runs anywhere
def python():
    """A Python that can import tensorflow, onnx and onnxruntime: STUDIO_TFLITE_PYTHON, then studio/.venv-tflite."""
    for p in (os.environ.get("STUDIO_TFLITE_PYTHON"), VENV_PY):
        if p and Path(p).exists(): return str(p)
    raise Unavailable(f"no TensorFlow environment: python -m venv {VENV_PY.parent.parent} && {VENV_PY.parent}/pip install tensorflow onnx onnxruntime numpy")


def export(onnx_bytes, samples=None, timeout=900):
    """reader.onnx bytes -> (reader.tflite bytes, report), by running `convert` (htr/export/rebuild_backend.py) in the TensorFlow
    environment. samples: optional numpy array [N, H, W] of real cells, scaled as the reader takes them. Raises Unavailable, or
    ValueError with the reason."""
    py = python()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "reader.onnx").write_bytes(onnx_bytes); cmd = [py, str(Path(__file__).resolve()), "convert", str(d / "reader.onnx"), str(d / "reader.tflite")]
        if samples is not None:
            import numpy as np
            np.save(d / "cells.npy", samples); cmd += ["--samples", str(d / "cells.npy")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": ""})
        last = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode != 0 or not (d / "reader.tflite").exists():
            raise ValueError("TFLite conversion failed: " + (last if last.startswith("ERROR") else (r.stderr.strip().splitlines() or [last or "no output"])[-1])[:400])
        return (d / "reader.tflite").read_bytes(), json.loads(last)


def manifest_entry(report, reader):
    """What a pack's manifest says about reader.tflite (reader = the manifest's `reader` block, for the scaling it shares with the ONNX model)."""
    i = reader["input"]
    return {"file": "reader.tflite", "input": {"name": report["input"]["name"], "layout": "NHWC", "height": i["height"], "width": i["width"], "mean": i["mean"], "std": i["std"]},
            "output": {"name": report["output"]["name"], "layout": "TBV"}, "dtype": report["dtype"], "ops": report["ops"], "parity": report["parity"]}


def add_to_pack(pack_path, out_dir=None):
    """Write the next version of a pack with reader.tflite added (everything else, including held-out metrics, is unchanged: same weights)."""
    sys.path.insert(0, str(HERE.parent / "cardpack")); import cardpack as cp
    pack_path = Path(pack_path); m = cp.read(pack_path); man = json.loads(m["manifest.json"])
    blob, report = export(m["reader.onnx"])
    m["reader.tflite"] = blob; man["reader"]["tflite"] = manifest_entry(report, man["reader"])
    out_dir = Path(out_dir) if out_dir else pack_path.parent
    have = [int(p.stem.split("_v")[1]) for p in out_dir.glob(f"{man['id']}_v*.cardpack") if p.stem.split("_v")[1].isdigit()]
    man["version"] = max(have + [man["version"]]) + 1
    man["files"] = {n: cp.sha256(m[n]) for n in cp.hashed(m)}; m["manifest.json"] = json.dumps(man, indent=1).encode()
    out = out_dir / f"{man['id']}_v{man['version']}.cardpack"; cp.build(m, out)
    return {"pack": str(out), "version": man["version"], "errors": cp.validate(out), "tflite": report}


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "convert":
        try:
            print(json.dumps(convert(a[1], a[2], a[a.index("--samples") + 1] if "--samples" in a else None)))
        except Exception as e:
            print(f"ERROR {type(e).__name__}: {e}"); sys.exit(1)
    elif len(a) == 2 and a[0] == "add":
        try: print(json.dumps(add_to_pack(a[1]), indent=1))
        except (Unavailable, ValueError) as e: sys.exit(str(e))
    else: sys.exit(__doc__.split("CLI:")[1].strip())
